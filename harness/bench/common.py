"""Shared pieces of the benchmarks: RCON, fake humans, the agent under test, and tailing its run log."""
import json, os, socket, struct, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INBOX = os.path.join(ROOT, "runs", "inbox.jsonl")


class Rcon:
    def __init__(self, host, port, password):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.req = 0
        if self._send(3, password)[0] == -1:
            sys.exit("RCON auth failed")

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:   # the other end closed (a relay with nothing behind it): no endless spin
                raise ConnectionError("RCON connection closed")
            buf += chunk
        return buf

    def _send(self, kind, body):
        self.req += 1
        data = struct.pack("<ii", self.req, kind) + body.encode() + b"\x00\x00"
        self.sock.sendall(struct.pack("<i", len(data)) + data)
        (length,) = struct.unpack("<i", self._read(4))
        rid, _ = struct.unpack("<ii", self._read(8))
        return rid, self._read(length - 8)[:-2].decode("utf-8", "replace")

    def cmd(self, command):
        return self._send(2, command)[1]


def server_rcon(server_dir):
    """RCON from the server's own server.properties when it is on this machine, else from RCON_HOST / RCON_PORT /
    RCON_PASSWORD (environment or .env), e.g. through a tunnel when the agents run on another machine."""
    path = os.path.join(server_dir, "server.properties")
    if os.path.exists(path):
        props = {}
        with open(path, encoding="latin-1") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.rstrip("\n").split("=", 1)
                    props[k] = v
        return Rcon(props.get("server-ip") or "127.0.0.1", int(props["rcon.port"]), props["rcon.password"])
    env = {}
    dotenv = os.path.join(ROOT, ".env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            env.update(line.strip().split("=", 1) for line in f if "=" in line and not line.startswith("#"))
    env.update(os.environ)   # the environment wins over .env (a test pointed at another server)
    return Rcon(env["RCON_HOST"], int(env["RCON_PORT"]), env["RCON_PASSWORD"])


def start_tester(name):
    """A fake human that stands still; write lines to .stdin to make it chat."""
    return subprocess.Popen(["node", os.path.join(HERE, "tester.js"), name], stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, text=True,
                            env={**os.environ, "NODE_PATH": os.path.join(ROOT, "body", "node_modules")})


class AgentUnderTest:
    def __init__(self, name, owners, vision, harness, extra=(), max_steps=40, memory_dir=None, host=None, port=None):
        if host:
            extra = [*extra, "--host", host]
        if port:
            extra = [*extra, "--port", str(port)]
        self.name = name
        self.proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(ROOT, "agent.py"), "--name", name, "--vision", vision,
             "--harness", harness, "--quiet", "--owners", ",".join(owners), "--max-steps", str(max_steps),
             "--memory-dir", memory_dir or os.path.join(HERE, "results", "memory-scratch"), *extra],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.log_path = None
        threading.Thread(target=self._pump, daemon=True).start()
        for _ in range(180):
            if self.log_path:
                break
            time.sleep(0.5)
        else:
            sys.exit("agent did not start")
        self.offset = os.path.getsize(self.log_path)

    def _pump(self):
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        out = open(os.path.join(ROOT, "runs", f"{self.name}.bench.out"), "w")  # the agent's console, for debugging
        for line in self.proc.stdout:
            out.write(line)
            out.flush()
            if line.startswith("log:"):
                self.log_path = line.split(":", 1)[1].strip()

    def order(self, sender, text):
        with open(INBOX, "a") as f:
            f.write(json.dumps({"t": time.time(), "to": [self.name], "from": sender, "text": text}) + "\n")

    def mark(self):
        """Start collecting records from here on."""
        self.offset = os.path.getsize(self.log_path)

    def records(self):
        """All records written since the last mark()."""
        with open(self.log_path) as f:
            f.seek(self.offset)
            return [json.loads(line) for line in f if line.strip()]

    def wait_finish(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            recs = self.records()
            if any(r["kind"] == "instruction" and r["event"] == "finish" for r in recs):
                return recs
            time.sleep(1)
        return self.records()

    def stop(self):
        self.proc.terminate()
        self.proc.wait(10)
