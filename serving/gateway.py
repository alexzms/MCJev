#!/usr/bin/env python3
"""One address for many DJev-serve fronts: agents send /api/evaluate here, it goes to the least busy healthy one.

  python serving/gateway.py --listen 127.0.0.1:8780 --backends-file serving/backends.example.txt
  agents: JEV_URL=http://127.0.0.1:8780 (JEV_AUTH as before: the key passes through untouched)

Backends are host:port, one per line in --backends-file (# comments; re-read when it changes, so adding a node is
one line), each a DJev-serve front (server_vllm.py) or a relay to one. Every 2 s the gateway sends each a tiny
evaluate (one yes/no question, with the key from JEV_AUTH in the environment or in the repo's .env) and uses only
those that answer it. A request is read whole (they are a few KB), sent to the healthy backend with the fewest
requests in flight, and if that one fails before answering (connection refused, closed without a byte, e.g. a relay
whose front is gone), sent to the next; the answer is passed back as it comes. GET /gateway/status answers here:
each backend's state, in-flight, counts and latency. Standard library only; run one per machine that runs agents.
"""
import argparse, asyncio, base64, json, os, time

HEALTH_EVERY = 2.0
MAX_BODY = 64 << 20


def load_auth(root):
    auth = os.environ.get("JEV_AUTH")
    env = os.path.join(root, ".env")
    if not auth and os.path.exists(env):
        for line in open(env):
            if line.startswith("JEV_AUTH="):
                auth = line.split("=", 1)[1].strip()
    return "Basic " + base64.b64encode(auth.encode()).decode() if auth else None


class Backend:
    def __init__(self, addr):
        self.addr = addr
        self.host, port = addr.rsplit(":", 1)
        self.port = int(port)
        self.healthy = False
        self.why = "not checked yet"
        self.inflight = self.done = self.failed = 0
        self.ms = []                       # recent latencies

    def status(self):
        ms = sorted(self.ms[-200:])
        return {"healthy": self.healthy, "why": self.why, "inflight": self.inflight, "done": self.done,
                "failed": self.failed, "median_ms": round(ms[len(ms) // 2], 1) if ms else None}


async def http(host, port, head, body=b"", timeout=30.0):
    """Send one request, return the whole response (by its Content-Length, else up to EOF); b"" if the other end
    closed without answering."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port, limit=MAX_BODY), 3)

    async def answer():
        try:
            h = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError as e:
            return e.partial
        n = None
        for line in h.split(b"\r\n")[1:]:
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":", 1)[1])
        return h + (await reader.readexactly(n) if n is not None else await reader.read(-1))

    try:
        writer.write(head + body)
        await writer.drain()
        return await asyncio.wait_for(answer(), timeout)
    finally:
        writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8780")
    ap.add_argument("--backends-file", required=True)
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for a backend's answer")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    auth = load_auth(root)
    backends, mtime = {}, [None]

    def log(msg):
        print(time.strftime("%F %T"), msg, flush=True)

    def reload():
        try:
            m = os.path.getmtime(args.backends_file)
        except OSError:
            return
        if m == mtime[0]:
            return
        mtime[0] = m
        want = [l.split("#", 1)[0].strip() for l in open(args.backends_file)]
        want = [w for w in want if w]
        for a in list(backends):
            if a not in want:
                del backends[a]
        for a in want:
            backends.setdefault(a, Backend(a))
        log("backends: " + ", ".join(want))

    probe = json.dumps({"states": [{"id": "health", "state": "The gateway checks that you answer.",
                                    "questions": {"q": {"type": "boolean", "instructions": "Are you answering?"}}}]}).encode()

    async def check(b):
        # a real (tiny) evaluate, not GET /api/health: a front's health flag can be wrong while it answers fine
        head = (f"POST /api/evaluate HTTP/1.1\r\nHost: {b.addr}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(probe)}\r\nConnection: close\r\n"
                + (f"Authorization: {auth}\r\n" if auth else "") + "\r\n").encode()
        try:
            resp = await http(b.host, b.port, head, probe, timeout=5)
            body = resp.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in resp else b""
            ok = resp.startswith(b"HTTP/1.1 200") and bool(json.loads(body or b"{}").get("states"))
            why = "answers" if ok else (resp.split(b"\r\n", 1)[0].decode(errors="replace") or "no answer")
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as e:
            ok, why = False, f"{type(e).__name__}: {e}"[:80]
        if ok != b.healthy:
            log(f"{b.addr}: {'healthy' if ok else 'down (' + why + ')'}")
        b.healthy, b.why = ok, why

    async def health_loop():
        while True:
            reload()
            await asyncio.gather(*(check(b) for b in list(backends.values())))
            await asyncio.sleep(HEALTH_EVERY)

    async def read_request(reader):
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        n = 0
        for line in head.split(b"\r\n")[1:]:
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":", 1)[1])
        if n > MAX_BODY:
            raise ValueError("body too big")
        body = await asyncio.wait_for(reader.readexactly(n), 30) if n else b""
        lines = [l for l in head.rstrip(b"\r\n").split(b"\r\n") if not l.lower().startswith(b"connection:")]
        return b"\r\n".join(lines + [b"Connection: close"]) + b"\r\n\r\n", body, head.split(b" ", 2)[:2]

    def reply(writer, code, obj):
        body = json.dumps(obj, indent=1).encode()
        writer.write(f"HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                     f"Connection: close\r\n\r\n".encode() + body)

    async def handle(reader, writer):
        try:
            head, body, (method, path) = await read_request(reader)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError, ConnectionError):
            writer.close()
            return
        try:
            if path == b"/gateway/status":
                reply(writer, "200 OK", {a: b.status() for a, b in backends.items()})
                return
            tried = []
            while True:
                live = [b for b in backends.values() if b.healthy and b not in tried]
                if not live:
                    reply(writer, "503 Service Unavailable", {"error": "no healthy DJev-serve backend",
                                                               "tried": [b.addr for b in tried]})
                    return
                b = min(live, key=lambda x: (x.inflight, x.done))
                tried.append(b)
                b.inflight += 1
                t0 = time.perf_counter()
                try:
                    resp = await http(b.host, b.port, head, body, args.timeout)
                except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError) as e:
                    resp, err = b"", e
                else:
                    err = None
                finally:
                    b.inflight -= 1
                if resp:
                    b.done += 1
                    b.ms = b.ms[-500:] + [(time.perf_counter() - t0) * 1000]
                    writer.write(resp)
                    return
                b.failed += 1
                b.healthy, b.why = False, f"request failed: {err or 'closed without answering'}"[:80]
                log(f"{b.addr}: {b.why}; trying another")
        finally:
            try:
                await writer.drain()
            except ConnectionError:
                pass
            writer.close()

    async def serve():
        reload()
        lhost, lport = args.listen.rsplit(":", 1)
        server = await asyncio.start_server(handle, lhost, int(lport), limit=MAX_BODY)
        log(f"gateway {args.listen}, auth for health checks: {'yes' if auth else 'NO (JEV_AUTH unset)'}")
        asyncio.create_task(health_loop())
        async with server:
            await server.serve_forever()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
