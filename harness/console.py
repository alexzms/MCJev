#!/usr/bin/env python3
"""Local web console: every agent run, the exact context Jev read, and its decisions, live.

  ./console.py               http://127.0.0.1:8770
  ./console.py --port 9000

It only tails runs/*.jsonl (written by agent.py), so it can start before or after the agents,
shows any number of them, and replays old runs the same way. Commands typed into it are appended
to runs/inbox.jsonl, which every running agent reads.
"""
import argparse, json, os, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PAGE = os.path.join(HERE, "console.html")
LIVE_SECONDS = 15  # agents write at least a heartbeat every 5 s while running
RUN_NAME = re.compile(r"^[\w.-]+\.jsonl$")
INBOX = os.path.join(RUNS, "inbox.jsonl")


def first_record(path):
    with open(path, encoding="utf-8") as f:
        line = f.readline()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def last_record(path, size):
    with open(path, "rb") as f:
        f.seek(max(0, size - 65536))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def list_runs():
    out = []
    for name in os.listdir(RUNS) if os.path.isdir(RUNS) else []:
        if not RUN_NAME.match(name) or name == "inbox.jsonl":
            continue
        path = os.path.join(RUNS, name)
        st = os.stat(path)
        head, tail = first_record(path), last_record(path, st.st_size)
        out.append({"file": name, "agent": head.get("agent", name), "started": head.get("t"),
                    "updated": st.st_mtime, "bytes": st.st_size,
                    "live": tail.get("kind") != "end" and time.time() - st.st_mtime < LIVE_SECONDS})
    return sorted(out, key=lambda r: (not r["live"], -r["updated"]))


def overview(run):
    """Latest state of one run from the tail of its log: current instruction and last step."""
    path = os.path.join(RUNS, run["file"])
    with open(path, "rb") as f:
        f.seek(max(0, run["bytes"] - 400_000))
        lines = f.read().splitlines()
    step, working, image = None, None, None
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r["kind"] == "step":
            step, working = r, r["instruction"]
        elif r["kind"] == "instruction":
            working = r["text"] if r["event"] == "start" else None
        if r.get("images"):
            image = r["images"][0]
    if step:
        step = {k: step.get(k) for k in ("t", "step", "action", "done", "effect", "seconds", "instruction")} | {
            "p": step["answers"]["action"]["probabilities"].get(step["action"])}
    return {**run, "instruction": working, "last_step": step, "image": image}


def add_command(to, sender, text):
    os.makedirs(RUNS, exist_ok=True)
    with open(INBOX, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "to": to, "from": sender, "text": text}, ensure_ascii=False) + "\n")


def read_records(name, offset):
    """Complete lines from byte offset on -> (records, next offset)."""
    path = os.path.join(RUNS, name)
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b"\n") + 1
    records = []
    for line in chunk[:end].splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, offset + end


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            with open(PAGE, "rb") as f:
                return self.send(200, f.read(), "text/html")
        if url.path == "/api/runs":
            return self.send(200, list_runs())
        if url.path == "/api/overview":
            return self.send(200, sorted((overview(r) for r in list_runs() if r["live"]), key=lambda r: r["agent"]))
        if url.path == "/api/records":
            name = q.get("file", "")
            if not RUN_NAME.match(name) or not os.path.exists(os.path.join(RUNS, name)):
                return self.send(404, {"error": "no such run"})
            records, nxt = read_records(name, int(q.get("from", 0)))
            return self.send(200, {"records": records, "next": nxt})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/command":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            text, sender, to = body["text"].strip(), body["from"].strip(), body["to"]
        except (ValueError, KeyError, AttributeError):
            return self.send(400, {"error": "need {to: '*' | [names], from, text}"})
        if not text or len(text) > 200 or not re.fullmatch(r"\w{1,16}", sender):
            return self.send(400, {"error": "text 1-200 chars, from a player name"})
        if to != "*" and not (isinstance(to, list) and all(isinstance(n, str) for n in to)):
            return self.send(400, {"error": "to must be '*' or a list of agent names"})
        add_command(to, sender, text)
        self.send(200, {"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Jev console on http://{args.host}:{args.port}  (runs: {RUNS})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
