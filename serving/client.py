#!/usr/bin/env python3
"""Minimal DJev client: send one request, or hammer the server with concurrent requests.

  python client.py                                  # built-in example, prints the answers
  python client.py --input req.json                 # your own {"states": [...]} file
  python client.py --concurrency 16 --requests 64   # load test: latency + throughput
  python client.py --images cat.jpg https://…/x.png  # visual example: what is in the image?
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

EXAMPLE = {"states": [
    {"id": "smart_home:0",
     "state": "Home log: user wants the dining room fan set to on; access=yes; occupants=2; clock=20:00.",
     "questions": {
         "device": {"type": "choice",
                    "instructions": "Select exactly the room and device named in the request. Ignore authorization and occupancy for this question.",
                    "criteria": {"unit_0": "balcony speaker", "unit_1": "dining room fan"}},
         "execute": {"type": "boolean",
                     "instructions": "Is execution allowed? Rule: authorization is required for every command; additionally any heater command requires at least one person present. Other devices do not require occupancy."},
         "risk": {"type": "score",
                  "instructions": "Classify the control request using these mutually exclusive rules.",
                  "criteria": ["Authorized, and either not a heater or at least one person present",
                               "Authorized heater request with zero people present",
                               "Not authorized, regardless of device or occupancy"]}}},
    {"id": "maze:0",
     "state": "5x5 local view, '#' wall, '.' open, 'A' agent at the centre:\n#####\n#..A#\n#.#.#\n#...#\n#####",
     "questions": {
         "north_safe": {"type": "boolean", "instructions": "Is the cell directly north of A open?"},
         "west_safe": {"type": "boolean", "instructions": "Is the cell directly west of A open?"},
         "move": {"type": "choice", "instructions": "Which single move keeps A on an open cell?",
                  "criteria": {"north": "move one cell up", "south": "move one cell down",
                               "east": "move one cell right", "west": "move one cell left"}}}}]}


AUTH = None  # "user:password" for a tunnel protected by basic auth (see --auth)


VISUAL_QUESTIONS = {
    "content": {"type": "choice", "instructions": "What does Image 1 mainly show?",
                "criteria": {"animal": "an animal", "people": "one or more people", "scene": "a landscape, street or building",
                             "object": "a single object or product", "graphic": "a chart, diagram, drawing or synthetic shapes",
                             "text": "mostly text or a screenshot"}},
    "photo": {"type": "boolean", "instructions": "Is Image 1 a photograph (not a drawing, chart or rendering)?"},
    "complexity": {"type": "score", "instructions": "How visually complex is Image 1?",
                   "criteria": ["very simple", "simple", "moderate", "complex", "very complex"]},
}


def image_source(path_or_url):
    """Local file -> data URI; URL passes through."""
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    import base64
    import mimetypes
    mime = mimetypes.guess_type(path_or_url)[0] or "image/png"
    with open(path_or_url, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def visual_example(images):
    return {"states": [{"id": "visual:0", "state": "Look at the attached image(s) and judge each question.",
                        "images": [image_source(i) for i in images], "questions": VISUAL_QUESTIONS}]}


def post(url, payload, timeout=300):
    headers = {"Content-Type": "application/json"}
    if AUTH and ":" in AUTH:
        import base64
        headers["Authorization"] = "Basic " + base64.b64encode(AUTH.encode()).decode()
    elif AUTH:
        headers["Authorization"] = "Bearer " + AUTH
    req = urllib.request.Request(url + "/api/evaluate", data=json.dumps(payload).encode(), headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    with opener.open(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.perf_counter() - t0


def show(body):
    for s in body["states"]:
        where = f", worker {s['worker']}" if "worker" in s else ""
        print(f"== {s['id']}  (prompt {s.get('prompt_tokens')} tokens, {s.get('images', 0)} image(s){where})")
        for qid, a in s["answers"].items():
            probs = ", ".join(f"{k}={v:.3f}" for k, v in a["probabilities"].items())
            val = a.get("choice", a.get("p_true", a.get("score")))
            print(f"   {qid:14s} [{a['type']:7s}] value={val}  mass={a['candidate_mass']:.3f}  {probs}")
    ex = body["execution"]
    print(f"-- {ex['states']} states, {ex['questions']} questions, {ex['server_evaluation_seconds']*1000:.0f} ms server time, "
          f"backend={ex.get('backend', 'hf')}, readout={ex['readout']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--input", help="JSON file with {'states': [...]}")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--auth", "--api-key", dest="auth",
                   help="API key (sent as Bearer), or user:password for basic auth; default: $DJEV_API_KEY")
    p.add_argument("--images", nargs="+", metavar="FILE_OR_URL",
                   help="send the built-in visual example about these image(s) (local files are inlined as data URIs)")
    a = p.parse_args()
    global AUTH
    import os
    AUTH = a.auth or os.environ.get("DJEV_API_KEY")
    payload = json.load(open(a.input)) if a.input else (visual_example(a.images) if a.images else EXAMPLE)

    if a.concurrency == 1 and a.requests == 1:
        body, dt = post(a.url, payload)
        show(body)
        print(f"-- round trip {dt*1000:.0f} ms")
        return

    lat, errors, lock = [], [], threading.Lock()
    todo = list(range(a.requests))

    def run():
        while True:
            with lock:
                if not todo:
                    return
                todo.pop()
            try:
                _, dt = post(a.url, payload)
                with lock:
                    lat.append(dt)
            except Exception as e:
                with lock:
                    errors.append(str(e))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=run) for _ in range(a.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    n_states = len(payload["states"])
    n_q = sum(len(s["questions"]) for s in payload["states"])
    lat.sort()
    print(f"{len(lat)} ok, {len(errors)} errors, concurrency {a.concurrency}, wall {wall:.2f}s")
    if lat:
        print(f"latency ms: p50 {statistics.median(lat)*1000:.0f}  p90 {lat[int(0.9*(len(lat)-1))]*1000:.0f}  "
              f"max {lat[-1]*1000:.0f}")
        print(f"throughput: {len(lat)/wall:.1f} req/s = {len(lat)*n_states/wall:.1f} states/s = {len(lat)*n_q/wall:.1f} questions/s")
    for e in errors[:3]:
        print("error:", e[:300])


if __name__ == "__main__":
    main()
