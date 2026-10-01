#!/usr/bin/env python3
"""How fast does a djev pool answer N fighting text bots at once?  N closed-loop clients, each a bot that sends its
next step as soon as the last one is answered, on a real p_uhc_pro_v10 step (probes/data/uhc_pro_step85.json, ~4.5k
characters, 16 options). Like real bots: each client has its own name (Jev1..JevN, so bots share almost no prompt
prefix) and each step changes only what changes in a match - the numbers from the "Now:" line on; the rules above it
stay, so a bot's own previous step can serve as its cached prefix.

    python probes/stress_text.py --url http://127.0.0.1:8765 --bots 1 2 4 8 12 16 20 24 --seconds 12

Per level: steps/s, round trip median/p90/p99 (a bot's step waits that long for Jev), prompt tokens served from the
prefix cache (mean), errors.
"""
import argparse
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--request", default=os.path.join(HERE, "data", "uhc_pro_step85.json"))
    ap.add_argument("--bots", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20, 24])
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--warm", type=float, default=2)
    a = ap.parse_args()
    base = json.load(open(a.request))["states"][0]
    head, sep, tail = base["state"].partition("\nNow:")
    assert sep, "the state has no 'Now:' line"
    url = a.url.rstrip("/") + "/api/evaluate"

    def body(bot):
        dyn = re.sub(r"\d", lambda m: str(random.randint(0, 9)), sep + tail)
        st = dict(base, state=head.replace("You are JevPro", f"You are Jev{bot}", 1) + dyn)
        return json.dumps({"states": [st]}).encode()

    print(f"{url} | {len(base['state'])} chars, {len(base['questions']['action']['criteria'])} options | "
          f"{a.seconds:.0f} s per level after {a.warm:.0f} s warm-up", flush=True)
    for n in a.bots:
        times, cached, errs, stages = [], [], [0], []
        lock = threading.Lock()
        t_warm = time.time() + a.warm
        stop = t_warm + a.seconds

        def client(bot):
            while time.time() < stop:
                data = body(bot)
                t0 = time.perf_counter()
                try:
                    r = json.load(urllib.request.urlopen(urllib.request.Request(
                        url, data=data, headers={"Content-Type": "application/json"}), timeout=60))
                    ms = (time.perf_counter() - t0) * 1000
                    if time.time() >= t_warm:
                        tm = (r.get("execution") or {}).get("timing_ms") or []
                        with lock:
                            times.append(ms)
                            cached.append(r["states"][0].get("cached_prompt_tokens", 0))
                            if tm:   # an engine started with DJEV_TIMING=1: where the round trip went
                                st = {k: v for k, v in tm[0].items() if k != "read"}
                                st.update({f"read.{k}": v for k, v in (tm[0].get("read") or {}).items()})
                                st["client"] = ms
                                stages.append(st)
                except Exception:
                    with lock:
                        errs[0] += 1
                    time.sleep(0.05)

        th = [threading.Thread(target=client, args=(i + 1,)) for i in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        q = sorted(times)
        pick = lambda f: q[min(len(q) - 1, int(f * len(q)))] if q else float("nan")
        print(f"{n:3d} bots: {len(q) / a.seconds:6.1f} steps/s | median {pick(0.5):6.1f} ms  p90 {pick(0.9):6.1f}  "
              f"p99 {pick(0.99):6.1f} | cached {statistics.mean(cached) if cached else 0:5.0f} tokens | "
              f"{errs[0]} errors", flush=True)
        if stages:   # a batched read's split has a "batch" size too
            keys = list(dict.fromkeys(k for s in stages for k in s))
            print("        median ms: " + "  ".join(f"{k} {statistics.median(s[k] for s in stages if k in s):.2f}"
                                               for k in keys), flush=True)


if __name__ == "__main__":
    main()
