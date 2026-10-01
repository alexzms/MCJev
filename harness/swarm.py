#!/usr/bin/env python3
"""Run several Jev agents at once, each its own agent.py process and bot.

  ./swarm.py 10                         Jev01 .. Jev10 with default agent options
  ./swarm.py 5 --prefix Bot             Bot01 .. Bot05
  ./swarm.py 10 -- --vision off         everything after -- goes to every agent.py

Each agent's stdout goes to runs/<name>.out; the console shows them all. Ctrl-C / SIGTERM stops them all.
Give orders from the console's command bar (all agents or one), or in game chat (every agent hears it).
"""
import argparse, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("count", type=int)
    ap.add_argument("--prefix", default="Jev")
    ap.add_argument("--first", type=int, default=1, help="number of the first agent")
    ap.add_argument("--stagger", type=float, default=1.5, help="seconds between two agents joining")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args, extra = ap.parse_args(argv[:split]), argv[split + 1:]  # options after -- go to agent.py

    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    procs = {}

    def stop(*_):
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for i in range(args.first, args.first + args.count):
        name = f"{args.prefix}{i:02d}"
        out = open(os.path.join(HERE, "runs", f"{name}.out"), "w")
        procs[name] = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "agent.py"), "--name", name, *extra],
                                       stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"started {name} (output: runs/{name}.out)", flush=True)
        time.sleep(args.stagger)

    while True:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                print(f"{name} exited with {p.returncode}; see runs/{name}.out", flush=True)
                del procs[name]
        if not procs:
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
