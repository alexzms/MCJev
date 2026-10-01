#!/usr/bin/env python3
"""Jev agent: a mineflayer bot whose every decision is one Jev forward pass.

  ./agent.py                  join localhost:25565 as "Jev", Jev URL/auth from .env
  ./agent.py --mock           random decisions, no GPU server needed (tests the plumbing)
  ./agent.py -v               also print the full state text Jev reads each step

Loop: observe -> (new human chat? ask Jev whether it is an instruction / a stop) ->
if an instruction is active: render state + questions -> Jev -> press one control -> remember.
Every Jev call is logged to runs/<start time>-<name>.jsonl; ./console.py shows them live.
"""
import argparse, collections, json, math, os, re, signal, sys, time

import context
from body import Body, BodyError, free_port
from jev import Jev, JevError, MockJev

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(HERE, ".env")):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k, v)


class Memory:
    def __init__(self):
        self.chat = collections.deque(maxlen=50)      # every chat message seen, incl. our own
        self.history = collections.deque(maxlen=50)   # steps taken on the current instruction
        self.finished = collections.deque(maxlen=5)   # earlier instructions and how they ended
        self.instruction = None                       # {"from", "text", "t", "steps", "once"}
        self.sight = None                             # vision: {"est": world x/z, "step"} where the target was last seen
        self.scanned = 0.0                            # vision: degrees turned without seeing the target
        self.done_streak = 0                          # consecutive steps with p(done) over the threshold
        self.seen = 0                                 # last chat seq pulled from the body


class RunLog:
    """runs/<start>-<agent>.jsonl, one record per event; console.py tails these files."""

    def __init__(self, agent):
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        self.agent = agent
        self.path = os.path.join(HERE, "runs", f"{time.strftime('%Y%m%d-%H%M%S')}-{agent}.jsonl")
        self.f = open(self.path, "a", encoding="utf-8")
        self.last = 0

    def write(self, kind, **record):
        self.f.write(json.dumps({"t": time.time(), "agent": self.agent, "kind": kind, **record},
                                ensure_ascii=False) + "\n")
        self.f.flush()
        self.last = time.time()


class Inbox:
    """Commands typed into the console: runs/inbox.jsonl, one {"t", "to": [names] | "*", "from", "text"} per line.
    Only commands sent after the agent started are read."""

    PATH = os.path.join(HERE, "runs", "inbox.jsonl")

    def __init__(self, name):
        self.name = name
        self.offset = os.path.getsize(self.PATH) if os.path.exists(self.PATH) else 0
        self.n = 0

    def read(self):
        if not os.path.exists(self.PATH):
            return []
        with open(self.PATH, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        self.offset += end
        out = []
        for line in chunk[:end].splitlines():
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            if c.get("to") == "*" or self.name in c.get("to", []):
                self.n += 1
                out.append({"seq": f"console{self.n}", "t": c["t"] * 1000, "username": c["from"],
                            "message": c["text"], "via": "console"})
        return out


# Our bot accounts by name (Jev, Jev<n> the hub's pool, JevPro<n>, JevBench, JevCal, JevShooter): bots_by_uuid's
# fallback for a name not in the player list, or a server where nobody has a Mojang UUID.
BOT_NAME = re.compile(r"Jev(\d+|Pro\d*|Bench|Cal|Shooter)?", re.I)


class Agent:
    # Which players are bots. False: any with "jev" in the name. True (the user, 2026-09-27: people may take any name):
    # by UUID - people come through the online-mode proxy with Mojang UUIDs (version 4), our bots join the server
    # directly with offline ones (version 3), and nobody else can (the server listens on 127.0.0.1 only).
    bots_by_uuid = False

    def __init__(self, body, jev, cfg):
        self.body, self.jev, self.cfg = body, jev, cfg
        self.uuid_versions = {}   # the players in the last observation: name -> their UUID's version
        self.mem = Memory()
        self.inbox = Inbox(body.name)
        self.log = RunLog(body.name)
        self.log.write("start", jev="mock" if cfg.mock else cfg.jev_url,
                       cfg={k: sorted(v) if isinstance(v, set) else v for k, v in vars(cfg).items() if k != "jev_auth"})

    def say(self, text):
        if not self.cfg.quiet:
            self.body.say(text)

    def ask(self, states):
        """-> ({state_id: answers}, execution info, seconds)"""
        t0 = time.time()
        answers, raw = self.jev.evaluate(states)
        return answers, raw.get("execution"), time.time() - t0

    def see_players(self, obs):
        self.uuid_versions = {p["name"]: int(p["uuid"][14], 16) for p in obs.get("players", []) if p.get("uuid")}

    def is_bot(self, name):
        if not self.bots_by_uuid:
            return "jev" in name.lower()
        v = self.uuid_versions.get(name)
        if v is not None and 4 in self.uuid_versions.values():   # behind the proxy: offline UUIDs are ours
            return v == 3
        return bool(BOT_NAME.fullmatch(name))

    def is_human(self, name, me):
        """Whose chat can be an order: the owners if given, else any player that is not a bot (is_bot)."""
        if self.cfg.owners:
            return name in self.cfg.owners
        return name != me and not self.is_bot(name)

    def addressed_to_me(self, text, me, players):
        """Every bot hears all game chat. Bots (is_bot: players with "jev" in the name) can be named alone or as a group:
        "cjev1" is CJev01, "cjev"/"all cjevs" is every CJev; plain "jev" means everyone. A message naming
        other bots or groups but not mine is not for me; one naming no bot is for everyone."""
        norm = lambda n: re.sub(r"0+(\d)", r"\1", n.lower())
        group = lambda n: re.sub(r"\d+$", "", n)
        text = norm(text)
        named = lambda token: re.search(rf"(?<![a-z0-9]){re.escape(token)}s?(?![a-z0-9])", text)
        mine = {norm(me), group(norm(me))}
        if any(named(t) for t in mine):
            return True
        bots = [norm(p) for p in players if self.is_bot(p)]
        others = {t for b in bots for t in (b, group(b))} - mine - {"jev"}
        return not any(named(t) for t in others)

    # ---------------------------------------------------------------- instructions

    def start(self, msg, once):
        if self.mem.instruction:
            self.finish("replaced by a newer instruction")
        self.mem.instruction = {"from": msg["username"], "text": msg["message"], "t": msg["t"], "steps": 0,
                                "once": once}
        self.mem.history.clear()
        self.mem.sight, self.mem.scanned, self.mem.done_streak = None, 0.0, 0
        self.log.write("instruction", event="start", **{k: self.mem.instruction[k] for k in ("from", "text", "once")})
        print(f'>> {"one-time" if once else "ongoing"} instruction from {msg["username"]}: "{msg["message"]}"')
        self.say(self.cfg.say_ack)

    def finish(self, outcome):
        ins = self.mem.instruction
        ins["outcome"] = outcome
        self.mem.finished.append(ins)
        self.mem.instruction = None
        self.log.write("instruction", event="finish", outcome=outcome, steps=ins["steps"],
                       **{k: ins[k] for k in ("from", "text")})
        print(f'<< "{ins["text"]}": {outcome} after {ins["steps"]} steps')

    def take_chat(self, obs):
        """New in-game chat plus console commands -> Jev judges each: instruction, stop, or chatter."""
        self.see_players(obs)
        if obs["chat"]:
            self.mem.seen = obs["chat"][-1]["seq"]
        new = obs["chat"] + self.inbox.read()
        self.mem.chat.extend(new)
        others = [p["name"] for p in obs["players"]]
        msgs = [m for m in new if m.get("via") == "console" or (
            self.is_human(m["username"], obs["name"]) and (m.get("whisper") or   # a whisper to me is for me, whoever it names
                                                            self.addressed_to_me(m["message"], obs["name"], others)))]
        if not msgs:
            return
        states = [{"id": f"msg{m['seq']}", "state": context.message_state(obs, self.mem, m),
                   "questions": context.MESSAGE_QUESTIONS} for m in msgs]
        answers, execution, dt = self.ask(states)
        decisions = []
        for m in msgs:
            a = answers[f"msg{m['seq']}"]
            stop, order, once = a["is_stop"]["p_true"], a["is_instruction"]["p_true"], a["once"]["p_true"]
            decision = "stop" if stop >= 0.5 else "instruction" if order >= 0.5 else "chatter"
            decisions.append({"from": m["username"], "text": m["message"], "via": m.get("via", "chat"),
                              "decision": decision, "p_instruction": order, "p_stop": stop, "p_once": once})
            print(f'   {m.get("via", "chat")} {m["username"]}: "{m["message"]}" -> {decision} '
                  f'(instruction {order:.2f}, stop {stop:.2f}, once {once:.2f}, {dt * 1000:.0f}ms)')
        self.log.write("message", states=states, answers=answers, execution=execution, seconds=dt,
                       decisions=decisions)
        for m, d in zip(msgs, decisions):
            if d["decision"] == "stop" and self.mem.instruction:
                self.finish(f"stopped by {m['username']}")
                self.say(self.cfg.say_stop)
            elif d["decision"] == "instruction":
                self.start(m, once=d["p_once"] >= 0.5)

    # ---------------------------------------------------------------- control

    def step(self, obs):
        cfg, ins = self.cfg, self.mem.instruction
        questions, answers, images, look, execution, dt = {}, {}, [], None, [], 0.0
        if cfg.vision != "off":
            # pass 1: look at the screen. Answer slots are denoised together, so an action slot cannot use
            # the sees/where answers of the same pass; pass 2 gets them as text instead.
            images = [self.body.snapshot()]
            look_q = context.look_questions(ins["from"])
            ans, ex, t = self.ask([{"id": "look", "state": context.LOOK_STATE, "images": images, "questions": look_q}])
            look = ans["look"]
            questions.update(look_q); answers.update(look); execution.append(ex); dt += t
            seen, angle, dist = context.read_look(look)
            if seen:
                self.mem.sight = {"est": context.project(obs, angle, dist), "step": ins["steps"]}
                self.mem.scanned = 0.0
        view = (context.view_lines(obs, look, ins["from"], self.mem.sight, self.mem.scanned, ins["steps"])
                if look else None)
        state = context.step_state(obs, self.mem, cfg, view=view)
        toward = context.target_side(obs, cfg, ins["from"], look, self.mem.sight)
        step_q = context.step_questions(cfg, ins["from"], toward)
        if cfg.verbose:
            print("-" * 100 + "\n" + state + "\n" + "-" * 100)
        ans, ex, t = self.ask([{"id": "step", "state": state, "questions": step_q}])
        questions.update(step_q); answers.update(ans["step"]); execution.append(ex); dt += t
        a = answers
        probs, done = a["action"]["probabilities"], a["done"]["p_true"]
        action = max(probs, key=probs.get)
        record = dict(instruction=ins["text"], step=ins["steps"] + 1, state=state, images=images,
                      questions=questions, answers=a, execution=execution[-1], seconds=dt, action=action, done=done)
        target = next((p for p in obs["players"] if p["name"] == ins["from"] and p["pos"]), None)
        if target:  # ground truth for evaluation (console / logs only, never shown to Jev in vision-only mode)
            rel = context.relative(obs, target["pos"])
            record["truth"] = {"target": target["name"], "dist": round(rel["dist"], 2), "angle": round(rel["angle"], 1),
                               "pos": obs["pos"], "yaw": obs["yaw"]}
        # With vision one misjudged frame (a player 4 blocks off read as "big") could end the instruction, so done
        # must hold on two consecutive steps; the first time the chosen action still runs, giving a new view.
        confirmed = cfg.vision == "off" or self.mem.done_streak >= 1
        self.mem.done_streak = self.mem.done_streak + 1 if done >= cfg.done_threshold else 0
        if ins["once"] and done >= cfg.done_threshold and confirmed:  # ongoing requests run until stopped or replaced
            self.log.write("step", **record, act=None, effect="judged the instruction done")
            self.finish("completed")
            self.say(cfg.say_done)
            return

        result = self.body.act(action, ms=cfg.move_ms, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg)
        h = {"action": action, "result": result}
        if look:  # what Jev saw before acting; the history then records where it has already looked
            where = max(look["where"]["probabilities"], key=look["where"]["probabilities"].get)
            h["sees"] = (ins["from"], f"in view, {where.replace('_', ' ')}" if seen else "not in view")
            if not seen:
                self.mem.scanned += abs(math.degrees(math.atan2(math.sin(result["yaw"] - obs["yaw"]),
                                                                math.cos(result["yaw"] - obs["yaw"]))))
        if target:  # the target's last seen position, as seen before and after this step
            before, after = context.relative(obs, target["pos"]), context.relative({**obs, **result}, target["pos"])
            h["target"] = (target["name"], before["dist"], after["dist"], before["angle"], after["angle"])
        self.mem.history.append(h)
        ins["steps"] += 1
        line = context.history_line(ins["steps"], h, context.action_options(cfg), show_target=cfg.vision != "only")
        self.log.write("step", **record, act=result, effect=line.split(" -> ", 1)[1])
        print(f"   p={probs[action]:.2f} mass={a['action']['candidate_mass']:.2f} done={done:.2f} "
              f"{dt * 1000:4.0f}ms  {line}")
        if ins["once"] and ins["steps"] >= cfg.max_steps:
            self.finish("gave up (step limit)")
            self.say(cfg.say_giveup)

    def run(self):
        print(f"log: {self.log.path}")
        while True:
            reason = self.body.lost()
            if reason is not None:
                sys.exit(f"bot left the server: {reason}")
            try:
                obs = self.body.observe(self.mem.seen)
                self.take_chat(obs)
                if not self.mem.instruction:
                    if time.time() - self.log.last > 5:  # lets the console tell idle from dead
                        self.log.write("heartbeat", pos=obs["pos"], players=[p["name"] for p in obs["players"]],
                                       images=[self.body.snapshot()] if self.cfg.vision != "off" else [])
                    time.sleep(self.cfg.idle_poll)
                    continue
                self.step(obs)
            except JevError as e:
                print(f"!! jev: {e}; retrying in 2s")
                time.sleep(2)


# Harness versions: name -> (module, class). The p_<mode>_v<n> names are what the server asks for (/play).
HARNESSES = {
    "v1": ("agent", "Agent"),
    "v2": ("agent_v2", "AgentV2"), "c1": ("agent_v2", "AgentV2"),
    "c2": ("agent_survive", "AgentC2"), "c3": ("agent_survive", "AgentC3"),
    "p1": ("agent_pvp", "AgentP1"), "p_sumo_v1": ("agent_pvp", "AgentSumo1"),
    "u1": ("agent_uhc", "AgentU1"), "p_uhc_v1": ("agent_uhc", "AgentU1"),
    "p_uhc_raw_v1": ("agent_uhc", "AgentURaw"),
    "p_uhc_pro_v1": ("agent_uhc_pro", "AgentUPro"), "p_uhc_pro_v2": ("agent_uhc_pro", "AgentUPro2"),
    "p_uhc_pro_v3": ("agent_uhc_pro", "AgentUPro3"), "p_uhc_pro_v4": ("agent_uhc_pro", "AgentUPro4"),
    "p_uhc_pro_v5": ("agent_uhc_pro", "AgentUPro5"), "p_uhc_pro_v6": ("agent_uhc_pro", "AgentUPro6"),
    "p_uhc_pro_v7": ("agent_uhc_pro", "AgentUPro7"), "p_uhc_pro_v8": ("agent_uhc_pro", "AgentUPro8"),
    "p_uhc_pro_v8_official": ("agent_uhc_pro", "AgentUPro8Official"), "p_uhc_pro_v9": ("agent_uhc_pro", "AgentUPro9"), "p_uhc_pro_v10": ("agent_uhc_pro", "AgentUPro10"), "p_uhc_pro_v11": ("agent_uhc_pro", "AgentUPro11"), "p_uhc_pro_team_v1": ("agent_uhc_pro", "AgentUProTeam"),
    "p_uhc_v2": ("agent_uhc", "AgentU2"), "p_uhc_raw_v2": ("agent_uhc", "AgentURaw2"),
    "p_uhc_vis_body_2step_v2": ("agent_uhc", "AgentUVisBody2"),
    "p_uhc_vis_body_2step_v3": ("agent_uhc", "AgentUVisBody3"), "p_uhc_vis_body_1step_v1": ("agent_uhc", "AgentUVisBody1"),
    "p_uhc_vis_body_pure_2step_v1": ("agent_uhc", "AgentUVisPure"), "p_uhc_vis_body_pure_1step_v1": ("agent_uhc", "AgentUVisPure1"),
    "p_uhc_vis_body_2step_v1": ("agent_uhc", "AgentUVisBody"), "p_uhc_vis_raw_2step_v1": ("agent_uhc", "AgentUVisRaw"),
    "p_uhc_vis_raw_1step_v1": ("agent_uhc", "AgentUVisRaw1"),
    "p_uhc_vis_raw_v2": ("agent_uhc", "AgentUVisRawV2"),
    "p_uhc_raw_v3": ("agent_uhc", "AgentURaw3"),
    "p_uhc_vis_raw_v3": ("agent_uhc", "AgentUVisRawV3"),
}


def known_harness(name):
    return name in HARNESSES


def harness_class(name):
    module, cls = HARNESSES[name]
    if module == "agent":
        return globals()[cls]
    import importlib
    return getattr(importlib.import_module(module), cls)



def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="Jev", help="bot username (server runs offline-mode)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25565)
    ap.add_argument("--owners", default="", help="comma-separated players the bot obeys (default: every human)")
    ap.add_argument("--jev-url", default=os.environ.get("JEV_URL"))
    ap.add_argument("--jev-auth", default=os.environ.get("JEV_AUTH"), help="user:password for the tunnel")
    ap.add_argument("--mock", action="store_true", help="random decisions instead of the Jev server")
    ap.add_argument("--harness", default="v1", metavar="{" + ",".join(HARNESSES) + "}",
                    help="v1: context.py; v2: context_v2.py sections, per-instruction target, long-term memory; "
                         "creeper scene: c1 = v2 (does not know mobs), c2 = runs from danger, "
                         "c3 = uses the surroundings (agent_survive.py); p_sumo_v1 (= p1) = PvP, sumo (agent_pvp.py); "
                         "p_uhc_v1 (= u1) = Block UHC (agent_uhc.py). The server can switch it: a whisper "
                         "'harness <version>' from the server console")
    ap.add_argument("--memory-dir", default=os.path.join(HERE, "memory"), help="v2 long-term memory, shared by agents")
    ap.add_argument("--vision", choices=["off", "on", "only"], default="on",
                    help="off: text only; on: first-person screenshot + text; "
                         "only: screenshot, and no text derived from other players' coordinates")
    ap.add_argument("--browser", help="Chromium-based browser for the camera (default: Chrome/Edge/Chromium)")
    ap.add_argument("--camera-size", help="start the camera at this size (e.g. 256x256) whatever --vision says, for "
                                          "harnesses that take their own pictures (p_uhc_vis_*)")
    ap.add_argument("--camera-fps", type=float, default=0, help="render continuously at this rate: a snapshot is then the "
                                                                "latest frame, ready at once (0: render on demand)")
    ap.add_argument("--camera-gpu", action="store_true", help="browser camera: render on the GPU (Vulkan)")
    ap.add_argument("--camera", choices=["browser", "vox"], default="browser",
                    help="browser: prismarine-viewer in headless Chromium; vox: body/voxcam, a CPU ray-caster (no "
                         "browser, a few ms a picture, never an old frame)")
    ap.add_argument("--move-ms", type=int, default=400, help="how long a walk/jump key is held")
    ap.add_argument("--turn-deg", type=float, default=30, help="mouse step left/right")
    ap.add_argument("--fine-turn-deg", type=float, default=10, help="small mouse step left/right, for aiming")
    ap.add_argument("--pitch-deg", type=float, default=15, help="mouse step up/down")
    ap.add_argument("--max-steps", type=int, default=80, help="give up a one-time instruction after this many steps")
    ap.add_argument("--done-threshold", type=float, default=0.7, help="p(done) needed to finish an instruction")
    ap.add_argument("--history", type=int, default=8, help="past steps shown to Jev")
    ap.add_argument("--idle-poll", type=float, default=0.3, help="seconds between observations while idle")
    ap.add_argument("--quiet", action="store_true", help="do not send the canned chat acknowledgements")
    ap.add_argument("--say-ack", default="Got it")
    ap.add_argument("--say-done", default="Done")
    ap.add_argument("--say-stop", default="OK, stopping")
    ap.add_argument("--say-giveup", default="Couldn't do it, stopping")
    ap.add_argument("--say-remember", default="Noted")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the full state text every step")
    cfg = ap.parse_args()
    if not known_harness(cfg.harness):
        ap.error(f"--harness {cfg.harness}: not a harness (see --help)")
    cfg.owners = {o for o in cfg.owners.split(",") if o}

    if cfg.mock:
        jev = MockJev()
    elif cfg.jev_url:
        jev = Jev(cfg.jev_url, cfg.jev_auth)
    else:
        sys.exit("no Jev server: set JEV_URL / JEV_AUTH (or .env), or pass --mock")

    wants_camera = cfg.vision != "off" or cfg.camera_size
    body = Body(cfg.name, cfg.host, cfg.port, camera_port=free_port() if wants_camera and cfg.camera == "browser" else None,
                browser=cfg.browser, camera_size=cfg.camera_size, camera_fps=cfg.camera_fps, camera_gpu=cfg.camera_gpu,
                camera_kind=cfg.camera if wants_camera else "browser")
    print(f"{body.name} joined {cfg.host}:{cfg.port}; decisions from {'mock' if cfg.mock else cfg.jev_url}; "
          f"harness {cfg.harness}; vision {cfg.vision}{' (' + body.camera + ')' if body.camera else ''}")
    from agent_v2 import SwitchHarness
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("terminated"))
    try:
        while True:
            agent = harness_class(cfg.harness)(body, jev, cfg)
            try:
                agent.run()
            except SwitchHarness as s:
                if not known_harness(s.version):
                    print(f"!! unknown harness {s.version}; staying on {cfg.harness}")
                    continue
                agent.log.write("end", reason=f"harness switched to {s.version}")
                print(f"== harness {cfg.harness} -> {s.version}")
                cfg.harness = s.version
    except (KeyboardInterrupt, BodyError, SystemExit) as e:
        reason = f"{e.__class__.__name__}{': ' + str(e) if str(e) else ''}"
        agent.log.write("end", reason=reason)
        print(f"\nstopping ({reason})")
    finally:
        body.close()


if __name__ == "__main__":
    main()
