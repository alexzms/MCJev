"""Harnesses c2 and c3 for the creeper scene (./agent.py --harness c2|c3); c1 is harness v2 as it is.

c1 (= v2): no notion of mobs at all, so a creeper is invisible to it.
c2: sees hostile mobs (distance, closing in, lit fuse), keeps keys held between decisions (it runs while Jev
    thinks), may pick a mob as an instruction's target, marks every movement option toward / away from the
    danger, and reacts on its own when a hostile mob comes near and nobody has said anything.
c3: c2 + Minecraft facts recalled from knowledge.md, the landmarks and animals around, and a plan asked every
    step in parallel with the action (keep running / get to a landmark / get to an animal); the plan's target
    is marked on the options from the next step on.
"""
import math, os, time

import context as v1
import context_survive as cs
import context_v2 as v2
import memory as ltm
from agent import HERE
from agent_v2 import AgentV2, MemoryV2, best
from jev import JevError

SELF_TRIGGER = 12       # blocks: a hostile mob this close starts the "stay safe" goal when there is no instruction
PLAN_SWITCH = 0.6       # a new plan needs this probability on two steps in a row


class MemorySurvive(MemoryV2):
    def __init__(self):
        super().__init__()
        self.facts = []          # c3: recalled facts for the current instruction
        self.plan = None         # c3: "flee", "thing:i", "mob:id"
        self.plan_pending = None
        self.deaths_seen = None
        self.vmem = {}           # vision: last sighting of the danger
        self.last_look = 0.0


class AgentC2(AgentV2):
    level = 2

    def __init__(self, body, jev, cfg):
        super().__init__(body, jev, cfg)
        self.mem = MemorySurvive()
        self.all_facts = cs.load_facts(os.path.join(HERE, "knowledge.md")) if self.level >= 3 else []

    # ---------------------------------------------------------------- instruction start

    def prepare(self, obs):
        """As v2, plus mobs as possible targets; c3 also scans landmarks and recalls facts, all in one request."""
        ins = self.mem.instruction
        ins["yaw0"] = obs["yaw"]
        self.mem.looked, self.mem.plan, self.mem.plan_pending = [], None, None
        places, entries = self.store.places(), self.store.entries()
        self.mem.things = self.body.landmarks() if self.level >= 3 else []
        cands = v2.target_candidates(obs, ins["from"], places, self.mem.things, self.cfg)
        mobs = {m["id"]: m for m in obs.get("mobs", [])}
        for m in mobs.values():
            r = v1.relative(obs, m["pos"])
            cands[f"mob:{m['id']}"] = f"the {cs.mob_name(m)}, {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
        cands["none"] = cands.pop("none")
        states = [{"id": "target", "state": f'You are {obs["name"]}, a bot in Minecraft. {ins["from"]} (human) just '
                                            f'told you: "{ins["text"]}"',
                   "questions": {"target": {"type": "choice", "instructions": v2.TARGET_QUESTION, "criteria": cands}}}]
        if entries:
            states.extend(ltm.recall_requests(entries, ins["text"], ins["from"]))
        if self.all_facts:
            situation = f'You were told: "{ins["text"]}". ' + (" ".join(cs.danger_lines(obs, None)) or "")
            states.extend(cs.recall_states(self.all_facts, situation))
        answers, execution, dt = self.ask(states)
        key = best(answers["target"]["target"])
        if key.startswith("mob:"):
            m = mobs[int(key.split(":", 1)[1])]
            ins["target"] = {"kind": "mob", "id": m["id"], "name": f"the {cs.mob_name(m)}", "label": f"the {cs.mob_name(m)}"}
        else:
            ins["target"] = v2.resolve_target(key, ins["from"], places, self.mem.things, ins["text"])
        self.mem.recalled = ltm.pick_recalled(entries, answers) if entries else []
        if self.all_facts:
            scored = sorted(((cs.recalled_p(answers, i), f) for i, f in enumerate(self.all_facts)), reverse=True)
            self.mem.facts = [f for p, f in scored if p >= 0.5][:5]
        self.log.write("prepare", states=states, answers=answers, execution=execution, seconds=dt,
                       target=ins["target"], recalled=[e["file"] for e in self.mem.recalled], facts=self.mem.facts)
        print(f'   target: {ins["target"].get("name") or "none"}; facts: {len(self.mem.facts)} ({dt * 1000:.0f}ms)')

    def start_self(self, obs, mob):
        """No instruction, a hostile mob close by: take up "stay safe" as an ongoing goal of one's own."""
        msg = {"username": obs["name"], "message": f"(a {cs.mob_name(mob)} is coming: stay safe)",
               "t": time.time() * 1000, "seq": "self"}
        self.start(msg, once=False)
        self.mem.instruction["self"] = True
        self.prepare(obs)

    # ---------------------------------------------------------------- vision

    def perceive(self, obs):
        """Vision only: two pictures (in front, behind) -> obs whose mobs are the estimated danger. The body still
        knows real mob positions; they are replaced here and kept only for evaluation."""
        if self.cfg.vision == "off":
            return obs, None
        front, rear = self.body.snapshot(0.0), self.body.snapshot(math.pi)
        q = cs.look2_questions()
        answers, execution, dt = self.ask([{"id": "look", "state": cs.LOOK2_STATE, "images": [front, rear], "questions": q}])
        seen = cs.read_look2(answers["look"])
        self.mem.last_look = time.time()
        return cs.vision_obs(obs, seen, self.mem.vmem), {"images": [front, rear], "answers": answers["look"],
                                                         "questions": q, "seconds": dt, "seen": seen}

    # ---------------------------------------------------------------- loop

    def run(self):
        print(f"log: {self.log.path}")
        while True:
            reason = self.body.lost()
            if reason is not None:
                raise SystemExit(f"bot left the server: {reason}")
            try:
                obs = self.body.observe(self.mem.seen)
                if self.mem.deaths_seen is None:
                    self.mem.deaths_seen = obs["deaths"]
                if obs["deaths"] > self.mem.deaths_seen:
                    self.mem.deaths_seen = obs["deaths"]
                    self.log.write("death", pos=obs["pos"])
                    if self.mem.instruction:
                        self.finish("died")
                    self.body.act("stop", hold=True)
                self.take_chat(obs)
                if not self.mem.instruction:
                    seen_obs = obs
                    if self.cfg.vision != "off":  # look around once a second while idle
                        if time.time() - self.mem.last_look < 1.0:
                            time.sleep(self.cfg.idle_poll)
                            continue
                        seen_obs, _ = self.perceive(obs)
                    near = [m for m, r in cs.threats(seen_obs) if r["dist"] <= SELF_TRIGGER]
                    if near:
                        self.start_self(seen_obs, near[0])
                    else:
                        if time.time() - self.log.last > 5:
                            self.log.write("heartbeat", pos=obs["pos"], players=[p["name"] for p in obs["players"]])
                        time.sleep(self.cfg.idle_poll)
                        continue
                self.step(obs)
            except JevError as e:
                print(f"!! jev: {e}; retrying in 1s")
                time.sleep(1)

    # ---------------------------------------------------------------- control

    def step(self, obs):
        cfg, mem, ins = self.cfg, self.mem, self.mem.instruction
        if "target" not in (self.mem.instruction or {}):  # preparing failed (Jev unreachable): try it again first
            self.prepare(obs)
            return
        tgt = ins["target"]
        real = obs
        obs, look = self.perceive(obs)  # vision: mobs become what was seen; coordinates: unchanged
        threat = cs.threats(obs)
        if mem.history and "after" not in mem.history[-1]:
            # keys stay held between decisions, so what a decision did shows only now: fill it in
            h = mem.history[-1]
            h["after"] = True
            h["result"]["moved"] = round(math.hypot(obs["pos"]["x"] - h["pos0"]["x"], obs["pos"]["z"] - h["pos0"]["z"]), 2)
            h["danger1"] = threat[0][1]["dist"] if threat else None
        state = cs.step_state(obs, mem, cfg, self.level)
        goal = None  # something to move toward, marked on the options
        if self.level >= 3:
            goal = cs.plan_target(obs, mem.plan, mem.things)
        if goal is None and tgt["kind"] in ("place", "thing", "player"):
            pos = tgt.get("pos") or next((p["pos"] for p in obs["players"] if p["name"] == tgt.get("name") and p["pos"]), None)
            if pos:
                goal = (tgt["label"] if tgt["kind"] != "thing" else tgt["where"], v1.relative(obs, pos)["angle"])
        questions = {"action": cs.action_question(cfg, obs, goal)}
        states = [{"id": "step", "state": state, "questions": questions}]
        if ins["once"]:
            states.append({"id": "verify", "state": v2.verify_state(mem, obs, cfg, None), "questions": v2.VERIFY_QUESTION})
        if self.level >= 3:
            states.append(cs.plan_state(obs, mem.prev_obs, mem.things, mem.facts))
        answers, execution, dt = self.ask(states)
        a = dict(answers["step"])
        if "verify" in answers:
            a["done"] = answers["verify"]["done"]
        probs = a["action"]["probabilities"]
        action = max(probs, key=probs.get)
        done = a["done"]["p_true"] if "done" in a else 0.0
        record = dict(harness=f"c{self.level}", vision=cfg.vision, instruction=ins["text"], step=ins["steps"] + 1, state=state,
                      questions={**questions, **({"done": v2.VERIFY_QUESTION["done"]} if "done" in a else {})},
                      answers=a, execution=execution, seconds=dt, action=action, done=done,
                      danger=[{"name": m["name"], "dist": round(r["dist"], 2), "angle": round(r["angle"]),
                               "fuse": m.get("fuse")} for m, r in cs.threats(real)],
                      perceived=[{"dist": round(r["dist"], 2), "angle": round(r["angle"]), "seen": m.get("seen")}
                                 for m, r in threat] if look else None,
                      images=look["images"] if look else [], look=look and {k: look[k] for k in ("answers", "seconds")})
        if self.level >= 3:
            p = answers["plan"]["plan"]["probabilities"]
            choice = max(p, key=p.get)
            if choice != mem.plan and p[choice] >= PLAN_SWITCH:
                if mem.plan_pending == choice or mem.plan is None:
                    mem.plan, mem.plan_pending = choice, None
                else:
                    mem.plan_pending = choice
            else:
                mem.plan_pending = None
            record.update(plan=mem.plan, plan_answer=answers["plan"]["plan"], plan_state=states[-1]["state"])
        if ins["once"] and done >= cfg.done_threshold:
            self.log.write("step", **record, act=None, effect="judged the instruction done")
            self.finish("completed")
            self.body.act("stop", hold=True)
            self.say(cfg.say_done)
            return
        result = self.body.act(action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg)
        h = {"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"],
             "danger0": threat[0][1]["dist"] if threat else None,
             "danger_name": f"the {cs.mob_name(threat[0][0])}" if threat else None}
        mem.history.append(h)
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        near = f"{threat[0][0]['name']} {threat[0][1]['dist']:.1f}b{' FUSE' if threat[0][0].get('fuse') else ''}" if threat else "-"
        self.log.write("step", **record, act=result, effect=f"{action}; keys {', '.join(result.get('held', [])) or 'none'}")
        print(f"   p={probs[action]:.2f} {dt * 1000:4.0f}ms  {action:16s} danger {near}"
              f"{'  plan ' + str(mem.plan) if self.level >= 3 else ''}")
        if ins["once"] and ins["steps"] >= cfg.max_steps:
            self.finish("gave up (step limit)")
            self.body.act("stop", hold=True)


class AgentC3(AgentC2):
    level = 3

