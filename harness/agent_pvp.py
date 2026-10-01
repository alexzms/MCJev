"""Harness P1 (./agent.py --harness p1): PvP, sumo style - knock the other player off a small ring.

Everything from the earlier harnesses in one: v2's chat judgement, target choice and memory; the survival
harness's held keys, recalled facts (knowledge.md) and options marked with what they would do; plus a fighter's
view (context_pvp.py): the ring's edge around you, the ground behind your opponent in the direction a hit pushes
them, reach, recharge, sprint freshness for w-tapping, and a log of hits given and taken. Jev picks one control a
step from moves, turns, aim, aim-and-hit and w-tap-and-hit; nothing is scripted. A player who hits an idle bot
becomes its opponent.
"""
import math, time

import context as v1
import context_pvp as cp
import context_survive as cs
import context_v2 as v2
from agent_survive import AgentC2
from jev import JevError


class AgentP1(AgentC2):
    level = 3  # recall facts from knowledge.md at the start of a fight

    def start_fight(self, obs, name):
        msg = {"username": name, "message": f"(you were hit by {name}: fight back and knock {name} off)",
               "t": time.time() * 1000, "seq": "self"}
        self.start(msg, once=False)
        self.prepare(obs)

    def run(self):
        print(f"log: {self.log.path}")
        seen_hurt = 0
        while True:
            reason = self.body.lost()
            if reason is not None:
                raise SystemExit(f"bot left the server: {reason}")
            try:
                obs = self.body.observe(self.mem.seen)
                if self.mem.deaths_seen is None:
                    self.mem.deaths_seen = obs["deaths"]
                if obs["deaths"] > self.mem.deaths_seen:  # fell into the water (the arena kills you there): round lost
                    self.mem.deaths_seen = obs["deaths"]
                    self.log.write("death", pos=obs["pos"])
                    if self.mem.instruction:
                        self.finish("died")
                    self.body.act("stop", hold=True)
                self.take_chat(obs)
                hurt = [c for c in obs.get("combat", []) if c["kind"] == "hurt" and c["who"] and c["t"] > seen_hurt]
                if hurt:
                    seen_hurt = hurt[-1]["t"]
                    if not self.mem.instruction:  # hit while idle: fight whoever hit you
                        self.start_fight(obs, hurt[-1]["who"])
                if not self.mem.instruction:
                    if time.time() - self.log.last > 5:
                        self.log.write("heartbeat", pos=obs["pos"], players=[p["name"] for p in obs["players"]])
                    time.sleep(0.1)
                    continue
                self.step(obs)
            except JevError as e:
                print(f"!! jev: {e}; retrying in 1s")
                time.sleep(1)

    def step(self, obs):
        cfg, mem, ins = self.cfg, self.mem, self.mem.instruction
        if "target" not in (ins or {}):
            self.prepare(obs)
            return
        if mem.history and "after" not in mem.history[-1]:
            h = mem.history[-1]
            h["after"] = True
            h["result"]["moved"] = round(math.hypot(obs["pos"]["x"] - h["pos0"]["x"], obs["pos"]["z"] - h["pos0"]["z"]), 2)
        tgt = ins["target"]
        name = tgt.get("name") if tgt["kind"] == "player" else None
        state = cp.step_state(obs, mem, cfg)
        question = cp.options(obs, name) if name else cs.action_question(cfg, obs)
        answers, execution, dt = self.ask([{"id": "step", "state": state, "questions": {"action": question}}])
        a = answers["step"]
        probs = a["action"]["probabilities"]
        action = max(probs, key=probs.get)
        f = cp.fighter(obs, name) if name else None
        record = dict(harness="p1", instruction=ins["text"], step=ins["steps"] + 1, state=state,
                      questions={"action": question}, answers=a, execution=execution, seconds=dt, action=action, done=0.0,
                      fight=f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2),
                                   "behind_them": cp.behind_them(obs, f), "behind_you": cp.behind_you(obs, f),
                                   "reach": round(cp.reach_of(obs, f), 2)})
        result = self.body.act(action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg, target=name, guard=True)  # the body stops you at the edge
        mem.history.append({"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        self.log.write("step", **record, act=result, effect=f"{action}; {result.get('note') or 'keys ' + ', '.join(result.get('held', []))}")
        print(f"   p={probs[action]:.2f} {dt * 1000:4.0f}ms  {action:12s} {result.get('note', '')}"
              f"{'  dist %.1f behind-them %s behind-you %s' % (record['fight']['dist'], record['fight']['behind_them'], record['fight']['behind_you']) if f else ''}")


class AgentSumo1(AgentP1):
    """p_sumo_v1: P1 in the hub's sumo, where bots are known by UUID and people may take any name (bots_by_uuid)."""
    bots_by_uuid = True
