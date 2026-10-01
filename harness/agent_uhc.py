"""Harness U1 (./agent.py --harness u1): Block UHC - kill the other player with a sword, a bow, buckets and blocks.

The P1 loop (chat judgement, target, recalled facts, a fighter's view, one Jev choice per step, held keys; a
player who hits an idle bot becomes its opponent; dying ends the fight) with the Block UHC state and options of
context_uhc.py. The body lets go of held keys at a drop of 4 or more blocks (one that would hurt), not at every
step off a rise, and auto-jumps one-block steps (the 1.11 client's Auto-Jump option). Rounds, resets and the rules
themselves are the server's (the JevArena plugin).
"""
import math, random, time

import context as v1
import context_pvp as cp
import context_survive as cs
import context_uhc as cu
import context_uhc_raw as cr
import context_uhc_vis as cv
from agent_pvp import AgentP1

GUARD = 4          # drops of this many blocks or more stop your keys: from 4 blocks a fall hurts
LEAD_NET_S = 0.1   # pure vision: how late the turn reaches the server after the decision (network, one way)
LEAD_MAX = 30      # degrees at most added to lead a moving opponent


def shuffled(question):
    """The options in a random order: Jev leans to the options listed first (offline, on 12 recorded raw steps, the
    one that put the crosshair on the opponent was picked 5/12 times in the harness's order, 10/12 shuffled)."""
    items = list(question["criteria"].items())
    random.shuffle(items)
    return {**question, "criteria": dict(items)}


class AgentU1(AgentP1):
    harness_name = "u1"
    ctx = cu                    # the state and the options
    v = 1                       # harness version: v2 adds shuffled options, what a hit feels like, the bow drawn and held
    shuffle = False

    def act_params(self, option):
        return option, {}

    def focus_camera(self, name):
        """The camera draws only the opponent (a player tells them apart by the name tag; other players, e.g. in the
        next arena behind an invisible wall, are left out)."""
        if name and getattr(self, "focused", None) != name and self.body.camera:
            try:
                self.body.call("camera_focus", names=[name])
                self.focused = name
            except Exception as e:
                print(f"!! camera focus: {e}")

    def start_fight(self, obs, name):
        """In Block UHC the server starts the fights (and burning near a player is not being hit by them)."""
        return

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
        state = self.ctx.step_state(obs, mem, cfg, v=self.v)
        question = self.ctx.options(obs, name, mem.history, v=self.v) if name else cs.action_question(cfg, obs)
        if self.shuffle:
            question = shuffled(question)
        answers, execution, dt = self.ask([{"id": "step", "state": state, "questions": {"action": question}}])
        a = answers["step"]
        probs = a["action"]["probabilities"]
        action = max(probs, key=probs.get)
        f = cp.fighter(obs, name) if name else None
        record = dict(harness=self.harness_name, instruction=ins["text"], step=ins["steps"] + 1, state=state,
                      questions={"action": question}, answers=a, execution=execution, seconds=dt, action=action, done=0.0,
                      fight=f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2), "their_health": f.get("health"),
                                   "health": obs["health"], "held": obs.get("held_item")})
        body_action, params = self.act_params(action)
        result = self.body.act(body_action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg, target=name, guard=GUARD, autojump=True, **params)
        mem.history.append({"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        self.log.write("step", **record, act=result, effect=f"{action}; {result.get('note') or 'keys ' + ', '.join(result.get('held', []))}")
        fight = record["fight"]
        print(f"   p={probs[action]:.2f} {dt * 1000:4.0f}ms  {action:12s} {result.get('note', '')}"
              + (f"  dist {fight['dist']:.1f} hp {fight['health']:.0f} vs {fight['their_health']}" if fight else ""))


class AgentURaw(AgentU1):
    """p_uhc_raw_v1: the raw controls (context_uhc_raw.py): keys, mouse steps, buttons, hotbar; no aim help."""
    harness_name = "p_uhc_raw_v1"
    ctx = cr

    def act_params(self, option):
        return cr.act_params(option)


class AgentU2(AgentU1):
    harness_name = "p_uhc_v2"
    v, shuffle = 2, True


class AgentURaw2(AgentURaw):
    harness_name = "p_uhc_raw_v2"
    v, shuffle = 2, True


class AgentUVis(AgentU1):
    """Vision: each step a 256x256 picture (context_uhc_vis.py). Needs the camera (agent.py --camera-size 256x256).
    2step: a look call about the picture only, then the decision reads what was seen as a sentence.
    1step: one call - the picture and the text state (no perception sentence) - and the decision; the look questions
    ride along as a second state in the same request, only for the log (the decision cannot see their answers)."""
    variant = "body"
    one_step = False
    image_every = 10          # keep every 10th picture in the run log

    def act_params(self, option):
        if self.variant == "pure":
            if option in cv.PURE_ACTIONS:   # aim where the picture showed them, led by how fast they cross the view
                dx, dist = getattr(self, "pure_tgt", None) or (0.0, 3.0)
                w = self.omega()
                if w is not None and getattr(self, "pure_pic_t", None):
                    age = time.time() - self.pure_pic_t + LEAD_NET_S
                    dx += max(-LEAD_MAX, min(LEAD_MAX, max(-200.0, min(200.0, w)) * age))
                return cv.PURE_ACTIONS[option], {"dyaw": dx, "dist": dist}
            if option.startswith("look_"):
                return cr.act_params(option)
            return option, {}
        if option.startswith("turn_") and option.endswith("s") and option[5] in "lr":   # turn_r30s: held turning
            return "turn_rate", {"n": int(option[6:-1]) * (1 if option[5] == "r" else -1)}
        if option == "turn_stop":
            return "turn_rate", {"n": 0}
        return cr.act_params(option) if self.variant == "raw" else (option, {})

    def omega(self):
        """How fast the opponent moves across the view (°/s, + right), from the last sightings and the turn since:
        their bearing is where you face plus where they were in the picture."""
        pts = getattr(self, "bearings", [])[-3:]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.08 or pts[-1][0] - pts[0][0] > 1.0:
            return None
        d = (pts[-1][1] - pts[0][1] + 180) % 360 - 180
        return d / (pts[-1][0] - pts[0][0])

    def note_bearing(self, seen, yaw):
        """Keep where a sighting puts them: facing (° right of north) plus their place in the picture; and how big."""
        now = time.time()
        self.bearings = [b for b in getattr(self, "bearings", []) if now - b[0] < 1.0]
        self.sizes = [z for z in getattr(self, "sizes", []) if now - z[0] < 1.0]
        if seen and seen["player"]:
            self.bearings.append((now, -math.degrees(yaw) + cv.X_DEG[seen["x"]]))
            self.sizes.append((now, cv.SIZE_RANK[seen["size"]]))

    def question(self, obs, name, seen, last):
        if self.variant == "pure":
            looked = [b for _, b in getattr(self, "looked", [])]
            if seen is not None and not seen["player"]:
                looked.append(math.degrees(obs["yaw"]))
            self.pure_tgt = cv.pure_target(seen, last)
            # when the picture the aim comes from was taken: this step's (2step) or the previous one's (1step)
            self.pure_pic_t = self.pic_t if seen is not None and seen["player"] else getattr(self, "prev_pic_t", self.pic_t)
            q = cv.pure_options(obs, name, seen, self.mem.history, last, looked, v=self.v, sizes=getattr(self, "sizes", []))
            return shuffled(q) if self.shuffle else q
        if self.variant != "raw":
            q = cv.body_options(obs, name, seen, self.mem.history, last, v=self.v, sizes=getattr(self, "sizes", []))
        else:
            looked = [b for _, b in getattr(self, "looked", [])]
            if seen is not None and not seen["player"]:
                looked.append(math.degrees(obs["yaw"]))   # this picture: nobody here either
            q = cv.raw_options(obs, name, seen, self.mem.history, last, looked, omega=self.omega())
        return shuffled(q) if self.shuffle else q

    def step(self, obs):
        cfg, mem, ins = self.cfg, self.mem, self.mem.instruction
        if "target" not in (ins or {}):
            self.prepare(obs)
            return
        if not self.body.camera:
            raise SystemExit(f"{self.harness_name} needs the camera: start the agent with --camera-size 256x256")
        if mem.history and "after" not in mem.history[-1]:
            h = mem.history[-1]
            h["after"] = True
            h["result"]["moved"] = round(math.hypot(obs["pos"]["x"] - h["pos0"]["x"], obs["pos"]["z"] - h["pos0"]["z"]), 2)
        tgt = ins["target"]
        name = tgt.get("name") if tgt["kind"] == "player" else None
        self.focus_camera(name)
        t0 = time.time()
        image = self.body.snapshot()
        t_snap = time.time() - t0
        self.prev_pic_t, self.pic_t = getattr(self, "pic_t", t0), t0
        probe = {"id": "look", "state": cv.LOOK_STATE, "images": [image],
                 "questions": cv.LOOK_QUESTIONS_RAW if self.variant == "raw" else cv.LOOK_QUESTIONS}
        if self.one_step:
            # the picture now, plus what the previous picture was read as (one step old): the text of the past
            prev = getattr(self, "prev_look", None)
            prev_t = None
            if prev:
                n0, t0s, was, yaw0 = prev
                turned = math.degrees(math.atan2(math.sin(yaw0 - obs["yaw"]), math.cos(yaw0 - obs["yaw"])))
                prev_t = (ins["steps"] - n0, time.time() - t0s, was, turned)
            state = cv.step_state(obs, mem, cfg, None, self.variant, picture=True, prev=prev_t, v=self.v, omega=self.omega(),
                                  sizes=getattr(self, "sizes", []))
            last = prev_t if prev_t and prev_t[2]["player"] else None
            question = self.question(obs, name, None, last)
            answers, execution, dt = self.ask([{"id": "step", "state": state, "images": [image], "questions": {"action": question}}, probe])
            look, t_look = answers, 0.0
            seen = cv.read_look(answers["look"])
            self.prev_look = (ins["steps"], time.time(), seen, obs["yaw"])
            self.note_bearing(seen, obs["yaw"])
        else:
            look, _, t_look = self.ask([probe])
            seen = cv.read_look(look["look"])
            self.note_bearing(seen, obs["yaw"])
            last = None
            if seen["player"]:
                self.sighting = (ins["steps"], time.time(), seen, obs["yaw"])
            elif getattr(self, "sighting", None) and ins["steps"] - self.sighting[0] <= 20:
                n0, t0s, was, yaw0 = self.sighting
                turned = math.degrees(math.atan2(math.sin(yaw0 - obs["yaw"]), math.cos(yaw0 - obs["yaw"])))   # + right
                last = (ins["steps"] - n0, time.time() - t0s, was, turned)
            state = cv.step_state(obs, mem, cfg, seen, self.variant, last=last, v=self.v, omega=self.omega(),
                                  sizes=getattr(self, "sizes", []))
            question = self.question(obs, name, seen, last)
            answers, execution, dt = self.ask([{"id": "step", "state": state, "questions": {"action": question}}])
        a = answers["step"]
        probs = a["action"]["probabilities"]
        action = max(probs, key=probs.get)
        now = time.time()
        self.looked = [(t, b) for t, b in getattr(self, "looked", []) if now - t < 5]
        if not seen["player"]:
            self.looked.append((now, math.degrees(obs["yaw"])))
        else:
            self.looked = []
        f = cp.fighter(obs, name) if name else None   # ground truth for the log only, never in the state
        truth = f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2), "angle": round(v1.relative(obs, f["pos"])["angle"], 1),
                       "their_health": f.get("health")}
        record = dict(harness=self.harness_name, instruction=ins["text"], step=ins["steps"] + 1, state=state, seen=seen,
                      look_answers=look["look"], questions={"action": question}, answers=a, execution=execution,
                      seconds=dt, seconds_look=t_look, seconds_snapshot=round(t_snap, 3), action=action, done=0.0,
                      truth=truth, fight=truth and {**truth, "health": obs["health"], "held": obs.get("held_item")},
                      image=image if ins["steps"] % self.image_every == 0 else None)
        body_action, params = self.act_params(action)
        result = self.body.act(body_action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg, target=name, guard=GUARD, autojump=True, **params)
        mem.history.append({"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        self.log.write("step", **record, act=result, effect=f"{action}; {result.get('note') or 'keys ' + ', '.join(result.get('held', []))}")
        sees = f"sees {seen['x']}/{seen['size']}" if seen["player"] else "sees nobody"
        print(f"   p={probs[action]:.2f} {(t_snap + t_look + dt) * 1000:4.0f}ms  {action:12s} {result.get('note', '')}"
              f"  [{sees}; truth {truth['angle'] if truth else '-'}° {truth['dist'] if truth else '-'}b]")


class AgentUVisBody(AgentUVis):
    harness_name = "p_uhc_vis_body_2step_v1"
    variant = "body"


class AgentUVisBody2(AgentUVisBody):
    harness_name = "p_uhc_vis_body_2step_v2"
    v, shuffle = 2, True


class AgentUVisBody3(AgentUVisBody):
    """v3: the attack indicator and 'coming at you' (growing fast) in the state and on the sword/bow options."""
    harness_name = "p_uhc_vis_body_2step_v3"
    v, shuffle = 3, True


class AgentUVisBody1(AgentUVisBody):
    """One Jev call a step: the picture now with last step's reading of it, plus the v3 cues (about half the time
    of 2step, which waits for the look call before deciding)."""
    harness_name = "p_uhc_vis_body_1step_v1"
    v, shuffle, one_step = 3, True, True


class AgentUVisPure(AgentUVis):
    """p_uhc_vis_body_pure_*: vision+body with no coordinates at all: the body aims only where the picture showed
    the opponent (body/vis_aim.js), and cannot aim at someone not seen."""
    harness_name = "p_uhc_vis_body_pure_2step_v1"
    variant = "pure"
    v, shuffle = 3, True


class AgentUVisPure1(AgentUVisPure):
    harness_name = "p_uhc_vis_body_pure_1step_v1"
    one_step = True


class AgentUVisRaw(AgentUVis):   # not released yet: improved in place
    harness_name = "p_uhc_vis_raw_2step_v1"
    variant = "raw"
    shuffle = True


class AgentUVisRaw1(AgentUVis):
    harness_name = "p_uhc_vis_raw_1step_v1"
    variant = "raw"
    one_step = True
    shuffle = True


class AgentUVisRawV2(AgentU1):
    """p_uhc_vis_raw_v2 (context_uhc_visraw2.py): the picture only, raw keys and mouse, Jev every step. One request a
    step: the decision (the picture now, the compact state) and a look at the same picture, whose answers are the
    next step's 'last look'. No held turns, no coordinates."""
    harness_name = "p_uhc_vis_raw_v2"
    image_every = 10

    def step(self, obs):
        import context_uhc_visraw2 as c2
        cfg, mem, ins = self.cfg, self.mem, self.mem.instruction
        if "target" not in (ins or {}):
            self.prepare(obs)
            return
        if not self.body.camera:
            raise SystemExit(f"{self.harness_name} needs the camera: start the agent with --camera vox")
        if mem.history and "after" not in mem.history[-1]:
            h = mem.history[-1]
            h["after"] = True
            h["result"]["moved"] = round(math.hypot(obs["pos"]["x"] - h["pos0"]["x"], obs["pos"]["z"] - h["pos0"]["z"]), 2)
        tgt = ins["target"]
        name = tgt.get("name") if tgt["kind"] == "player" else "them"
        self.focus_camera(name if tgt["kind"] == "player" else None)
        t_pic = time.time()
        image = self.body.snapshot()
        look = getattr(self, "look", None)
        ago = t_pic - getattr(self, "look_t", t_pic)
        yaw0 = getattr(self, "look_yaw", obs["yaw"])
        turned = math.degrees(math.atan2(math.sin(yaw0 - obs["yaw"]), math.cos(yaw0 - obs["yaw"])))   # + right
        self.looked = [(t, b) for t, b in getattr(self, "looked", []) if t_pic - t < 4]
        ls = getattr(self, "lastseen", None)          # (time, bearing) of the latest sighting
        lastseen = (t_pic - ls[0], ls[1]) if ls else None
        state = c2.state(obs, name, look, ago, turned, mem.history, lastseen)
        question = shuffled(c2.options(obs, name, look, turned, [b for _, b in self.looked], mem.history, lastseen))
        answers, execution, dt = self.ask([
            {"id": "step", "state": state, "images": [image], "questions": {"action": question}},
            {"id": "look", "state": c2.LOOK_STATE, "images": [image], "questions": c2.LOOK}])
        probs = answers["step"]["action"]["probabilities"]
        action = max(probs, key=probs.get)
        seen = c2.read(answers["look"])
        self.look, self.look_t, self.look_yaw = seen, t_pic, obs["yaw"]
        if seen["player"]:
            self.looked = []
            self.lastseen = (t_pic, -math.degrees(obs["yaw"]) + cv.X_DEG[seen["x"]])
        else:
            self.looked.append((t_pic, math.degrees(obs["yaw"])))
        f = cp.fighter(obs, name)   # the truth, for the log only
        truth = f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2), "angle": round(v1.relative(obs, f["pos"])["angle"], 1),
                       "their_health": f.get("health")}
        body_action, params = cr.act_params(action)
        result = self.body.act(body_action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg, target=name, guard=GUARD, autojump=True, **params)
        mem.history.append({"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        self.log.write("step", harness=self.harness_name, instruction=ins["text"], step=ins["steps"], state=state,
                       seen=seen, questions={"action": question}, answers=answers["step"], execution=execution, seconds=dt,
                       action=action, truth=truth, fight=truth and {**truth, "health": obs["health"], "held": obs.get("held_item")},
                       image=image if ins["steps"] % self.image_every == 0 else None, act=result,
                       effect=f"{action}; {result.get('note') or ''}")
        sees = f"sees {seen['x']}/{seen['size']}/{seen['side']}" if seen["player"] else "sees nobody"
        print(f"   p={probs[action]:.2f} {dt * 1000:4.0f}ms  {action:12s} {(result.get('note') or '')[:40]}"
              f"  [{sees}; truth {truth['angle'] if truth else '-'}° {truth['dist'] if truth else '-'}b]")


class AgentUVisRawV3(AgentU1):
    """p_uhc_vis_raw_v3 (context_uhc_visraw3.py): the pictures only, Jev every step with three hands at once - the
    keys held, a mouse move, a button. Two requests a step: first two looks at the same moment (the view, and a zoomed
    view of its middle: a scope), then the decision from what they saw (text only: with its picture too, the request
    went past the fast engine's one-pass size and took 105 ms instead of 47). No coordinates."""
    harness_name = "p_uhc_vis_raw_v3"
    image_every = 10

    def step(self, obs):
        import context_uhc_visraw3 as c3
        cfg, mem, ins = self.cfg, self.mem, self.mem.instruction
        if "target" not in (ins or {}):
            self.prepare(obs)
            return
        if not self.body.camera:
            raise SystemExit(f"{self.harness_name} needs the camera: start the agent with --camera vox")
        if mem.history and "after" not in mem.history[-1]:
            h = mem.history[-1]
            h["after"] = True
            h["result"]["moved"] = round(math.hypot(obs["pos"]["x"] - h["pos0"]["x"], obs["pos"]["z"] - h["pos0"]["z"]), 2)
        tgt = ins["target"]
        name = tgt.get("name") if tgt["kind"] == "player" else "them"
        self.focus_camera(name if tgt["kind"] == "player" else None)
        t_pic = time.time()
        image, zoom = self.body.snapshot_zoom(c3.ZOOM)
        heading = -math.degrees(obs["yaw"])     # degrees right of north
        looks, _, dt_look = self.ask([
            {"id": "look", "state": c3.LOOK_STATE, "images": [image], "questions": c3.LOOK},
            {"id": "scope", "state": c3.SCOPE_STATE, "images": [zoom], "questions": c3.SCOPE}])
        seen = c3.read(looks["look"], looks["scope"])
        fused = c3.fuse(seen)
        self.sightings = [(t, b) for t, b in getattr(self, "sightings", []) if t_pic - t < c3.SIGHTINGS_S]
        self.looked = [(t, b) for t, b in getattr(self, "looked", []) if t_pic - t < 4]
        if fused:
            self.looked = []
            self.sightings.append((t_pic, heading + fused["x"]))
        else:
            self.looked.append((t_pic, math.degrees(obs["yaw"])))
        now = time.time()
        est = c3.estimate(fused, t_pic, heading, self.sightings, now, heading)
        if fused:     # the latest sighting: where they go when they leave the view
            self.lastseen = {"t": t_pic, "bearing": heading + fused["x"], "omega": est["omega"], "x": fused["x"],
                             "heading": heading}
        lastseen = getattr(self, "lastseen", None)
        knock = c3.knock_bearing(obs) if not est else None
        state = c3.state(obs, name, est, fused, now - t_pic, lastseen, mem.history, now)
        qk = shuffled(c3.keys_question(obs, est, knock))
        qm = shuffled(c3.mouse_question(obs, est, lastseen, [b for _, b in self.looked], knock, now))
        qb = shuffled(c3.button_question(obs, name, est, mem.history))
        answers, execution, dt = self.ask([{"id": "step", "state": state, "questions": {"keys": qk, "mouse": qm, "button": qb}}])
        pick = {q: max(answers["step"][q]["probabilities"], key=answers["step"][q]["probabilities"].get)
                for q in ("keys", "mouse", "button")}
        f = cp.fighter(obs, name)   # the truth, for the log only
        truth = f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2), "angle": round(v1.relative(obs, f["pos"])["angle"], 1),
                       "their_health": f.get("health")}
        dyaw, dpitch = c3.mouse_move(pick["mouse"])
        result = self.body.act("input", hold=True, target=name, guard=GUARD, autojump=True, keys=c3.KEYS[pick["keys"]],
                               dyaw=dyaw, dpitch=dpitch, button=None if pick["button"] == "none" else pick["button"])
        action = f"{pick['keys']}|{pick['mouse']}|{pick['button']}"
        mem.history.append({"action": action, "button": pick["button"], "words": c3.words(pick["keys"], pick["mouse"], pick["button"]),
                            "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        self.log.write("step", harness=self.harness_name, instruction=ins["text"], step=ins["steps"], state=state,
                       seen=seen, est=est, questions={"keys": qk, "mouse": qm, "button": qb}, answers=answers["step"],
                       execution=execution, seconds=dt, seconds_look=dt_look, action=action, truth=truth,
                       in_reach=c3.cv.in_reach(obs), fight=truth and {**truth, "health": obs["health"], "held": obs.get("held_item")},
                       image=image if ins["steps"] % self.image_every == 0 else None,
                       zoom=zoom if ins["steps"] % self.image_every == 0 else None, act=result,
                       effect=f"{action}; {result.get('note') or ''}")
        sees = f"x {est['x']:+.0f}{'*' if est['fine'] else ''} {est['range']} w{est['omega']}" if est else "-"
        print(f"   {dt_look * 1000:3.0f}+{dt * 1000:3.0f}ms  {action:28s} {(result.get('note') or '')[:30]:30s}"
              f"  [{sees}; truth {truth['angle'] if truth else '-'}° {truth['dist'] if truth else '-'}b]")


class AgentURaw3(AgentU1):
    """p_uhc_raw_v3 (context_uhc_raw3.py): the game's data in, raw controls out - every step Jev chooses the keys held,
    the mouse (left/right, up/down) and a button, four questions of one text state answered in one pass. The body
    only presses what it is told (plus the client's auto-jump and the edge guard); no aiming, no swings of its own."""
    harness_name = "p_uhc_raw_v3"

    def step(self, obs):
        import context_uhc_raw3 as c
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
        f = cp.fighter(obs, name) if name else None
        now = time.time()
        self.track = [(t, p) for t, p in getattr(self, "track", []) if now - t < c.TRACK_S] + ([(now, f["pos"])] if f else [])
        vel = c.velocity(self.track)
        if f and f.get("holding") == "bow" and f.get("using"):      # drawing: they walk at a fifth - aim at them, no lead
            vel = (vel[0] * 0.2, vel[1], vel[2] * 0.2)
        a = c.aim(obs, f, vel) if f else None
        obs["_vel"] = vel
        state = c.state(obs, name or "them", f, a, vel, mem.history)
        qk = shuffled(c.keys_question(obs, f, name, a))
        qy, qp = (shuffled(q) for q in c.mouse_questions(obs, f, name, a))
        qb = shuffled(c.button_question(obs, f, name, a, mem.history))
        answers, execution, dt = self.ask([{"id": "step", "state": state,
                                            "questions": {"keys": qk, "yaw": qy, "pitch": qp, "button": qb}}])
        ans = answers["step"]
        pick = {q: max(ans[q]["probabilities"], key=ans[q]["probabilities"].get) for q in ("keys", "yaw", "pitch", "button")}
        dyaw, dpitch = c.mouse_move(pick["yaw"], pick["pitch"])
        result = self.body.act("input", hold=True, target=name, guard=GUARD, autojump=True, arrowwarn=True,
                               keys=c.KEYS[pick["keys"]], dyaw=dyaw, dpitch=dpitch,
                               button=None if pick["button"] == "none" else pick["button"])
        action = f"{pick['keys']}|{pick['yaw']}|{pick['pitch']}|{pick['button']}"
        mem.history.append({"action": action, "button": pick["button"], "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"],
                            "words": c.words(pick["keys"], pick["yaw"], pick["pitch"], pick["button"])})
        mem.prev_obs, mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        fight = f and {"dist": round(cp.reach_of(obs, f), 2), "angle": round(v1.relative(obs, f["pos"])["angle"], 1),
                       "their_health": f.get("health"), "health": obs["health"], "held": obs.get("held_item")}
        self.log.write("step", harness=self.harness_name, instruction=ins["text"], step=ins["steps"], state=state,
                       questions={"keys": qk, "yaw": qy, "pitch": qp, "button": qb}, answers=ans, execution=execution,
                       seconds=dt, action=action, aim=a, vel=[round(v, 2) for v in vel], in_reach=c.cp.reach_of(obs, f) <= c.REACH + 0.3 if f else False,
                       indicator=bool(obs.get("cursor_entity") and obs["cursor_entity"].get("name") == name
                                      and obs["cursor_entity"]["distance"] <= c.REACH + 0.3),
                       fight=fight, act=result, effect=f"{action}; {result.get('note') or ''}")
        aimw = f"aim {a['yaw']:+.0f}/{a['pitch']:+.0f}" if a else "-"
        print(f"   {dt * 1000:3.0f}ms  {action:34s} {(result.get('note') or '')[:28]:28s} [{aimw}; "
              f"{fight['dist'] if fight else '-'}b hp {obs['health']:.0f} vs {fight['their_health'] if fight else '-'}]")
