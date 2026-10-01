"""Harness p_uhc_pro (./agent.py --harness p_uhc_pro_v1): Block UHC, text + body, split into head and hands.

The P1/U1 loop (chat judgement, target, recalled facts, one Jev choice per step, held keys; the server starts and
ends the fights; dying ends one) with context_uhc_pro.py: Jev chooses tactics and movement, the body's engage
controller does the sword work (body/body.js). Options are shuffled every step.
"""
import json, math, os, random, re, time

import context as v1
import context_pvp as cp
import context_survive as cs
import context_uhc_pro as pro
from agent_pvp import AgentP1
from jev import OfficialJev

SUDDEN_RE = re.compile(r"sudden death\b", re.I)
FINAL_RE = re.compile(r"final|sudden death", re.I)
FIGHT_RE = re.compile(r"(team )?fight\b", re.I)
NO_BLOCKS_RE = re.compile(r"cannot place blocks|no placing blocks", re.I)
PLACING = ("place_block", "build_cover", "build_window", "water_block", "water_escape")   # options that place blocks
GUARD = 4   # drops of 4 blocks or more stop your keys (a fall that hurts); one-block steps are auto-jumped


def shuffled(question):
    items = list(question["criteria"].items())
    random.shuffle(items)
    return {**question, "criteria": dict(items)}


class AgentUPro(AgentP1):
    harness_name = "p_uhc_pro_v1"
    v = 1

    def start_fight(self, obs, name):
        """The server starts the fights."""
        return

    def take_info(self, m):
        """The arena's "sudden death: from now on every hit does 2 times the damage ...; keep fighting the same enemy"
        (and "sudden death, final duel: ...") is news for the fight in progress: Jev judged it a new instruction,
        which replaced the fight's and, naming no one, left the target none for the rest of the round. Kept on the
        instruction instead; the state quotes it (context_uhc_pro.news_lines)."""
        if self.from_server(m) and SUDDEN_RE.match(m["message"].strip()):
            if self.mem.instruction:
                self.mem.instruction.setdefault("news", []).append({"t": m.get("t") or time.time(), "text": m["message"].strip()})
            self.log.write("info", text=m["message"])
            print(f'   {m["username"]}: "{m["message"]}" -> news for the fight')
            return True
        if self.v >= 10 and m.get("via") != "console":
            # The user: Jev only fights - it takes no orders from chat. The server's own whispers are read here by
            # rule, without asking Jev ("fight ...", "team fight ...", "stop"; harness orders and sudden death are
            # read before); the arena's announcements are news (the final ones go to the state), never orders; all
            # else is left alone. (The operator's console lines still go to Jev.)
            text = m["message"].strip()
            if self.from_server(m):
                if text.lower() == "stop":
                    if self.mem.instruction:
                        self.finish(f"stopped by {m['username']}")
                        self.say(self.cfg.say_stop)
                    print(f'   {m["username"]}: "stop" -> stop')
                    return True
                if FIGHT_RE.match(text):
                    print(f'   {m["username"]}: "{text}" -> fight')
                    self.start(m, once=False)
                    return True
            elif m["username"] == "Arena" and "Arena" not in self.uuid_versions:
                if self.mem.instruction and FINAL_RE.search(text):
                    self.mem.instruction.setdefault("news", []).append({"t": m.get("t") or time.time(), "text": text})
            self.log.write("info", text=text, sender=m["username"], taken=False)
            return True
        return False

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
        # the time between the two observations (prev_obs and this one): a step's own length (a 1.5 s shot) must
        # not count, or a distance change over 1.6 s divided by 0.07 s reads as 90 blocks a second
        obs_t = time.time()
        dt = obs_t - mem.prev_obs_t if mem.prev_obs and getattr(mem, "prev_obs_t", None) else None
        if self.v >= 10:
            self.ingest(obs)
            if getattr(self, "team_mode", False):
                name = self.refocus(obs) or name
            if not self.decision_moment(obs, name):
                time.sleep(0.02)   # the draw goes on in the body; Jev is asked at its moments
                return
            if getattr(self, "team_mode", False) and ins.get("team"):
                shared = self.shared()
                state = pro.step_state_team(obs, mem, cfg, self.match, ins["team"], shared, dt)
                question = shuffled(pro.options_team(obs, name, ins["team"], shared, mem.history))
            elif self.v >= 11:
                state = pro.step_state11(obs, mem, cfg, self.match, dt)
                question = shuffled(pro.options11(obs, name, mem.history)) if name else cs.action_question(cfg, obs)
            else:
                state = pro.step_state10(obs, mem, cfg, self.match, dt)
                question = shuffled(pro.options10(obs, name, mem.history)) if name else cs.action_question(cfg, obs)
            if any(NO_BLOCKS_RE.search(n["text"]) for n in ins.get("news", [])):   # the final duel: the server refuses them
                question = {**question, "criteria": {k: v for k, v in question["criteria"].items() if k not in PLACING}}
        elif self.v >= 9:
            state = pro.step_state9(obs, mem, cfg, dt)
            question = shuffled(pro.options9(obs, name, mem.history)) if name else cs.action_question(cfg, obs)
        elif self.v >= 8:
            state = pro.step_state8(obs, mem, cfg, dt)
            question = shuffled(pro.options8(obs, name, mem.history)) if name else cs.action_question(cfg, obs)
        elif self.v >= 7:
            state = pro.step_state7(obs, mem, cfg, dt)
            question = shuffled(pro.options7(obs, name, mem.history)) if name else cs.action_question(cfg, obs)
        elif self.v >= 5:
            state = pro.step_state5(obs, mem, cfg, dt, v=self.v)
            question = shuffled(pro.options5(obs, name, mem.history, v=self.v)) if name else cs.action_question(cfg, obs)
        else:
            state = pro.step_state(obs, mem, cfg, dt, v=self.v)
            question = (shuffled(pro.options(obs, name, mem.history, v=self.v, dt=dt, prev=mem.prev_obs)) if name
                        else cs.action_question(cfg, obs))
        answers, execution, secs = self.ask([{"id": "step", "state": state, "questions": {"action": question}}])
        a = answers["step"]
        probs = a["action"]["probabilities"]
        action = max(probs, key=probs.get)
        f = cp.fighter(obs, name) if name else None
        record = dict(harness=self.harness_name, instruction=ins["text"], step=ins["steps"] + 1, state=state,
                      questions={"action": question}, answers=a, execution=execution, seconds=secs, action=action, done=0.0,
                      fight=f and {"dist": round(v1.relative(obs, f["pos"])["dist"], 2), "their_health": f.get("health"),
                                   "health": obs["health"], "held": obs.get("held_item"), "engage": obs.get("engage")})
        body_action, params = pro.act_params(action)
        if body_action.startswith("focus_"):   # a team fight: the fight's focus moves to another enemy
            name = body_action[len("focus_"):]
            ins["target"] = {"kind": "player", "name": name}
            body_action = "wait"
        if ins.get("team"):
            params = {**params, "enemies": ins["team"]["enemies"], "allies": ins["team"]["allies"]}
        if self.v >= 4:   # the body's bow rule: the sword as soon as they come close (an early release stays Jev's call)
            params = {**params, "bowfull": False, "bowswitch": 4.0}
        if self.v >= 5:   # and it does not walk into lava or fire (its own trap included)
            params["lavaguard"] = True
        if self.v >= 7:   # and a bucket or blocks go back for the sword when they come close
            params["weaponswitch"] = 4.5
        if self.v >= 8:   # and it sidesteps an arrow it sees coming
            params["dodge"] = True
        if os.environ.get("JEV_LEAD_ROTATE"):   # a comparison of lead modes against a person (whispered per arrow)
            params["leadrotate"] = True
        result = self.body.act(body_action, hold=True, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg, target=name, guard=GUARD, autojump=True, **params)
        mem.history.append({"action": action, "result": result, "yaw0": obs["yaw"], "pos0": obs["pos"]})
        if ins.get("team"):
            self.publish(obs, name, action)
        mem.prev_obs, mem.last_step_t, mem.prev_obs_t = obs, time.time(), obs_t
        ins["steps"] += 1
        self.log.write("step", **record, act=result, effect=f"{action}; {result.get('note') or 'keys ' + ', '.join(result.get('held', []))}")
        fight = record["fight"]
        print(f"   p={probs[action]:.2f} {secs * 1000:4.0f}ms  {action:14s} {result.get('note', '')}"
              + (f"  dist {fight['dist']:.1f} hp {fight['health']:.0f} vs {fight['their_health']}" if fight else ""))

    def finish(self, outcome):
        """Whatever ends the fight (stop, death, a new instruction) also ends the body's engage."""
        try:
            self.body.act("disengage", hold=True)
        except Exception:
            pass
        return super().finish(outcome)


class AgentUPro2(AgentUPro):
    """p_uhc_pro_v2: switching the engage style is marked as a switch (v1 flipped styles every step), keeping the
    fight going is a clear option, and an early release of the bow says what it loses."""
    harness_name = "p_uhc_pro_v2"
    v = 2


class AgentUPro3(AgentUPro):
    """p_uhc_pro_v3: v2 plus tactics beyond the sword trade: the race says when a trade is a coin flip or lost and
    what else takes their health; a lava trap laid in their path; the bow on a straight runner; the opponent's sword
    timing (step in while it recharges, out while yours does); backing off from a burning opponent."""
    harness_name = "p_uhc_pro_v3"
    v = 3


class AgentUPro4(AgentUPro):
    """p_uhc_pro_v4: v3 with the body's bow rule - a bow in hand (drawn or not) is dropped for the sword as soon as the
    opponent is within 4 blocks - and no drawing up close; releasing early stays a choice, marked with what it loses."""
    harness_name = "p_uhc_pro_v4"
    v = 4


class AgentUPro5(AgentUPro):
    """p_uhc_pro_v5: the user's techniques and when to use them, no warnings (context_uhc_pro.step_state5/options5):
    close - sword; far - the bow with lead; under arrows - cover or a zigzag; fairly close and coming - a lava or water
    trap. Each option says what it does and which technique it is; the state says which techniques match now. Keeps
    v4's body rule (the bow dropped for the sword within 4 blocks) and shuffled options."""
    harness_name = "p_uhc_pro_v5"
    v = 5


class AgentUPro6(AgentUPro):
    """p_uhc_pro_v6: v5 with a full-power shot as one choice (the body draws 1 s with lead, then lets go; v5 let go
    of 60 draws after 0.0-0.2 s), 'draw and hold' kept for waiting behind cover, and the lava guard (v5 walked into
    its own lava trap; the guard came with v5's fix, and v6 keeps it)."""
    harness_name = "p_uhc_pro_v6"
    v = 6


class AgentUPro7(AgentUPro):
    """p_uhc_pro_v7: the user's play distilled into the techniques (run at an archer in a zigzag, lava at their feet
    rather than a trap they can walk around, the sword back right after an item), the body's weapon rule (anything but
    a weapon goes back for the sword within 4.5 blocks: v6 kept an empty bucket in hand while being hit), and poured
    liquids predicted before they are poured (the body simulates the server's bucket ray and only pours where it lands
    as wanted and, for lava, 2+ blocks from itself; the options say where it would land, or are left out)."""
    harness_name = "p_uhc_pro_v7"
    v = 7


class AgentUPro8(AgentUPro):
    """p_uhc_pro_v8: v7 plus moving between shots (body: shoot_hop - a full-power shot, then a hop sideways, alternating
    sides), a steadier lead (a least-squares fit over the last half second of their positions, capped at a player's
    top speed), the body's arrow dodge (an arrow they shoot on a line that meets you gets a sidestep across it), crits
    that land while Jev holds jump (v7's crit style never swung in the air: 0 swings in 3 s of a close fight), and
    running in on an archer as a mid-range move (v7 ran at the user from 35-40 blocks and died to the arrows)."""
    harness_name = "p_uhc_pro_v8"
    v = 8


class AgentUPro8Official(AgentUPro8):
    """p_uhc_pro_v8_official: v8 with its decisions from the official Jev (TypeSafe System One API, model
    JEV_OFFICIAL_MODEL, default jev-latest) instead of the local djev. Text only, as every Pro version."""
    harness_name = "p_uhc_pro_v8_official"

    def __init__(self, body, jev, cfg):
        super().__init__(body, OfficialJev(os.environ["TYPESAFE_API_KEY"], os.environ.get("JEV_OFFICIAL_MODEL", "jev-latest")), cfg)


class AgentUPro9(AgentUPro):
    """p_uhc_pro_v9: the user's notes on v8 (it beat them 3-0): cover, footwork up close, better arrows.
    - Cover: the body knows the spots within 3 blocks where blocks cut the opponent's line of fire; take_cover walks
      there, cover_shot draws behind the cover and shoots over it from the top of a jump.
    - Footwork: circle - in to hit when the sword is ready, out of their reach while it recharges, always sideways.
    - Arrows: a steady mover is led by the last 0.3 s, a weaver aimed at the middle of the weave, a jumper along the
      jump's arc; a full draw waits up to 0.3 s for a moment their path is set.
    - Dodging: the arrow is flown ahead as the server flies it against your box (v8's straight-line check never
      fired), and a bow drawn 0.6 s+ and aimed at you starts you across its line before the release.
    - When: shoot_hop only while they hold a bow (else back-to-back shots); no run-in from beyond 15 blocks at an
      archer while you can shoot back."""
    harness_name = "p_uhc_pro_v9"
    v = 9


ROUND_RE = re.compile(r"Round (\d+): (\S+) wins \((.+?), ([\d.]+)s\)\. (\S+) (\d+) - (\d+) (\S+)")


def new_round(n):
    return {"n": n, "t0": time.time(), "my_shots": 0, "my_hits": 0, "their_shots": 0, "their_hits": 0, "dodged": 0,
            "lost_by": {}}


class AgentUPro10(AgentUPro):
    """p_uhc_pro_v10: a layered state (context_uhc_pro.step_state10) and the draw as a task.
    - Layers: fixed (rules, techniques); the match (score, a line a round, the opponent so far: their arrows' hit
      rate, yours, their turn rhythm) and this round (time, health both ways, arrows both ways, what hurt you) -
      counted up here from the body's events and the arena's round messages; the last seconds (events with times);
      now (them, you, the draw's details, items, the last three choices, repeats folded).
    - The draw: bow_draw starts it and returns; each step Jev may let go (bow_release), hold (wait), lower it
      (bow_cancel) or jump over cover (jump_shot), seeing its power, whether the shot is clear, the flight and
      the lead. The body still lowers it to dodge and drops it for the sword within 4 blocks.
    - v9's bow macros (shoot_full, shoot_hop, bow_barrage, quick_shot, cover_shot) are gone: Jev composes them.
    - While drawing, Jev is asked at the draw's moments (decision_moment), not every step."""
    harness_name = "p_uhc_pro_v10"
    v = 10
    bots_by_uuid = True   # bots by UUID, people by any name (agent.Agent.bots_by_uuid); v11 and team inherit it

    def __init__(self, body, jev, cfg):
        super().__init__(body, jev, cfg)
        self.match = {"rounds": [], "cur": None, "score": None}
        self.seen_event, self.seen_chat = 0, None
        self.was_full, self.woke, self.woke_event = False, 0.0, 0

    WAKE = ("their_shot", "exposed", "covered", "they_switch", "hurt", "they_hurt", "drop_draw", "reflex", "my_result")

    def decision_moment(self, obs, name):
        """While a draw is in progress, Jev is asked at its moments - full power, then every 0.5 s; they come out of
        or go behind cover; an arrow of theirs, an item in their hand, a hit either way, a reflex; them within 8
        blocks - not every 40 ms, where a small chance each step of letting go early adds up over a draw."""
        k = obs.get("task")
        now = time.time()
        if not k:
            self.was_full, self.woke = False, now
            return True
        ev = [e for e in obs.get("events", []) if e["id"] > self.woke_event and e["kind"] in self.WAKE]
        f = cp.fighter(obs, name) if name else None
        near = f and v1.relative(obs, f["pos"])["dist"] < 8
        full_now = k["power"] >= 1 and not self.was_full
        if ev or near or full_now or now - self.woke >= 0.5:
            self.was_full = k["power"] >= 1
            self.woke = now
            if ev:
                self.woke_event = max(e["id"] for e in ev)
            return True
        return False

    def ingest(self, obs):
        """The round's counts from the body's new events; a round closed by the arena's message."""
        chat = list(self.mem.chat)
        for m in chat:
            seq = m.get("seq")
            if self.seen_chat is not None and isinstance(seq, int) and seq <= self.seen_chat:
                continue
            if isinstance(seq, int):
                self.seen_chat = seq
            mm = ROUND_RE.search(m.get("message", ""))
            if mm:
                n = int(mm.group(1))
                if self.match["rounds"] and n <= self.match["rounds"][-1]["n"]:   # a new match: its own counts
                    self.match = {"rounds": [], "cur": self.match["cur"], "score": None}
                cur = self.match["cur"] or new_round(n)
                cur.update(n=int(mm.group(1)), won=mm.group(2) == obs["name"], how=mm.group(3), secs=float(mm.group(4)))
                self.match["rounds"].append(cur)
                self.match["score"] = {mm.group(5): int(mm.group(6)), mm.group(8): int(mm.group(7))}
                self.match["cur"] = None
        if self.match["cur"] is None:
            self.match["cur"] = new_round(len(self.match["rounds"]) + 1)
        cur = self.match["cur"]
        for e in obs.get("events", []):
            if e["id"] <= self.seen_event:
                continue
            self.seen_event = e["id"]
            k = e["kind"]
            if k == "my_shot":
                cur["my_shots"] += 1
            elif k == "my_result" and e.get("hit"):
                cur["my_hits"] += 1
            elif k == "their_shot":
                cur["their_shots"] += 1
            elif k == "dodged":
                cur["dodged"] += 1
            elif k == "hurt":
                if e["cause"] == "a hit" and e["dmg"] >= 10:   # no known cause and a big step: an effect ending, a reset
                    continue
                cur["lost_by"][e["cause"]] = cur["lost_by"].get(e["cause"], 0) + e["dmg"]
                if e["cause"] == "their arrow":
                    cur["their_hits"] += 1


class AgentUPro11(AgentUPro10):
    """p_uhc_pro_v11: v10 plus taking the fight to a hider (the user: "if the opponent keeps hiding, take the initiative and attack").
    Two v10s in a bot duel stood 46 blocks apart, each behind cover with a full draw, waiting. v11's state says how
    long they have been out of your line of fire and since the last arrow either way; after 3 s of it, find_angle
    (the body walks to a spot nearby with a clear shot, closer first) and a zigzag run-in at any range are offered;
    technique 6 says to take the fight to a hider."""
    harness_name = "p_uhc_pro_v11"
    v = 11


class AgentUProTeam(AgentUPro10):
    """p_uhc_pro_team_v1: 2v2 Block UHC, v10 plus a teammate.
    - The arena whispers "team fight - allies: <ally>; enemies: <e1>, <e2>: ..."; prepare reads it without asking Jev
      and fights the nearest enemy; when that one is out of the fight, the focus moves to the other.
    - The state adds your team (the teammate's place, health, hand, and whom they fight and what they just chose, from
      the team's shared notes in runs/team/) and the other enemy (and whether they are on your teammate); technique 7:
      two on one, help a teammate under attack, keep out of each other's line of fire.
    - focus_<name> options switch the fight; the body watches every enemy's arrows and holds an arrow that would meet
      an ally."""
    harness_name = "p_uhc_pro_team_v1"
    v = 10
    team_mode = True
    TEAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "team")

    def prepare(self, obs):
        ins = self.mem.instruction
        team = pro.parse_team(ins.get("text"))
        if not team:
            return super().prepare(obs)
        ins["team"], ins["yaw0"] = team, obs["yaw"]
        alive = [e for e in team["enemies"] if pro.in_fight(obs, e)] or team["enemies"]
        first = min(alive, key=lambda e: v1.relative(obs, pro.in_fight(obs, e)["pos"])["dist"] if pro.in_fight(obs, e) else 1e9)
        ins["target"] = {"kind": "player", "name": first}
        self.log.write("prepare", target=ins["target"], team=team)
        print(f"   team: allies {team['allies']}, enemies {team['enemies']}; fighting {first}")

    def refocus(self, obs):
        """When the enemy you fight is out of the fight, the other one."""
        ins = self.mem.instruction
        team, focus = ins.get("team"), ins["target"].get("name")
        if not team or pro.in_fight(obs, focus):
            return None
        alive = [e for e in team["enemies"] if pro.in_fight(obs, e)]
        if not alive:
            return None
        ins["target"] = {"kind": "player", "name": alive[0]}
        return alive[0]

    def shared(self):
        out = {}
        for a in (self.mem.instruction.get("team") or {}).get("allies", []):
            try:
                with open(os.path.join(self.TEAM_DIR, f"{a}.json")) as f:
                    out[a] = json.load(f)
            except (OSError, ValueError):
                pass
        return out

    def publish(self, obs, target, action):
        try:
            os.makedirs(self.TEAM_DIR, exist_ok=True)
            path = os.path.join(self.TEAM_DIR, f"{obs['name']}.json")
            with open(path + ".tmp", "w") as f:
                json.dump({"t": time.time(), "target": target, "action": cu_label(action), "health": obs["health"]}, f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass


def cu_label(action):
    import context_uhc as cu
    return cu.LABELS.get(action, action).replace("_", " ")
