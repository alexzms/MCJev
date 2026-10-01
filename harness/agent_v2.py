"""Harness v2 agent (./agent.py --harness v2): v1's loop and perception, with context_v2's state, a target
chosen per instruction (any player or a remembered place), and long-term memory (memory.py).

What changes against v1 (agent.Agent, which stays as it is):
- chat judgement also asks "is this asking you to remember something?" (and which kind of memory), and right
  after an instruction ended, whether the message praises or corrects what was done; both are written to the
  memory directory by the harness
- when an instruction starts, one request picks the target and recalls the relevant memories
- steps render context_v2's sectioned state; everything else (vision passes, done hysteresis, option marks)
  is v1's
"""
import math, re, time

import context as v1
import context_v2 as ctx
import memory as ltm
from agent import Agent, Memory

FEEDBACK_WINDOW = 60  # seconds after an instruction ends in which a human's message can be feedback on it
SERVER_SENDERS = {"Rcon", "Server", "CONSOLE"}   # the server's own messages (a referee, a plugin): orders, never feedback
HARNESS_ORDER = re.compile(r"^harness\s+(\S+)$")   # from the server: play the next match with this harness version


class SwitchHarness(Exception):
    """Raised out of run() when the server asks for another harness version; agent.py starts it on the same body."""
    def __init__(self, version):
        super().__init__(version)
        self.version = version

MESSAGE_QUESTIONS = {
    **v1.MESSAGE_QUESTIONS,
    "is_remember": {"type": "boolean",
                    "instructions": "Is the new message asking you to remember something for later "
                                    "(like 'remember this place as home' or 'remember that Bob is my brother')?"},
    # no "none" option: whether to remember is is_remember's job; with "none" offered, Chinese requests to remember
    # a place or a preference read "none" (2 of 2), without it 7 of 7 kinds were right
    "memory_type": {"type": "choice", "instructions": "What kind of thing does the new message ask you to remember?",
                    "criteria": dict(ltm.TYPES)},
}
FEEDBACK_QUESTION = {"type": "choice", "instructions": "How does the new message react to what you just did?",
                     "criteria": {"praise": "praises it or says it was right",
                                  "correction": "says it was wrong or should be done differently",
                                  "neither": "neither; it is about something else"}}


class MemoryV2(Memory):
    def __init__(self):
        super().__init__()
        self.recalled = []      # long-term memories recalled for the current instruction
        self.looked = []        # vision: bearings (° clockwise from north) looked at without ever seeing the target
        self.things = []        # landmarks (clusters of uncommon blocks) found at instruction start; not for vision-only
        self.prev_obs = None    # observation at the previous step, for "changes since your last step"
        self.last_step_t = 0.0

    def messages_since_start(self):
        ins = self.instruction
        if not ins:
            return []
        is_bot = getattr(self, "is_bot", lambda n: "jev" in n.lower())   # the agent's rule (AgentV2 sets it)
        return [m for m in self.chat if m["t"] > ins["t"] and not m.get("self") and not is_bot(m["username"])]


def best(answer):
    return max(answer["probabilities"], key=answer["probabilities"].get)


class AgentV2(Agent):
    def __init__(self, body, jev, cfg):
        super().__init__(body, jev, cfg)
        self.mem = MemoryV2()
        self.mem.is_bot = self.is_bot
        self.store = ltm.MemoryStore(cfg.memory_dir)
        self.chat_started = False   # the first chat read skips what an earlier harness on this body already handled

    def finish(self, outcome):
        self.mem.instruction["ended"] = time.time()
        super().finish(outcome)

    # ---------------------------------------------------------------- chat

    def recent_finished(self):
        if self.mem.finished and time.time() - self.mem.finished[-1].get("ended", 0) < FEEDBACK_WINDOW:
            return self.mem.finished[-1]
        return None

    def take_info(self, m):
        """A harness may take a message as news for what it is doing rather than something to judge (True)."""
        return False

    def from_server(self, m):
        """The server's own message (a referee, a plugin): from Server, Rcon or CONSOLE - and with bots_by_uuid, a
        whisper the body saw come from the console, not from a person who took one of those names."""
        return m["username"] in SERVER_SENDERS and (m.get("console", True) if self.bots_by_uuid else True)

    def take_chat(self, obs):
        self.see_players(obs)
        if obs["chat"]:
            self.mem.seen = obs["chat"][-1]["seq"]
        new = obs["chat"]
        order = lambda m: self.from_server(m) and HARNESS_ORDER.match(m["message"].strip())
        if not self.chat_started:
            # After a harness switch this agent runs on the same body, whose chat still holds everything said
            # before, the orders to switch included: re-read, an old order to another version switched straight
            # back, forever. Everything up to the order that started this version was the earlier harness's.
            self.chat_started = True
            mine = [i for i, m in enumerate(new) if order(m) and order(m).group(1) == self.cfg.harness]
            if mine:
                new = new[mine[-1] + 1:]
        new = new + self.inbox.read()
        self.mem.chat.extend(new)
        orders = [order(m).group(1) for m in new if order(m)]
        if orders and orders[-1] != self.cfg.harness:   # the server switching this bot's harness: the latest order counts
            if self.mem.instruction:
                self.finish(f"harness switched to {orders[-1]}")
            self.body.act("stop", hold=True)
            raise SwitchHarness(orders[-1])
        new = [m for m in new if not order(m)]
        others = [p["name"] for p in obs["players"]]
        msgs = [m for m in new if m.get("via") == "console" or (
            self.is_human(m["username"], obs["name"]) and (m.get("whisper") or   # a whisper to me is for me, whoever it names
                                                            self.addressed_to_me(m["message"], obs["name"], others)))]
        msgs = [m for m in msgs if not self.take_info(m)]
        if not msgs:
            return
        last = self.recent_finished()
        questions = {**MESSAGE_QUESTIONS, **({"feedback": FEEDBACK_QUESTION} if last else {})}
        states = []
        for m in msgs:
            state = v1.message_state(obs, self.mem, m)
            if last:
                state += (f'\n\nYou just finished an earlier instruction from {last["from"]}: "{last["text"]}" - '
                          f'{last["outcome"]} after {last["steps"]} steps, {ltm.age_words(last["ended"])}.')
            states.append({"id": f"msg{m['seq']}", "state": state, "questions": questions})
        answers, execution, dt = self.ask(states)
        decisions = []
        for m in msgs:
            a = answers[f"msg{m['seq']}"]
            p = {k: a[k]["p_true"] for k in ("is_instruction", "is_stop", "once", "is_remember")}
            decision = ("remember" if p["is_remember"] >= 0.5 else "stop" if p["is_stop"] >= 0.5
                        else "instruction" if p["is_instruction"] >= 0.5 else "chatter")
            d = {"from": m["username"], "text": m["message"], "via": m.get("via", "chat"), "decision": decision,
                 "p_instruction": p["is_instruction"], "p_stop": p["is_stop"], "p_once": p["once"],
                 "p_remember": p["is_remember"], "memory_type": best(a["memory_type"]),
                 "feedback": best(a["feedback"]) if last else None}
            decisions.append(d)
            print(f'   {d["via"]} {m["username"]}: "{m["message"]}" -> {decision}'
                  f'{" (" + d["memory_type"] + ")" if decision == "remember" else ""}'
                  f'{", feedback " + d["feedback"] if d["feedback"] not in (None, "neither") else ""} '
                  f'(instruction {p["is_instruction"]:.2f}, stop {p["is_stop"]:.2f}, once {p["once"]:.2f}, '
                  f'remember {p["is_remember"]:.2f}, {dt * 1000:.0f}ms)')
        self.log.write("message", states=states, answers=answers, execution=execution, seconds=dt,
                       decisions=decisions)
        for m, d in zip(msgs, decisions):
            if self.from_server(m) and d["decision"] == "remember":
                d["decision"] = "chatter"
            if d["feedback"] in ("praise", "correction") and not self.from_server(m):
                self.remember_feedback(m, d["feedback"], last)
            if d["decision"] == "remember":
                self.remember(obs, m, d["memory_type"])
            elif d["decision"] == "stop" and self.mem.instruction:
                self.finish(f"stopped by {m['username']}")
                self.say(self.cfg.say_stop)
            elif d["decision"] == "instruction":
                self.start(m, once=d["p_once"] >= 0.5)
                self.prepare(obs)

    def remember(self, obs, m, kind):
        kind = kind if kind in ltm.TYPES else "fact"
        pos = obs["pos"] if kind == "place" else None
        body = (f"Where {obs['name']} stood when it was said: x={pos['x']:.1f} y={pos['y']:.1f} z={pos['z']:.1f}."
                if pos else "")
        name = self.store.add(kind, f'{m["username"]} said: "{m["message"]}"', body, m["username"], pos)
        self.log.write("memory", event="add", file=name, type=kind, text=m["message"], by=m["username"], pos=pos)
        print(f"   memory + {name}")
        self.say(self.cfg.say_remember)

    def remember_feedback(self, m, kind, last):
        desc = (f'{m["username"]} on "{last["text"]}" ({last["outcome"]}, {last["steps"]} steps): '
                f'"{m["message"]}"')
        body = (f"{kind.capitalize()}.\n\n**Why:** said right after the instruction ended ({last['outcome']}).\n"
                f"**How to apply:** when given a similar instruction.")
        name = self.store.add("feedback", desc, body, m["username"])
        self.log.write("memory", event="add", file=name, type="feedback", feedback=kind, text=m["message"],
                       by=m["username"])
        print(f"   memory + {name} ({kind})")

    # ---------------------------------------------------------------- instruction start

    def prepare(self, obs):
        """One request, two states: which player or place the instruction is about, and which memories help."""
        ins = self.mem.instruction
        ins["yaw0"] = obs["yaw"]
        self.mem.looked = []
        places, entries = self.store.places(), self.store.entries()
        self.mem.things = self.body.landmarks() if self.cfg.vision != "only" else []
        cands = ctx.target_candidates(obs, ins["from"], places, self.mem.things, self.cfg)
        states = [{"id": "target",
                   "state": f'You are {obs["name"]}, a bot in Minecraft. {ins["from"]} (human) just told you: '
                            f'"{ins["text"]}"',
                   "questions": {"target": {"type": "choice", "instructions": ctx.TARGET_QUESTION,
                                            "criteria": cands}}}]
        if entries:
            states.extend(ltm.recall_requests(entries, ins["text"], ins["from"]))
        answers, execution, dt = self.ask(states)
        key = best(answers["target"]["target"])
        ins["target"] = ctx.resolve_target(key, ins["from"], places, self.mem.things, ins["text"])
        self.mem.recalled = ltm.pick_recalled(entries, answers) if entries else []
        self.log.write("prepare", states=states, answers=answers, execution=execution, seconds=dt,
                       target=ins["target"], recalled=[e["file"] for e in self.mem.recalled])
        print(f'   target: {ins["target"].get("name") or "none"}; recalled: '
              f'{[e["description"][:40] for e in self.mem.recalled] or "nothing"} ({dt * 1000:.0f}ms)')

    # ---------------------------------------------------------------- control

    def step(self, obs):
        cfg, ins = self.cfg, self.mem.instruction
        if "target" not in (self.mem.instruction or {}):  # preparing failed (Jev unreachable): try it again first
            self.prepare(obs)
            return
        tgt = ins["target"]
        questions, answers, images, look, execution, dt = {}, {}, [], None, [], 0.0
        seen = False
        if cfg.vision != "off" and tgt["kind"] in ("player", "sought"):
            images = [self.body.snapshot()]
            look_q = v1.look_questions(tgt["name"]) if tgt["kind"] == "player" else ctx.look_questions_thing(ins["text"])
            ans, ex, t = self.ask([{"id": "look", "state": v1.LOOK_STATE, "images": images, "questions": look_q}])
            look = ans["look"]
            questions.update(look_q); answers.update(look); execution.append(ex); dt += t
            seen, angle, dist = v1.read_look(look)
            if seen:
                self.mem.sight = {"est": v1.project(obs, angle, dist), "step": ins["steps"]}
                self.mem.scanned = 0.0
            elif not self.mem.sight:
                self.mem.looked.append((-math.degrees(obs["yaw"])) % 360)
        view = (ctx.view_lines(obs, look, tgt["name"], self.mem.sight, self.mem.looked, ins["steps"],
                               thing=tgt["kind"] == "sought") if look else None)
        state = ctx.step_state(obs, self.mem, cfg, view=view)
        toward, mark = ctx.target_side(obs, cfg, tgt, look, self.mem.sight), tgt["label"] or "the target"
        if toward is None:
            toward = ctx.search_side(obs, cfg, tgt, look, self.mem.sight, self.mem.looked)
            mark = "where you have not looked yet"
        step_q = {"action": v1.step_questions(cfg, mark, toward)["action"]}
        if cfg.verbose:
            print("-" * 100 + "\n" + state + "\n" + "-" * 100)
        states = [{"id": "step", "state": state, "questions": step_q}]
        if ins["once"]:  # one-time requests: an independent check of whether it is done, in the same request
            verify = ctx.verify_state(self.mem, obs, cfg, look)
            states.append({"id": "verify", "state": verify, "questions": ctx.VERIFY_QUESTION})
        ans, ex, t = self.ask(states)
        questions.update(step_q); answers.update(ans["step"]); execution.append(ex); dt += t
        if "verify" in ans:
            questions["done"] = ctx.VERIFY_QUESTION["done"]
            answers["done"] = ans["verify"]["done"]
        a = answers
        probs = a["action"]["probabilities"]
        done = a["done"]["p_true"] if "done" in a else 0.0
        action = max(probs, key=probs.get)
        record = dict(harness="v2", instruction=ins["text"], step=ins["steps"] + 1, state=state, images=images,
                      verify_state=states[1]["state"] if len(states) > 1 else None,
                      questions=questions, answers=a, execution=execution[-1], seconds=dt, action=action, done=done)
        truth_pos = tgt.get("pos") if tgt["kind"] in ("place", "thing") else next(
            (p["pos"] for p in obs["players"] if p["name"] == tgt.get("name") and p["pos"]), None)
        if truth_pos:  # evaluation only; never shown to Jev in vision-only mode
            rel = v1.relative(obs, truth_pos)
            record["truth"] = {"target": tgt.get("name"), "dist": round(rel["dist"], 2),
                               "angle": round(rel["angle"], 1), "pos": obs["pos"], "yaw": obs["yaw"]}
        confirmed = cfg.vision == "off" or tgt["kind"] not in ("player", "sought") or self.mem.done_streak >= 1
        self.mem.done_streak = self.mem.done_streak + 1 if done >= cfg.done_threshold else 0
        if ins["once"] and done >= cfg.done_threshold and confirmed:
            self.log.write("step", **record, act=None, effect="judged the instruction done")
            self.finish("completed")
            self.say(cfg.say_done)
            return

        result = self.body.act(action, ms=cfg.move_ms, turn_deg=cfg.turn_deg, fine_deg=cfg.fine_turn_deg,
                               pitch_deg=cfg.pitch_deg)
        h = {"action": action, "result": result, "yaw0": obs["yaw"]}
        if look:
            h["sees"] = (tgt["name"], f"in view, {best(look['where']).replace('_', ' ')}" if seen else "not in view")
            if not seen:
                d = result["yaw"] - obs["yaw"]
                self.mem.scanned += abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))
        if truth_pos and cfg.vision != "only":
            before, after = v1.relative(obs, truth_pos), v1.relative({**obs, **result}, truth_pos)
            h["target"] = (tgt["label"], before["dist"], after["dist"], before["angle"], after["angle"])
        self.mem.history.append(h)
        self.mem.prev_obs, self.mem.last_step_t = obs, time.time()
        ins["steps"] += 1
        line = v1.history_line(ins["steps"], h, v1.action_options(cfg), show_target=cfg.vision != "only")
        self.log.write("step", **record, act=result, effect=line.split(" -> ", 1)[1])
        print(f"   p={probs[action]:.2f} mass={a['action']['candidate_mass']:.2f} done={done:.2f} "
              f"{dt * 1000:4.0f}ms  {line}")
        if ins["once"] and ins["steps"] >= cfg.max_steps:
            self.finish("gave up (step limit)")
            self.say(cfg.say_giveup)
