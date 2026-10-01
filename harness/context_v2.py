"""Harness v2: the state Jev reads, as named sections in a fixed order (after Claude Code's context layout).

Differences from v1 (context.py, whose geometry, map and vision pieces are reused as they are):
- every section is always present, "(none)" when empty, so positions stay stable from step to step
- the target is chosen per instruction (a player or a remembered place), not always the speaker
- recalled long-term memories, earlier instructions with their age, and the humans' words since the
  instruction started, verbatim
- action history in tiers: the last 3 steps in full, the 5 before as one line each, older ones folded
  into a worklog of runs ("steps 1-6: turn left x6, turned 180°")
- CHANGES since the last step (the target moved, players came or went) and triggered NOTES
Jev writes nothing, so all of this text is written here.
"""
import math, re

import context as v1
from memory import age_words, memory_lines

RECENT_FULL, RECENT_SHORT = 3, 5

PREAMBLE = (
    "You are {name}, a bot playing Minecraft (Java 1.11.2) on a server together with human players. "
    "You control your character like a player at a keyboard and mouse: every step you choose one control "
    "(walk, jump, turn or tilt your view, left-click, right-click). You cannot type in chat. Humans give you "
    "instructions in chat; your job is to carry out the current instruction.")


# Vision agents are told how their sight works, as a fact about themselves; whether to turn stays Jev's call.
SENSES = ("You see only what is in front of you, about a third of the way around you (110°); what is beside or "
          "behind you is out of view until you turn your view toward it.")


def view_lines(obs, look, who, sight, scanned, step, thing=False):
    """Pass 1's answers as what you know. When the target is out of view it is said as missing knowledge, with how
    much of the surroundings has been looked over: on 31 logged 'come to me' states where Tester had not been
    seen, 'Tester is NOT in your view, and you have not seen them yet' led to walking forward 30 times; this
    wording (with SENSES in the preamble) led to turning 31 times."""
    seen, _, _ = v1.read_look(look)
    if seen:
        best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
        where, size = best("where"), best("distance")
        side = {"far_left": "at the left edge of your view", "left": "to the LEFT of your crosshair",
                "center": "right under your crosshair", "right": "to the RIGHT of your crosshair",
                "far_right": "at the right edge of your view"}[where]
        if where.startswith("far_"):  # sizes at the edge of the view are unreliable (a player 6 blocks off read "big")
            return [f"You looked at your screen just now: {who} is in your view, {side} (how far is unclear there)."]
        if size == "close":
            front = "right in front of you" if where == "center" else side
            return [f"You looked at your screen just now: {who} is right next to you, {front} "
                    f"(within about 2 blocks): you have reached {'it' if thing else 'them'}."]
        return [f"You looked at your screen just now: {who} is in your view, {side}, "
                f"{v1.DISTANCE[size][2] if size in v1.DISTANCE else 'at an unclear distance'}."]
    if sight:
        r = v1.relative(obs, sight["est"])
        ago = step - sight["step"]
        spot = "right next to you" if r["dist"] < 1.5 else f"about {r['dist']:.0f} blocks away"
        return [f"You cannot see {who} right now. You last saw {'it' if thing else 'them'} {ago} step{'s' if ago != 1 else ''} ago; judging "
                f"by how you have moved since, that spot is now {v1.direction_words(r['angle'])}, {spot}."]
    unseen = unlooked_words(obs, scanned)
    pron = "it is" if thing else "they are"
    return [f"You do not know where {who} is: {pron} out of your view, somewhere around you. "
            + (f"You have not looked {unseen} yet." if unseen else
               f"You have looked all around without seeing {'it' if thing else 'them'}.")]


def look_questions_thing(text):
    """Pass 1 when the target is a thing the vision-only bot must find: described by the instruction itself."""
    what = f'what the instruction "{text}" is about'
    return {
        "sees": {"type": "boolean", "instructions": f"Can you see {what} in Image 1?"},
        "where": {"type": "choice", "instructions": f"Where in Image 1 is it, left to right?",
                  "criteria": {**{k: v[0] for k, v in v1.WHERE.items()}, "none": "it is not visible"}},
        "distance": {"type": "choice", "instructions": "How close does it look?",
                     "criteria": {"close": "right in front of you: it fills much of the image or runs off its edges",
                                  "near": "a few blocks away", "far": "far away: it looks small",
                                  "none": "it is not visible"}},
    }


def unlooked(obs, looked):
    """Relative angles (degrees, + right) of the 8 compass directions not yet covered by any look."""
    out = []
    for k in range(8):
        world = k * 45  # bearing clockwise from north
        if not any(abs((world - b + 180) % 360 - 180) <= 55 for b in looked):
            facing = (-math.degrees(obs["yaw"])) % 360
            out.append((world - facing + 180) % 360 - 180)
    return out


def unlooked_words(obs, looked):
    rel = unlooked(obs, looked)
    if not rel:
        return ""
    names = []
    for a in sorted(rel, key=abs):
        w = ("behind you" if abs(a) > 135 else f"behind you on the {'right' if a > 0 else 'left'}" if abs(a) > 100
             else f"to your {'right' if a > 0 else 'left'}" if abs(a) > 60 else f"ahead-{'right' if a > 0 else 'left'}")
        if w not in names:
            names.append(w)
    return ", ".join(names[:-1]) + (" or " if len(names) > 1 else "") + names[-1]


def unlooked_side(obs, looked):
    """Which way to turn to reach the nearest direction not looked at yet (None if there is none)."""
    rel = unlooked(obs, looked)
    if not rel:
        return None
    a = min(rel, key=abs)
    return "right" if a > 0 else "left"


def section(title, lines):
    return f"== {title} ==\n" + "\n".join(lines or ["(none)"])


# ---------------------------------------------------------------- target

def thing_description(c):
    """A cluster of blocks from body.landmarks, in words: 'a column of 7 diamond block blocks, 7 blocks tall'."""
    size = {k: c["max"][k] - c["min"][k] for k in "xyz"}
    w, h = max(size["x"], size["z"]), size["y"]
    name = c["name"].replace("_", " ")
    if c["count"] == 1:
        return f"a single {name} block"
    if h >= 3 and h >= 2 * w:
        return f"a column of {c['count']} {name} blocks, {h} blocks tall"
    if h == 1:
        return f"a flat patch of {c['count']} {name} blocks, {size['x']} by {size['z']}"
    return f"a pile of {c['count']} {name} blocks, {size['x']} by {size['z']}, {h} tall"


def box_relative(obs, c):
    """Distance to the nearest point of a block cluster's box, and the direction of its middle."""
    p = obs["pos"]
    dx = max(c["min"]["x"] - p["x"], 0, p["x"] - c["max"]["x"])
    dz = max(c["min"]["z"] - p["z"], 0, p["z"] - c["max"]["z"])
    r = v1.relative(obs, c["center"])
    return dict(r, dist=math.hypot(dx, dz))


def target_candidates(obs, sender, places, things=(), cfg=None):
    """Choice options for "what is this instruction about": players, remembered places, things, nothing.

    Things come from the block scan (body.landmarks) when the bot may use it; a vision-only bot gets one
    option instead, "something the instruction names that you have to find by looking"."""
    opts = {"speaker": f"{sender}, the player who gave the instruction"}
    for p in obs["players"]:
        if p["name"] != sender:
            opts[f"player:{p['name']}"] = f"the player {p['name']}"
    for e in places:
        opts[f"place:{e['file']}"] = f"a remembered place: {e['description']}"
    for i, c in enumerate(things):
        r = box_relative(obs, c)
        opts[f"thing:{i}"] = f"{thing_description(c)}, {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
    if cfg is not None and cfg.vision == "only":
        opts["sought"] = "some other thing or place the instruction names, which you have to find by looking"
    opts["none"] = "no particular player or place: the instruction is only about your own movement, like jumping or turning around"
    return opts


# With examples of self-only instructions, 16/16 test instructions got the right target (incl. walk forward,
# look up, 跳一下, 转身, hit Bob, stay away from Bob); without, "turn around" and "jump" picked the speaker.
TARGET_QUESTION = ("Which player, place or thing does the instruction tell you to go to, look at, hit, follow or keep "
                   "away from? If it only asks you to move yourself (like jump or turn around), answer none.")


def resolve_target(key, sender, places, things=(), text=""):
    if key.startswith("thing:"):
        c = things[int(key.split(":", 1)[1])]
        desc = thing_description(c)
        return {"kind": "thing", "name": desc, "label": "it", "box": c, "pos": c["center"], "where": f"the {desc[2:]}"}
    if key == "sought":
        return {"kind": "sought", "name": f'what "{text}" is about', "label": "it", "text": text}
    if key == "speaker":
        return {"kind": "player", "name": sender, "label": sender}
    if key.startswith("player:"):
        name = key.split(":", 1)[1]
        return {"kind": "player", "name": name, "label": name}
    if key.startswith("place:"):
        e = next(e for e in places if e["file"] == key.split(":", 1)[1])
        m = re.match(r'(.*?) said: "(.*)"$', e["description"])
        where = f'the place where {m.group(1)} said "{m.group(2)}"' if m else f"the place ({e['description']})"
        return {"kind": "place", "name": e["description"], "label": "the remembered place", "pos": e["pos"],
                "where": where}
    return {"kind": "none", "label": None}


def place_line(obs, target):
    """'Arrived at your target' links the place to the instruction through the target Jev itself chose; with
    it the verifier read 12/12 (1.00 / 0.00) over go home, 回到矿洞入口, 'take me to where we started', while
    'the remembered place is right here: you are there' read 0.00 standing on it."""
    r = v1.relative(obs, target["pos"])
    if r["dist"] < 2:
        return f"you have arrived at your target, {target['where']}: you are standing on it."
    return (f"you have not arrived at your target yet: {target['where']} is {r['dist']:.0f} blocks away, "
            f"{v1.direction_words(r['angle'])}, {v1.height_words(r['dy'])}.")


def thing_line(obs, target):
    r = box_relative(obs, target["box"])
    if r["dist"] <= 1.5:
        return f"you have arrived at your target, {target['where']}: you are right next to it."
    return (f"you have not arrived at your target yet: {target['where']} is {r['dist']:.0f} blocks away, "
            f"{v1.direction_words(r['angle'])}.")


def target_lines(obs, cfg, target, view):
    if target["kind"] == "thing":
        line = thing_line(obs, target)
        return [line[0].upper() + line[1:]]
    if target["kind"] == "sought":
        return view or ["You have not looked yet."]
    if target["kind"] == "place":
        line = place_line(obs, target)
        return [line[0].upper() + line[1:]]
    if target["kind"] == "player":
        if cfg.vision != "off":
            return view or [f"{target['name']}: you have not looked yet."]
        p = next((p for p in obs["players"] if p["name"] == target["name"]), None)
        return [v1.player_line(obs, p, None)] if p else [f"{target['name']} is not online."]
    return ["No particular player or place."]


def target_side(obs, cfg, target, look, sight):
    if target["kind"] in ("place", "thing"):
        a = v1.relative(obs, target["pos"])["angle"]
        return None if abs(a) <= 10 else ("right" if a > 0 else "left")
    if target["kind"] == "player":
        return v1.target_side(obs, cfg, target["name"], look, sight)
    if target["kind"] == "sought" and look:
        return v1.target_side(obs, cfg, None, look, sight)
    return None


def search_side(obs, cfg, target, look, sight, looked):
    """Vision, target never seen: the side with directions not looked at yet (marked on the turn options)."""
    if cfg.vision == "off" or target["kind"] not in ("player", "sought") or sight or (look and v1.read_look(look)[0]):
        return None
    return unlooked_side(obs, looked)


# ---------------------------------------------------------------- history tiers

def no_effect(h):
    r = h["result"]
    if h["action"] in ("forward", "back", "left", "right", "jump_forward"):
        return r["moved"] < 0.2
    return any(w in r.get("note", "") for w in ("nothing", "swung at the air"))


def turned_deg(h):
    d = h["result"]["yaw"] - h.get("yaw0", h["result"]["yaw"])
    return abs(math.degrees(math.atan2(math.sin(d), math.cos(d))))


def short_effect(h):
    r, a = h["result"], h["action"]
    if a in ("forward", "back", "left", "right", "jump_forward"):
        return "blocked" if r["moved"] < 0.2 else f"moved {r['moved']:.1f}"
    if a.startswith(("turn_", "look_")):
        return "turned"
    return r.get("note") or "waited"


def recent_lines(history, cfg):
    opts = v1.action_options(cfg)
    hist = list(history)
    n0 = len(hist)
    full = hist[-RECENT_FULL:]
    short = hist[-(RECENT_FULL + RECENT_SHORT):-RECENT_FULL] if len(hist) > RECENT_FULL else []
    first = n0 - len(full) - len(short) + 1
    lines = [f"{first + k}. {opts[h['action']].split(' (')[0]} -> {short_effect(h)}" for k, h in enumerate(short)]
    base = n0 - len(full) + 1
    lines += [v1.history_line(base + k, h, opts, show_target=cfg.vision != "only") for k, h in enumerate(full)]
    return lines


def worklog_lines(history, cfg):
    """Steps older than the recent tiers, folded into runs of the same control with their summed effect."""
    hist = list(history)[:-(RECENT_FULL + RECENT_SHORT)]
    if not hist:
        return []
    opts = v1.action_options(cfg)
    runs, start = [], 0
    for i in range(1, len(hist) + 1):
        if i == len(hist) or hist[i]["action"] != hist[start]["action"]:
            runs.append((start, i))
            start = i
    lines = []
    for a, b in runs:
        chunk = hist[a:b]
        act = chunk[0]["action"]
        label = opts[act].split(" (")[0]
        steps = f"step {a + 1}" if b - a == 1 else f"steps {a + 1}-{b}"
        if act in ("forward", "back", "left", "right", "jump_forward"):
            moved = sum(h["result"]["moved"] for h in chunk)
            blocked = sum(h["result"]["moved"] < 0.2 for h in chunk)
            eff = f"moved {moved:.0f} blocks" + (f", blocked {blocked} times" if blocked else "")
        elif act.startswith("turn_"):
            eff = f"turned {sum(turned_deg(h) for h in chunk):.0f}°"
        else:
            notes = {h["result"].get("note") for h in chunk} - {"", None}
            eff = "; ".join(sorted(notes)) or "no effect"
        looks = [h["sees"][1] != "not in view" for h in chunk if h.get("sees")]
        if looks:
            eff += f"; target in view {sum(looks)} of {len(looks)} times" if any(looks) else "; target never in view"
        lines.append(f"{steps}: {label} x{b - a}, {eff}")
    return lines[-6:] if len(lines) <= 6 else [f"(+{len(lines) - 6} earlier runs)"] + lines[-6:]


# ---------------------------------------------------------------- changes and notes

def changes_lines(prev, obs, target, cfg):
    if not prev:
        return []
    out = []
    before = {p["name"]: p["pos"] for p in prev["players"]}
    now = {p["name"]: p["pos"] for p in obs["players"]}
    for name in now.keys() - before.keys():
        out.append(f"{name} joined the game.")
    for name in before.keys() - now.keys():
        out.append(f"{name} left the game.")
    if cfg.vision == "off" and target["kind"] == "player":
        a, b = before.get(target["name"]), now.get(target["name"])
        if a and b:
            moved = math.hypot(b["x"] - a["x"], b["z"] - a["z"])
            if moved >= 1:
                out.append(f"{target['name']} moved {moved:.0f} blocks since your last step.")
    return out


def notes_lines(mem, cfg):
    hist = list(mem.history)
    notes = []
    moves = [h for h in hist[-5:] if h["action"] in ("forward", "back", "left", "right", "jump_forward")]
    if len(moves) >= 4 and sum(h["result"]["moved"] for h in moves) < 0.5:
        notes.append("You have hardly moved over your last moves: something is blocking you; turn or jump.")
    last3 = hist[-3:]
    if len(last3) == 3 and len({h["action"] for h in last3}) == 1 and all(no_effect(h) for h in last3):
        label = v1.action_options(cfg)[last3[0]["action"]].split(" (")[0]
        notes.append(f"'{label}' had no effect 3 times in a row: try something else.")
    if cfg.vision != "off" and mem.instruction["target"]["kind"] == "player":
        unseen = 0
        for h in reversed(hist):
            if not h.get("sees") or h["sees"][1] != "not in view":
                break
            unseen += 1
        if unseen >= 10:
            notes.append(f"You have not seen {mem.instruction['target']['name']} for {unseen} steps.")
    new_msgs = [m for m in mem.messages_since_start() if m["t"] / 1000 > mem.last_step_t]
    if new_msgs:
        notes.append(f"{new_msgs[-1]['username']} said something new while you were working (see Player messages).")
    return notes[:3]


# ---------------------------------------------------------------- progress and verification

def facing_words(yaw0, yaw):
    """How the bot faces now relative to when it was told, as a relation rather than a number to compare."""
    d = (math.degrees(yaw0 - yaw) + 180) % 360 - 180  # + = turned right (mineflayer yaw grows to the left)
    a, side = abs(d), ("right" if d > 0 else "left")
    if a < 20:
        return "you face the same way as when you were told"
    if a >= 150:
        return "you now face the opposite way from when you were told"
    if a >= 110:  # "turned 130°" alone read as done for "turn around"
        return (f"you have turned {a:.0f}° to the {side} of where you faced when told: most of the way round, "
                f"but not yet facing the opposite way")
    if a >= 70:
        return f"you have turned sideways, {a:.0f}° to the {side} of where you faced when told"
    return f"you have turned {a:.0f}° to the {side} of where you faced when told"


def done_so_far(mem, obs):
    """Everything the steps of this instruction achieved, counted by the harness."""
    hits, used, jumps, walked = {}, {}, 0, 0.0
    for h in mem.history:
        note = h["result"].get("note", "")
        if note.startswith("hit "):
            hits[note[4:]] = hits.get(note[4:], 0) + 1
        elif note.startswith("right-clicked "):
            used[note[14:]] = used.get(note[14:], 0) + 1
        jumps += h["action"] == "jump_forward"
        walked += h["result"].get("moved", 0)
    times = lambda n: f"{n} time{'s' if n > 1 else ''}"
    parts = [f"hit {w} {times(n)}" for w, n in hits.items()] + [f"right-clicked {w} {times(n)}" for w, n in used.items()]
    if jumps:
        parts.append(f"jumped {times(jumps)}")
    if walked >= 0.5:
        parts.append(f"walked {walked:.0f} blocks")
    facing = facing_words(mem.instruction["yaw0"], obs["yaw"])
    if not parts and facing.startswith("you face the same way"):
        return "nothing yet; " + facing
    return "; ".join(parts + [facing])


def now_sentence(obs, cfg, target, look):
    """Where the target stands relative to you, in one plain sentence with you as the subject."""
    if target["kind"] == "place":
        return place_line(obs, target)
    if target["kind"] == "thing":
        return thing_line(obs, target)
    if target["kind"] not in ("player", "sought"):
        return "nothing in particular."
    name = target["name"]
    if target["kind"] == "sought":  # vision-only: what the instruction is about, as seen on the screen
        seen, angle, _ = v1.read_look(look) if look else (False, None, None)
        if not seen:
            return f"you cannot see {name}."
        best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
        where = {"far_left": "far to your left", "left": "a little to your left", "center": "right in front of you",
                 "right": "a little to your right", "far_right": "far to your right"}[best("where")]
        if best("distance") == "close" and not best("where").startswith("far_"):
            return f"you have arrived at your target: {name} is right in front of you, {where}."
        return f"you have not arrived at your target yet: {name} is {DISTANCE_WORDS.get(best('distance'), 'in view')}, {where}."
    if cfg.vision != "off":
        seen, angle, _ = v1.read_look(look) if look else (False, None, None)
        if not seen:
            return f"you cannot see {name}."
        best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
        close = best("distance") == "close"
        where = {"far_left": "far to your left", "left": "a little to your left", "center": "right in front of you",
                 "right": "a little to your right", "far_right": "far to your right"}[best("where")]
        near = "right next to you (you have reached them)" if close else DISTANCE_WORDS.get(best("distance"), "in view")
        aim = ("you are looking straight at them (your crosshair is on them)" if best("where") == "center"
               else "your crosshair is not on them")
        return f"{name} is {near}, {where}; {aim}."
    p = next((p for p in obs["players"] if p["name"] == name and p["pos"]), None)
    if not p:
        return f"you do not know where {name} is."
    r = v1.relative(obs, p["pos"])
    near = "right next to you (you have reached them)" if r["dist"] < 2 else f"{r['dist']:.0f} blocks away"
    aimed = abs(r["angle"]) <= max(5.0, math.degrees(math.atan2(0.3, max(r["dist"], 0.3))))
    aim = "you are looking straight at them (your crosshair is on them)" if aimed else "your crosshair is not on them"
    return f"{name} is {near}, {v1.direction_words(r['angle'])}; {aim}."


DISTANCE_WORDS = {"near": "a few blocks away", "far": "far away"}

# The "is it done?" question gets its own small state, sent in the same request as the action state. In the
# full state Jev kept seeing reasons to go on (after hitting Tester: p(done) 0.0-0.3 with "Aim: within reach"
# in view). Two further findings shaped it: the facts must read with "you" as the subject throughout (a
# third-person frame around second-person facts answered "is the bot looking at Tester?" 0.00 while the
# crosshair was on Tester), and relations are stated as plain sentences ("you are looking straight at them",
# "you are not there yet"). On 127 logged states of eight task kinds: 119 right, against 96 before.
VERIFY_QUESTION = {"done": {"type": "boolean",
                            "instructions": "Have you already done what you were asked to do? What you have already "
                                            "done counts, even if you could do more."}}


def verify_state(mem, obs, cfg, look):
    ins = mem.instruction
    return (f'You are a bot in Minecraft. {ins["from"]} (human) told you: "{ins["text"]}". It is a one-time request: '
            f"it is finished as soon as you have done it once.\n"
            f"Right now: {now_sentence(obs, cfg, ins['target'], look)}\n"
            f"What you have done since you were told: {done_so_far(mem, obs)}.")


# ---------------------------------------------------------------- state

def instruction_lines(mem, cfg, obs):
    ins = mem.instruction
    kind = ("This is a one-time request: it is complete as soon as you have done it once."
            if ins["once"] else "This is an ongoing request: keep doing it until you are told to stop.")
    limit = f" (limit {cfg.max_steps})" if ins["once"] else ""
    t = ins["target"]
    tgt = {"player": f"Target: {t.get('name')}.", "place": f"Target: {t.get('where')}.",
           "thing": f"Target: {t.get('where')}.", "sought": f"Target: {t.get('name')}; find it by looking.",
           "none": "Target: no particular player or place."}[t["kind"]]
    return [f'{ins["from"]} (human) said {v1.ago(ins["t"])}: "{ins["text"]}"', kind, tgt,
            f"You have taken {ins['steps']} steps on it so far{limit}.",
            f"Done so far: {done_so_far(mem, obs)}."]


def earlier_lines(mem):
    return [f'"{o["text"]}" from {o["from"]}: {o["outcome"]} after {o["steps"]} steps, {age_words(o.get("ended", 0))}.'
            for o in list(mem.finished)[-3:]]


def players_lines(obs, cfg, target):
    others = [p for p in obs["players"] if not (target["kind"] == "player" and p["name"] == target["name"])]
    if not others:
        return []
    if cfg.vision == "only":
        return [f"Online: {', '.join(p['name'] for p in others)}. You only know where someone is when you see them."]
    return [v1.player_line(obs, p, None) for p in others]


def landmark_lines(obs, things):
    out = []
    for c in things[:5]:
        r = box_relative(obs, c)
        out.append(f"{thing_description(c)[0].upper()}{thing_description(c)[1:]}: {r['dist']:.0f} blocks away, "
                   f"{v1.direction_words(r['angle'])}.")
    return out


def step_state(obs, mem, cfg, view=None):
    t = mem.instruction["target"]
    return "\n\n".join(x for x in [
        PREAMBLE.format(name=obs["name"]) + (" " + SENSES if cfg.vision != "off" else ""),
        section("Memory", memory_lines(mem.recalled) if mem.recalled else []),
        section("Earlier instructions", earlier_lines(mem)),
        section("Current instruction", instruction_lines(mem, cfg, obs)),
        section("Player messages since it started",
                [f"[{v1.ago(m['t'])}] {m['username']}: {m['message']}" for m in mem.messages_since_start()[-3:]]),
        section("You", v1.you_lines(obs, show_entity=cfg.vision != "only")),
        section("Target", target_lines(obs, cfg, t, view)),
        section("Other players", players_lines(obs, cfg, t)),
        section("Landmarks (uncommon blocks nearby)", landmark_lines(obs, mem.things)) if cfg.vision != "only" else None,
        section("Around you", v1.map_lines(obs, show_players=cfg.vision != "only")),
        section("Worklog (older steps)", worklog_lines(mem.history, cfg)),
        section("Your last actions (oldest first)", recent_lines(mem.history, cfg)),
        section("Changes since your last step", changes_lines(mem.prev_obs, obs, t, cfg)),
        section("Notes", notes_lines(mem, cfg)),
    ] if x)
