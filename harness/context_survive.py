"""Harnesses c2 and c3 (the creeper scene): what a bot needs to stay alive, on top of context_v2.

c2: danger (hostile mobs: where they are, whether they close in, a lit fuse), continuous control (keys stay held
    between decisions), and every movement option marked toward / away from the danger.
c3: c2 + Minecraft facts recalled from knowledge.md, landmarks and animals nearby, and a plan chosen every step
    (keep running, or get to some landmark or animal) whose target is marked on the options as well.
Jev still only picks: all text is written here, facts are sentences a player would know.
"""
import math

import context as v1
import context_v2 as v2
from memory import age_words

HOSTILE = {"creeper", "zombie", "skeleton", "spider", "cave_spider", "witch", "slime", "enderman", "husk", "stray",
           "zombie_villager", "silverfish", "blaze", "ghast", "magma_cube", "guardian"}
DANGER_RANGE = 16

PREAMBLE_HOLD = ("The keys you press stay held until you choose something else: if you sprint, you keep running "
                 "while you think.")


def mob_name(m):
    return m["name"].replace("_", " ")


def threats(obs):
    out = []
    for m in obs.get("mobs", []):
        if m["name"] in HOSTILE:
            r = v1.relative(obs, m["pos"])
            if r["dist"] <= DANGER_RANGE:
                out.append((r["dist"], m, r))
    return [(m, r) for _, m, r in sorted(out, key=lambda x: x[0])]


def animals(obs):
    return [(m, v1.relative(obs, m["pos"])) for m in obs.get("mobs", []) if m["name"] not in HOSTILE]


def mob_line(obs, m, r, prev):
    s = (f"The {mob_name(m)} is {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}, "
         f"{v1.height_words(r['dy'])}")
    old = next((p for p in (prev or {}).get("mobs", []) if p["id"] == m["id"]), None)
    if old:
        d0 = v1.relative(prev, old["pos"])["dist"]
        if d0 - r["dist"] > 0.3:
            s += f"; it came {d0 - r['dist']:.1f} blocks closer since your last step"
        elif r["dist"] - d0 > 0.3:
            s += f"; it fell back {r['dist'] - d0:.1f} blocks since your last step"
        else:
            s += "; about as far as at your last step"
    if m.get("fuse"):
        s += ". Its fuse is lit: it is about to explode"
    if m.get("seen"):
        s += f" ({m['seen']})"
    return s + "."


STRATEGY = ("Strategy: in a small walled yard you cannot outrun a creeper for long. Keep an obstacle, such as a "
            "pillar, between you and it: run around the obstacle as the creeper follows, so it never gets close.")


def cover_lines(obs):
    """Obstacles you could keep between you and the danger, and whether they are between you now."""
    t = threats(obs)
    out = []
    for center, half in find_pillars(obs)[:2]:
        r = v1.relative(obs, center)
        line = f"A pillar is {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
        if t:
            line += ("; it is between you and the " if blocked(obs, obs["pos"], t[0][0]["pos"]) else
                     "; it is not between you and the ") + mob_name(t[0][0])
        out.append(line + ".")
    return out


def danger_lines(obs, prev):
    lines = [mob_line(obs, m, r, prev) for m, r in threats(obs)]
    return lines + cover_lines(obs) + [STRATEGY] if lines else []


def held_words(obs):
    held = set(obs.get("held", []))
    if not held:
        return "You are standing still (no keys held)."
    moves = [w for k, w in (("sprint", "sprinting"), ("forward", "forward"), ("back", "backward"),
                            ("left", "sideways left"), ("right", "sideways right"), ("jump", "jumping")) if k in held]
    return "Keys held: " + ", ".join(moves) + " (you keep moving like this until you choose something else)."


# ---------------------------------------------------------------- open ground

def free_run(obs, rel_deg, limit=7.5):
    """Blocks of open ground from the bot in a direction (° right of facing), within the observed grid."""
    g, R = obs["grid"], obs["grid"]["r"]
    (fx, fz), (rx, rz) = v1.basis(obs["yaw"])
    a = math.radians(rel_deg)
    dx, dz = math.cos(a) * fx + math.sin(a) * rx, math.cos(a) * fz + math.sin(a) * rz
    px, pz = obs["pos"]["x"], obs["pos"]["z"]
    bx, bz = math.floor(px), math.floor(pz)
    d = 0.5
    while d <= limit:
        cx, cz = math.floor(px + dx * d) - bx, math.floor(pz + dz * d) - bz
        if abs(cx) > R or abs(cz) > R:
            return None
        feet, head = (g["layers"][k][cz + R][cx + R] for k in (1, 2))
        if feet not in ".bv" or head not in ".bv":
            return max(0.0, d - 0.5)
        d += 0.5
    return None  # open as far as the grid reaches


def find_pillars(obs, max_cells=9, clearance=2):
    """Obstacles to run around: small solid clusters (feet and head both blocked, at most 3x3) with open ground
    all round them, within the observed grid. -> [(center {x, z}, half size)] nearest first."""
    g, R = obs["grid"], obs["grid"]["r"]
    solid = lambda dx, dz: g["layers"][1][dz + R][dx + R] == "s" and g["layers"][2][dz + R][dx + R] == "s"
    free = lambda dx, dz: g["layers"][1][dz + R][dx + R] in ".bv" and g["layers"][2][dz + R][dx + R] in ".bv"
    seen, out = set(), []
    bx, bz = math.floor(obs["pos"]["x"]), math.floor(obs["pos"]["z"])
    for dz in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if (dx, dz) in seen or not solid(dx, dz):
                continue
            comp, stack = [], [(dx, dz)]
            seen.add((dx, dz))
            while stack:
                x, z = stack.pop()
                comp.append((x, z))
                for nx, nz in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
                    if -R <= nx <= R and -R <= nz <= R and (nx, nz) not in seen and solid(nx, nz):
                        seen.add((nx, nz))
                        stack.append((nx, nz))
            xs, zs = [c[0] for c in comp], [c[1] for c in comp]
            x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
            if len(comp) > max_cells or x1 - x0 > 2 or z1 - z0 > 2:
                continue
            ring = [(x, z) for x in range(x0 - clearance, x1 + clearance + 1) for z in range(z0 - clearance, z1 + clearance + 1)
                    if not (x0 <= x <= x1 and z0 <= z <= z1)]
            if any(not (-R <= x <= R and -R <= z <= R) or not free(x, z) for x, z in ring):
                continue
            center = {"x": bx + (x0 + x1 + 1) / 2, "y": obs["pos"]["y"], "z": bz + (z0 + z1 + 1) / 2}
            out.append((center, max(x1 - x0 + 1, z1 - z0 + 1) / 2))
    return sorted(out, key=lambda c: math.hypot(c[0]["x"] - obs["pos"]["x"], c[0]["z"] - obs["pos"]["z"]))


def blocked(obs, a, b):
    """Is there a solid block (feet and head height) on the straight line from a to b? Cells outside the grid count
    as open."""
    g, R = obs["grid"], obs["grid"]["r"]
    bx, bz = math.floor(obs["pos"]["x"]), math.floor(obs["pos"]["z"])
    n = max(2, int(math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / 0.25))
    for i in range(1, n):
        x = a["x"] + (b["x"] - a["x"]) * i / n
        z = a["z"] + (b["z"] - a["z"]) * i / n
        cx, cz = math.floor(x) - bx, math.floor(z) - bz
        if abs(cx) <= R and abs(cz) <= R and g["layers"][1][cz + R][cx + R] == "s" and g["layers"][2][cz + R][cx + R] == "s":
            return True
    return False


LOOKAHEAD = 0.45                        # seconds: about one decision
SPEED = {"sprint": 5.6, "walk": 4.3}    # blocks per second
CREEPER_SPEED = 3.0                     # measured in the arena


def predict(obs, move_deg, sprinting, threat):
    """Where one decision of travel in direction move_deg (° right of facing) leaves you, and how that stands to the
    threat: (distance to it, whether a block is between you). The threat walks straight at your new spot."""
    run = free_run(obs, move_deg)
    d = min(SPEED["sprint" if sprinting else "walk"] * LOOKAHEAD, 7.5 if run is None else run)
    (fx, fz), (rx, rz) = v1.basis(obs["yaw"])
    a = math.radians(move_deg)
    me = {"x": obs["pos"]["x"] + d * (math.cos(a) * fx + math.sin(a) * rx),
          "z": obs["pos"]["z"] + d * (math.cos(a) * fz + math.sin(a) * rz)}
    c = threat["pos"]
    gap = math.hypot(me["x"] - c["x"], me["z"] - c["z"])
    step = min(CREEPER_SPEED * LOOKAHEAD, max(0.0, gap - 1))
    c2 = {"x": c["x"] + (me["x"] - c["x"]) / gap * step, "z": c["z"] + (me["z"] - c["z"]) / gap * step} if gap else c
    return math.hypot(me["x"] - c2["x"], me["z"] - c2["z"]), blocked(obs, me, c2)


def run_words(n):
    if n is None:
        return "open for 8+ blocks"
    if n < 1:
        return "something solid right there"
    return f"open for {n:.0f} block{'s' if round(n) != 1 else ''}, then something solid"


def open_lines(obs):
    parts = [f"{name} {run_words(free_run(obs, deg))}" for name, deg in
             (("ahead", 0), ("behind", 180), ("to your left", -90), ("to your right", 90))]
    return ["Open ground: " + "; ".join(parts) + "."]


# ---------------------------------------------------------------- options

def hold_options(cfg):
    return {
        "sprint": "sprint forward, the fastest way to move",
        "sprint_turn_around": "turn around (180°) and sprint that way",
        "sprint_turn_left": "turn 90° to the left and sprint that way",
        "sprint_turn_right": "turn 90° to the right and sprint that way",
        "forward": "walk forward",
        "back": "walk backward (slower than sprinting)",
        "left": "move sideways to your left",
        "right": "move sideways to your right",
        "sprint_jump": "sprint and jump forward (to get onto a 1-block step)",
        "turn_left": f"turn {cfg.turn_deg:g}° to the left",
        "turn_right": f"turn {cfg.turn_deg:g}° to the right",
        "turn_left_wide": "turn 90° to the left",
        "turn_right_wide": "turn 90° to the right",
        "turn_around": "turn around, 180°",
        "stop": "stop moving",
        "wait": "keep doing what you are doing",
        "use": "right-click the block under the crosshair (open or close a door, press a button)",
        "attack": "left-click: hit what is under the crosshair",
    }


# (turn in ° to the right, direction of travel relative to the new facing; None = the keys you already hold)
GEOMETRY = {"sprint": (0, 0), "forward": (0, 0), "sprint_jump": (0, 0), "back": (0, 180), "left": (0, -90),
            "right": (0, 90), "sprint_turn_around": (180, 0), "sprint_turn_left": (-90, 0), "sprint_turn_right": (90, 0),
            "turn_left": ("-turn", None), "turn_right": ("turn", None), "turn_left_wide": (-90, None),
            "turn_right_wide": (90, None), "turn_around": (180, None), "wait": (0, None)}


def held_direction(obs):
    held = set(obs.get("held", []))
    dirs = [d for k, d in (("forward", 0), ("back", 180), ("left", -90), ("right", 90)) if k in held]
    return dirs[0] if len(dirs) == 1 else None


def relation(a, m):
    d = (a - m + 180) % 360 - 180
    return "toward" if abs(d) < 45 else "away from" if abs(d) > 135 else "sideways to"


def mark_options(opts, obs, cfg, about, threat=None):
    """Say what each option does, as geometry: which way you would then travel relative to each thing in `about`
    (the danger, a goal) and how much open ground lies that way. The choice stays Jev's."""
    for k in opts:
        if k not in GEOMETRY:
            continue
        turn, move = GEOMETRY[k]
        turn = {"turn": cfg.turn_deg, "-turn": -cfg.turn_deg}.get(turn, turn)
        if move is None:
            held = held_direction(obs)
            if held is None:
                if turn and about:  # standing: a turn only changes where you face
                    opts[k] += " - " + ", ".join(f"then facing {relation(a, turn).replace('sideways to', 'sideways to')} {label}"
                                                 for label, a in about)
                continue
            move = held
        m = turn + move
        rels = ", ".join(f"{relation(a, m)} {label}" for label, a in about)
        opts[k] += f" - you would move {rels + ', ' if rels else ''}{run_words(free_run(obs, m))}"
        if threat:
            sprinting = k.startswith("sprint") or (GEOMETRY[k][1] is None and "sprint" in obs.get("held", []))
            dist, cover = predict(obs, m, sprinting, threat)
            opts[k] += (f"; then about {dist:.0f} blocks from the {mob_name(threat)}, "
                        + ("with a block between you" if cover else "in the open"))
    return opts


def action_question(cfg, obs, goal=None):
    opts = hold_options(cfg)
    about = []
    t = threats(obs)
    if t:
        about.append((f"the {mob_name(t[0][0])}", t[0][1]["angle"]))
    if goal:
        about.append(goal)
    return {"type": "choice", "instructions": "Which control should you use now?",
            "criteria": mark_options(opts, obs, cfg, about, t[0][0] if t else None)}


def recent_lines(history, cfg, n=8):
    """Hold mode: what each decision led to by the next one (distance covered, how far the danger was)."""
    opts = hold_options(cfg)
    hist = list(history)
    out = []
    for k, h in enumerate(hist[-n:], start=len(hist[-n:]) and len(hist) - len(hist[-n:]) + 1):
        label = opts.get(h["action"], h["action"]).split(" (")[0].split(",")[0]
        eff = f"moved {h['result']['moved']:.1f} blocks" if "after" in h else "just now"
        if h.get("danger0") is not None and h.get("danger1") is not None:
            eff += f"; {h['danger_name']} {h['danger0']:.0f} -> {h['danger1']:.0f} blocks away"
        if h["result"].get("note"):
            eff += f"; {h['result']['note']}"
        out.append(f"{k}. {label} -> {eff}")
    return out


# ---------------------------------------------------------------- c3: knowledge, surroundings, plan

def load_facts(path):
    with open(path, encoding="utf-8") as f:
        return [line[2:].strip() for line in f if line.startswith("- ")]


RECALL_CHUNK = 12  # facts per state: all their answers must fit in half of Jev's canvas (256 tokens, ~5 per yes/no)


def recall_states(facts, situation):
    """One yes/no per fact, in states of RECALL_CHUNK facts (ids facts0, facts1, ...; question ids f<fact index>),
    all sent in the same request."""
    return [{"id": f"facts{k // RECALL_CHUNK}",
             "state": f"A bot in Minecraft is in this situation: {situation} Each question below is about one thing "
                      f"a Minecraft player knows.",
             "questions": {f"f{i}": {"type": "boolean",
                                     "instructions": f"Would knowing this help the bot right now? \"{facts[i]}\""}
                           for i in range(k, min(k + RECALL_CHUNK, len(facts)))}}
            for k in range(0, len(facts), RECALL_CHUNK)]


def recalled_p(answers, i):
    return answers[f"facts{i // RECALL_CHUNK}"][f"f{i}"]["p_true"]


def surroundings(obs, things):
    """Landmarks and animals as candidate places to go, each with its current direction."""
    out = []
    for i, c in enumerate(things):
        r = v2.box_relative(obs, c)
        out.append((f"thing:{i}", v2.thing_description(c), r))
    for m, r in animals(obs):
        out.append((f"mob:{m['id']}", f"a {mob_name(m)} (animal)", r))
    return out


def surroundings_lines(obs, things):
    return [f"{d[0].upper()}{d[1:]}: {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}."
            for _, d, r in surroundings(obs, things)[:8]]


def plan_state(obs, prev, things, facts):
    """Asked in parallel with the action every step; its answer steers the next step."""
    near = surroundings(obs, things)[:8]
    opts = {"flee": "keep running away from the danger in the open"}
    for key, desc, r in near:
        opts[key] = f"get to {desc}, {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
    danger = " ".join(danger_lines(obs, prev)) or "There is no danger in sight."
    known = " ".join(facts) if facts else "nothing in particular"
    state = (f"You are a bot in Minecraft. {danger}\nWhat you know: {known}\n"
             f"Around you: " + ("; ".join(f"{d}, {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
                                         for _, d, r in near) or "open ground only") + ".")
    return {"id": "plan", "state": state,
            "questions": {"plan": {"type": "choice", "instructions": "What is the best way to stay safe right now?",
                                   "criteria": opts}}}


def plan_line(obs, plan, things):
    if not plan or plan == "flee":
        return ["Keep running away from the danger in the open."]
    for key, desc, r in surroundings(obs, things):
        if key == plan:
            there = "you are there" if r["dist"] <= 1.5 else f"{r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}"
            return [f"Get to {desc}: {there}."]
    return ["Keep running away from the danger in the open."]


def plan_target(obs, plan, things):
    for key, desc, r in surroundings(obs, things):
        if key == plan:
            return (desc, r["angle"])
    return None


# ---------------------------------------------------------------- state

def instruction_lines(mem, cfg, obs):
    ins = mem.instruction
    if ins.get("self"):
        return [f"Nobody told you anything: you noticed the danger yourself. Stay safe.",
                f"Done so far: {v2.done_so_far(mem, obs)}."]
    if ins["target"]["kind"] == "mob":  # v2 has no mob targets
        kind = ("This is a one-time request: it is complete as soon as you have done it once."
                if ins["once"] else "This is an ongoing request: keep doing it until you are told to stop.")
        return [f'{ins["from"]} (human) said {v1.ago(ins["t"])}: "{ins["text"]}"', kind,
                f"Target: {ins['target']['name']}.", f"Done so far: {v2.done_so_far(mem, obs)}."]
    return v2.instruction_lines(mem, cfg, obs)


def step_state(obs, mem, cfg, level):
    t = mem.instruction["target"]
    c3 = level >= 3
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + PREAMBLE_HOLD,
        v2.section("Danger", danger_lines(obs, mem.prev_obs)),
        v2.section("What you know", mem.facts) if c3 else None,
        v2.section("Your plan", plan_line(obs, mem.plan, mem.things)) if c3 else None,
        v2.section("Memory", [f"- {e['description']}" for e in mem.recalled]),
        v2.section("Current instruction", instruction_lines(mem, cfg, obs)),
        v2.section("Player messages since it started",
                   [f"[{v1.ago(m['t'])}] {m['username']}: {m['message']}" for m in mem.messages_since_start()[-3:]]),
        v2.section("You", v1.you_lines(obs) + [held_words(obs)]),
        v2.section("Target", target_lines(obs, cfg, t, mem.prev_obs)),
        v2.section("Around you: places and animals", surroundings_lines(obs, mem.things)) if c3 else None,
        v2.section("Other players", v2.players_lines(obs, cfg, t)),
        v2.section("Map", v1.map_lines(obs) + open_lines(obs)),
        v2.section("Your last actions (oldest first)", recent_lines(mem.history, cfg)),
        v2.section("Notes", v2.notes_lines(mem, cfg)),
    ] if x)


def target_lines(obs, cfg, t, prev):
    if t["kind"] == "mob":
        m = next((m for m in obs.get("mobs", []) if m["id"] == t["id"]), None)
        return [mob_line(obs, m, v1.relative(obs, m["pos"]), prev)] if m else [f"You cannot see {t['name']} any more."]
    return v2.target_lines(obs, cfg, t, None)


# ---------------------------------------------------------------- vision (c2/c3 with --vision only)
# No mob coordinates: two screenshots per look, in front and behind (the camera alone is turned for the second),
# and the questions are about the pictures. The code turns the answers into an estimated position, which the rest
# of this module then uses exactly as it uses a known one.

KINDS = {"creeper": "a creeper", "ocelot": "an ocelot or a cat", "person": "a person",
         "other": "some other creature", "none": "there is no creature"}
HARMLESS = {"ocelot", "person"}

LOOK2_STATE = ("Image 1 is what you see in front of you in Minecraft, in first person. Image 2 is what is behind "
               "you, seen by turning your head around.")


def look2_questions():
    """Asked about the two pictures only. The viewer draws creepers in odd colours: "is there a creeper in Image 2?"
    read 0.01 with one there, "is there a creature?" 0.66-0.76, so a creature that is not an ocelot or a person
    counts as the danger."""
    q = {}
    for k in (1, 2):
        q[f"any_{k}"] = {"type": "boolean",
                         "instructions": f"Is there a creature (an animal, a monster or a person) in Image {k}?"}
        q[f"kind_{k}"] = {"type": "choice", "instructions": f"What is the creature in Image {k}?", "criteria": KINDS}
        q[f"where_{k}"] = {"type": "choice", "instructions": f"Where in Image {k} is the creature, left to right?",
                           "criteria": {**{w: v[0] for w, v in v1.WHERE.items()}, "none": "no creature"}}
        q[f"size_{k}"] = {"type": "choice", "instructions": f"How big is the creature in Image {k}, from bottom to top?",
                          "criteria": {**{s: v[0] for s, v in v1.DISTANCE.items()}, "none": "no creature"}}
    return q


def read_look2(look):
    """-> [(kind, angle ° right of facing, distance, view)] for the threatening creatures seen."""
    best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
    out = []
    for k, view in ((1, "in front of you"), (2, "behind you")):
        if look[f"any_{k}"]["p_true"] < 0.5 or best(f"kind_{k}") in HARMLESS | {"none"} or best(f"where_{k}") == "none":
            continue
        a = v1.WHERE[best(f"where_{k}")][1]
        angle = a if k == 1 else (180 + a + 180) % 360 - 180  # image 2 looks backwards: its right is your left
        dist = v1.DISTANCE.get(best(f"size_{k}"), (None, 6))[1]
        out.append(("creeper", angle, dist, view))
    return out


def vision_obs(obs, seen, memory):
    """A copy of obs whose mobs are what was just seen (estimated positions), or, if nothing was seen, the last
    sighting carried along for a few steps. memory: {"mob": ..., "age": steps} updated in place."""
    mobs = []
    for kind, angle, dist, view in seen:
        pos = v2_project(obs, angle, dist)
        mobs.append({"id": -1, "name": kind, "pos": pos, "fuse": None, "seen": f"seen {view}, distance estimated"})
    if mobs:
        memory.update(mob=mobs[0], age=0)
    elif memory.get("mob") and memory["age"] < 5:
        memory["age"] += 1
        m = dict(memory["mob"])
        m["seen"] = f"not in view now; last seen {memory['age']} step{'s' if memory['age'] > 1 else ''} ago, there or near"
        mobs.append(m)
    return {**obs, "mobs": mobs}


def v2_project(obs, angle, dist):
    return v1.project(obs, angle, dist)
