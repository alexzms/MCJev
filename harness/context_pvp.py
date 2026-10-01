"""Harness P1 (PvP, sumo style): what a bot needs to knock another player off a small raised ring.

Built on the survival harness (held keys, options marked with what they would do, recalled facts) and v2 (target,
memory). Everything geometric is worked out here and stated as plain facts: where the ring's edge is around you,
how much ground is behind your opponent in the direction a hit from you would push them, whether they are within
reach, whether your sprint is fresh for extra knockback. The ring is found from the blocks under your feet (the
floor region you stand on), not from known coordinates, so any platform works.
"""
import math

import context as v1
import context_survive as cs
import context_v2 as v2

REACH = 3.0

RULES = ["Never walk or sprint off the edge of the ring. Falling into the water below kills you, and you lose the round.",
         "Do not get knocked off: keep ground behind you, the way their hits push you. When there is little, get back "
         "toward the middle, or step sideways out of the line to the edge, before anything else."]

STRATEGY = ("Sumo: you win by knocking the other player off the ring and lose if you fall off. Stay near the middle "
            "and never between your opponent and the edge. Hit whenever they are within reach, sprinting: a sprint hit "
            "knocks them much farther, and after one your sprint is spent, so let go and sprint again (w-tap) for the "
            "next. Move sideways while facing them so their hits miss. When there is little ground behind them, "
            "keep hitting them that way.")


# ---------------------------------------------------------------- the floor and its edge

def _floor(obs, x, z):
    """Solid ground under world point (x, z) at your feet level: True / False, None outside the observed grid."""
    g, R = obs["grid"], obs["grid"]["r"]
    cx, cz = math.floor(x) - math.floor(obs["pos"]["x"]), math.floor(z) - math.floor(obs["pos"]["z"])
    if abs(cx) > R or abs(cz) > R:
        return None
    return g["layers"][0][cz + R][cx + R] == "s"


def edge_distance(obs, p, ux, uz, limit=9.0):
    """Blocks of ground from p along the unit vector (ux, uz) before the edge (None if it runs past the grid)."""
    d = 0.25
    while d <= limit:
        f = _floor(obs, p["x"] + ux * d, p["z"] + uz * d)
        if f is None:
            return None
        if not f:
            return max(0.0, d - 0.3)  # a player is 0.6 wide: most of the body over the edge means falling
        d += 0.25
    return None


def world_dir(obs, rel_deg):
    (fx, fz), (rx, rz) = v1.basis(obs["yaw"])
    a = math.radians(rel_deg)
    return math.cos(a) * fx + math.sin(a) * rx, math.cos(a) * fz + math.sin(a) * rz


def ring(obs):
    """The floor region you stand on, from the blocks one below your feet: its middle and size, or None."""
    g, R = obs["grid"], obs["grid"]["r"]
    solid = lambda dx, dz: -R <= dx <= R and -R <= dz <= R and g["layers"][0][dz + R][dx + R] == "s"
    if not solid(0, 0):
        return None
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        x, z = stack.pop()
        for n in ((x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)):
            if n not in seen and solid(*n):
                seen.add(n)
                stack.append(n)
    if any(abs(x) == R or abs(z) == R for x, z in seen):
        return None  # the floor runs past what you can see: not a ring
    bx, bz = math.floor(obs["pos"]["x"]), math.floor(obs["pos"]["z"])
    cx = bx + sum(x for x, _ in seen) / len(seen) + 0.5
    cz = bz + sum(z for _, z in seen) / len(seen) + 0.5
    return {"center": {"x": cx, "y": obs["pos"]["y"], "z": cz}, "across": round(math.sqrt(len(seen)))}


def edge_words(d):
    if d is None:
        return "far"
    return "right at the edge" if d < 0.5 else f"{d:.0f} block{'s' if round(d) != 1 else ''}"


def ring_lines(obs):
    rg = ring(obs)
    if not rg:
        return ["You are not on a small ring (the ground goes on further than you can see)."]
    r = v1.relative(obs, rg["center"])
    mid = "right at the middle" if r["dist"] < 1 else f"{r['dist']:.1f} blocks from the middle, which is {v1.direction_words(r['angle'])}"
    edges = "; ".join(f"{name} {edge_words(edge_distance(obs, obs['pos'], *world_dir(obs, deg)))}"
                      for name, deg in (("ahead", 0), ("behind you", 180), ("to your left", -90), ("to your right", 90)))
    return [f"You are on a ring about {rg['across']} blocks across, {mid}.", f"Ground before the edge: {edges}."]


# ---------------------------------------------------------------- the opponent

def fighter(obs, name):
    return next((f for f in obs.get("fighters", []) if f["name"] == name and f.get("pos")), None)


def behind_them(obs, f):
    """Ground behind the opponent in the direction a hit from you pushes them (from you toward them)."""
    dx, dz = f["pos"]["x"] - obs["pos"]["x"], f["pos"]["z"] - obs["pos"]["z"]
    n = math.hypot(dx, dz) or 1.0
    return edge_distance(obs, f["pos"], dx / n, dz / n)


def behind_you(obs, f):
    """Ground behind you in the direction a hit from the opponent pushes you (from them toward you)."""
    dx, dz = obs["pos"]["x"] - f["pos"]["x"], obs["pos"]["z"] - f["pos"]["z"]
    n = math.hypot(dx, dz) or 1.0
    return edge_distance(obs, obs["pos"], dx / n, dz / n)


def reach_of(obs, f):
    eye = {"x": obs["pos"]["x"], "y": obs["pos"]["y"] + 1.62, "z": obs["pos"]["z"]}
    return math.sqrt((f["pos"]["x"] - eye["x"]) ** 2 + (f["pos"]["y"] + 0.9 - eye["y"]) ** 2 + (f["pos"]["z"] - eye["z"]) ** 2)


def opponent_lines(obs, name, prev):
    f = fighter(obs, name)
    if not f:
        return [f"You cannot see {name}."]
    r = v1.relative(obs, f["pos"])
    lines = [f"{name} is {r['dist']:.1f} blocks away, {v1.direction_words(r['angle'])}, {v1.height_words(r['dy'])}."]
    p = next((q for q in (prev or {}).get("fighters", []) if q["name"] == name and q.get("pos")), None)
    if p:
        d0 = v1.relative(prev, p["pos"])["dist"]
        if abs(d0 - r["dist"]) > 0.3:
            lines[-1] = lines[-1][:-1] + f" ({'closer' if r['dist'] < d0 else 'farther'} than at your last step, {d0:.1f})."
    reach = reach_of(obs, f)
    aimed = abs(r["angle"]) <= max(6.0, math.degrees(math.atan2(0.3, max(r["dist"], 0.3))))
    lines.append(("Within reach: a hit now lands." if reach <= REACH + 0.4 else f"Out of reach (a hit needs {REACH:.0f} "
                  f"blocks or less): it would miss.") + (" Your crosshair is on them." if aimed else " Your crosshair is not on them."))
    b = behind_them(obs, f)
    lines.append(f"Ground behind {name}, the way a hit from you would push them: {edge_words(b)}"
                 + (". One good hit could knock them off." if b is not None and b < 1.5 else "."))
    b = behind_you(obs, f)
    lines.append(f"Ground behind you, the way a hit from {name} would push you: {edge_words(b)}"
                 + (". One hit could knock you off: get back toward the middle first." if b is not None and b < 1.5 else "."))
    if f.get("sprinting"):
        lines.append(f"{name} is sprinting (their hits knock you back farther).")
    return lines


def you_lines(obs):
    lines = [f"Your hit is {'recharged' if obs.get('recharge', 1) >= 1 else 'recharging'}. "
             + ("Your sprint is fresh: your next hit while sprinting knocks back extra."
                if obs.get("sprint_fresh") else "Your sprint is spent (or you are not sprinting): let go and sprint again "
                                                "(w-tap) for extra knockback."),
             cs.held_words(obs)]
    if obs.get("edge_stop") is not None and obs["edge_stop"] < 3:
        lines.append(f"{obs['edge_stop']:.1f}s ago you nearly walked off the edge: your keys were let go just in time.")
    return lines


def combat_lines(obs, me):
    out = []
    for c in reversed(obs.get("combat", [])[-6:]):
        if c["ago"] > 6:
            continue
        if c["kind"] == "hit":
            out.append(f"{c['ago']:.1f}s ago: you hit {c['who']}.")
        elif c["kind"] == "hurt":
            out.append(f"{c['ago']:.1f}s ago: {c['who'] or 'someone'} hit you.")
        elif c["kind"] == "swing":
            out.append(f"{c['ago']:.1f}s ago: you swung and missed.")
    return out[:5]


# ---------------------------------------------------------------- options

MOVES = {  # id: (description, turn °, travel ° relative to the new facing or None)
    "sprint": ("sprint forward", 0, 0),
    "forward": ("walk forward", 0, 0),
    "back": ("walk backward", 0, 180),
    "left": ("move sideways to your left (strafe)", 0, -90),
    "right": ("move sideways to your right (strafe)", 0, 90),
    "turn_left": ("turn 30° to the left", -30, None),
    "turn_right": ("turn 30° to the right", 30, None),
    "turn_around": ("turn around, 180°", 180, None),
    "stop": ("stop moving", 0, "stop"),
    "wait": ("keep doing what you are doing", 0, None),
    "jump": ("jump (your other keys stay as they are)", 0, None),
}


def options(obs, name):
    f = fighter(obs, name)
    opts = {}
    rel = v1.relative(obs, f["pos"]) if f else None
    rg = ring(obs)
    mid = v1.relative(obs, rg["center"]) if rg else None
    for k, (desc, turn, travel) in MOVES.items():
        text = desc
        if travel == "stop":
            opts[k] = text
            continue
        if travel is None:
            travel = cs.held_direction(obs)
        if travel is not None:
            m = turn + travel
            ux, uz = world_dir(obs, m)
            e = edge_distance(obs, obs["pos"], ux, uz)
            parts = []
            if rel:
                parts.append(f"{cs.relation(rel['angle'], m)} {name}")
            if mid and mid["dist"] >= 1:
                parts.append(f"{cs.relation(mid['angle'], m)} the middle")
            off = 1.5 if k == "sprint" or "sprint" in obs.get("held", []) else 0.8
            parts.append("off the edge: you would fall into the water and lose" if e is not None and e < off else
                         f"{edge_words(e)} of ground that way" if e is not None else "plenty of ground that way")
            text += " - you would move " + ", ".join(parts)
        opts[k] = text
    if f:
        reach = reach_of(obs, f)
        b = behind_them(obs, f)
        push = (f"it pushes them toward the edge {edge_words(b)} behind them" if b is not None else "it pushes them back")
        lands = reach <= REACH + 0.4
        opts["aim"] = f"aim at {name} (turn to face them)"
        opts["aim_attack"] = (f"aim at {name} and hit - " + (f"lands: {push}" if lands else f"misses: they are {reach:.1f} blocks away"))
        opts["wtap_attack"] = (f"w-tap: let go of forward, sprint again, aim at {name} and hit - "
                               + (f"lands with sprint knockback: {push}, much farther" if lands else
                                  f"misses: they are {reach:.1f} blocks away"))
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


# ---------------------------------------------------------------- state

def step_state(obs, mem, cfg):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", RULES),
        v2.section("Strategy", [STRATEGY]),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("Opponent", opponent_lines(obs, name, mem.prev_obs) if name else ["No opponent."]),
        v2.section("The ring", ring_lines(obs)),
        v2.section("You", v1.you_lines(obs)[:1] + you_lines(obs)),
        v2.section("Fight so far", combat_lines(obs, obs["name"])),
        v2.section("Your last actions (oldest first)", cs.recent_lines(mem.history, cfg, n=6)),
    ] if x)
