"""Harness p_uhc_raw_v1: Block UHC with the raw controls of a player - keys, mouse steps, the two buttons and the
hotbar - and no help from the body in aiming. Jev turns the view itself until the crosshair is on the opponent,
left-clicks to hit, right-clicks to use what it holds (holding the button to draw a bow or eat), and picks hotbar
slots itself. The state says where the crosshair is relative to the opponent, in degrees, and every mouse option
says where the crosshair would then be; the other options say what they would do from where the crosshair is.
Kept from p_uhc_v1: the text view of the opponent, you, your items, lava and fire, the fight so far; auto-jump and
the edge guard (settings and reflexes, not skill).
"""
import math

import context as v1
import context_pvp as cp
import context_survive as cs
import context_uhc as cu
import context_v2 as v2

REACH = 3.0
YAW_STEPS = (3, 10, 30, 90)
PITCH_STEPS = (5, 15, 45)
FACES = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (1, 0, 0)]

STRATEGY = ("You aim yourself: turn your view in steps until your crosshair is on your opponent, then left-click to "
            "hit. A sword hit does full damage once recharged (every 0.6 s). Sprint into a hit for knockback; after "
            "it let go of forward and sprint again (w-tap). A hit while falling from a jump is a critical hit. "
            "For the bow, hold the slot with it, press and hold the right button for 1 s with the crosshair a "
            "little above them, then let go. Right-click a bucket or a block where the crosshair points.")

MOVES = {k: v for k, v in cu.MOVES.items() if k not in ("turn_left", "turn_right", "turn_around")}


def aim_error(obs, pos, height=1.2):
    """Degrees from the crosshair to a point `height` above pos: (right +, up +), and the distance from your eyes."""
    eye_y = obs["pos"]["y"] + 1.62
    r = v1.relative(obs, pos)
    horiz = max(0.05, math.hypot(pos["x"] - obs["pos"]["x"], pos["z"] - obs["pos"]["z"]))
    up = math.degrees(math.atan2(pos["y"] + height - eye_y, horiz)) - math.degrees(obs["pitch"])
    return r["angle"], up, math.hypot(horiz, pos["y"] + height - eye_y)


def on_body(dyaw, dup, dist):
    """Would the crosshair be on a player's body (0.6 wide, 1.8 tall) at these errors?"""
    return abs(dyaw) <= math.degrees(math.atan2(0.3, dist)) + 1 and abs(dup) <= math.degrees(math.atan2(0.8, dist)) + 1


def error_words(dyaw, dup, dist, name):
    if on_body(dyaw, dup, dist):
        return f"on {name}"
    parts = []
    if abs(dyaw) >= 1:
        parts.append(f"{abs(dyaw):.0f}° {'left' if dyaw > 0 else 'right'} of {name}")   # the crosshair, not the target
    if abs(dup) >= 1:
        parts.append(f"{abs(dup):.0f}° {'below' if dup > 0 else 'above'} them")
    return ", ".join(parts) or f"just off {name}"


def bow_lift(dist):
    """Degrees above the target an arrow needs at this distance (a full draw)."""
    t = dist / 3.0
    return math.degrees(math.atan2(0.025 * t * t, max(dist, 0.1)))


def aim_lines(obs, name, f):
    if not f:
        return []
    dyaw, dup, dist = aim_error(obs, f["pos"])
    ce = obs.get("cursor_entity")
    lines = [f"Your crosshair is {error_words(dyaw, dup, dist, name)}."]
    if ce and ce.get("name") == name:
        lines[0] = f"Your crosshair is on {name}, {ce['distance']:.1f} blocks from your eyes" + (
            " - within reach: a left-click hits." if ce["distance"] <= REACH + 0.3 else " - out of reach for a hit.")
    if obs.get("held_item") == "bow" and dist > 6:
        lines.append(f"For an arrow at {dist:.0f} blocks, the crosshair should be about {bow_lift(dist):.0f}° above them.")
    return lines


def hotbar_lines(obs):
    hb, slot = obs.get("hotbar") or [], obs.get("slot")
    out = []
    for i, it in enumerate(hb):
        if it:
            out.append(f"{i + 1}: {cu.item_words(it['name'])}" + (f" x{it['count']}" if it["count"] > 1 else "")
                       + (" (in your hand)" if i == slot else ""))
    lines = ["Hotbar - " + "; ".join(out) + "."]
    if obs.get("using_s") is not None:
        held = obs.get("held_item")
        lines.append(f"You are holding the right button down ({obs['using_s']:.1f} s)"
                     + (": the bow is fully drawn." if held == "bow" and obs["using_s"] >= 1 else
                        ": the bow is still drawing." if held == "bow" else "."))
    return lines


def cursor_spot(obs):
    """The cell a right-click with a block or bucket would fill: next to the face under the crosshair, or None."""
    c = obs.get("cursor")
    if not c or c.get("face") is None or c["distance"] > 4.5:
        return None
    fx, fy, fz = FACES[c["face"]]
    return {"x": c["pos"]["x"] + fx + 0.5, "y": c["pos"]["y"] + fy, "z": c["pos"]["z"] + fz + 0.5}


def spot_words(obs, spot, name, f):
    me = math.hypot(spot["x"] - obs["pos"]["x"], spot["z"] - obs["pos"]["z"])
    words = f"{me:.1f} blocks from you"
    if f:
        d = math.hypot(spot["x"] - f["pos"]["x"], spot["z"] - f["pos"]["z"])
        words += f", {d:.1f} from {name}" + (" (where they stand)" if d < 0.8 else "")
    return words


def options(obs, name, history=(), v=1):
    f = cp.fighter(obs, name) if name else None
    rel = v1.relative(obs, f["pos"]) if f else None
    opts = {}
    base = cu.options(obs, name, history, v=1)["criteria"]   # the move marks only
    for k in MOVES:
        opts[k] = base[k]
    # the mouse: where the crosshair would be
    err = aim_error(obs, f["pos"]) if f else None
    for step in YAW_STEPS:
        for sign, side in ((-1, "left"), (1, "right")):
            k = f"look_{side[0]}{step}"
            text = f"turn your view {step}° to the {side}"
            if err:
                text += f" - the crosshair would be {error_words(err[0] - sign * step, err[1], err[2], name)}"
            opts[k] = text
    opts["turn_around"] = "turn around, 180°" + (f" - you would face away from {name}" if rel and abs(rel["angle"]) < 90 else
                                                 f" - you would face {name}" if rel else "")
    for step in PITCH_STEPS:
        for sign, way in ((1, "up"), (-1, "down")):
            k = f"look_{way[0]}{step}"
            text = f"tilt your view {step}° {way}"
            if err:
                text += f" - the crosshair would be {error_words(err[0], err[1] - sign * step, err[2], name)}"
            opts[k] = text
    # the buttons
    held = obs.get("held_item")
    ce = obs.get("cursor_entity")
    on = ce and ce.get("name") == name
    if on and ce["distance"] <= REACH + 0.3:
        weapon = cu.is_weapon(held)
        strength = ("full strength" if weapon and obs.get("recharge", 1) >= 1 else "weak: still recharging" if weapon
                    else f"weak: you hold {'a ' + cu.item_words(held) if held else 'nothing'}")
        click = f"hits {name}, {strength}" + ("; a critical hit (you are falling)" if obs.get("falling") else "")
    elif on:
        click = f"misses: {name} is under your crosshair but {ce['distance']:.1f} blocks away"
    else:
        click = "swings at the air: nobody under your crosshair within reach"
    opts["click"] = f"left-click - {click}"
    spot = cursor_spot(obs)
    if obs.get("using_s") is not None:
        opts["use_release"] = ("let go of the right button - " + (
            f"shoots the arrow ({'full power' if obs['using_s'] >= 1 else 'weak: not fully drawn'})" if held == "bow"
            else "stops using it"))
        if v >= 2 and held == "bow" and f:
            if cs.blocked(obs, obs["pos"], f["pos"]):
                opts["use_release"] += "; blocks are between you: the arrow would hit them - keep drawing until they step out"
            if cp.reach_of(obs, f) <= REACH + 1.5:
                opts["use_release"] += f"; {name} is right next to you: a sword hit is better"
    elif held == "bow":
        opts["use"] = "press and hold the right button: start drawing the bow"
    elif held in ("golden_apple", "cooked_beef"):
        opts["use"] = "press and hold the right button: eat it (1.6 s)"
    elif held in cu.BLOCKS:
        opts["use"] = "right-click: place a block " + (f"there, {spot_words(obs, spot, name, f)}" if spot else
                                                        "- nothing within reach under your crosshair")
    elif held in ("lava_bucket", "water_bucket"):
        what = "lava" if held == "lava_bucket" else "water"
        opts["use"] = f"right-click: pour the {what} " + (f"there, {spot_words(obs, spot, name, f)}" if spot else
                                                          "- nothing within reach under your crosshair")
    elif held == "bucket":
        srcs = obs.get("sources", [])
        opts["use"] = "right-click: scoop up the liquid under your crosshair" + (
            f" (the nearest: {srcs[0]['kind']} {srcs[0]['dist']:.1f} blocks away)" if srcs else " - no source within reach")
    # the hotbar
    close = bool(f) and cp.reach_of(obs, f) <= REACH + 1.5
    for i, it in enumerate(obs.get("hotbar") or []):
        if it and i != obs.get("slot"):
            opts[f"slot_{i + 1}"] = (f"press {i + 1}: hold your {cu.item_words(it['name'])}"
                                     + (" (the sword's recharge starts over)" if it["name"].endswith("_sword") else ""))
            if v >= 2 and it["name"].endswith("_sword") and close:
                opts[f"slot_{i + 1}"] += f" - {name} is close: switch now" + (
                    " (drops the drawn arrow)" if obs.get("using_s") is not None and held == "bow" else "")
    for k in opts:
        why = cu.failed_last(history, k)
        if why:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


def act_params(option):
    """Option id -> (body action, params)."""
    if option.startswith("look_") and option[5] in "lrud":
        way, step = option[5], float(option[6:])
        return "look", {"dyaw": {"l": -step, "r": step}.get(way, 0), "dpitch": {"u": step, "d": -step}.get(way, 0)}
    if option.startswith("slot_"):
        return "slot", {"n": int(option[5:])}
    return option, {}


def step_state(obs, mem, cfg, v=1):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    f = cp.fighter(obs, name) if name else None
    you = [line for line in cu.you_lines(obs) if not line.startswith("In your hand")]
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", cu.RULES),
        v2.section("Strategy", [STRATEGY + (" If they come close while you hold the bow, press your sword's slot at "
                                            "once; if blocks are between you, keep the bow drawn and let go when they "
                                            "step out. When you are hurt and do not see them, look where the hit came "
                                            "from." if v >= 2 else "")]),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("Opponent", (cu.opponent_lines(obs, name, mem.prev_obs) + aim_lines(obs, name, f)) if name else ["No opponent."]),
        v2.section("You", you + (cu.feel_lines(obs) if v >= 2 else [])),
        v2.section("Your hotbar", hotbar_lines(obs) + cu.item_lines(obs)[1:]),
        v2.section("Around you", cu.hazard_lines(obs)),
        v2.section("Fight so far", cu.combat_lines(obs)),
        v2.section("Your last actions (oldest first)", cu.recent_lines(mem.history)),
    ] if x)
