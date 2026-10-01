"""Harness p_uhc_vis_raw_v3: Block UHC from pictures alone, raw keys, mouse and buttons, Jev deciding every step.

v2 let Jev press one control a step (a turn, or a key, or a click): up close it spent 110 of 137 steps turning left
and right, never moving and aiming at once, and put the attack indicator on the opponent twice. v3:
- three hands a step: the keys held, a mouse move and a button are three questions of one state, answered together,
  the way a player strafes, moves the mouse and clicks in the same moment (the body does them in that order);
- a scope: with the view, a zoomed view of its middle (field of view 30°), where Jev tells left / on / right of the
  cross much better (offline, 2026-09-26: turning by it put the cross on the body 37/40 times, by the view 27/40);
- lead: the bearings of the latest sightings give how fast the opponent crosses the view; the mouse is marked by
  where they will be when the move lands.
Still nothing from coordinates: everything about the opponent is what Jev saw (and what a player sees on screen:
the attack indicator, the hit flash, their own health and the push of a hit).
"""
import math

import context as v1
import context_uhc as cu
import context_uhc_vis as cv
import context_uhc_visraw2 as c2

ZOOM = 30            # the scope's field of view, degrees
ACT_DELAY = 0.05     # seconds from the picture to the move landing: Jev's answer and the body
SIGHTINGS_S = 0.5    # sightings this recent give the opponent's angular speed
LEAD_MAX = 12        # degrees at most added to lead them
RECHARGE_S = 0.625   # a diamond sword's recharge (attack speed 1.6)


# what the scope's answers mean, in degrees off the cross: measured, not the bins' share of the picture (offline,
# 2026-09-26, 60 views at 1.5-6 blocks, -25..25°: "center" spans about ±7°, "left"/"right" about 9-12°, and a body
# half out of the scope's edge (15° or more off) is still "left" or "far_right")
ZOOM_DEG = {"far_left": -12, "left": -9, "slightly_left": -4, "center": 0, "slightly_right": 4, "right": 9, "far_right": 12}


def zoom_deg(b):
    return ZOOM_DEG[b]


LOOK_STATE = cv.LOOK_STATE
LOOK = {k: c2.LOOK[k] for k in ("player", "x", "y", "size")}
SCOPE_STATE = ("A zoomed-in view (a scope) of the middle of a first-person view from a Minecraft game. A small cross, "
               "open in its middle, marks the middle of the picture.")
SCOPE = {k: c2.LOOK[k] for k in ("player", "x", "y", "size")}
# how far they are, from how big they look in the view and in the scope (offline, 2026-09-26, 70 views at 1.5-8
# blocks: the view says large/huge up to 2.5 blocks, medium/small from 3 to 8; the scope says huge up to ~3.5 blocks,
# large from 4.5 to 8)
RANGE = {"close": "right next to you", "reach": "about 3 blocks away, at the edge of reach", "mid": "a few blocks away",
         "far": "far off"}
# half a body's width in degrees at that range; a reading from the plain view is off by up to ~5° itself
TOL = {"close": 8, "reach": 5, "mid": 3, "far": 1.5}


def range_of(w, z):
    if w["player"] and w["size"] in ("huge", "large"):
        return "close"
    if z["player"] and z.get("size") == "huge":
        return "reach"
    if (z["player"] and z.get("size") in ("large", "medium")) or (w["player"] and w["size"] in ("medium", "small")):
        return "mid"
    return "far"


def read(look, scope):
    return {"wide": {"player": look["player"]["p_true"] >= 0.5, "x": cv.best(look["x"]), "y": cv.best(look["y"]),
                     "size": cv.best(look["size"])},
            "zoom": {"player": scope["player"]["p_true"] >= 0.5, "x": cv.best(scope["x"]), "y": cv.best(scope["y"]),
                     "size": cv.best(scope["size"])}}


def fuse(seen):
    """One reading from both pictures: {x: degrees right of the cross, y, range, fine: from the scope}, or None."""
    w, z = seen["wide"], seen["zoom"]
    # the scope only where the view puts them near the cross: 15° or more off, its answers stop growing
    if z["player"] and (not w["player"] or abs(cv.X_DEG[w["x"]]) <= 6):
        return {"x": round(zoom_deg(z["x"]), 1), "y": z["y"], "range": range_of(w, z), "fine": True}
    if w["player"]:
        return {"x": cv.X_DEG[w["x"]], "y": w["y"], "range": range_of(w, z), "fine": False}
    return None


def wrap(d):
    return (d + 180) % 360 - 180


def omega(sightings):
    """Degrees a second they cross the view (+ right), fitted to [(time, bearing)], or 0 when too few."""
    if len(sightings) < 3 or sightings[-1][0] - sightings[0][0] < 0.15:
        return 0.0
    t0, b0 = sightings[-1]
    pts = [(t - t0, b0 + wrap(b - b0)) for t, b in sightings]
    mt = sum(t for t, _ in pts) / len(pts)
    mb = sum(b for _, b in pts) / len(pts)
    var = sum((t - mt) ** 2 for t, _ in pts)
    w = sum((t - mt) * (b - mb) for t, b in pts) / var if var else 0.0
    return max(-180.0, min(180.0, w))


def estimate(fused, look_t, look_heading, sightings, now, heading):
    """Where they are now from the last reading: x (degrees right of where you face, at the move), lead (included)."""
    if not fused:
        return None
    w = omega(sightings)
    lead = w * (now + ACT_DELAY - look_t) if abs(w) >= 30 else 0.0
    lead = max(-LEAD_MAX, min(LEAD_MAX, lead))
    x = wrap(look_heading + fused["x"] + lead - heading)
    return {**fused, "x": round(x, 1), "omega": round(w), "lead": round(lead, 1), "ago": round(now - look_t, 2)}


def dist_class(est, obs):
    if cv.in_reach(obs):
        return "near"
    if not est:
        return None
    return "near" if est["range"] in ("close", "reach") else est["range"]


def recharge_left(obs):
    held = obs.get("held_item")
    if not (held and held.endswith("_sword")):
        return None
    return max(0.0, (1 - obs.get("recharge", 1)) * RECHARGE_S)


FIGHT = ("Each moment you choose three things at once, as with two hands: the keys you hold, a mouse move, and a "
         "button. How to fight: keep your sword in hand and close in; the bow only when they are far off (hold the "
         "right button about 1 s, then let go). A left-click lands only with the cross on them within 3 blocks - the "
         "attack indicator shows it - at full strength every 0.6 s; a click at the air starts the recharge over. Up "
         "close, keep the cross on them with small mouse moves while you strafe around them (forward and sideways when "
         "your sword is ready, back and sideways while it recharges) and click when the indicator shows and the sword "
         "is ready; jump and click while falling for a critical hit. If you lose them, turn toward where you last saw "
         "them or where a hit came from.")


LOST_S = 1.5   # how long after they leave the view where they went still says where they are


def gone(lastseen, now, heading):
    """Out of view: where they probably are now (+ right of where you face), from the last sighting and how they
    moved then; None when that is too old. lastseen: {t, bearing, omega, x, heading}."""
    if not lastseen or now - lastseen["t"] > LOST_S:
        return None
    age = now - lastseen["t"]
    b = lastseen["bearing"] + max(-120, min(120, lastseen["omega"] * min(age, 1.0)))
    rel_then = wrap(b - lastseen["heading"])
    if abs(lastseen["x"]) >= 18 and abs(rel_then) < 45:   # they left over the edge of the view: past it
        b = lastseen["heading"] + math.copysign(45, lastseen["x"])
    return wrap(b - heading), age


def seen_line(name, est, fused, look_ago, lastseen, heading, now):
    if not fused:
        g = gone(lastseen, now, heading)
        if g:
            d, age = g
            moving = (f" moving {'right' if lastseen['omega'] > 0 else 'left'}" if abs(lastseen["omega"]) >= 30 else "")
            return (f"Last look ({look_ago:.2f} s ago): {name} not in view. You saw them {age:.1f} s ago{moving}: now "
                    f"probably about {abs(d):.0f}° to your {'right' if d > 0 else 'left'}.")
        return f"Last look ({look_ago:.2f} s ago): {name} not in view."
    x = est["x"]
    where = "on the cross" if abs(x) < 1.5 else f"{abs(x):.0f}° {'right' if x > 0 else 'left'} of the cross"
    out = (f"Last look ({look_ago:.2f} s ago{', with the scope' if fused['fine'] else ''}): {name} {RANGE[fused['range']]}, "
           f"{cv.Y_WORDS[fused['y']]}; now about {where}")
    if est["omega"]:
        out += f" (they cross your view to the {'right' if est['omega'] > 0 else 'left'} at {abs(est['omega'])}°/s)"
    return out + "."


def you_line(obs):
    line = c2.you_line(obs)
    left = recharge_left(obs)
    if left:
        line = line.replace("sword in hand, recharging", f"sword in hand, recharging (ready in {left:.1f} s)")
    return line


def state(obs, name, est, fused, look_ago, lastseen, history, now):
    heading = -math.degrees(obs["yaw"])
    lines = [f"You are {obs['name']}, fighting {name} in Minecraft 1.11 Block UHC: to the death, no healing. You play "
             f"with keys and mouse; the picture is your view, the small cross in its middle is your crosshair.",
             FIGHT, "", seen_line(name, est, fused, look_ago, lastseen, heading, now)]
    if cv.in_reach(obs):
        lines.append("The attack indicator is on: your crosshair is on someone within reach.")
    lines.append(you_line(obs))
    ev = c2.events_lines(obs, name, bool(fused))
    if ev:
        lines.append("Lately: " + " ".join(ev))
    last = [h["words"] for h in list(history)[-2:] if "words" in h]
    if last:
        lines.append("Your last moves: " + "; ".join(last) + ".")
    return "\n".join(lines)


KEYS = {"none": [], "sprint": ["forward", "sprint"], "sprint_jump": ["forward", "sprint", "jump"],
        "forward": ["forward"], "forward_left": ["forward", "left"], "forward_right": ["forward", "right"],
        "left": ["left"], "right": ["right"], "back": ["back"], "back_left": ["back", "left"],
        "back_right": ["back", "right"], "jump": ["jump"]}
KEY_WORDS = {"none": "let go of all keys", "sprint": "sprint forward",
             "sprint_jump": "sprint forward jumping", "forward": "walk forward", "forward_left": "forward and strafe left",
             "forward_right": "forward and strafe right", "left": "strafe left", "right": "strafe right",
             "back": "walk backward", "back_left": "back and strafe left", "back_right": "back and strafe right",
             "jump": "jump in place"}
YAW = (2, 5, 10, 20, 45, 90)
PITCH = (5, 15)


def mouse_move(m):
    """Mouse option id -> (dyaw right, dpitch up)."""
    if m == "still":
        return 0, 0
    if m == "around":
        return 180, 0
    way, step = m[0], float(m[1:])
    return {"l": -step, "r": step}.get(way, 0), {"u": step, "d": -step}.get(way, 0)


def keys_question(obs, est, knock):
    dc = dist_class(est, obs)
    x = est["x"] if est else None
    left = recharge_left(obs)
    ready = left is not None and left <= 0.05
    held = [k for k in ("forward", "back", "left", "right", "sprint", "jump") if k in obs.get("held", [])]
    opts = {}
    for k, words in KEY_WORDS.items():
        text = words
        keys = KEYS[k]
        fwd, back, side = "forward" in keys, "back" in keys, ("left" in keys) != ("right" in keys)
        if dc == "near":
            if "sprint" in keys and not side:
                text += " → into them: you run past them and lose them"
            elif fwd and side:
                text += (" → stay in reach, circling them: your sword is ready" if ready else
                         " → stay in their reach while your sword recharges")
            elif back and side:
                text += (" → out of their reach, circling, while your sword recharges" if not ready else
                         " → away from them though your sword is ready")
            elif side:
                text += " → around them (their swing misses more)"
            elif back:
                text += (" → out of their reach while your sword recharges (back in when it is ready)" if not ready else
                         " → away from them though your sword is ready")
            elif not keys:
                text += " → standing still: easy to hit"
            elif keys == ["jump"]:
                text += " → a click while falling is a critical hit" if ready else " → up and down"
            elif fwd:
                text += " → closer (you are in reach already)"
        elif dc in ("mid", "far") and x is not None:
            facing = abs(x) <= 30
            if fwd and facing:
                text += " → toward them, into reach (a hit needs 3 blocks or less)" + (
                    ": fastest" if "jump" in keys and "sprint" in keys else ", slowly" if "sprint" not in keys else "")
            elif fwd:
                text += f" → not toward them: they are {abs(x):.0f}° to your {'right' if x > 0 else 'left'}"
            elif back and facing:
                text += " → away from them: out of your reach, no hit from there"
            elif side:
                text += " → sideways: arrows miss more" if dc == "far" else " → sideways: no closer to them"
            elif not keys:
                text += " → standing still: no closer to them"
        if sorted(keys) == sorted(held):
            text += " (what you hold now)"
        opts[k] = text
    return {"type": "choice", "instructions": "Which keys do you hold now?", "criteria": opts}


def mouse_question(obs, est, lastseen, looked, knock, now):
    heading = -math.degrees(obs["yaw"])
    on = cv.in_reach(obs)
    x = est["x"] if est else None
    tol = (TOL[est["range"]] if est["fine"] else max(TOL[est["range"]], 5)) if est else None
    aimed = on or (x is not None and abs(x) <= tol)
    back = None      # unseen: where a hit came from, or where they went (+ right of where you face)
    why = None
    g = gone(lastseen, now, heading) if x is None else None
    if x is None and knock is not None:
        back, why = knock, "where the hit came from"
    elif g:
        back, why = g[0], "where they went"
    turns = {"still": 0, **{f"{s[0]}{step}": sign * step for step in YAW for sign, s in ((-1, "left"), (1, "right"))},
             "around": 180}
    goal = x if x is not None else back
    best = min(turns, key=lambda m: abs(wrap(goal - turns[m]))) if goal is not None else None
    opts = {}
    def mark(m):
        turn = turns[m]
        if x is not None:
            after = wrap(x - turn)
            if m == best and abs(after) <= tol:
                return " → onto them" + (" (the middle of their body)" if aimed and turn else "")
            if abs(after) <= tol:
                return " → near the edge of their body"
            if aimed:
                return " → off them (you are on them now)"
            way = "right" if after > 0 else "left"
            return f" → they are then {abs(after):.0f}° {way} of the cross" + (" (away from them)" if abs(after) > abs(x) else "")
        if back is not None:
            return f" → toward {why}" if m == best and abs(wrap(back - turn)) <= max(12, abs(turn) * 0.4) else ""
        if looked:
            fresh = cv.unseen_after(math.degrees(obs["yaw"]), turn, looked)
            return f" → shows {fresh}° not looked at lately" if fresh else " → shows what you just saw"
        return ""
    opts["still"] = "keep the mouse still" + mark("still")
    for step in YAW:
        for s in ("left", "right"):
            opts[f"{s[0]}{step}"] = f"turn {step}° {s}" + mark(f"{s[0]}{step}")
    opts["around"] = "turn around" + mark("around")
    y = est["y"] if est else None
    for step in PITCH:
        for sign, w in ((1, "up"), (-1, "down")):
            after = math.degrees(obs["pitch"]) + sign * step
            text = f"tilt {step}° {w}"
            if y in ("above", "below") and (y == "above") == (sign > 0):
                text += " → toward them"
            elif y == "level" and aimed and step > 5:
                text += " → off them (you are on them now)"
            elif abs(after) <= 5:
                text += " → level"
            else:
                text += f" → looking {abs(after):.0f}° {'up' if after > 0 else 'down'}"
            opts[f"{w[0]}{step}"] = text
    return {"type": "choice", "instructions": "How do you move the mouse now?", "criteria": opts}


def button_question(obs, name, est, history):
    size = {"close": "large", "reach": "medium", "mid": "small", "far": "tiny"}[est["range"]] if est else "tiny"
    look = ({"player": True, "x": "center", "y": est["y"], "size": size, "side": "none"} if est else
            {"player": False, "x": "center", "y": "level", "size": "tiny", "side": "none"})
    base = c2.options(obs, name, look, 0, (), (), None)["criteria"]
    opts = {"none": "no button"}
    left = recharge_left(obs)
    on = cv.in_reach(obs)
    if left and left > 0.05:
        opts["none"] += f" → your sword recharges (ready in {left:.1f} s)"
    elif left is not None and not on:
        opts["none"] += " → your sword stays ready for a full hit when the indicator shows"
    for k, text in base.items():
        if k in ("click", "use", "use_release") or k.startswith("slot_"):
            if k == "click" and not on and left is not None:
                text += " and starts your recharge over"
            opts[k] = text
    for k in opts:
        for h in list(history)[-2:]:
            note = h["result"].get("note") or ""
            if h.get("button") == k and any(w in note for w in ("did not", "cannot", "no ", "not ", "nothing")):
                opts[k] += f" (failed just now: {note[:40]})"
                break
    return {"type": "choice", "instructions": "Which button now?", "criteria": opts}


def knock_bearing(obs, max_ago=1.5):
    """Where the last push came from (+ right of where you face), if a hit pushed you lately."""
    k, drops = obs.get("knock"), [d for d in obs.get("drops", []) if d["ago"] <= max_ago]
    if not (k and drops and abs(k["ago"] - drops[-1]["ago"]) <= 0.6):
        return None
    (fx, fz), (rx, rz) = v1.basis(obs["yaw"])
    push = math.degrees(math.atan2(k["vx"] * rx + k["vz"] * rz, k["vx"] * fx + k["vz"] * fz))
    return wrap(push + 180)


def words(k, m, b):
    turn = ("mouse still" if m == "still" else "turned around" if m == "around" else
            f"{'turned' if m[0] in 'lr' else 'tilted'} {m[1:]}° {dict(l='left', r='right', u='up', d='down')[m[0]]}")
    return f"{KEY_WORDS[k]}, {turn}, {'no button' if b == 'none' else b.replace('_', ' ')}"
