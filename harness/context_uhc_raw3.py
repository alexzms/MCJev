"""Harness p_uhc_raw_v3: Block UHC from the game's data, with raw keys, mouse and buttons, Jev deciding every step.

The p_uhc_raw line (the controls of a player, no aim help from the body), rebuilt with what p_uhc_pro_v10 and
p_uhc_vis_raw_v3 taught:
- three hands a step: the keys held, the mouse (left/right and up/down, one question each) and a button, four
  questions of one state answered in one pass, the way a player strafes, moves the mouse and clicks at once; v1/v2
  chose one control a step, so aiming stopped the feet and moving stopped the aim;
- p_uhc_pro_v10's state, short: techniques, this round, the last seconds, them now (distance, where from your
  crosshair, how they move, their sword's recharge, their bow), you now, items, lava and fire;
- marks from the data: each mouse move says where the crosshair ends up against where they will be when it lands
  (their speed times the step's delay) - their body for the sword, the arrow's aim point (drop and lead) for a
  drawn bow; each key set what it does to the distance and the fight's rhythm; each button what it does now.
Text only, one request a step (~30 ms on the fast engine).
"""
import math

import context as v1
import context_pvp as cp
import context_uhc as cu
import context_uhc_pro as pro
import context_uhc_raw as cr

REACH = 3.0
LAT = 0.06           # seconds from the observation to the move reaching the server: Jev, the body, a tick
TRACK_S = 0.2        # their positions this recent give their speed (0.3 lagged behind a stop: arrows went ahead)
RECHARGE_S = 0.625   # a diamond sword's recharge
SPEED = {"sprint": 5.6, "walk": 4.3}
YAW = (1, 2, 4, 8, 15, 25, 40, 60, 90, 135)
PITCH = (1, 2, 4, 7, 12, 20, 40)


def wrap(d):
    return (d + 180) % 360 - 180


# ---------------------------------------------------------------- where they are going, where to aim

def speed_of(vel):
    return math.hypot(vel[0], vel[2])


def velocity(track):
    """Blocks a second (x, y, z) from [(t, pos)], oldest first; zeros when too few."""
    if len(track) < 2 or track[-1][0] - track[0][0] < 0.08:
        return (0.0, 0.0, 0.0)
    (t0, p0), (t1, p1) = track[0], track[-1]
    dt = t1 - t0
    return tuple((p1[k] - p0[k]) / dt for k in ("x", "y", "z"))


def draw_power(s):
    """A bow drawn s seconds: the arrow's power, 0-1 (1.9+: f = (s^2 + 2s) / 3)."""
    return max(0.0, min(1.0, (s * s + 2 * s) / 3))


def arrow_pitch(dh, dy, speed):
    """Pitch (degrees, + up) that brings an arrow of `speed` blocks a tick `dh` blocks away at height `dy`, and the
    ticks it flies; None if out of range. Arrows: drag 0.99 a tick, gravity 0.05 a tick squared."""
    def height_at(theta):
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
        x = y = 0.0
        for tick in range(1, 120):
            x, y = x + vx, y + vy
            vx, vy = vx * 0.99, vy * 0.99 - 0.05
            if x >= dh:
                return y, tick
        return None, None
    lo, hi = math.radians(-40), math.radians(40)
    if height_at(hi)[0] is None or height_at(hi)[0] < dy:
        return None
    for _ in range(24):
        mid = (lo + hi) / 2
        y, _ = height_at(mid)
        if y is None or y < dy:
            lo = mid
        else:
            hi = mid
    return math.degrees(hi), height_at(hi)[1]


def aim(obs, f, vel, full=False):
    """Where the crosshair should be: {yaw, pitch} errors (degrees, + right / + up) from where it is, the tolerance
    (half the body), the distance, and for a drawn or held bow the arrow's flight. Sword: their chest where they
    will be when the move lands; bow: the arrow's aim point for where they will be when it arrives."""
    eye = {"x": obs["pos"]["x"], "y": obs["pos"]["y"] + 1.62, "z": obs["pos"]["z"]}
    bow = obs.get("held_item") == "bow"
    s = obs.get("using_s") if bow and obs.get("using_s") is not None else 1.0
    speed = 3.0 if full else 3 * max(draw_power(s + 0.1), 0.3)
    t_ahead, flight, sol, short = LAT, None, None, False
    for _ in range(3 if bow else 1):
        p = {k: f["pos"][k] + vel[i] * t_ahead for i, k in enumerate(("x", "y", "z"))}
        dx, dz, dy = p["x"] - eye["x"], p["z"] - eye["z"], p["y"] + 0.9 - eye["y"]
        dh = math.hypot(dx, dz)
        if not bow or dh < 4:
            break
        s_ = arrow_pitch(dh, dy + 0.1, speed)
        if not s_:
            short = sol is None
            break
        sol, flight = s_, s_[1] / 20
        t_ahead = LAT + flight
    yaw_to = -math.degrees(math.atan2(-dx, -dz))                  # compass: degrees right of north
    pitch_to = math.degrees(math.atan2(dy, max(dh, 0.05)))
    if flight is not None:
        pitch_to = sol[0]
    facing = -math.degrees(obs["yaw"])
    dist = math.hypot(dh, dy)
    if bow and short and not full:   # not drawn enough to reach: aim for the full draw it is building up to
        return {**aim(obs, f, vel, full=True), "short": True}
    return {"yaw": wrap(yaw_to - facing), "pitch": pitch_to - math.degrees(obs["pitch"]), "dist": dist,
            "tol_yaw": math.degrees(math.atan2(0.3, max(dist, 0.3))), "tol_pitch": math.degrees(math.atan2(0.9, max(dist, 0.3))),
            "flight": flight, "bow": flight is not None, "short": bow and short}


# ---------------------------------------------------------------- the state

TECHNIQUES = [
    "1. Close (within about 4 blocks): the sword. A left-click hits only with the crosshair on them within 3 "
    "blocks; full strength once your sword has recharged (0.6 s after the last click - a click at the air starts it "
    "over). A click while falling from a jump is a critical hit (1.5x). A hit right after you press sprint knocks "
    "them back (a w-tap: let go of sprint, press it again). Circle them: in to hit when your sword is ready, out of "
    "their reach while it recharges, sideways all the time so their swings miss.",
    "2. At range (8 blocks or more): the bow. Hold the right button 1 s for full power, keep the crosshair on the "
    "arrow's aim point (the marks say where: above and ahead of them), let go. Take the sword before they are "
    "within 4 blocks.",
    "3. Their bow: move sideways, do not run straight at a drawn bow from far.",
    "4. Lava at their feet or in their path burns them (8 a second in it, 1 a second for 15 s after); never walk into "
    "lava or fire; water puts out fire and lava.",
]


def them_lines(obs, f, name, a, vel):
    if not f:
        return [f"You cannot see {name}."]
    rel = v1.relative(obs, f["pos"])
    d = cp.reach_of(obs, f)
    out = []
    where = (f"{abs(rel['angle']):.0f}° to your {'right' if rel['angle'] > 0 else 'left'}" if abs(rel["angle"]) >= 2
             else "straight ahead")
    hp = f" health {f['health']:.0f}/20;" if f.get("health") is not None else ""
    out.append(f"{name}: {d:.1f} blocks away (a hit needs 3 or less), {where}" +
               (f", {rel['dy']:+.0f} blocks up" if abs(rel["dy"]) >= 1 else "") + f";{hp} holding {cu.item_words(f.get('holding'))}.")
    # how they move, as seen from you
    fx, fz = rel["ahead"], rel["right"]
    n = math.hypot(fx, fz) or 1
    (bx, bz), (rx, rz) = v1.basis(obs["yaw"])
    va, vr = vel[0] * bx + vel[2] * bz, vel[0] * rx + vel[2] * rz     # their speed along your forward / right
    closing = -(va * fx + vr * fz) / n
    across = (vr * fx - va * fz) / n                                   # + moving to your right across your view
    sp = math.hypot(vel[0], vel[2])
    if sp < 0.5:
        out.append(f"{name} stands still.")
    else:
        way = ("coming at you" if closing > 1.5 else "backing off" if closing < -1.5 else "moving across")
        out.append(f"{name} is {way} ({sp:.1f} blocks/s" + (f", crossing your view to the {'right' if across > 0 else 'left'} at "
                                                           f"{math.degrees(abs(across) / max(d, 1)):.0f}°/s" if abs(across) > 1 else "") + ").")
    sw = pro.their_sword(f)
    if sw and d < 6:
        out.append(f"Their sword recharges for {sw[1]:.1f} s more: step in and hit." if sw[0] == "recharging"
                   else "Their sword is ready: their next hit is a full one.")
    if f.get("holding") == "bow" and f.get("using"):
        drawn = obs.get("their_draw_s")
        out.append(f"{name} is drawing their bow" + (f" ({drawn:.1f} s: {'full' if drawn >= 1 else 'not full yet'})" if drawn is not None else "")
                   + ": an arrow is coming.")
    if f.get("burning"):
        out.append(f"{name} is burning.")
    inc = obs.get("incoming")
    if inc:
        out.append(f"Their arrow is on its way and hits you in {inc['in_s']:.2f} s unless you step {inc['key']} (strafe) off its line.")
    return out


def crosshair_line(obs, f, name, a):
    if not f:
        return []
    ce = obs.get("cursor_entity")
    if ce and ce.get("name") == name and ce["distance"] <= REACH + 0.3:
        return [f"Your crosshair is on {name} within reach: the attack indicator shows."]
    what = "the arrow's aim point for them" if a["bow"] else name
    yaw = f"{abs(a['yaw']):.0f}° {'right' if a['yaw'] > 0 else 'left'}" if abs(a["yaw"]) > a["tol_yaw"] else "on"
    pitch = f"{abs(a['pitch']):.0f}° {'above' if a['pitch'] > 0 else 'below'}" if abs(a["pitch"]) > a["tol_pitch"] else "level"
    fly = f" (the arrow flies {a['flight']:.2f} s)" if a["bow"] else ""
    if yaw == "on" and pitch == "level":
        return [f"Your crosshair is on {what}{fly}."]
    return [f"Your crosshair: {what} is {yaw if yaw != 'on' else 'straight'}, {pitch} from it{fly}."]


def recharge_left(obs):
    held = obs.get("held_item")
    if not cu.is_weapon(held):
        return None
    return max(0.0, (1 - obs.get("recharge", 1)) * RECHARGE_S)


def you_line(obs):
    held = obs.get("held_item")
    parts = [f"health {obs['health']:.0f}/20"]
    left = recharge_left(obs)
    if left is not None:
        parts.append("sword in hand, " + ("recharged" if left <= 0.05 else f"recharging ({left:.1f} s)"))
    else:
        parts.append(f"{cu.item_words(held) if held else 'nothing'} in hand")
    if held == "bow" and obs.get("using_s") is not None:
        p = draw_power(obs["using_s"])
        parts.append(f"bow drawn {obs['using_s']:.1f} s (power {100 * p:.0f}%)" + ("" if p >= 1 else "; you walk at a fifth"))
    if obs.get("falling"):
        parts.append("falling: a click now is a critical hit")
    if obs.get("burning"):
        parts.append("burning")
    held_keys = [k for k in ("forward", "back", "left", "right", "sprint", "jump") if k in obs.get("held", [])]
    parts.append("keys: " + (", ".join(held_keys) if held_keys else "none"))
    if obs.get("sprint_fresh") and "sprint" in held_keys:
        parts.append("a fresh sprint: your next hit knocks them back")
    return "You: " + "; ".join(parts) + "."


def state(obs, name, f, a, vel, history):
    rnd = f"This round: you {obs['health']:.0f}/20" + (f", {name} {f['health']:.0f}/20." if f and f.get("health") is not None else ".")
    ev = pro.event_lines(obs, 4.0)[-5:]
    lines = [f"You are {obs['name']}, fighting {name} in Minecraft 1.11 Block UHC: to the death, no healing. You play "
             f"with keys and mouse: each moment you choose the keys you hold, how the mouse moves (left/right and "
             f"up/down) and a button.",
             "Techniques: " + " ".join(TECHNIQUES), "", rnd]
    if ev:
        lines.append("Last seconds: " + " ".join(ev))
    lines += them_lines(obs, f, name, a, vel) + (crosshair_line(obs, f, name, a) if a else []) + [you_line(obs)]
    lines += cr.hotbar_lines(obs)[:1]
    hz = cu.hazard_lines(obs)
    if not hz[0].startswith("No lava"):
        lines += hz[:2]
    last = [h["words"] for h in list(history)[-2:] if "words" in h]
    if last:
        lines.append("Your last moves: " + "; ".join(last) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------- the questions

KEYS = {"none": [], "sprint": ["forward", "sprint"], "sprint_jump": ["forward", "sprint", "jump"],
        "forward": ["forward"], "forward_left": ["forward", "left"], "forward_right": ["forward", "right"],
        "left": ["left"], "right": ["right"], "back": ["back"], "back_left": ["back", "left"],
        "back_right": ["back", "right"], "jump": ["jump"]}
KEY_WORDS = {"none": "let go of all keys", "sprint": "sprint forward", "sprint_jump": "sprint forward jumping",
             "forward": "walk forward", "forward_left": "forward and strafe left", "forward_right": "forward and strafe right",
             "left": "strafe left", "right": "strafe right", "back": "walk backward", "back_left": "back and strafe left",
             "back_right": "back and strafe right", "jump": "jump in place"}
MOVE_DEG = {"sprint": 0, "sprint_jump": 0, "forward": 0, "forward_left": -45, "forward_right": 45, "left": -90,
            "right": 90, "back": 180, "back_left": -135, "back_right": 135}


def into_hazard(obs, move_deg, within=2.0):
    """Lava or fire in the ground or at your feet within `within` blocks that way."""
    ux, uz = cp.world_dir(obs, move_deg)
    for h in cu.hazard_cells(obs, radius=3):
        if h["level"] == "at head height" or h["dist"] > within + 0.7:
            continue
        dx, dz = h["pos"]["x"] - obs["pos"]["x"], h["pos"]["z"] - obs["pos"]["z"]
        along = dx * ux + dz * uz
        if 0 < along <= within and abs(dx * uz - dz * ux) <= 0.8:
            return h["kind"]
    return None


def keys_question(obs, f, name, a):
    held = sorted(k for k in ("forward", "back", "left", "right", "sprint", "jump") if k in obs.get("held", []))
    left = recharge_left(obs)
    ready = left is not None and left <= 0.05
    rel = v1.relative(obs, f["pos"]) if f else None
    d = cp.reach_of(obs, f) if f else None
    sw = pro.their_sword(f) if f else None
    theirs_ready = not sw or sw[0] == "ready"
    drawing_at_you = f and f.get("holding") == "bow" and f.get("using")
    opts = {}
    for k, words in KEY_WORDS.items():
        keys = KEYS[k]
        text = words
        md = MOVE_DEG.get(k)
        hz = into_hazard(obs, md) if md is not None else None
        inc = obs.get("incoming")
        if hz:
            text += f" → into the {hz}!"
        elif inc:
            off = inc["key"] in keys and ("left" in keys) != ("right" in keys)
            text += (f" → off the arrow's line (it would hit you in {inc['in_s']:.2f} s)" if off else
                     f" → their arrow hits you in {inc['in_s']:.2f} s")
        elif f and md is not None:
            toward = math.cos(math.radians(md - rel["angle"]))
            side = abs(math.sin(math.radians(md - rel["angle"]))) > 0.5
            speed = SPEED["sprint"] if "sprint" in keys else SPEED["walk"] * (0.2 if obs.get("using_s") is not None else 1)
            if d > REACH + 0.3:
                if toward > 0.7:
                    text += f" → toward them: in reach in about {max(0.1, (d - REACH) / (speed * toward)):.1f} s"
                    if obs.get("held_item") == "bow" and d > 14:
                        text += "; closer, your arrows fly shorter and hit more"
                    if drawing_at_you and d > 8:
                        text += ", straight at their drawn bow"
                elif toward < -0.5:
                    text += " → away from them: out of your reach"
                elif side:
                    text += " → across: their arrow misses more" if drawing_at_you else " → across: no closer"
            else:
                if toward > 0.7 and not side:
                    text += " → into them: you push past them"
                elif toward > 0.2:
                    text += (" → stays in reach, circling: your sword is ready" if ready else
                             f" → stays in their reach while your sword recharges ({left:.1f} s)" if left is not None else " → stays in reach")
                elif toward < -0.2:
                    text += (" → out of their reach, though your sword is ready" if ready else
                             f" → out of their reach while your sword recharges ({left:.1f} s)"
                             + ("" if theirs_ready else " - theirs is recharging too"))
                else:
                    text += " → around them: their swing misses more"
            if "sprint" in keys and d <= 5 and cu.is_weapon(obs.get("held_item")):
                text += (" (holding sprint: no knockback bonus, let go of it for a step)" if "sprint" in held and not obs.get("sprint_fresh")
                         else " (a fresh sprint: your next hit knocks them back)" if "sprint" not in held else "")
        elif f and k == "jump" and d <= REACH + 0.3:
            text += " → a click while falling is a critical hit (1.5x)" if ready else " → up and down"
        elif f and k == "none":
            text += " → standing still: easy to hit" if d <= 6 else ""
        if sorted(keys) == held:
            text += " (what you hold now)"
        opts[k] = text
    return {"type": "choice", "instructions": "Which keys do you hold now?", "criteria": opts}


def mouse_questions(obs, f, name, a):
    yaw = {"still": 0, **{f"{s[0]}{st}": sg * st for st in YAW for sg, s in ((-1, "left"), (1, "right"))}, "around": 180}
    pitch = {"still": 0, **{f"{s[0]}{st}": sg * st for st in PITCH for sg, s in ((1, "up"), (-1, "down"))}}
    qy, qp = {}, {}
    what = "the aim point" if a and a["bow"] else "them"
    by = min(yaw, key=lambda m: abs(wrap(a["yaw"] - yaw[m]))) if a else None
    bp = min(pitch, key=lambda m: abs(a["pitch"] - pitch[m])) if a else None
    for m, st in yaw.items():
        text = ("keep the mouse still" if m == "still" else "turn around" if m == "around" else
                f"move the mouse {abs(st)}° {'right' if st > 0 else 'left'}")
        if a:
            err = a["yaw"]
            after = wrap(err - st)
            then = f"{what} then {abs(after):.0f}° {'right' if after > 0 else 'left'} of the crosshair"
            text += (f" → onto {what}" if m == by and abs(after) <= a["tol_yaw"] else
                     f" → on {what}, off its middle" if abs(after) <= a["tol_yaw"] else
                     f" → the nearest to {what}: {then}" if m == by else f" → {then}")
        qy[m] = text
    for m, st in pitch.items():
        text = "no mouse up or down" if m == "still" else f"move the mouse {abs(st)}° {'up' if st > 0 else 'down'}"
        if a:
            after = a["pitch"] - st
            then = f"{what} then {abs(after):.0f}° {'above' if after > 0 else 'below'} the crosshair"
            text += (f" → level with {what}" if m == bp and abs(after) <= a["tol_pitch"] else
                     f" → still on {what}" if abs(after) <= a["tol_pitch"] else
                     f" → the nearest to {what}: {then}" if m == bp else f" → {then}")
        qp[m] = text
    return ({"type": "choice", "instructions": "How do you move the mouse left or right now?", "criteria": qy},
            {"type": "choice", "instructions": "How do you move the mouse up or down now?", "criteria": qp})


def button_question(obs, f, name, a, history):
    base = cr.options(obs, name, history, v=2)["criteria"]
    held = obs.get("held_item")
    left = recharge_left(obs)
    ce = obs.get("cursor_entity")
    on = bool(ce) and ce.get("name") == name and ce["distance"] <= REACH + 0.3
    opts = {"none": "no button"}
    if on:
        if left is not None and left <= 0.05:
            opts["click"] = "left-click → hits them at full strength" + (
                ": a critical hit (you are falling)" if obs.get("falling") else "") + (
                ", knocking them back (fresh sprint)" if obs.get("sprint_fresh") and "sprint" in obs.get("held", []) else "")
        elif left is not None:
            opts["click"] = f"left-click → a weak hit ({100 * (1 - left / RECHARGE_S):.0f}% recharged; full in {left:.1f} s)"
            opts["none"] += f" → wait for a full hit ({left:.1f} s)"
        else:
            opts["click"] = f"left-click → a weak hit: you hold {cu.item_words(held) if held else 'nothing'}"
    else:
        opts["click"] = "left-click → swings at the air (the crosshair is not on anyone within reach)" + (
            ", starting your sword's recharge over" if left is not None else "")
        if left is not None and left <= 0.05:
            opts["none"] += " → your sword stays ready for a full hit"
    if held == "bow" and obs.get("using_s") is not None:
        p = draw_power(obs["using_s"])
        power = "at full power" if p >= 1 else f"at {100 * p:.0f}% power"
        full_in = max(0, 1 - obs["using_s"])
        vel = obs.get("_vel") or (0, 0, 0)
        if a and a.get("short"):
            text = f"let go of the right button → a weak arrow ({100 * p:.0f}%) that falls short of them: not drawn enough"
            opts["none"] = f"no button → keep drawing: full power in {full_in:.1f} s"
        elif a and a["bow"]:
            on_aim = abs(a["yaw"]) <= max(a["tol_yaw"], 1) and abs(a["pitch"]) <= max(a["tol_pitch"], 1)
            if on_aim:
                text = f"let go of the right button → shoots {power}, on the aim point: it reaches them in {a['flight']:.2f} s"
                if f and f.get("holding") == "bow" and f.get("using"):
                    text += "; they stand drawing their bow: a sure target"
                elif speed_of(vel) > 3:
                    text += (f"; they run across at {speed_of(vel):.0f} blocks/s: in its {a['flight']:.1f} s they may stop "
                             f"or turn - a gamble (they stand still when they draw)")
                elif speed_of(vel) < 1:
                    text += "; they stand still: a sure target"
                opts["none"] = "no button → keep the bow drawn" + (" (full power reached)" if p >= 1 else f" (full in {full_in:.1f} s)")
            else:
                off = []
                if abs(a["yaw"]) > max(a["tol_yaw"], 1):
                    off.append(f"{abs(a['yaw']):.0f}° {'left' if a['yaw'] > 0 else 'right'}")
                if abs(a["pitch"]) > max(a["tol_pitch"], 1):
                    off.append(f"{abs(a['pitch']):.0f}° {'below' if a['pitch'] > 0 else 'above'}")
                text = f"let go of the right button → a miss: the crosshair is {' and '.join(off)} the arrow's aim point"
                opts["none"] = "no button → keep the bow drawn and aim first (the mouse marks say where)"
        else:
            text = f"let go of the right button → shoots {power}"
        if f and cp.reach_of(obs, f) <= 4.5:
            text += " - they are close: the sword is better"
        opts["use_release"] = text
    elif "use" in base:
        opts["use"] = base["use"].replace("press and hold the right button: ", "hold the right button → ").replace(
            "right-click: ", "right-click → ")
        if held == "bow" and f and cp.reach_of(obs, f) <= 4.5:
            opts["use"] += " - they are close: the sword is better"
        elif held == "bow" and f and obs.get("items", {}).get("arrow"):
            opts["use"] += ": full power in 1 s" + ("; they are drawing too - be first" if f.get("using") else "")
            opts["none"] += " → your bow stays lowered: no arrow on its way to them"
    d = cp.reach_of(obs, f) if f else None
    for k, text in base.items():
        if not k.startswith("slot_"):
            continue
        i = int(k[5:]) - 1
        it = (obs.get("hotbar") or [None] * 9)[i]
        t = text.replace(": hold your", " → hold your")
        if it and d is not None:
            n = it["name"]
            if n.endswith("_sword"):
                t += " - they are close: switch now" if d <= 4.5 and not cu.is_weapon(held) else ""
            elif n == "bow":
                t += (" - they are far: a bow fight" if d >= 8 and obs.get("items", {}).get("arrow") else
                      " - they are close: the sword is better" if d < 6 else "")
            elif n == "lava_bucket":
                t += " - lava for their feet or path (3-5 blocks off)" if 3 <= d <= 6 else ""
            elif n == "water_bucket":
                t += " - you burn: water at your feet puts it out" if obs.get("burning") else ""
        opts[k] = t
    for k in opts:
        for h in list(history)[-2:]:
            note = h["result"].get("note") or ""
            if h.get("button") == k and any(w in note for w in ("did not", "cannot", "no ", "not ", "nothing")):
                opts[k] += f" (failed just now: {note[:40]})"
                break
    return {"type": "choice", "instructions": "Which button now?", "criteria": opts}


def words(k, my, mp, b):
    mouse = []
    if my != "still":
        mouse.append("turned around" if my == "around" else f"mouse {my[1:]}° {'left' if my[0] == 'l' else 'right'}")
    if mp != "still":
        mouse.append(f"mouse {mp[1:]}° {'up' if mp[0] == 'u' else 'down'}")
    return f"{KEY_WORDS[k]}, {', '.join(mouse) or 'mouse still'}, {'no button' if b == 'none' else b.replace('_', ' ')}"


def mouse_move(my, mp):
    dyaw = 0 if my == "still" else 180 if my == "around" else (1 if my[0] == "r" else -1) * float(my[1:])
    dpitch = 0 if mp == "still" else (1 if mp[0] == "u" else -1) * float(mp[1:])
    return dyaw, dpitch
