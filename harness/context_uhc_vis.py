"""Block UHC with vision: the opponent is only what Jev sees in a small first-person picture (256x256, crosshair in
the middle). Nothing about the opponent comes from coordinates: no distance, direction, reach, health or what
they hold. What a player knows without looking is still text: your own health, burning, falling, the recharge,
your keys, and the hotbar (too small to read in the picture).

Each step is two Jev calls: `look` asks only about the picture (is a player there, where from the crosshair, how
big, lava or fire), then the decision reads that as a sentence among the rest of the state.

  p_uhc_vis_body_2step_v1: look, then decide; the body aims for Jev (aim, hit, bow, lava at them: p_uhc_v1's actions)
  p_uhc_vis_raw_2step_v1:  look, then decide; raw controls (p_uhc_raw_v1): Jev puts the crosshair on them itself
  p_uhc_vis_raw_1step_v1:  one call: the picture with the text state (no perception sentence), raw controls
"""
import math

import context_survive as cs
import context_uhc as cu
import context_uhc_raw as cr
import context_v2 as v2

LOOK_STATE = ("A first-person view from a Minecraft game. A small cross, open in its middle, marks the middle of the "
              "picture.")

X_WORDS = {"far_left": "far to the left of the cross", "left": "left of the cross", "slightly_left": "slightly left of the cross",
           "center": "right on the cross", "slightly_right": "slightly right of the cross", "right": "right of the cross",
           "far_right": "far to the right of the cross"}
X_DEG = {"far_left": -28, "left": -16, "slightly_left": -6, "center": 0, "slightly_right": 6, "right": 16, "far_right": 28}
Y_WORDS = {"above": "higher than the cross", "level": "at the height of the cross", "below": "lower than the cross"}
SIZE_WORDS = {"huge": "fills much of the picture", "large": "large", "medium": "medium-sized", "small": "small", "tiny": "tiny"}
SIZE_NEAR = {"huge": "right next to you, within reach", "large": "close, about within reach", "medium": "a few blocks away",
             "small": "far away", "tiny": "very far away"}
SIDE_WORDS = {"none": "no", "left": "on the left", "middle": "in the middle", "right": "on the right"}
# how far off the cross still counts as on them, by how big they look (half a body width), plus the bins' own error
AIM_TOL = {"huge": 10, "large": 5, "medium": 3, "small": 2, "tiny": 1}
BIN_ERR = 3

LOOK_QUESTIONS = {
    "player": {"type": "boolean", "instructions": "Is there a person-shaped figure (a Minecraft player) in the picture?"},
    "x": {"type": "choice", "instructions": "Where is the person-shaped figure, left to right, compared with the white cross?",
          "criteria": X_WORDS},
    "y": {"type": "choice", "instructions": "Is the person-shaped figure higher or lower than the white cross?",
          "criteria": Y_WORDS},
    "size": {"type": "choice", "instructions": "How big is the person-shaped figure in the picture?",
             "criteria": {"huge": "it fills much of the picture", "large": "large: about a third of the picture's height",
                          "medium": "medium: about a sixth of the picture's height", "small": "small",
                          "tiny": "tiny: a few pixels"}},
    "lava": {"type": "choice", "instructions": "Is there orange lava or fire in the picture, and where?",
             "criteria": {"none": "no lava or fire", "left": "on the left", "middle": "in the middle", "right": "on the right"}},
}


# raw aiming up close: the finest cue a picture gives, which side of the cross the body is on
SIDE_QUESTION = {"type": "choice", "instructions": "Look at the white cross in the middle of the picture. Where is most "
                 "of the person-shaped figure's body compared with the cross?",
                 "criteria": {"left": "mostly left of the cross", "on": "the cross is on the body",
                              "right": "mostly right of the cross", "none": "no person-shaped figure"}}
LOOK_QUESTIONS_RAW = {**LOOK_QUESTIONS, "side": SIDE_QUESTION}


def best(answer):
    p = answer["probabilities"]
    return max(p, key=p.get)


def read_look(answers):
    out = {"player": answers["player"]["p_true"] >= 0.5, "p_player": round(answers["player"]["p_true"], 2),
           "x": best(answers["x"]), "y": best(answers["y"]), "size": best(answers["size"]), "lava": best(answers["lava"])}
    if "side" in answers:
        out["side"] = best(answers["side"])
    return out


def view_lines(seen, name, last=None):
    """last: (steps ago, seconds ago, what was seen then, the turn since in degrees) of the latest sighting."""
    if not seen:
        return ["(no picture yet)"]
    if not seen["player"]:
        if last:
            n, secs, was, turned = last
            where = f"{X_WORDS[was['x']]}, {SIZE_WORDS[was['size']]}"
            since = f"; you have turned {abs(turned):.0f}° {'right' if turned > 0 else 'left'} since" if abs(turned) >= 1 else ""
            lines = [f"In your view: no player now. {secs:.1f} s ago ({n} step{'s' if n > 1 else ''} back) you saw "
                     f"{name} {where}{since}. A far player is tiny and easy to miss."]
        else:
            lines = ["In your view: no player. They may be behind you or to a side: turn to look for them."]
    else:
        lines = [f"In your view: a player (likely {name}) {X_WORDS[seen['x']]}, {Y_WORDS[seen['y']]}, "
                 f"{SIZE_WORDS[seen['size']]} - {SIZE_NEAR[seen['size']]}."]
    if seen["lava"] != "none":
        lines.append(f"Lava or fire in view, {SIDE_WORDS[seen['lava']]}.")
    return lines


def you_lines(obs):
    return cu.you_lines(obs)


# ---------------------------------------------------------------- options

def _moves(obs, seen, name, keys, last=None):
    """Move options marked with the player you see - or, just after losing sight, where you last saw them."""
    opts = {}
    x = X_DEG[seen["x"]] if seen and seen["player"] else None
    what, size = "the player you see", seen["size"] if seen and seen["player"] else None
    if x is None and last:
        x, what, size = X_DEG[last[2]["x"]] - last[3], "where you last saw them", last[2]["size"]
    for k in keys:
        desc, turn, travel = cu.MOVES[k]
        text = desc
        if travel == "stop":
            opts[k] = text
            continue
        if travel is None:
            travel = cs.held_direction(obs)
        if travel is not None:
            m = turn + travel
            parts = []
            if x is not None:
                rel = cs.relation(x, m)
                far = size in ("medium", "small", "tiny")
                if rel == "toward" and far and k in ("sprint", "sprint_jump", "forward"):
                    parts.append(f"closer to {what}: you must be within 3 blocks to hit (a sprint covers about 5 blocks a second)")
                else:
                    parts.append(f"{rel} {what}")
            wall = cu.ahead_words(obs, m)   # your own feet and the block before them: felt, not seen
            if wall:
                parts.append(wall)
            if parts:
                text += " - you would move " + ", ".join(parts)
        opts[k] = text
    return opts


def body_options(obs, name, seen, history=(), last=None, v=1, sizes=()):
    """p_uhc_vis_body_*: the p_uhc_v* actions, marked only with what the picture shows (v3: and the attack
    indicator and how fast they grow)."""
    it = obs.get("items", {})
    held = obs.get("held_item")
    opts = _moves(obs, seen, name, list(cu.MOVES), last)
    near = seen and seen["player"] and seen["size"] in ("huge", "large")
    rushing = v >= 3 and closing_words(list(sizes)) is not None
    if v >= 3:
        near = near or in_reach(obs) or (last is not None and last[2]["player"] and last[2]["size"] in ("huge", "large"))
    weapon = cu.is_weapon(held)
    strength = ("full strength" if weapon and obs.get("recharge", 1) >= 1 else "weak: still recharging" if weapon else
                f"weak: you hold {'a ' + cu.item_words(held) if held else 'nothing'}")
    if not (seen and seen["player"]):
        reach = "you do not see them"
    else:
        reach = "they look close enough to hit" if near else "they look too far to hit"
    opts["aim"] = f"aim at {name} (turn to face them)"
    opts["aim_attack"] = f"aim at {name} and hit - {reach}; {strength}"
    opts["wtap_attack"] = f"w-tap: let go of forward, sprint again, aim at {name} and hit - {reach}; {strength}"
    opts["crit_attack"] = (f"jump and hit {name} on the way down (a critical hit) - "
                           + ("you are in the air already" if not obs.get("on_ground", True) else f"{reach}; {strength}"))
    drawing = obs.get("using_s") is not None and held == "bow"
    if v < 2 and it.get("bow") and it.get("arrow"):
        opts["shoot_bow"] = f"draw your bow for 1 s and shoot {name} ({it['arrow']} arrows left)"
    elif drawing:
        drawn = obs["using_s"]
        opts["release_bow"] = (f"let go: shoot {name} - " + ("fully drawn" if drawn >= 1 else f"only {drawn:.1f} s drawn: weak")
                               + ("; they look close enough for your sword: a sword hit is better" if near else "")
                               + "; if blocks are between you in the picture, keep drawing until they step out")
        opts["wait"] = "keep drawing (your bow stays on them)"
    elif it.get("bow") and it.get("arrow"):
        opts["draw_bow"] = (f"start drawing your bow at {name} ({it['arrow']} arrows; you walk slowly while drawing)"
                            + (" - they look close: take your sword instead" if near else
                               " - they are coming at you: they will be on you before it is drawn" if rushing else ""))
    if it.get("lava_bucket"):
        opts["lava_at"] = f"pour lava at {name}'s feet (a bucket reaches about 4 blocks)"
    if any(n.endswith("_sword") for n in it) and not (held or "").endswith("_sword"):
        opts["hold_sword"] = "take your sword in hand (its recharge starts over: 0.6 s)"
        if v >= 2 and (near or rushing):
            opts["hold_sword"] += (" - they are coming at you: switch now" if rushing and not near else
                                   " - they look close: switch now") + (" (drops the drawn arrow)" if drawing else "")
    srcs = obs.get("sources", [])
    if it.get("bucket"):
        for kind in ("lava", "water"):
            s = next((x for x in srcs if x["kind"] == kind), None)
            if s:
                opts[f"{kind}_pickup"] = f"take back the {kind} {s['dist']:.1f} blocks away with your empty bucket"
    if it.get("water_bucket"):
        opts["water_here"] = "pour water where you stand" + (" - puts out the fire on you" if obs.get("burning") else "")
    if any(it.get(b) for b in cu.BLOCKS):
        opts["place_block"] = "place a block in front of you (again for a wall two high)"
        opts["pillar_up"] = "pillar up: jump and place a block under you"
    for k in opts:
        why = cu.failed_last(history, k)
        if why:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


HALF_VIEW = 35   # the picture covers 70° across


def unseen_after(yaw_now_deg, turn, looked):
    """Degrees of view a turn would show that were not looked at lately. looked: compass bearings (deg) of recent
    pictures without the player; turn: + right."""
    new_center = yaw_now_deg - turn          # mineflayer yaw grows to the left
    fresh = 0
    for d in range(-HALF_VIEW, HALF_VIEW, 5):
        b = new_center + d
        if all(abs(((b - l) + 180) % 360 - 180) > HALF_VIEW for l in looked):
            fresh += 5
    return fresh


TURN_RATES = (30, 90)   # a mouse held moving: slow and fast, degrees a second
SIZE_RANK = {"tiny": 0, "small": 1, "medium": 2, "large": 3, "huge": 4}


def in_reach(obs):
    """The 1.9 attack indicator: the crosshair on someone within reach (what a player sees on screen)."""
    ce = obs.get("cursor_entity")
    return bool(ce) and ce.get("distance", 9) <= cr.REACH + 0.3


def closing_words(sizes):
    """sizes: [(time, rank)] of recent sightings. How their size changed lately, if it grew."""
    if len(sizes) < 2:
        return None
    t1, r1 = sizes[-1]
    older = [(t, r) for t, r in sizes[:-1] if t1 - t <= 0.8]
    if not older:
        return None
    t0, r0 = min(older, key=lambda x: x[1])
    if r1 - r0 >= 1:
        names = {v: k for k, v in SIZE_RANK.items()}
        return f"they grew from {names[r0]} to {names[r1]} in {t1 - t0:.1f} s: they are coming at you"
    return None


def closeness_lines(obs, sizes):
    lines = []
    if in_reach(obs):
        lines.append("Your crosshair shows them within reach (the attack indicator): a sword hit lands now.")
    c = closing_words(sizes)
    if c:
        lines.append(c[0].upper() + c[1:] + ".")
    return lines


def rate_words(rate):
    return "not turning" if not rate else f"turning {'right' if rate > 0 else 'left'} at {abs(rate)}°/s"


def raw_options(obs, name, seen, history=(), last=None, looked=(), omega=None):
    """p_uhc_vis_raw_*: the raw controls, marked only with what the picture shows. looked: bearings of recent
    pictures without the player, for searching when there is nothing to go on."""
    opts = _moves(obs, seen, name, list(cr.MOVES), last)
    x = X_DEG[seen["x"]] if seen and seen["player"] else None
    what, size = "the player you see", seen["size"] if seen and seen["player"] else None
    if x is None and last:   # not in view: estimate from the last sighting and the turn since
        x, what, size = X_DEG[last[2]["x"]] - last[3], "where you last saw them", last[2]["size"]
    tol = AIM_TOL.get(size, 2) + BIN_ERR
    ce = obs.get("cursor_entity")   # the crosshair on someone within reach: seen in the picture as the cross on them
    on_them = bool(ce) and ce.get("distance", 9) <= cr.REACH + 0.3
    near = size in ("huge", "large")
    # up close the bins are coarser than a body is wide: only the crosshair itself says whether it is on them
    aimed = on_them or (x is not None and abs(x) <= tol and not near)
    side = seen.get("side") if seen and seen["player"] and near else None
    if side == "on":
        aimed = True
    for step in cr.YAW_STEPS:
        for sign, side_ in ((-1, "left"), (1, "right")):
            text = f"turn your view {step}° to the {side_}"
            if aimed:
                text += f" - you are already aimed at them; this moves the cross about {step}° off"
            elif side in ("left", "right") and step <= 30:
                text += (f" - toward their body, which is {side} of the cross" if side[0] == side_[0] else
                         f" - away from their body, which is {side} of the cross")
            elif x is not None:
                after = x - sign * step
                text += (f" - {what} would be on the cross" if abs(after) <= tol else
                         f" - {what} would then be about {abs(after):.0f}° {'right' if after > 0 else 'left'} of the cross")
            elif looked:   # nothing to go on: how much new ground the turn shows
                fresh = unseen_after(math.degrees(obs["yaw"]), sign * step, looked)
                text += (f" - shows {fresh}° you have not looked at in the last few seconds" if fresh else
                         " - shows only what you looked at a moment ago")
            opts[f"look_{side_[0]}{step}"] = text
    # the mouse held moving: to follow someone who runs sideways
    rate_now = obs.get("turn_rate", 0)
    best_rate = None
    if omega is not None and abs(omega) >= 15:
        best_rate = min([r * sgn for r in TURN_RATES for sgn in (1, -1)], key=lambda r: abs(r - omega))
    for r in TURN_RATES:
        for sgn, side in ((1, "right"), (-1, "left")):
            rate = r * sgn
            if rate == rate_now:
                continue
            text = f"keep turning your view {side} {'slowly' if r == 30 else 'fast'} ({r}°/s) until you choose otherwise"
            if best_rate == rate:
                text += f" - about how fast {name} moves across your view: keeps them near the cross"
            opts[f"turn_{side[0]}{r}s"] = text
    if rate_now:
        opts["turn_stop"] = "stop turning your view" + (" - they are not moving across your view" if omega is not None and abs(omega) < 15 else "")
    opts["turn_around"] = "turn around, 180°" + (
        f" - shows {unseen_after(math.degrees(obs['yaw']), 180, looked)}° you have not looked at in the last few seconds"
        if x is None and looked else "")
    pitch = math.degrees(obs["pitch"])          # + up
    y = seen["y"] if seen and seen["player"] else None
    for step in cr.PITCH_STEPS:
        for way, sign in (("up", 1), ("down", -1)):
            after = pitch + sign * step
            text = f"tilt your view {step}° {way}"
            if y in ("above", "below") and (y == "above") == (sign > 0) and step <= 15:
                text += " - toward the player you see"
            elif abs(after) <= 5:
                text += " - your view would be level, at the height of players"
            else:
                text += f" - you would look {abs(after):.0f}° {'up' if after > 0 else 'down'}" + (
                    ": mostly sky" if after > 30 else ": mostly ground" if after < -40 else "")
            opts[f"look_{way[0]}{step}"] = text
    held = obs.get("held_item")
    ce = obs.get("cursor_entity")   # what the crosshair touches: in the game you see the target outline
    if ce and ce.get("distance", 9) <= cr.REACH + 0.3:
        weapon = cu.is_weapon(held)
        strength = "full strength" if weapon and obs.get("recharge", 1) >= 1 else "weak"
        click = f"hits what is under your crosshair ({strength})"
    else:
        click = "swings at the air: nothing under your crosshair within reach"
    opts["click"] = f"left-click - {click}"
    base = cr.options(obs, None, history)["criteria"]
    for k in ("use", "use_release"):
        if k in base:
            opts[k] = base[k].split(", ")[0] if k == "use" else base[k]
    for i, itm in enumerate(obs.get("hotbar") or []):
        if itm and i != obs.get("slot"):
            opts[f"slot_{i + 1}"] = f"press {i + 1}: hold your {cu.item_words(itm['name'])}"
    for k in opts:
        why = cu.failed_last(history, k)
        if why:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


# ---------------------------------------------------------------- vision+body, aiming from the picture ("pure")

DIST_EST = {"huge": 1.5, "large": 3.0, "medium": 7.0, "small": 18.0, "tiny": 35.0}   # from logged sightings


def pure_target(seen, last):
    """Where to aim, from the picture only: (degrees right of the cross, estimated blocks away), or None.
    seen: this step's reading (2step); last: the latest earlier sighting with the turn since (1step, or lost)."""
    if seen and seen["player"]:
        return X_DEG[seen["x"]], DIST_EST[seen["size"]]
    if last and last[2]["player"] and last[0] <= 2:
        return X_DEG[last[2]["x"]] - last[3], DIST_EST[last[2]["size"]]
    return None


def pure_options(obs, name, seen, history=(), last=None, looked=(), v=3, sizes=()):
    """p_uhc_vis_body_pure_*: the body acts, but aims only where the picture showed them. Unseen, it cannot aim:
    it can look for them."""
    it = obs.get("items", {})
    held = obs.get("held_item")
    opts = _moves(obs, seen, name, list(cu.MOVES), last)
    tgt = pure_target(seen, last)
    near = in_reach(obs) or (tgt is not None and tgt[1] <= 3.0)
    rushing = closing_words(list(sizes)) is not None
    weapon = cu.is_weapon(held)
    strength = ("full strength" if weapon and obs.get("recharge", 1) >= 1 else "weak: still recharging" if weapon else
                f"weak: you hold {'a ' + cu.item_words(held) if held else 'nothing'}, not your sword")
    drawing = obs.get("using_s") is not None and held == "bow"
    if tgt:
        dx, dist = tgt
        where = "right where you look" if abs(dx) < 3 else f"{abs(dx):.0f}° {'right' if dx > 0 else 'left'}"
        reach = ("the attack indicator shows them within reach" if in_reach(obs) else
                 "they look within reach" if dist <= 3.0 else f"they look about {dist:.0f} blocks away: too far to hit")
        opts["aim"] = f"turn to where you see {name} ({where})"
        opts["aim_attack"] = f"turn to where you see {name} and hit - {reach}; {strength}"
        opts["wtap_attack"] = f"w-tap: let go of forward, sprint again, turn to {name} and hit - {reach}; {strength}"
        opts["crit_attack"] = (f"jump, turn to {name} and hit on the way down (a critical hit) - "
                               + ("you are in the air already" if not obs.get("on_ground", True) else f"{reach}; {strength}"))
        if drawing:
            drawn = obs["using_s"]
            opts["release_bow"] = (f"let go: shoot where you see {name} - " + ("fully drawn" if drawn >= 1 else f"only {drawn:.1f} s drawn: weak")
                                   + ("; they look close: a sword hit is better" if near else ""))
            opts["wait"] = "keep drawing"
        elif it.get("bow") and it.get("arrow"):
            opts["draw_bow"] = (f"start drawing your bow at where you see {name} ({it['arrow']} arrows; you walk slowly while drawing)"
                                + (" - they look close: take your sword instead" if near else
                                   " - they are coming at you: they will be on you before it is drawn" if rushing else ""))
        if it.get("lava_bucket"):
            opts["lava_at"] = (f"pour lava where you see {name} standing - " + (
                f"they look about {dist:.0f} blocks away: too far (about 4 at most)" if dist > 4.8 else
                "they look right next to you: the lava would be at your feet too" if dist < 2 else
                "sets them burning and slows them"))
    else:
        for step in (10, 30, 90):
            for sign, side in ((-1, "left"), (1, "right")):
                fresh = unseen_after(math.degrees(obs["yaw"]), sign * step, looked) if looked else None
                opts[f"look_{side[0]}{step}"] = f"turn your view {step}° to the {side}" + (
                    f" - shows {fresh}° you have not looked at in the last few seconds" if fresh else
                    " - shows only what you looked at a moment ago" if fresh == 0 else "")
        opts["turn_around"] = "turn around, 180° (you do not see them: they may be behind you)"
    if any(n.endswith("_sword") for n in it) and not (held or "").endswith("_sword"):
        opts["hold_sword"] = "take your sword in hand (its recharge starts over: 0.6 s)" + (
            " - they are coming at you: switch now" if rushing and not near else " - they look close: switch now" if near else "")
    srcs = obs.get("sources", [])
    if it.get("bucket"):
        for kind in ("lava", "water"):
            src = next((x for x in srcs if x["kind"] == kind), None)
            if src:
                opts[f"{kind}_pickup"] = f"take back the {kind} {src['dist']:.1f} blocks away with your empty bucket"
    if it.get("water_bucket"):
        opts["water_here"] = "pour water where you stand" + (" - puts out the fire on you" if obs.get("burning") else "")
    if any(it.get(b) for b in cu.BLOCKS):
        opts["place_block"] = "place a block in front of you (again for a wall two high)"
        opts["pillar_up"] = "pillar up: jump and place a block under you"
    for k in opts:
        why = cu.failed_last(history, k)
        if why:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


PURE_ACTIONS = {"aim": "vis_aim", "aim_attack": "vis_hit", "wtap_attack": "vis_wtap", "crit_attack": "vis_crit",
                "draw_bow": "vis_draw", "release_bow": "vis_release", "lava_at": "vis_lava"}


# ---------------------------------------------------------------- state

STRATEGY = {
    "body": ("Keep the player you are fighting in view; sprint toward them while they are far. Close in and hit with "
             "your sword whenever it is recharged (every 0.6 s); sprint into your hits. At range, shoot your bow. Pour "
             "lava at their feet when they are close, and water on yourself if you burn. Only when you have not seen "
             "them for a while, turn to look for them."),
    "pure": ("Keep the player you are fighting in view: you can only aim at them while you see them, and your aim "
             "goes where you saw them. Sprint toward them while they are far. Close in and hit with your sword whenever "
             "it is recharged (every 0.6 s); sprint into your hits. At range, draw your bow and let go when it is fully "
             "drawn; if they come close, take your sword at once. When you lose them or are hit and do not see them, "
             "turn to look for them."),
    "raw": ("When the player is far (small in the picture) and roughly ahead, sprint toward them: small aiming "
            "errors do not matter until you are close. Close up, turn your view until they are on the cross, then "
            "left-click to hit (within about 3 blocks, sword recharged every 0.6 s); a hit while falling from a jump "
            "is a critical hit. Only when you have not seen them for a while, turn to look for them. Hold the bow's "
            "slot and the right button 1 s, then let go, to shoot."),
}


def last_step_lines(prev, name):
    """1step: what the look questions said about the picture of the previous step (one step old)."""
    if not prev:
        return []
    n, secs, was, turned = prev
    if not was["player"]:
        return [f"At your last step ({secs:.1f} s ago) you saw no player."]
    since = f"; you have turned {abs(turned):.0f}° {'right' if turned > 0 else 'left'} since" if abs(turned) >= 1 else ""
    return [f"At your last step ({secs:.1f} s ago) you saw {name} {X_WORDS[was['x']]}, {Y_WORDS[was['y']]}, "
            f"{SIZE_WORDS[was['size']]} - {SIZE_NEAR[was['size']]}{since}."]


def motion_lines(obs, omega, name):
    lines = []
    if omega is not None and abs(omega) >= 15:
        lines.append(f"{name} moves across your view to your {'right' if omega > 0 else 'left'}, about {abs(omega):.0f}°/s.")
    if obs.get("turn_rate"):
        lines.append(f"Your mouse keeps {rate_words(obs['turn_rate'])}.")
    return lines


def view_pitch_lines(obs):
    p = math.degrees(obs["pitch"])
    if p > 20:
        return [f"You are looking {p:.0f}° up: the picture is mostly sky; players stand near the horizon, low in or "
                f"below the picture."]
    if p < -30:
        return [f"You are looking {-p:.0f}° down: the picture is mostly ground; a player a few blocks away may be "
                f"above the picture."]
    return []


def step_state(obs, mem, cfg, seen, variant, picture=False, last=None, prev=None, v=1, omega=None, sizes=()):
    """picture=True (1step): the picture comes with this state and nothing has been read from it; prev: what was
    read from the previous step's picture. last: the latest sighting when the player is not seen now."""
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else "them"
    hot = cr.hotbar_lines(obs) if variant == "raw" else cu.item_lines(obs)[:1]
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", cu.RULES),
        v2.section("Strategy", [STRATEGY[variant] + (" With the bow: start drawing, let go when fully drawn; if they "
                                                      "come close, take your sword at once. When hurt and you do not "
                                                      "see them, look where the hit came from."
                                                      if v >= 2 and variant == "body" else "")]),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("What you see", [f"The picture is your view right now; the small cross in its middle is your "
                                    f"crosshair. Find {name} in it yourself."] + last_step_lines(prev, name)
                   + (view_pitch_lines(obs) + motion_lines(obs, omega, name) if variant == "raw" else [])
                   + (closeness_lines(obs, sizes) if v >= 3 else [])
                   if picture else view_lines(seen, name, last)
                   + (view_pitch_lines(obs) + motion_lines(obs, omega, name) if variant == "raw" else [])
                   + (closeness_lines(obs, sizes) if v >= 3 else [])),
        v2.section("You", [line for line in cu.you_lines(obs, v) if variant != "raw" or not line.startswith("In your hand")]
                   + (cu.feel_lines(obs, sees_them=bool(seen and seen["player"]) if not picture else None)
                      if v >= 2 or variant == "raw" else [])),
        v2.section("Your hotbar" if variant == "raw" else "Your items", hot),
        v2.section("Fight so far", cu.combat_lines(obs)),
        v2.section("Your last actions (oldest first)", cu.recent_lines(mem.history)),
    ] if x)
