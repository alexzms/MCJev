"""Harness p_uhc_vis_raw_v2: Block UHC from the picture alone, with raw keys and mouse, Jev deciding every step.

Built to be fast (one request a step: the decision with the picture now, and a look at the same picture whose answers
are the text of the next step) and short (a compact state; the 8765 engine takes at most ~2500 tokens), with the
layers of the strongest text player (p_uhc_pro_v10): how to fight, what was seen, what happened lately, you, and the
last choices. What a picture cannot tell is left out: no distances in blocks (Jev tells only right next to you /
some way off / far), no health of theirs, no coordinates. What a player does see on screen is in: the attack
indicator (the crosshair on someone within reach), the hit flash (you hit them), your own health, recharge and
fall, and the push of a hit (where it came from).
"""
import math

import context_uhc as cu
import context_uhc_raw as cr
import context_uhc_vis as cv

LOOK_STATE = cv.LOOK_STATE
# the opponent's looks, as a player knows them (the default skin): a far corner pillar with a lantern on top passed
# for "a person-shaped figure"
WHO = "the Minecraft player (a blocky person: brown hair, cyan shirt, blue trousers)"


def _about(q, text):
    return {**q, "instructions": text}


LOOK = {
    "player": {"type": "boolean", "instructions": f"Is there a Minecraft player in the picture - a blocky person with "
                                                  f"brown hair, a cyan shirt and blue trousers? (Pillars, posts and lamps do not count.)"},
    "x": _about(cv.LOOK_QUESTIONS["x"], f"Where is {WHO}, left to right, compared with the white cross?"),
    "y": _about(cv.LOOK_QUESTIONS["y"], f"Is {WHO} higher or lower than the white cross?"),
    "size": _about(cv.LOOK_QUESTIONS["size"], f"How big is {WHO} in the picture?"),
    "side": _about(cv.SIDE_QUESTION, f"Look at the white cross in the middle of the picture. Where is most of {WHO}'s "
                                     f"body compared with the cross?"),
}   # (a "pulling back a bow?" question answered yes to a sword fighter again and again: left out)
# what the size answers are good for (offline, known distances): "large"/"huge" = within about 2 blocks,
# "small"/"medium" = anywhere from 2.5 to 10+, "tiny" = far
RANGE = {"huge": "right next to you", "large": "right next to you", "medium": "some way off", "small": "some way off",
         "tiny": "far off"}
NEAR = ("huge", "large")
YAW = (3, 10, 30, 90)   # 90: to turn back to someone who got beside you
PITCH = (5, 15)


def read(ans):
    return {"player": ans["player"]["p_true"] >= 0.5, "x": cv.best(ans["x"]), "y": cv.best(ans["y"]),
            "size": cv.best(ans["size"]), "side": cv.best(ans["side"])}


FIGHT = ("How to fight: keep your sword in hand and close in on them; only when they are far off (tiny in the picture) "
         "is the bow worth it (hold the right button about 1 s, then let go). A left-click lands only with the cross "
         "on them within 3 blocks - the attack indicator shows it - at full strength every 0.6 s. Jump and left-click "
         "while falling: a critical hit (1.5x). When they are right next to you, stop running (running on takes you "
         "past them): turn the cross onto them and click; while your sword recharges, strafe (left/right) around them "
         "so their swing misses. Keep them in view; if you lose them, turn toward where you last saw them or where a "
         "hit came from.")


def seen_line(name, look, ago, turned, lastseen=None, heading=None):
    """look: the last reading (None: nothing yet); turned: degrees you turned since (+ right)."""
    if not look:
        return f"You have not looked yet."
    if not look["player"]:
        if lastseen and lastseen[0] <= 3 and heading is not None:
            d = wrap(lastseen[1] - heading)
            return (f"Last look ({ago:.2f} s ago): {name} not in view. You saw them {lastseen[0]:.1f} s ago, "
                    f"{abs(d):.0f}° to your {'right' if d > 0 else 'left'} of where you face now.")
        return f"Last look ({ago:.2f} s ago): {name} not in view."
    since = f"; you turned {abs(turned):.0f}° {'right' if turned > 0 else 'left'} since" if abs(turned) >= 2 else ""
    return (f"Last look ({ago:.2f} s ago): {name} {cv.X_WORDS[look['x']]}, {cv.Y_WORDS[look['y']]}, "
            f"{RANGE[look['size']]}{since}.")


def events_lines(obs, name, sees):
    out = []
    for c in reversed(obs.get("combat", [])[-4:]):
        if c["ago"] > 3:
            continue
        if c["kind"] == "hit":
            out.append(f"{c['ago']:.1f} s ago you hit {name}.")
        elif c["kind"] == "swing":
            out.append(f"{c['ago']:.1f} s ago you swung and missed.")
    out += cu.feel_lines(obs, sees_them=sees)[:2]
    return out[:4]


def you_line(obs):
    held = obs.get("held_item")
    parts = [f"health {obs['health']:.0f}/20"]
    if held and held.endswith("_sword"):
        parts.append("sword in hand, " + ("recharged" if obs.get("recharge", 1) >= 1 else "recharging"))
    else:
        parts.append(f"{cu.item_words(held) if held else 'nothing'} in hand")
    if obs.get("using_s") is not None and held == "bow":
        parts.append(f"drawing the bow {obs['using_s']:.1f} s" + (" (full)" if obs["using_s"] >= 1 else ""))
    if obs.get("falling"):
        parts.append("falling: a left-click now is a critical hit")
    if obs.get("burning"):
        parts.append("burning")
    p = math.degrees(obs["pitch"])
    if abs(p) > 20:
        parts.append(f"looking {abs(p):.0f}° {'up' if p > 0 else 'down'}")
    keys = cu.cs.held_words(obs).split(" (")[0].replace("Keys held: ", "keys: ")
    parts.append(keys)
    return "You: " + "; ".join(parts) + "."


def state(obs, name, look, ago, turned, history, lastseen=None):
    lines = [f"You are {obs['name']}, fighting {name} in Minecraft 1.11 Block UHC: to the death, no healing. You play "
             f"with keys and mouse; the picture is your view, the small cross in its middle is your crosshair.",
             FIGHT, "", seen_line(name, look, ago, turned, lastseen, -math.degrees(obs["yaw"]))]
    if cv.in_reach(obs):
        lines.append("The attack indicator is on: your crosshair is on someone within reach.")
    lines.append(you_line(obs))
    ev = events_lines(obs, name, bool(look and look["player"]))
    if ev:
        lines.append("Lately: " + " ".join(ev))
    last = [f"{h['action']}" + (f" ({h['result'].get('note')[:40]})" if h["result"].get("note") else "")
            for h in list(history)[-3:]]
    if last:
        lines.append("Your last choices: " + "; ".join(last) + ".")
    return "\n".join(lines)


def wrap(d):
    return (d + 180) % 360 - 180


def options(obs, name, look, turned, looked=(), history=(), lastseen=None):
    """Raw controls, each marked with what it does given the last look (and the turn since). lastseen: (seconds ago,
    bearing in degrees right of north) of the latest sighting, to turn back to when they are out of view."""
    it = obs.get("items", {})
    held = obs.get("held_item")
    seen = look and look["player"]
    x = cv.X_DEG[look["x"]] - turned if seen else None
    near = seen and look["size"] in NEAR
    on = cv.in_reach(obs)
    side = look["side"] if seen and near and look["side"] in ("left", "right", "on") else None
    tol = 10 if near else 5
    aimed = on or (x is not None and abs(x) <= tol and not near) or side == "on"
    opts = {}
    for k, desc in (("sprint", "sprint forward"), ("sprint_jump", "sprint forward jumping"), ("forward", "walk forward"),
                    ("back", "walk backward"), ("left", "strafe left"), ("right", "strafe right"), ("stop", "stop"),
                    ("jump", "jump"), ("wait", "keep doing what you do")):
        text = desc
        if x is not None and k in ("sprint", "sprint_jump", "forward") and abs(x) <= 25:
            text += " → into them: you run past them, stop to aim" if near else " → toward them (a hit needs 3 blocks or less)"
        elif near and k == "stop":
            text += " → stand and aim at them"
        elif x is not None and k in ("left", "right") and near:
            text += " → around them (their swing misses more)"
        opts[k] = text
    heading = -math.degrees(obs["yaw"])          # degrees right of north
    back = None                                   # where they were last seen, from where you face now (+ right)
    if not seen and lastseen and lastseen[0] <= 3:
        back = wrap(lastseen[1] - heading)
    def toward_back(turn):
        return back is not None and abs(wrap(back - turn)) <= max(12, abs(turn) * 0.4)
    for step in YAW:
        for sign, s in ((-1, "left"), (1, "right")):
            text = f"turn {step}° {s}"
            if toward_back(sign * step):
                text += f" → toward where you saw them {lastseen[0]:.1f} s ago"
            elif aimed:
                text += " → off them (you are on them now)"
            elif side in ("left", "right") and step <= 10:
                text += " → toward their body" if side == s else " → away from their body"
            elif x is not None:
                after = x - sign * step
                text += " → onto them" if abs(after) <= tol else f" → still {abs(after):.0f}° {'right' if after > 0 else 'left'} of them"
            elif looked:
                fresh = cv.unseen_after(math.degrees(obs["yaw"]), sign * step, looked)
                text += f" → shows {fresh}° not looked at lately" if fresh else " → shows what you just saw"
            opts[f"look_{s[0]}{step}"] = text
    if seen:
        opts["turn_around"] = "turn around → away from them"
    elif toward_back(180):
        opts["turn_around"] = f"turn around → toward where you saw them {lastseen[0]:.1f} s ago"
    else:
        fresh = cv.unseen_after(math.degrees(obs["yaw"]), 180, looked) if looked else None
        opts["turn_around"] = "turn around" + (f" → shows {fresh}° not looked at lately" if fresh else
                                               " → shows what you just saw" if fresh == 0 else "")
    y = look["y"] if seen else None
    for step in PITCH:
        for sign, w in ((1, "up"), (-1, "down")):
            after = math.degrees(obs["pitch"]) + sign * step
            text = f"tilt {step}° {w}"
            if y in ("above", "below") and (y == "above") == (sign > 0):
                text += " → toward them"
            elif abs(after) <= 5:
                text += " → level"
            else:
                text += f" → looking {abs(after):.0f}° {'up' if after > 0 else 'down'}"
            opts[f"look_{w[0]}{step}"] = text
    weapon = cu.is_weapon(held)
    if on:
        opts["click"] = "left-click → hits them" + (" (full strength)" if weapon and obs.get("recharge", 1) >= 1 else
                                                    " (weak: recharging)" if weapon else " (weak: not your sword)")
        if obs.get("falling"):
            opts["click"] += ", a critical hit"
    else:
        opts["click"] = "left-click → swings at the air (the cross is not on anyone within reach)"
    if obs.get("using_s") is not None and held == "bow":
        drawn = obs["using_s"]
        opts["use_release"] = ("let go of the right button → shoots" + (" (full power)" if drawn >= 1 else " (weak: not drawn)")
                               + (" - they are right next to you: the sword is better" if near else ""))
    elif held == "bow" and it.get("arrow"):
        opts["use"] = "hold the right button → draw the bow (you walk slowly while drawing)" + (
            " - they are right next to you: the sword is better" if near else "")
    elif held in ("lava_bucket", "water_bucket") or held in cu.BLOCKS:
        base = cr.options(obs, None, history)["criteria"].get("use")
        if base:
            opts["use"] = base.replace("right-click: ", "right-click → ")
    for i, itm in enumerate(obs.get("hotbar") or []):
        if itm and i != obs.get("slot") and itm["name"] in ("diamond_sword", "bow", "lava_bucket", "water_bucket", "cobblestone"):
            text = f"press {i + 1} → hold your {cu.item_words(itm['name'])}"
            far = seen and look["size"] == "tiny"
            if itm["name"].endswith("_sword"):
                text += " - they are right next to you" if near else "" if far else " - they are within a sprint: have it ready"
            elif itm["name"] == "bow":
                text += " - they are far off" if far else " - they are not far off: the sword is better" if seen else ""
            opts[f"slot_{i + 1}"] = text
    for k in opts:
        why = cu.failed_last(history, k)
        if why:
            opts[k] += f" (failed just now: {why[:40]})"
    return {"type": "choice", "instructions": "Which control now?", "criteria": opts}
