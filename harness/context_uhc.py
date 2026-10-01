"""Harness U1 (Block UHC): what a bot needs to kill another player with a sword, a bow, a lava and a water bucket
and a stack of blocks, with no natural regeneration.

Built like P1 (held keys, options marked with what they would do, recalled facts, a fighter's view of the
opponent), without the ring and with what this mode adds: health that does not come back, the 1.9 attack recharge
of the item in hand, critical hits, what each player holds and whether they burn, lava and fire around you, liquids
you can take back, and whether blocks stand between you. Only options you can carry out are offered (no bow
without arrows, no scooping without an empty bucket and a source in reach); the fishing rod is left out, since in
1.9+ it does nothing to players.
"""
import math

import context as v1
import context_pvp as cp
import context_survive as cs
import context_v2 as v2

REACH = 3.0
BUCKET_REACH = 4.8
BLOCKS = ("cobblestone", "planks", "dirt", "stone", "stonebrick", "sandstone")

RULES = ["Block UHC: a fight to the death. There is no natural regeneration: health you lose stays lost.",
         "Never pour lava next to yourself, and never walk into lava or fire."]

STRATEGY = ("Close in and hit with your sword whenever it is recharged (a full-strength hit every 0.6 s; hitting "
            "sooner does little). Sprint into your hits for knockback; a jump hit on the way down is a critical hit. "
            "At range, shoot your bow, or close in fast while strafing so their arrows miss. Pour lava at their feet "
            "when they are close (not next to you) and take it back after; if you are burning, pour water at your "
            "feet and take it back. Put blocks between you and an archer; pillar up to get out of a sword's reach.")

STRATEGY_V2 = STRATEGY + (" With the bow: start drawing it (you walk slowly while drawing), and let go once it is fully "
                          "drawn; if blocks stand between you, keep it drawn and let go when they step out. If they come "
                          "close while you hold the bow, take your sword at once. When you are hurt and do not see them, "
                          "look where the hit came from.")

ITEM_WORDS = {"diamond_sword": "diamond sword", "bow": "bow", "fishing_rod": "fishing rod", "cobblestone": "cobblestone",
              "water_bucket": "water bucket", "lava_bucket": "lava bucket", "bucket": "empty bucket",
              "golden_apple": "golden apple", "arrow": "arrows"}

LABELS = {"sprint": "sprint forward", "sprint_jump": "sprint forward jumping", "forward": "walk forward",
          "back": "walk backward", "left": "strafe left", "right": "strafe right", "turn_left": "turn left",
          "turn_right": "turn right", "turn_around": "turn around", "stop": "stop", "wait": "keep going",
          "jump": "jump", "aim": "aim", "aim_attack": "aim and hit", "wtap_attack": "w-tap and hit",
          "crit_attack": "jump and hit", "hold_sword": "take the sword", "shoot_bow": "shoot the bow",
          "lava_at": "pour lava at them", "lava_pickup": "take the lava back", "water_here": "pour water at your feet",
          "water_pickup": "take the water back", "place_block": "place a block ahead", "pillar_up": "pillar up",
          "eat_gapple": "eat a golden apple", "break_block": "break the block ahead"}


def item_words(name):
    return ITEM_WORDS.get(name, (name or "nothing").replace("_", " "))


def hearts(h):
    return f"{h:.0f}/20 ({h / 2:.1f} hearts)"


def is_weapon(name):
    return bool(name) and (name.endswith("_sword") or name.endswith("_axe"))


# ---------------------------------------------------------------- hazards

def hazard_cells(obs, radius=4):
    """Lava and fire near you, in the ground (you would step into it) or at feet/head height."""
    g, R = obs["grid"], obs["grid"]["r"]
    bx, bz = math.floor(obs["pos"]["x"]), math.floor(obs["pos"]["z"])
    out = []
    for li, level in ((0, "in the ground"), (1, "at your feet"), (2, "at head height")):
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                c = g["layers"][li][dz + R][dx + R]
                if c in "lf":
                    pos = {"x": bx + dx + 0.5, "y": obs["pos"]["y"], "z": bz + dz + 0.5}
                    out.append({"kind": "lava" if c == "l" else "fire", "level": level, "pos": pos,
                                "dist": math.hypot(pos["x"] - obs["pos"]["x"], pos["z"] - obs["pos"]["z"])})
    return sorted(out, key=lambda h: h["dist"])


def hazard_lines(obs):
    hs = hazard_cells(obs)
    if not hs:
        return ["No lava or fire within 4 blocks."]
    lines, seen = [], set()
    for h in hs:
        r = v1.relative(obs, h["pos"])
        key = (h["kind"], v1.direction_words(r["angle"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{h['kind'].capitalize()} {h['dist']:.1f} blocks away, {v1.direction_words(r['angle'])}, {h['level']}.")
        if len(lines) == 3:
            break
    return lines


def ahead_words(obs, move_deg):
    """What stops you in a direction within a block: a step (auto-jump climbs it), a wall, or nothing."""
    g, R = obs["grid"], obs["grid"]["r"]
    ux, uz = cp.world_dir(obs, move_deg)
    bx, bz = math.floor(obs["pos"]["x"]), math.floor(obs["pos"]["z"])
    for d in (0.5, 0.9):
        cx, cz = math.floor(obs["pos"]["x"] + ux * d) - bx, math.floor(obs["pos"]["z"] + uz * d) - bz
        if (cx, cz) == (0, 0) or abs(cx) > R or abs(cz) > R:
            continue
        feet, head, above = (g["layers"][k][cz + R][cx + R] for k in (1, 2, 3))
        if feet == "s" and head != "s" and above != "s":
            return "a one-block step (you hop up it)"
        if feet == "s" or head == "s":
            return "a wall right there"
    return None


def hazard_toward(obs, move_deg, ahead=2.5):
    """The nearest lava or fire along a direction of travel (° right of facing), within `ahead` blocks."""
    ux, uz = cp.world_dir(obs, move_deg)
    for h in hazard_cells(obs, radius=3):
        if h["level"] == "at head height":
            continue
        dx, dz = h["pos"]["x"] - obs["pos"]["x"], h["pos"]["z"] - obs["pos"]["z"]
        along, across = dx * ux + dz * uz, abs(-dx * uz + dz * ux)
        if 0 < along <= ahead and across < 0.9:
            return h
    return None


# ---------------------------------------------------------------- the opponent

def opponent_lines(obs, name, prev):
    f = cp.fighter(obs, name)
    if not f:
        return [f"You cannot see {name}."]
    r = v1.relative(obs, f["pos"])
    line = f"{name} is {r['dist']:.1f} blocks away, {v1.direction_words(r['angle'])}, {v1.height_words(r['dy'])}"
    p = next((q for q in (prev or {}).get("fighters", []) if q["name"] == name and q.get("pos")), None)
    if p:
        d0 = v1.relative(prev, p["pos"])["dist"]
        if abs(d0 - r["dist"]) > 0.3:
            line += f" ({'closer' if r['dist'] < d0 else 'farther'} than at your last step, {d0:.1f})"
    lines = [line + "."]
    if f.get("health") is not None:
        h = f"{name}'s health: {hearts(f['health'])}"
        if p and p.get("health") is not None and p["health"] - f["health"] >= 0.5:
            h += f", down from {p['health']:.0f} at your last step"
        if f.get("absorption"):
            h += f", plus {f['absorption'] / 2:.0f} golden hearts"
        lines.append(h + ".")
    held = f.get("holding")
    what = f"{name} holds {'a ' + item_words(held) if held else 'nothing'}"
    if f.get("using") and held == "bow":
        what += ", drawing it: an arrow is coming"
    elif f.get("using"):
        what += ", using it"
    lines.append(what + ("; they are burning." if f.get("burning") else "."))
    reach = cp.reach_of(obs, f)
    aimed = abs(r["angle"]) <= max(6.0, math.degrees(math.atan2(0.3, max(r["dist"], 0.3))))
    lines.append(("Within sword reach." if reach <= REACH + 0.4 else f"Out of sword reach ({REACH:.0f} blocks at most).")
                 + (" Your crosshair is on them." if aimed else " Your crosshair is not on them."))
    if cs.blocked(obs, obs["pos"], f["pos"]):
        lines.append(f"Blocks stand between you and {name}: arrows cannot get through either way.")
    return lines


# ---------------------------------------------------------------- you

def you_lines(obs, v=1):
    lines = [v1.you_lines(obs)[0].split(" Health")[0]]
    h = f"Your health: {hearts(obs['health'])}"
    if obs.get("absorption"):
        h += f", plus {obs['absorption'] / 2:.0f} golden hearts"
    if any(e["id"] == 10 for e in obs.get("effects", [])):
        h += " (regenerating)"
    lines.append(h + ".")
    if obs.get("burning"):
        lines.append("You are burning! Fire keeps hurting until it goes out; water puts it out.")
    held = obs.get("held_item")
    if is_weapon(held):
        lines.append(f"In your hand: your {item_words(held)}, " + (
            "recharged: a full-strength hit." if obs.get("recharge", 1) >= 1 else
            f"recharging ({(1 - obs['recharge']) * obs.get('recharge_s', 0.6):.1f} s to go): a hit now is weak."))
    else:
        lines.append(f"In your hand: {'your ' + item_words(held) if held else 'nothing'} (a hit with it is as weak as a fist).")
    if obs.get("falling"):
        lines.append("You are falling: a hit now is a critical hit.")
    if v >= 2 and obs.get("using_s") is not None and obs.get("held_item") == "bow":
        lines.append(f"You are drawing your bow ({obs['using_s']:.1f} s; full power at 1 s) and walk slowly meanwhile.")
    lines.append(cs.held_words(obs))
    if obs.get("edge_stop") is not None and obs["edge_stop"] < 3:
        lines.append(f"{obs['edge_stop']:.1f}s ago your keys were let go at a drop, so you would not fall.")
    return lines


def item_lines(obs):
    it = obs.get("items", {})
    parts = []
    for k in ("arrow", "cobblestone", "water_bucket", "lava_bucket", "bucket", "golden_apple"):
        if it.get(k):
            parts.append(f"{it[k]} {item_words(k)}{'s' if it[k] > 1 and k not in ('arrow', 'cobblestone') else ''}")
    lines = ["You carry: " + (", ".join(parts) if parts else "no arrows, blocks or buckets") + "."]
    for s in obs.get("sources", [])[:2]:
        lines.append(f"{'Your own lava' if s.get('own') else s['kind'].capitalize()} {s['dist']:.1f} blocks away is within bucket reach.")
    return lines


def feel_lines(obs, sees_them=None):
    """What you feel without seeing: health lost lately and where the push of the last hit came from (the server
    pushes you away from whoever hit you). sees_them: False when a vision bot does not see its opponent now."""
    drops = [d for d in obs.get("drops", []) if d["ago"] <= 4]
    if not drops:
        return []
    lost = sum(d["from"] - d["to"] for d in drops)
    lines = [f"You lost {lost / 2:.1f} heart{'s' if lost > 2.5 else ''} in the last {max(d['ago'] for d in drops):.1f} s "
             f"(health {drops[0]['from']:.0f} -> {drops[-1]['to']:.0f}): something hurt you."]
    k = obs.get("knock")
    last = drops[-1]
    if k and abs(k["ago"] - last["ago"]) <= 0.6:
        (fx, fz), (rx, rz) = v1.basis(obs["yaw"])
        push = math.degrees(math.atan2(k["vx"] * rx + k["vz"] * rz, k["vx"] * fx + k["vz"] * fz))   # + right
        came = (push + 360) % 360 - 180
        lines.append(f"The hit pushed you {v1.direction_words(push).split(' (')[0]}, so it came from "
                     f"{v1.direction_words(came).split(' (')[0]}" + (
                         ": they may be there, close with a sword or far off with a bow - you do not see them now."
                         if sees_them is False else "."))
    elif obs.get("burning"):
        lines.append("You are burning: fire hurts you every second until it goes out.")
    elif sees_them is False:
        lines.append("A hit without a push: maybe fire, lava or a fall.")
    return lines


def combat_lines(obs):
    out = []
    for c in reversed(obs.get("combat", [])[-6:]):
        if c["ago"] > 6:
            continue
        out.append({"hit": f"{c['ago']:.1f}s ago: you hit {c['who']}.",
                    "hurt": f"{c['ago']:.1f}s ago: {c['who'] or 'something'} hurt you.",
                    "swing": f"{c['ago']:.1f}s ago: you swung and missed."}.get(c["kind"], ""))
    return [x for x in out if x][:5]


def recent_lines(history, n=6):
    history = list(history)  # a deque
    out = []
    for k, h in enumerate(history[-n:], start=max(1, len(history) - n + 1)):
        eff = f"moved {h['result']['moved']:.1f} blocks" if "after" in h else "just now"
        if h["result"].get("note"):
            eff += f"; {h['result']['note']}"
        out.append(f"{k}. {LABELS.get(h['action'], h['action'])} -> {eff}")
    return out


# ---------------------------------------------------------------- options

MOVES = {  # id: (description, turn °, travel ° relative to the new facing, None: keeps what is held, "stop")
    "sprint": ("sprint forward", 0, 0),
    "sprint_jump": ("sprint forward jumping (harder to hit)", 0, 0),
    "forward": ("walk forward", 0, 0),
    "back": ("walk backward", 0, 180),
    "left": ("strafe left", 0, -90),
    "right": ("strafe right", 0, 90),
    "turn_left": ("turn 30° to the left", -30, None),
    "turn_right": ("turn 30° to the right", 30, None),
    "turn_around": ("turn around, 180°", 180, None),
    "stop": ("stop moving", 0, "stop"),
    "wait": ("keep doing what you are doing", 0, None),
    "jump": ("jump (other keys stay as they are)", 0, None),
}


def failed_last(history, action):
    """The note of this action if it was tried at one of the last two steps and did not work, else None."""
    for h in list(history)[-2:]:
        note = h["result"].get("note") or ""
        if h["action"] == action and any(w in note for w in ("did not", "cannot", "no ", "too far", "not ", "nothing", "pressed", "blocked")):
            return note
    return None


def options(obs, name, history=(), v=1):
    f = cp.fighter(obs, name) if name else None
    rel = v1.relative(obs, f["pos"]) if f else None
    it = obs.get("items", {})
    held = obs.get("held_item")
    opts = {}
    for k, (desc, turn, travel) in MOVES.items():
        text = desc
        if travel == "stop":
            opts[k] = text
            continue
        if travel is None:
            travel = cs.held_direction(obs)
        if travel is not None:
            m = turn + travel
            parts = [f"{cs.relation(rel['angle'], m)} {name}"] if rel else []
            h = hazard_toward(obs, m)
            if h:
                parts.append(f"into {h['kind']} {h['dist']:.1f} blocks that way")
            wall = ahead_words(obs, m)
            if wall:
                parts.append(wall)
            if parts:
                text += " - you would move " + ", ".join(parts)
        opts[k] = text
    if f:
        reach = cp.reach_of(obs, f)
        lands = reach <= REACH + 0.4
        weapon = is_weapon(held)
        recharged = obs.get("recharge", 1) >= 1
        strength = ("full strength" if weapon and recharged else "weak: still recharging" if weapon else
                    f"weak: you hold {'a ' + item_words(held) if held else 'nothing'}, not your sword")
        miss = f"misses: they are {reach:.1f} blocks away"
        opts["aim"] = f"aim at {name} (turn to face them)"
        opts["aim_attack"] = f"aim at {name} and hit - " + (f"lands, {strength}" if lands else miss)
        opts["wtap_attack"] = (f"w-tap: let go of forward, sprint again, aim at {name} and hit - "
                               + (f"lands with sprint knockback, {strength}" if lands else miss))
        opts["crit_attack"] = (f"jump and hit {name} on the way down: a critical hit, 1.5x damage, no sprint knockback - "
                               + ("you are in the air already: it needs the ground to jump from" if not obs.get("on_ground", True)
                                  else f"lands, {strength}" if lands else miss))
        if v < 2 and it.get("bow") and it.get("arrow"):
            opts["shoot_bow"] = (f"draw your bow for 1 s and shoot {name} ({rel['dist']:.0f} blocks away; {it['arrow']} arrows left)"
                                 + (" - blocks are in the way" if cs.blocked(obs, obs["pos"], f["pos"]) else ""))
        if v >= 2:
            drawing = obs.get("using_s") is not None and held == "bow"
            close = reach <= REACH + 1.5
            blocked = cs.blocked(obs, obs["pos"], f["pos"])
            if drawing:
                drawn = obs["using_s"]
                text = f"let go: shoot {name} - " + ("fully drawn" if drawn >= 1 else f"only {drawn:.1f} s drawn: a weak arrow")
                if blocked:
                    text += "; blocks are between you: the arrow would hit them - keep drawing until they step out"
                if close:
                    text += f"; they are {rel['dist']:.1f} blocks away: a sword hit is better"
                opts["release_bow"] = text
                opts["wait"] = "keep drawing (your bow stays on them)" + (" - they are close: take your sword instead" if close else "")
            elif it.get("bow") and it.get("arrow"):
                text = f"start drawing your bow at {name} ({rel['dist']:.0f} blocks away; {it['arrow']} arrows; you walk slowly while drawing)"
                if close:
                    text += f" - they are only {rel['dist']:.1f} blocks away: take your sword instead"
                elif blocked:
                    text += " - blocks are between you: draw now and let go when they step out"
                opts["draw_bow"] = text
        if it.get("lava_bucket"):
            d = math.hypot(f["pos"]["x"] - obs["pos"]["x"], f["pos"]["z"] - obs["pos"]["z"])
            opts["lava_at"] = f"pour lava at {name}'s feet - " + (
                f"too far: they are {d:.1f} blocks away (about 4 at most)" if d > BUCKET_REACH - 0.6 else
                f"they are only {d:.1f} blocks away: the lava would be next to you too" if d < 2.0 else
                "sets them burning and slows them")
    if any(n.endswith("_sword") for n in it) and not (held or "").endswith("_sword"):
        opts["hold_sword"] = "take your sword in hand (its recharge starts over: 0.6 s)"
        if v >= 2 and f and cp.reach_of(obs, f) <= REACH + 1.5:
            opts["hold_sword"] += f" - {name} is close, {v1.relative(obs, f['pos'])['dist']:.1f} blocks away: switch now" + (
                " (drops the drawn arrow)" if obs.get("using_s") is not None and held == "bow" else "")
    srcs = obs.get("sources", [])
    if it.get("bucket"):
        lava = next((s for s in srcs if s["kind"] == "lava"), None)
        water = next((s for s in srcs if s["kind"] == "water"), None)
        if lava:
            opts["lava_pickup"] = f"take back the lava {lava['dist']:.1f} blocks away with your empty bucket"
        if water:
            opts["water_pickup"] = f"take back the water {water['dist']:.1f} blocks away with your empty bucket"
    if it.get("water_bucket"):
        opts["water_here"] = "pour water where you stand" + (" - puts out the fire on you" if obs.get("burning") else
                                                              " - it slows you and them")
    if any(it.get(b) for b in BLOCKS):
        facing_them = rel is not None and abs(rel["angle"]) < 45
        opts["place_block"] = ("place a block in front of you" + (f" - between you and {name}" if facing_them else "")
                               + " (again for a wall two high)")
        opts["pillar_up"] = "pillar up: jump and place a block under you (one block higher each time)"
    if it.get("golden_apple"):
        opts["eat_gapple"] = "eat a golden apple (1.6 s; you walk slowly meanwhile): golden hearts and regeneration"
    if any(n.endswith("_pickaxe") or n.endswith("_axe") for n in it):
        opts["break_block"] = "break the block in front of you"
    for k in opts:
        why = failed_last(history, k)
        if why:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


# ---------------------------------------------------------------- state

def step_state(obs, mem, cfg, v=1):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", RULES),
        v2.section("Strategy", [STRATEGY_V2 if v >= 2 else STRATEGY]),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("Opponent", opponent_lines(obs, name, mem.prev_obs) if name else ["No opponent."]),
        v2.section("You", you_lines(obs, v) + (feel_lines(obs) if v >= 2 else [])),
        v2.section("Your items", item_lines(obs)),
        v2.section("Around you", hazard_lines(obs)),
        v2.section("Fight so far", combat_lines(obs)),
        v2.section("Your last actions (oldest first)", recent_lines(mem.history)),
    ] if x)
