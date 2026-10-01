"""Harness p_uhc_pro (text + body): the strongest Block UHC player we can make, with a split of work.

The body has the hands: in "engage" it keeps the sword on the opponent and swings the moment a swing counts, as
sprint hits (a w-tap before each, for knockback), crits (jump-timed, 1.5x) or plain hits; it draws and aims the
bow, pours lava at the opponent's feet, puts out their lava, builds cover. Jev has the head: when to fight and in which style, where
to move while fighting (strafing around them makes their swings miss), when to open with the bow, lay lava, put
out fire, wall off or break away. The state gives it what a strong player keeps in mind: the damage race (hits
each side still needs), how fast they close in, what each item would do now, what a hit felt like; the options
say what each would do. Options are shuffled every step (Jev leans to the first ones listed).
"""
import math, re, time

import context as v1
import context_pvp as cp
import context_survive as cs
import context_uhc as cu
import context_v2 as v2

REACH = 3.0
SWORD, CRIT, ARROW = 7.0, 10.5, 8.0          # diamond sword, no armour: a hit, a crit, a full-power arrow (6-11)
LAVA_DPS, BURN_DPS, BURN_S = 8.0, 1.0, 15    # standing in lava (4 per half second), then burning 1 a second for 15 s

RULES = cu.RULES

STRATEGY = (
    "Win the damage race. With no armour a sword hit does 7 (3 kill from full health) and a crit 10.5 (2 kill), and "
    "their sword does the same to you. Fight with 'engage': the body swings the moment a swing counts, you choose the "
    "style and where to move - strafe around them while engaged so their swings miss, and back off a step when your "
    "sword is recharging and theirs is not. Crits kill fastest; sprint hits knock them back so they cannot answer. "
    "While they close in from afar, draw the bow and let go when fully drawn; switch to the sword before they are "
    "within 4 blocks. Lava at their feet as they come at you burns them badly (they burn on for 15 s unless they have "
    "water); do not stand in it yourself, and pour water at your feet if you burn. When they pour lava near you, pour "
    "your water on it at once: it goes out or turns to obsidian. Against their arrows, build cover (a wall two high) "
    "rather than standing in the open. When you shoot, the body leads them for the arrow's flight; someone running "
    "sideways can still dodge, so shoot when they come straight at you or stand still. If you are losing the race, "
    "break away behind cover.")

STRATEGY_V3 = (
    "Do not trade sword hits head-on with someone as fast as you: that is a coin flip. Win it with the other tools "
    "first. Open with the bow while they run at you from afar (running straight at you they are easy to hit; two full "
    "arrows take 16 of their 20). When they come within about 8 blocks, lay lava on the ground between you: running at "
    "you they run into it, lose about 8 health a second in it and burn 1 a second for 15 s after. Keep the lava "
    "between you, circle around it and let them burn; step away from their sword while yours recharges and in while "
    "theirs does. Finish with engage: crits kill fastest. If they pour lava at you, put it out with your water at once; "
    "if you burn, water at your feet. Against arrows, build cover.")

STYLES = {
    "crit": "crit hits: jump-timed, 10.5 each",
    "sprint": "sprint hits: 7 each, with a w-tap before each so it knocks them back",
    "plain": "plain hits: 7 each, standing",
}


def hits(hp, dmg):
    return max(1, math.ceil((hp - 0.01) / dmg)) if hp > 0 else 0


# ---------------------------------------------------------------- what a strong player keeps in mind

RACE_CTX = {}   # speed and history for race_extra (set by step_state)


def race_extra(obs, f, name, v):
    if v < 4:
        return []
    fits = fits_now(obs, f, name, RACE_CTX.get("speed"), RACE_CTX.get("history", ()))
    return ["What fits right now: " + "; ".join(fits) + "."] if fits else []


def race_lines(obs, f, name, v=1):
    if not f or f.get("health") is None:
        return []
    if v >= 3:
        th, mh = f["health"], obs["health"]
        need, needc, die = hits(th, SWORD), hits(th, CRIT), hits(mh, SWORD)
        it = obs.get("items", {})
        lines = [f"The race: {name} has {th:.0f} health, you {mh:.0f}. Sword for sword: {need} hits (or {needc} crits) kill "
                 f"them, {die} of theirs kill you - " + ("you are ahead." if need < die else
                                                        "a coin flip: do not rely on it." if need == die else
                                                        "you lose it: do not trade hits head-on.")]
        ways = []
        if it.get("lava_bucket"):
            ways.append(f"lava in their path: about {LAVA_DPS:.0f} a second while they are in it, then {BURN_DPS:.0f} a "
                        f"second for up to {BURN_S} s (they have {th:.0f})")
        if it.get("bow") and it.get("arrow"):
            ways.append(f"arrows: about {ARROW:.0f} each at full draw ({it['arrow']} left)")
        if f.get("burning"):
            ways.append("they already burn: 1 a second while you stay out of their reach")
        if ways:
            lines.append("Other ways to take their health: " + "; ".join(ways) + ".")
        return lines + race_extra(obs, f, name, v)
    th, mh = f["health"], obs["health"]
    need, needc, die = hits(th, SWORD), hits(th, CRIT), hits(mh, SWORD)
    verdict = ("you win a straight trade" if need < die else "a straight trade is even: whoever lands first wins" if need == die
               else "you lose a straight trade: land crits, use lava or the bow, or break away")
    return [f"The race: {name} has {th:.0f} health - {need} sword hit{'s' if need > 1 else ''} or {needc} crit{'s' if needc > 1 else ''} "
            f"kill them; you have {mh:.0f} - {die} of their hits kill you. So {verdict}."]


def closing_speed(obs, f, prev, dt, name):
    """Blocks a second they come closer (negative: moving away), or None."""
    if not f or not prev or not dt:
        return None
    p = cp.fighter(prev, name)
    if not p:
        return None
    v = (v1.relative(prev, p["pos"])["dist"] - v1.relative(obs, f["pos"])["dist"]) / dt
    return v if abs(v) <= 12 else None


def fits_now(obs, f, name, speed, history):
    """v4: which tools fit the moment - facts about timing, not orders."""
    it, held = obs.get("items", {}), obs.get("held_item")
    d = v1.relative(obs, f["pos"])["dist"]
    out = []
    to4 = (d - 4) / speed if speed and speed > 1 and d > 4 else None
    if it.get("bow") and it.get("arrow") and d >= 5:
        if to4 is not None and to4 < 1.0:
            out.append(f"the bow does not fit: they are within 4 blocks in {to4:.1f} s, before a full draw (1 s)")
        else:
            out.append("the bow fits: there is time for a full draw" + (f" ({to4:.1f} s before they are close)" if to4 else ""))
    if 3 <= d <= 9 and (it.get("lava_bucket") or it.get("water_bucket")):
        when = f", they reach it in about {max(0.0, (d - 2.5) / speed):.1f} s" if speed and speed > 1 else ""
        out.append(f"a trap fits: lava or water on the ground between you{when}")
    if f.get("holding") == "bow" and d > 6 and any(it.get(b) for b in cu.BLOCKS):
        out.append("cover fits: they hold a bow" + (" and are drawing it" if f.get("using") else ""))
    if d <= REACH + 1:
        out.append("the sword fits: they are in reach (engage)")
    lava = next((x for x in obs.get("sources", []) if x["kind"] == "lava"), None)
    if lava and it.get("water_bucket") and not own_lava(history) and lava["dist"] <= 3:
        out.append(f"water fits: their lava is {lava['dist']:.1f} blocks from you")
    return out


def own_lava(history):
    """Did you lay lava in the last few steps (your own trap, not theirs)?"""
    return any("laid lava" in (h["result"].get("note") or "") or "poured lava" in (h["result"].get("note") or "")
               for h in list(history)[-40:])


def closing_lines(obs, f, prev, dt, name):
    """How fast they close in, and when they will be in reach."""
    if not f or not prev or not dt:
        return []
    p = cp.fighter(prev, name)
    if not p:
        return []
    d0, d1 = v1.relative(prev, p["pos"])["dist"], v1.relative(obs, f["pos"])["dist"]
    speed = (d0 - d1) / dt
    if abs(speed) > 12:   # two sprinters meeting close at about 11 a second: more is a bad reading, not a fact
        return []
    if speed > 1.5 and d1 > REACH:
        return [f"{name} is closing in at {speed:.1f} blocks a second: in sword reach in about {(d1 - REACH) / speed:.1f} s."]
    if speed < -1.5:
        return [f"{name} is moving away at {-speed:.1f} blocks a second."]
    return []


OPP_RECHARGE = 0.625   # their diamond sword


def their_sword(f):
    """(state, seconds to ready) of the opponent's sword, from when they last swung; None if unknown."""
    ago = f.get("swung_ago")
    if ago is None or (f.get("holding") or "").endswith("_sword") is False:
        return None
    left = OPP_RECHARGE - ago
    return ("recharging", left) if left > 0.05 else ("ready", 0.0)


def threat_lines(obs, f, name, v=1):
    if not f:
        return []
    out = []
    if v >= 3:
        sw = their_sword(f)
        d = v1.relative(obs, f["pos"])["dist"]
        if sw and d < 6 and (f.get("holding") or "").endswith("_sword"):
            out.append(f"{name} swung {f['swung_ago']:.1f} s ago: their sword is recharging for {sw[1]:.1f} s more - step in and hit now."
                       if sw[0] == "recharging" else f"{name}'s sword is ready: their next hit is a full one.")
    held = f.get("holding")
    d = v1.relative(obs, f["pos"])["dist"]
    if held == "lava_bucket" and d <= 6:
        out.append(f"{name} holds a lava bucket {d:.0f} blocks from you: they may pour it at your feet - keep your water ready.")
    if held == "bow":
        out.append(f"{name} holds a bow{' and is drawing it: an arrow is coming' if f.get('using') else ''}.")
    if f.get("burning"):
        out.append(f"{name} is burning: they lose 1 health a second until it goes out.")
    return out


def reflex_lines(obs):
    r = obs.get("reflex")
    return [f"{r['ago']:.1f} s ago {r['what']}."] if r and r["ago"] <= 3 else []


def engage_lines(obs):
    e = obs.get("engage")
    if not e:
        return ["You are not engaged: the body swings only when you choose a hit."]
    return [f"You are engaged with {e['target']} ({STYLES[e['style']]}) for {e['s']:.0f} s: the body keeps your sword on them "
            f"and swings whenever a swing counts ({e['swings']} swings so far). You choose where to move."]


# ---------------------------------------------------------------- options

MOVES = {k: v for k, v in cu.MOVES.items() if k not in ("turn_left", "turn_right", "turn_around")}


def options(obs, name, history=(), v=1, dt=None, prev=None):
    f = cp.fighter(obs, name) if name else None
    it, held = obs.get("items", {}), obs.get("held_item")
    base = cu.options(obs, name, history, v=2)["criteria"]
    eng = obs.get("engage")
    opts = {k: base[k] for k in MOVES if k in base}
    if eng:   # engaged: the view is on them, so sideways keys circle them
        for k, words in (("left", "circle them to your left"), ("right", "circle them to your right")):
            if k in opts:
                opts[k] = opts[k].split(" - ")[0] + f" - you would {words}: their swings miss more"
        if "back" in opts:
            opts["back"] = opts["back"].split(" - ")[0] + " - you would back off from them (out of their reach while your sword recharges)"
    if f:
        rel = v1.relative(obs, f["pos"])
        th = f.get("health") or 20
        for style, words in STYLES.items():
            if eng and eng["style"] == style:
                continue
            k = f"engage_{style}"
            dmg = CRIT if style == "crit" else SWORD
            n = hits(th, dmg)
            if v >= 2 and eng:
                opts[k] = (f"switch from {eng['style']} hits to {words} - {n} such hit{'s' if n > 1 else ''} "
                           f"kill{'' if n > 1 else 's'} them; switching mid-fight gains nothing unless the situation changed")
            else:
                opts[k] = (f"engage {name} with {words} - the body swings whenever a swing counts; {n} such hit{'s' if n > 1 else ''} "
                           f"kill{'' if n > 1 else 's'} them" + (f"; they are {rel['dist']:.0f} blocks away: close in while engaged"
                                                                   if rel["dist"] > REACH + 1 else ""))
        if eng:
            opts["disengage"] = "stop fighting hand to hand (the body stops swinging)"
    for k in ("draw_bow", "release_bow", "lava_at", "lava_pickup", "water_here", "water_pickup", "place_block",
              "hold_sword", "eat_gapple"):
        if k in base:
            opts[k] = base[k]
    # their lava: put it out with your water
    lava = next((x for x in obs.get("sources", []) if x["kind"] == "lava"), None)
    if lava and it.get("water_bucket"):
        close = lava["dist"] <= 2.5 or obs.get("burning")
        opts["water_on_lava"] = (f"pour your water on the lava {lava['dist']:.1f} blocks away - it goes out or turns to obsidian"
                                 + (": it is right next to you, do it now" if close else ""))
    # cover against arrows
    if f and any(it.get(b) for b in cu.BLOCKS):
        bow = f.get("holding") == "bow"
        opts["build_cover"] = (f"build cover: a wall 3 wide and 2 high between you and {name} (about 1 s)"
                               + (f" - {name} holds a bow{' and is drawing it' if f.get('using') else ''}: the wall stops the arrow"
                                  if bow else " - it stops arrows and makes them come around it"))
    # the bow drawn: how much the aim leads them, and whether it will likely hit
    if v >= 2 and "release_bow" in opts and (obs.get("using_s") or 0) < 0.8:
        opts["release_bow"] = (f"let go now, only {obs.get('using_s') or 0:.1f} s drawn: a weak arrow (about 3), where a full "
                               f"draw at 1 s does about {ARROW:.0f} - better keep drawing a moment")
    d = obs.get("draw")
    if d and "release_bow" in opts:
        if d["speed"] < 1:
            opts["release_bow"] += f"; they are nearly still: a sure hit ({d['flight_s']:.2f} s flight)"
        else:
            opts["release_bow"] += (f"; the arrow flies {d['flight_s']:.2f} s and the aim leads them {d['lead']:.1f} blocks "
                                    f"(they move {d['speed']:.1f} blocks a second)" + ("; running sideways they may dodge it"
                                                                                      if d["speed"] > 3.5 else ""))
    if "wait" in base and "release_bow" in base:
        opts["wait"] = base["wait"]
    elif v >= 2 and eng and "wait" in opts:
        dmg = CRIT if eng["style"] == "crit" else SWORD
        th = (f.get("health") or 20) if f else 20
        opts["wait"] = (f"keep fighting as you are ({STYLES[eng['style']]}; {eng['swings']} swings so far, "
                        f"{hits(th, dmg)} more hits kill them) and keep your keys as they are")
    if f and "lava_at" in opts and "too far" not in opts["lava_at"] and "next to you" not in opts["lava_at"]:
        opts["lava_at"] += (f" - in the lava they lose about {LAVA_DPS:.0f} health a second and then burn "
                            f"{BURN_DPS:.0f} a second for up to {BURN_S} s unless they put it out")
    if f and "draw_bow" in opts:
        opts["draw_bow"] += f" - a full-power arrow does about {ARROW:.0f}"
    if v >= 3 and f:
        d = v1.relative(obs, f["pos"])["dist"]
        th, mh = f.get("health") or 20, obs["health"]
        if it.get("lava_bucket") and 3 <= d <= 9:
            opts["lava_trap"] = (f"lay lava on the ground 2-3 blocks ahead of you, in {name}'s path - coming at you they "
                                 f"step into it: about {LAVA_DPS:.0f} a second in it, then burning {BURN_DPS:.0f} a second "
                                 f"for up to {BURN_S} s; then keep it between you")
        for k in [k for k in opts if k.startswith("engage_")]:
            dmg = CRIT if k.endswith("crit") else SWORD
            if hits(th, dmg) >= hits(mh, SWORD):
                opts[k] += " - but hit for hit you do not win this trade; weaken them first"
        closing = closing_lines(obs, f, prev, dt, name) if prev and dt else []
        if "draw_bow" in opts and closing and d > 8:
            opts["draw_bow"] += " - they run straight at you: an easy target"
        mine = obs.get("recharge", 1) >= 1
        sw = their_sword(f)
        if eng and sw and d < 5:
            fwd = [k for k in ("forward", "sprint", "sprint_jump") if k in opts]
            if sw[0] == "recharging" and mine:
                for k in fwd:
                    opts[k] += f" - step in now: their sword needs {sw[1]:.1f} s more and yours is ready"
            elif sw[0] == "ready" and not mine and "back" in opts:
                opts["back"] += " - their sword is ready and yours is not: stay out of their reach a moment"
        if f.get("burning") and "back" in opts:
            opts["back"] += " - they are burning: every second away from their sword costs them 1 health"
        closing = closing_lines(obs, f, prev, dt, name) if prev and dt else []
        if "lava_at" in opts and closing and 3 <= d <= 5 and "too far" not in opts["lava_at"]:
            opts["lava_at"] += " - they are running at you: pour it now and they run into it"
    if v >= 4 and f:
        d = v1.relative(obs, f["pos"])["dist"]
        speed = closing_speed(obs, f, prev, dt, name)
        to4 = (d - 4) / speed if speed and speed > 1 and d > 4 else None
        if "draw_bow" in opts and to4 is not None and to4 < 1.0:
            opts["draw_bow"] += (f" - but they are within 4 blocks in {to4:.1f} s, before a full draw: the body drops the bow "
                                 f"for the sword then")
        reach = f", they reach it in about {max(0.0, (d - 2.5) / speed):.1f} s" if speed and speed > 1 else ""
        if "lava_trap" in opts:
            opts["lava_trap"] += reach
        if it.get("water_bucket") and 3 <= d <= 9:
            opts["water_trap"] = (f"lay water on the ground 2-3 blocks ahead of you, in {name}'s path - its flow slows them "
                                  f"and pushes them back (easy arrows, a late arrival){reach}; take it back after")
        if "water_on_lava" in opts and own_lava(history):
            opts["water_on_lava"] += " (it may be your own trap: put it out only if it reaches you)"
        if "build_cover" in opts and f.get("holding") == "bow":
            opts["build_cover"] += f"; a full arrow takes about {ARROW:.0f} of your {obs['health']:.0f}"
        drawing = obs.get("using_s") is not None and held == "bow"
        if drawing and (obs.get("using_s") or 0) < 0.95:
            opts["wait"] = f"keep drawing: full power in {max(0.0, 1 - obs['using_s']):.1f} s (the body holds it on them)"
        if d < 5:
            opts.pop("draw_bow", None)
            if held == "bow" and "hold_sword" in opts:
                opts["hold_sword"] = f"take your sword now - {name} is {d:.1f} blocks away: too close for the bow"
    for k in opts:
        why = cu.failed_last(history, k)
        if why and "(tried just now" not in opts[k]:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


def act_params(option):
    if option.startswith("engage_"):
        return "engage", {"style": option.split("_", 1)[1]}
    return option, {}


# ---------------------------------------------------------------- state

def step_state(obs, mem, cfg, dt=None, v=1):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    f = cp.fighter(obs, name) if name else None
    RACE_CTX.update(speed=closing_speed(obs, f, mem.prev_obs, dt, name), history=mem.history)
    you = [line for line in cu.you_lines(obs, 2) if not line.startswith("In your hand") or not obs.get("engage")]
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", RULES),
        v2.section("Strategy", [STRATEGY_V3 if v >= 3 else STRATEGY]),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("Opponent", (cu.opponent_lines(obs, name, mem.prev_obs) + closing_lines(obs, f, mem.prev_obs, dt, name)
                                + threat_lines(obs, f, name, v)) if name else ["No opponent."]),
        v2.section("The race", race_lines(obs, f, name, v)),
        v2.section("You", you + engage_lines(obs) + (reflex_lines(obs) if v >= 4 else []) + cu.feel_lines(obs)),
        v2.section("Your items", cu.item_lines(obs)),
        v2.section("Around you", cu.hazard_lines(obs)),
        v2.section("Fight so far", cu.combat_lines(obs)),
        v2.section("Your last actions (oldest first)", cu.recent_lines(mem.history)),
    ] if x)


# ---------------------------------------------------------------- v5: techniques and when to use them, no warnings

TECHNIQUES = [
    "1. They are close (within about 4 blocks): fight sword to sword - engage.",
    "2. They are far: draw the bow and shoot; the body leads them for the arrow's flight.",
    "3. They shoot at you with a bow: (a) build cover, or (b) weave left and right (zigzag) so their arrows miss.",
    "4. They are fairly close and coming (about 4-9 blocks): quickly lay lava or water in their path (a trap), or "
    "pour lava at their feet.",
]


TECHNIQUES_V6 = [TECHNIQUES[0],
                 "2. They are far: shoot full-power arrows (a full draw takes 1 s; the body keeps the aim on them and leads "
                 "them), one after another while they come.",
                 TECHNIQUES[2], TECHNIQUES[3]]


def techniques_now(obs, f, name):
    """Which techniques match the moment: the distance and what they are doing, as facts."""
    if not f:
        return []
    d = v1.relative(obs, f["pos"])["dist"]
    bow = f.get("holding") == "bow"
    now = []
    if d <= 4.5:
        now.append("1")
    if d > 9:
        now.append("2")
    if bow:
        now.append("3")
    if 4 <= d <= 9:
        now.append("4")
    what = f"{name} is {d:.0f} blocks away" + (", holding a bow" + (" and drawing it" if f.get("using") else "") if bow else "")
    return [f"Now: {what} - technique{'s' if len(now) > 1 else ''} {', '.join(now)} match{'' if len(now) > 1 else 'es'}."] if now else []


def their_sword_line(f, name):
    sw = their_sword(f)
    if not sw or not (f.get("holding") or "").endswith("_sword"):
        return []
    return [f"{name}'s sword recharges for {sw[1]:.1f} s more (they swung {f['swung_ago']:.1f} s ago)." if sw[0] == "recharging"
            else f"{name}'s sword is recharged."]


def options5(obs, name, history=(), v=5):
    f = cp.fighter(obs, name) if name else None
    it, held = obs.get("items", {}), obs.get("held_item")
    base = cu.options(obs, name, history, v=2)["criteria"]
    eng = obs.get("engage")
    opts = {k: base[k] for k in MOVES if k in base}
    if eng:
        for k, words in (("left", "circle them to your left"), ("right", "circle them to your right")):
            if k in opts:
                opts[k] = opts[k].split(" - ")[0] + f" - you would {words}"
        opts["wait"] = f"keep fighting as you are ({STYLES[eng['style']]}; {eng['swings']} swings so far)"
    if f:
        for style, words in STYLES.items():
            if eng and eng["style"] == style:
                continue
            opts[f"engage_{style}"] = ((f"switch from {eng['style']} hits to {words}" if eng else
                                        f"engage {name} with {words}: the body swings whenever a swing counts")
                                       + " (technique 1)")
        if eng:
            opts["disengage"] = "stop fighting hand to hand (the body stops swinging)"
        if held == "bow" or f.get("holding") == "bow" or v1.relative(obs, f["pos"])["dist"] > 6:
            opts["zigzag"] = "weave left and right (a snake walk; forward stays held if it is) - arrows aimed at you miss more (technique 3b)"
    tags = {"draw_bow": "2", "release_bow": "2", "lava_at": "4"}
    for k in ("draw_bow", "release_bow", "lava_at", "lava_pickup", "water_here", "water_pickup", "place_block", "hold_sword"):
        if k in base:
            opts[k] = base[k] + (f" (technique {tags[k]})" if k in tags else "")
    if "wait" in base and "release_bow" in base:
        opts["wait"] = base["wait"]
    d = obs.get("draw")
    if d and "release_bow" in opts:
        opts["release_bow"] += f"; the arrow flies {d['flight_s']:.2f} s and the aim leads them {d['lead']:.1f} blocks"
    if v >= 6 and f:
        dist = v1.relative(obs, f["pos"])["dist"]
        drawing = obs.get("using_s") is not None and held == "bow"
        if it.get("bow") and it.get("arrow") and not drawing and dist >= 5:
            opts["shoot_full"] = (f"shoot {name} with a full-power arrow: the body draws the bow for 1 s with the aim on them "
                                  f"and their lead, then lets go (about {ARROW:.0f}; {it['arrow']} arrows) (technique 2)")
        if "draw_bow" in opts:
            opts["draw_bow"] = (f"draw the bow and hold it drawn, to let go later - to wait behind cover or for {name} to "
                                f"step out (technique 2)")
    if f:
        dist = v1.relative(obs, f["pos"])["dist"]
        if 3 <= dist <= 9:
            if it.get("lava_bucket"):
                opts["lava_trap"] = (f"lay lava on the ground 2-3 blocks ahead of you, in {name}'s path: coming at you they step "
                                     f"into it (about {LAVA_DPS:.0f} a second in it, then they burn) (technique 4)")
            if it.get("water_bucket"):
                opts["water_trap"] = (f"lay water on the ground 2-3 blocks ahead of you, in {name}'s path: its flow slows and "
                                      f"pushes them (technique 4)")
        if any(it.get(b) for b in cu.BLOCKS):
            opts["build_cover"] = f"build cover: a wall 3 wide and 2 high between you and {name} (about 1 s) (technique 3a)"
    lava = next((x for x in obs.get("sources", []) if x["kind"] == "lava"), None)
    if lava and it.get("water_bucket"):
        opts["water_on_lava"] = f"pour your water on the lava {lava['dist']:.1f} blocks away: it goes out or turns to obsidian"
    for k in opts:
        why = cu.failed_last(history, k)
        if why and "(tried just now" not in opts[k]:
            opts[k] += f" (tried just now and it failed: {why})"
    return {"type": "choice", "instructions": "Which control should you use now?", "criteria": opts}


def step_state5(obs, mem, cfg, dt=None, v=5):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    f = cp.fighter(obs, name) if name else None
    you = [line for line in cu.you_lines(obs, 2) if not line.startswith("In your hand") or not obs.get("engage")]
    facts = [f"{name} has {f['health']:.0f} health; you have {obs['health']:.0f}. A sword hit does about {SWORD:.0f}, a crit "
             f"{CRIT:.1f}, a full arrow about {ARROW:.0f}."] if f and f.get("health") is not None else []
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", RULES),
        v2.section("Techniques (use each when its moment comes)",
                   (TECHNIQUES_V6 if v >= 6 else TECHNIQUES) + techniques_now(obs, f, name)),
        v2.section("What you know", mem.facts),
        v2.section("Current instruction", cs.instruction_lines(mem, cfg, obs)),
        v2.section("Opponent", (cu.opponent_lines(obs, name, mem.prev_obs) + closing_lines(obs, f, mem.prev_obs, dt, name)
                                + threat_lines(obs, f, name, 1) + (their_sword_line(f, name) if f else []) + facts)
                   if name else ["No opponent."]),
        v2.section("You", you + engage_lines(obs) + reflex_lines(obs) + cu.feel_lines(obs)
                   + (["You are weaving left and right (zigzag)."] if obs.get("zigzag") else [])
                   + ([f"{obs['lava_stop']:.1f} s ago your keys were let go just before lava or fire."]
                      if obs.get("lava_stop") is not None and obs["lava_stop"] < 3 else [])),
        v2.section("Your items", cu.item_lines(obs)),
        v2.section("Around you", cu.hazard_lines(obs)),
        v2.section("Fight so far", cu.combat_lines(obs)),
        v2.section("Your last actions (oldest first)", cu.recent_lines(mem.history)),
    ] if x)


# ---------------------------------------------------------------- v7: the user's play distilled, poured liquids previewed

TECHNIQUES_V7 = [
    TECHNIQUES[0],
    TECHNIQUES_V6[1],
    "3. They shoot at you from afar: run at them in a zigzag (weaving left and right while you close in), or build cover "
    "and move up behind it - standing still under arrows loses.",
    "4. They are close and coming (about 2-5 blocks): pour lava right at their feet - a trap laid farther off is one they "
    "can see and walk around.",
    "5. Right after you pour a bucket or place blocks, take your sword back (the body does it when they come within 4.5 "
    "blocks).",
]


def pour_words(p):
    return f"it would land {p['to_them']:.1f} blocks from them and {p['to_you']:.1f} from you"


def options7(obs, name, history=()):
    q = options5(obs, name, history, v=6)
    opts = q["criteria"]
    f = cp.fighter(obs, name) if name else None
    pour = obs.get("pour") or {}
    for k in ("lava_at", "lava_trap", "water_trap"):
        if k in opts:
            if pour.get(k):
                opts[k] = opts[k].split(" (technique")[0] + f" - {pour_words(pour[k])} (technique 4)"
            else:
                opts.pop(k)   # no clear, safe aim from here: said in the state instead
    if f and "lava_at" not in opts and pour.get("lava_at") and v1.relative(obs, f["pos"])["dist"] <= 5.5:
        opts["lava_at"] = f"pour lava at {name}'s feet - {pour_words(pour['lava_at'])} (technique 4)"
    if f and "zigzag" in opts:
        opts["zigzag_forward"] = (f"run at {name} in a zigzag, weaving left and right as you close in - their arrows miss "
                                  f"more (technique 3)")
    return q


def pour_lines(obs):
    pour, it = obs.get("pour") or {}, obs.get("items", {})
    out = []
    if it.get("lava_bucket") and not pour.get("lava_at") and not pour.get("lava_trap"):
        out.append("Lava: no clear, safe aim at them from here (it would land somewhere else, or near you).")
    return out


def step_state7(obs, mem, cfg, dt=None):
    s = step_state5(obs, mem, cfg, dt, v=6)
    s = s.replace("\n".join(TECHNIQUES_V6), "\n".join(TECHNIQUES_V7))
    extra = pour_lines(obs)
    if extra:
        s = s.replace("== Your items ==\n", "== Your items ==\n" + "\n".join(extra) + "\n", 1)
    return s


# ---------------------------------------------------------------- v8: move between shots

TECHNIQUES_V8 = list(TECHNIQUES_V7)
TECHNIQUES_V8[1] = ("2. They are far: shoot full-power arrows and move between them - shoot, hop sideways to a new spot, "
                    "shoot again - so they cannot line you up (the body leads them for the arrow's flight).")


TECHNIQUES_V8[2] = ("3. They shoot at you: keep moving across their line (the body also sidesteps an arrow it sees coming), "
                    "shoot back and move (technique 2), or build cover; run at them in a zigzag once you can reach them "
                    "before many more arrows - within about 12 blocks, or while they hold no bow.")


def options8(obs, name, history=()):
    q = options7(obs, name, history)
    opts = q["criteria"]
    f = cp.fighter(obs, name) if name else None
    if f and "zigzag_forward" in opts:
        d = v1.relative(obs, f["pos"])["dist"]
        secs = max(0.0, d - 3) / 5.6
        opts["zigzag_forward"] = (f"run at {name} in a zigzag, weaving left and right as you close in - about "
                                  f"{secs:.1f} s to reach sword range from {d:.0f} blocks (technique 3)")
    if "shoot_full" in opts:
        it = obs.get("items", {})
        opts["shoot_hop"] = (f"shoot {name} with a full-power arrow, then hop sideways to a new spot - harder for them to "
                             f"hit you back ({it.get('arrow', 0)} arrows) (technique 2)")
    return q


def step_state8(obs, mem, cfg, dt=None):
    return (step_state7(obs, mem, cfg, dt).replace(TECHNIQUES_V7[1], TECHNIQUES_V8[1])
            .replace(TECHNIQUES_V7[2], TECHNIQUES_V8[2]))


# ---------------------------------------------------------------- v9: cover, footwork, fire when they do not shoot back

TECHNIQUES_V9 = [
    "1. They are close (within about 4 blocks): fight sword to sword (engage) and circle them (circle): the body steps "
    "in to hit when your sword is ready and back out of their reach while it recharges, sideways all the time.",
    "2. They are far: keep them under fire - arrow after arrow (bow_barrage, or one at a time: shoot_full back to back "
    "while they are not shooting, shoot_hop while they are), or over cover (cover_shot). Volume wins a bow fight.",
    "3. They shoot at you: shoot back - the body sidesteps every arrow it sees coming and steps off a bow aimed at "
    "you, so your hands stay on the bow; move between shots (shoot_hop, which also closes in from far off), shoot "
    "over cover (cover_shot), and run in (zigzag_forward) from about 15 blocks.",
    TECHNIQUES_V7[3],
    TECHNIQUES_V7[4],
]


def cover_lines(obs, name):
    c, out = obs.get("cover"), []
    if c:
        if c["covered"]:
            out.append(f"You are behind cover: blocks cut {name}'s line of fire to you"
                       + (" (a jump lifts you over it for a shot)." if c["over"] else "."))
        elif c["spot"] is not None:
            out.append(f"Cover: a spot out of {name}'s line of fire is {c['spot']:.1f} blocks away.")
        else:
            out.append(f"Cover: no spot out of {name}'s line of fire within 3 blocks.")
    if obs.get("in_water"):
        dry = obs.get("dry")
        out.append("You are in water: you swim slowly and cannot sprint" + (f"; dry ground is {dry['k']} blocks off." if dry else "."))
    f = cp.fighter(obs, name) if name else None
    if f and f.get("in_water"):
        out.append(f"{name} is in water: they move slowly.")
    if obs.get("circle"):
        out.append(f"You are circling {name} (footwork).")
    if obs.get("going_cover"):
        out.append("You are moving to the spot behind cover.")
    if obs.get("their_draw_s") is not None:
        out.append(f"{name} has been drawing their bow for {obs['their_draw_s']:.1f} s (full power at 1 s).")
    return out


def options9(obs, name, history=()):
    q = options8(obs, name, history)
    opts = q["criteria"]
    f = cp.fighter(obs, name) if name else None
    if not f:
        return q
    it, c = obs.get("items", {}), obs.get("cover") or {}
    d = v1.relative(obs, f["pos"])["dist"]
    their_bow = f.get("holding") == "bow"
    can_shoot = bool(it.get("bow") and it.get("arrow"))
    if not their_bow:
        opts.pop("shoot_hop", None)
        if "shoot_full" in opts:
            opts["shoot_full"] = opts["shoot_full"].replace(" (technique 2)", f" - {name} is not shooting: back to back, the fastest fire (technique 2)")
    if their_bow and can_shoot and d > 12:
        opts.pop("zigzag_forward", None)
    if their_bow and can_shoot and d > 6:
        for k in ("sprint", "sprint_jump", "forward"):   # a straight run at someone shooting is the easiest target
            opts.pop(k, None)
    if obs.get("in_water"):   # the ways out of water: take it back, block out, or shoot from it
        dry = obs.get("dry")
        if any(it.get(b) for b in cu.BLOCKS):
            opts["water_escape"] = ("a step out of the water on a block: one placed ahead and climbed onto (about 0.4 s)"
                                    + (f", toward dry ground {dry['k']} blocks off" if dry else ""))
        src = next((x for x in obs.get("sources", []) if x["kind"] == "water"), None)
        if src and it.get("bucket"):
            opts["water_pickup"] = f"scoop up the water source {src['dist']:.1f} blocks away with your empty bucket: the water around you goes with it"
        f_me = obs.get("pos") or {}
        on_src = src and math.floor(f_me.get("x", 0)) == src["pos"]["x"] and math.floor(f_me.get("z", 0)) == src["pos"]["z"]
        if src and any(it.get(b) for b in cu.BLOCKS) and not on_src:
            opts["water_block"] = f"plug the water source {src['dist']:.1f} blocks away with a block: the water it feeds drains away"
    if can_shoot and (4 <= d <= 12 or (obs.get("in_water") and d > 3)):
        opts["quick_shot"] = (f"a quick arrow at {name}: a 0.7 s draw at about 3/4 power, the aim set for it"
                              + (" - you are in water, where a sword fight is slow to reach" if obs.get("in_water") else "")
                              + " (technique 2)")
    if can_shoot and d > 8:
        opts.pop("zigzag", None)   # the body dodges their arrows itself; a weave in place only stops your own fire
    if can_shoot and d >= 12:
        opts["bow_barrage"] = (f"keep {name} under fire: two arrows with a hop between them (the body dodges their arrows; "
                               f"it stops early if they come within 10 blocks or you lose 6 health) ({it.get('arrow', 0)} "
                               f"arrows) (technique 2)")
    if d <= 8 and not obs.get("circle"):
        opts["circle"] = (f"circle {name}: the body steps in to hit when your sword is ready and out of their reach while it "
                          f"recharges, sideways all the time (technique 1)")
    if c.get("spot") is not None and not c.get("covered"):
        opts["take_cover"] = f"step {c['spot']:.1f} blocks to a spot behind cover, out of {name}'s line of fire (technique 3)"
    # water: for your own fire and for their lava, never your own traps
    opts.pop("water_trap", None)
    if not obs.get("burning"):
        opts.pop("water_here", None)
    opts.pop("water_on_lava", None)
    theirs = next((x for x in obs.get("sources", []) if x["kind"] == "lava" and not x.get("own")), None)
    if theirs and it.get("water_bucket"):
        opts["water_on_lava"] = (f"pour your water on {name}'s lava {theirs['dist']:.1f} blocks away: it goes out or "
                                 f"turns to obsidian")
    if any(it.get(b) for b in cu.BLOCKS) and d >= 5:
        opts["build_cover"] = (f"build a full wall (3 wide, 2 high) between you and {name}, in about 0.3 s: it stops all "
                               f"their arrows - for when you do not want to trade shots (technique 3)")
        if can_shoot:
            opts["build_window"] = (f"build a wall with a window (凹: the sides 2 high, the middle 1 high) in about 0.25 s: "
                                    f"covered below the chest, and you shoot through the gap - for trading shots from "
                                    f"cover (technique 3)")
    if c.get("covered") and c.get("over") and can_shoot and d >= 5:
        opts["cover_shot"] = (f"draw behind the cover, then jump and shoot over it at {name} at the top of the jump, dropping "
                              f"back behind it (technique 3)")
    return q


def step_state9(obs, mem, cfg, dt=None):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    s = step_state7(obs, mem, cfg, dt).replace("\n".join(TECHNIQUES_V7), "\n".join(TECHNIQUES_V9))
    extra = cover_lines(obs, name) if name else []
    if extra:
        s = s.replace("== You ==\n", "== You ==\n" + "\n".join(extra) + "\n", 1)
    return s


# ---------------------------------------------------------------- v10: a layered state, the draw as a task
# Fixed (rules, techniques) first; then the match and this round (counted up from the body's events by the agent);
# then the last seconds (events with times, the draw in progress); then now (them, you, the draw's details, items,
# the last few choices). Old steps are not listed one by one: they live on as counts and events.

TECHNIQUES_V10 = [
    "1. Close (within about 4 blocks): sword to sword (engage), circling them (circle): in to hit when your sword is "
    "ready, out of their reach while it recharges.",
    "2. At range, the bow is a draw you hold and let go: draw (bow_draw) and choose the moment to let go "
    "(bow_release) - full power at 1 s; as they come out of cover, just after they turn, while they draw their own "
    "bow (a drawing player moves at a fifth). Hold it while they hide; lower it (bow_cancel) to move or fight.",
    "3. They shoot at you: shoot back - the body sidesteps arrows it sees coming (lowering your draw when it must). "
    "A wall (build_cover) stops their arrows; a wall with a window (build_window) lets you shoot through it; a "
    "jump shot (jump_shot) goes over a wall. Run in (zigzag_forward) from about 12 blocks.",
    TECHNIQUES_V7[3],
    "5. In water: out on blocks (water_escape), plug the source (water_block), scoop it (water_pickup), or shoot.",
]

CAUSE_WORDS = {"their arrow": "arrows", "their sword": "sword", "fire": "fire", "a hit": "other"}


def round_words(r, name):
    """One line for a round's counts: arrows both ways, health lost by cause."""
    by = ", ".join(f"{CAUSE_WORDS.get(c, c)} {d:.0f}" for c, d in sorted(r["lost_by"].items(), key=lambda kv: -kv[1]))
    return (f"your arrows {r['my_hits']}/{r['my_shots']}, theirs {r['their_hits']}/{r['their_shots']}"
            + (f" ({r['dodged']} dodged)" if r["dodged"] else "") + (f"; you lost {sum(r['lost_by'].values()):.0f}"
                                                                      + (f" ({by})" if by else "") if r["lost_by"] else ""))


def match_lines(match, name, obs):
    out = []
    sc = match.get("score")
    if sc:
        out.append(f"Score: you {sc.get(obs['name'], 0)}, {name} {sc.get(name, 0)}.")
    for r in match["rounds"][-4:]:
        out.append(f"Round {r['n']}: {'you won' if r['won'] else 'they won'} in {r['secs']:.0f} s ({r['how']}) - {round_words(r, name)}.")
    allr = match["rounds"] + ([match["cur"]] if match.get("cur") else [])
    ts, th = sum(r["their_shots"] for r in allr), sum(r["their_hits"] for r in allr)
    ms, mh = sum(r["my_shots"] for r in allr), sum(r["my_hits"] for r in allr)
    prof = []
    if ts >= 3:
        prof.append(f"their arrows hit you {th} of {ts} ({100 * th / ts:.0f}%)")
    if ms >= 3:
        prof.append(f"yours hit them {mh} of {ms} ({100 * mh / ms:.0f}%)")
    rh = obs.get("their_rhythm")
    if rh:
        prof.append(f"they change direction about every {rh['run_s']:.1f} s")
    if prof:
        out.append(f"{name} so far: " + "; ".join(prof) + ".")
    return out


def this_round_lines(match, obs, name, f):
    cur = match.get("cur")
    if not cur:
        return []
    secs = time.time() - cur["t0"]
    them = f" {name} {f['health']:.0f}/20" if f and f.get("health") is not None else ""
    return [f"{secs:.0f} s in. You {obs['health']:.0f}/20;{them}. " + round_words(cur, name)[0].upper() + round_words(cur, name)[1:] + "."]


def news_lines(ins):
    """The arena's news for this fight (sudden death, the final duel), in its own words, the last two."""
    return [f'{time.time() - n["t"]:.0f} s ago the arena told you: "{n["text"]}"' for n in (ins or {}).get("news", [])[-2:]]


def event_lines(obs, secs=5.0):
    out, last = [], None
    for e in obs.get("events", []):
        if e["ago"] > secs:
            continue
        if last and e["kind"] == last["kind"] == "they_draw":
            continue
        out.append(f"{e['ago']:.1f} s ago: {e['text']}.")
        last = e
    return out[-8:]


def task_lines(obs, name):
    k = obs.get("task")
    if not k:
        return []
    if k["power"] >= 1:
        line = f"You are drawing your bow: full power ({k['s']:.1f} s drawn; holding longer adds nothing); you walk at a fifth while drawing."
    else:
        line = (f"You are drawing your bow: {k['s']:.1f} s, power {100 * k['power']:.0f}% - full in {k['to_full']:.1f} s"
                + (f" (a full arrow reaches them in {k['full_flight_s']:.2f} s)" if "full_flight_s" in k else "")
                + "; you walk at a fifth while drawing.")
    out = [line]
    if "clear" in k:
        out.append((f"The shot at {name} is clear" + (f" - only their {k['part']} shows past the cover, and the aim is on it"
                                                         if k.get("part") else "")
                    if k["clear"] else f"A block stands between your arrow and {name}")
                   + f": let go now and it flies {k['flight_s']:.2f} s, the aim {k['lead']:.1f} blocks ahead of them"
                   + {"steady": " (they move steadily)", "still": " (they stand still)", "weave": " (they weave: aimed at the middle)",
                      "turning": " (they are turning)", "slowed": " (they are drawing too: slow)", "airborne": " (mid-jump)"}.get(k.get("path"), "")
                   + ".")
    return out


def move_words(m, name):
    if not m:
        return []
    dd = m["d1"] - m["d0"]
    way = ("came closer" if dd < -1.5 else "went back" if dd > 1.5 else "kept the distance")
    how = ("stood still" if m["went"] < 0.8 else f"weaving ({m['turns']} turns)" if m["turns"] >= 2
           else "moving across your line" if m["across"] > 1.5 else "moving straight")
    return [f"Over the last {m['secs']:.0f} s {name} {way} ({m['d0']:.0f} -> {m['d1']:.0f} blocks), {how}"
            + ("; they are behind cover from you now." if m["covered"] else ".")]


def intent_words(obs, f, name):
    """What they look to be doing, from what they hold and how they move: a fact to weigh, not a certainty."""
    if not f:
        return []
    m, held = obs.get("their_move") or {}, f.get("holding") or ""
    d = v1.relative(obs, f["pos"])["dist"]
    closing = m and m["d0"] - m["d1"] > 2
    if held.endswith("_sword") and closing:
        return [f"{name} looks to be rushing you: sword in hand, closing in."]
    if held in ("lava_bucket", "water_bucket") and d < 9:
        return [f"{name} looks to be about to pour {'lava' if held == 'lava_bucket' else 'water'}: the bucket in hand, {d:.0f} blocks off."]
    if held == "bow" and m and m.get("covered"):
        return [f"{name} looks to be shooting from cover."]
    if held == "bow" and (f.get("using") or (m and m["went"] < 1.5)):
        return [f"{name} looks to be sniping: bow in hand" + (", drawing." if f.get("using") else ", hardly moving.")]
    if held in cu.BLOCKS:
        return [f"{name} looks to be building: blocks in hand."]
    if m and m["d1"] - m["d0"] > 3:
        return [f"{name} looks to be backing off."]
    return []


def now_lines(obs, mem, name, f):
    opp = [l for l in cu.opponent_lines(obs, name, None)]
    seen, opp2 = set(), []
    for l in opp:
        key = "drawing" if "drawing it" in l else l
        if key in seen:
            continue
        seen.add(key)
        opp2.append(l)
    opp2 += move_words(obs.get("their_move"), name) + intent_words(obs, f, name)
    if f and obs.get("their_draw_s") is not None:
        opp2.append(f"{name} has drawn their bow for {obs['their_draw_s']:.1f} s (full power at 1 s).")
    you = [l for l in cu.you_lines(obs, 2) if not l.startswith("Position") and not (l.startswith("You are drawing") and obs.get("task"))]
    you += engage_lines(obs) + [l for l in cover_lines(obs, name) if "drawing their bow" not in l]
    return opp2, you


def recent3(history):
    hist = list(history)[-6:]
    out, prev = [], None
    for h in hist:
        label = cu.LABELS.get(h["action"], h["action"])
        note = (h["result"].get("note") or "")[:90]
        if prev and prev[0] == label and not note:
            prev[1] += 1
            continue
        prev = [label, 1, note]
        out.append(prev)
    return [f"{lab}" + (f" x{n}" if n > 1 else "") + (f" -> {note}" if note else "") for lab, n, note in out[-3:]]


def dedupe(lines):
    """Each fact once: repeated lines go, and a plain 'holds a bow' goes when a 'drawing it' line says more."""
    drawing = any("drawing it" in l for l in lines)
    out, seen = [], set()
    for l in lines:
        key = re.sub(r"[^a-z ]", "", l.lower())
        if "drawing it" in l:
            key = "drawing"
        elif re.search(r" holds a bow\.?$", l) and drawing:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out


def step_state10(obs, mem, cfg, match, dt=None):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    f = cp.fighter(obs, name) if name else None
    opp, you = now_lines(obs, mem, name, f) if name else ([f"No opponent."], [])
    return "\n\n".join(x for x in [
        v2.PREAMBLE.format(name=obs["name"]) + " " + cs.PREAMBLE_HOLD,
        v2.section("Rules", RULES),
        v2.section("Techniques (use each when its moment comes)", TECHNIQUES_V10 + techniques_now(obs, f, name)),
        v2.section("The match", match_lines(match, name, obs) or ["The first round."]),
        v2.section("This round", this_round_lines(match, obs, name, f) + news_lines(mem.instruction)),
        v2.section("Last seconds", event_lines(obs) + task_lines(obs, name)[:1] or ["Nothing yet."]),
        v2.section("Now: " + (name or "no opponent"), dedupe(opp + closing_lines(obs, f, mem.prev_obs, dt, name) + threat_lines(obs, f, name, 1)[:2])),
        v2.section("Now: you", you + task_lines(obs, name)[1:] + reflex_lines(obs) + cu.feel_lines(obs)[:1]),
        v2.section("Your items", cu.item_lines(obs)),
        v2.section("Around you", cu.hazard_lines(obs)),
        v2.section("Your last choices", recent3(mem.history)),
    ] if x)


def options10(obs, name, history=()):
    q = options9(obs, name, history)
    opts = q["criteria"]
    for k in ("shoot_full", "shoot_hop", "bow_barrage", "quick_shot", "cover_shot", "draw_bow", "release_bow"):
        opts.pop(k, None)
    f = cp.fighter(obs, name) if name else None
    if not f:
        return q
    it, k = obs.get("items", {}), obs.get("task")
    d = v1.relative(obs, f["pos"])["dist"]
    if (obs.get("cover") or {}).get("covered"):   # a wall is already between you: no second one
        opts.pop("build_cover", None)
        opts.pop("build_window", None)
    if k:   # drawing: let go, hold, lower, jump over
        p = 100 * k["power"]
        clear = k.get("clear", True)
        opts["bow_release"] = (("let go now: a full-power arrow" if p >= 99 else
                                f"let go now: only {p:.0f}% drawn (full in {k.get('to_full', 0):.1f} s) - a weak arrow, slower, it drops more")
                               + (f", {k.get('flight_s', 0):.2f} s to them" if clear else
                                  f" - but a block stands between: the arrow would stop in it and not reach {name}") + " (technique 2)")
        if not clear:
            opts["wait"] = (f"keep the draw, the aim on {name}: they are behind cover now; the moment they come out you "
                            f"choose again" + ("" if p >= 99 else f" (full power in {k.get('to_full', 0):.1f} s)") + " (technique 2)")
        else:
            opts["wait"] = ("keep holding the full draw, the aim on them, and let go when the moment comes (technique 2)" if p >= 99
                            else f"keep drawing: full power in {k.get('to_full', 0):.1f} s, the aim stays on them (technique 2)")
        opts["bow_cancel"] = "lower the bow without shooting: full speed again, the sword or another tool next"
        if not clear and p >= 90:
            their = obs.get("their_draw_s")
            opts["jump_shot"] = ("jump and let go at the top of the jump, over the cover: the jump shows you for about "
                                 "0.3 s" + (f", and {name}'s bow has been drawn {their:.1f} s" + (" (full)" if their >= 1 else "")
                                             if their is not None else f"; {name} is not drawing") + " (technique 3)")
        for m in ("left", "right", "back", "forward", "zigzag", "zigzag_forward", "circle", "take_cover"):
            if m in opts:
                opts[m] = opts[m].split(" (")[0] + " - slowly, the bow still drawn"
    elif it.get("bow") and it.get("arrow") and d >= 4:
        opts["bow_draw"] = (f"draw your bow at {name}: the aim follows them; full power after 1 s; you choose when to let "
                            f"go ({it.get('arrow', 0)} arrows) (technique 2)")
    return q


# ---------------------------------------------------------------- v11: take the fight to a hider

TECHNIQUES_V11 = TECHNIQUES_V10 + [
    "6. They keep hiding (out of your line of fire for several seconds): take the fight to them in the gaps - "
    "right after they let go an arrow (a full draw takes them 1 s again) run in in a zigzag or step to an angle "
    "with a clear shot (find_angle); while their bow is drawn full, a step out is a step into their arrow, so stay "
    "behind your cover until they shoot - unless you lead the race. A stand-off where both hide ends only when "
    "someone moves: at long range an arrow flies most of a second and hits a runner seldom, so the side with health "
    "to spare breaks it - a zigzag run-in, an angle, a peek (peek_shot: lean out past your cover's edge, shoot, "
    "lean back) or a jump shot (jump_shot). Waiting only draws the round out.",
]


def race_words(obs, f, name):
    """How many hits end it either way, arrows and swords."""
    if not f or f.get("health") is None:
        return []
    mine, theirs = obs["health"], f["health"]
    a_me, a_them = math.ceil(theirs / ARROW), math.ceil(mine / ARROW)
    lead = "you lead" if a_me < a_them else "they lead" if a_me > a_them else "even"
    return [f"The race: {a_me} arrow{'s' if a_me > 1 else ''} of yours end{'' if a_me > 1 else 's'} {name} ({theirs:.0f} health); "
            f"{a_them} of theirs end{'s' if a_them == 1 else ''} you ({mine:.0f}) - {lead}."]


def their_bow_words(obs, name):
    """Where their bow is in its beat: drawn (how long), or loosed a moment ago (time to their next full draw)."""
    drawn = obs.get("their_draw_s")
    if drawn is not None:
        return (f"{name}'s bow is drawn full ({drawn:.1f} s): they shoot as you come out" if drawn >= 1
                else f"{name} is drawing ({drawn:.1f} s, full in {1 - drawn:.1f} s)")
    shot = [e["ago"] for e in obs.get("events", []) if e["kind"] == "their_shot"]
    if shot and min(shot) < 1.5:
        return f"{name} let go {min(shot):.1f} s ago: about {max(0.0, 1.0 - min(shot)):.1f} s before their next full draw"
    return f"{name}'s bow is not drawn"


def stall_lines(obs, name):
    """How long they have been out of your line of fire, and since the last arrow either way."""
    out = []
    hid = obs.get("their_hidden_s")
    if hid is not None and hid >= 1.5:
        out.append(f"{name} has been out of your line of fire for {hid:.0f} s; " + their_bow_words(obs, name) + ".")
    shots = [e["ago"] for e in obs.get("events", []) if e["kind"] in ("my_shot", "their_shot")]
    quiet = min(shots) if shots else None
    if quiet is None or quiet >= 5:
        out.append("No arrow either way for " + (f"{quiet:.0f} s." if quiet is not None else "the last 20 s."))
    return out


def step_state11(obs, mem, cfg, match, dt=None):
    t = mem.instruction["target"]
    name = t.get("name") if t["kind"] == "player" else None
    s = step_state10(obs, mem, cfg, match, dt).replace("\n".join(TECHNIQUES_V10), "\n".join(TECHNIQUES_V11))
    f = cp.fighter(obs, name) if name else None
    extra = (stall_lines(obs, name) + race_words(obs, f, name)) if name else []
    if extra:
        s = s.replace("== This round ==\n", "== This round ==\n" + "\n".join(extra) + "\n", 1)
    return s


def options11(obs, name, history=()):
    q = options10(obs, name, history)
    opts = q["criteria"]
    f = cp.fighter(obs, name) if name else None
    if f and obs.get("task") and obs.get("peek"):
        opts["peek_shot"] = (f"peek: lean out {obs['peek']['d']:.1f} blocks past your cover's edge with the bow drawn, "
                             f"let go the moment the shot is clear, and lean back - {their_bow_words(obs, name)} (technique 6)")
    hid = obs.get("their_hidden_s")
    if f and hid is not None and hid >= 3:
        bow = their_bow_words(obs, name)
        if obs.get("angle"):
            opts["find_angle"] = (f"step to an angle on {name}: {obs['angle']['d']:.1f} blocks to a spot with a clear shot "
                                  f"at them - {bow} (technique 6)")
        d = v1.relative(obs, f["pos"])["dist"]
        opts["zigzag_forward"] = (f"run at {name} in a zigzag, weaving left and right as you close in - about "
                                  f"{max(0.0, d - 3) / 5.6:.1f} s to sword range from {d:.0f} blocks; {bow}, and their "
                                  f"arrow takes about {d / 55:.1f} s to reach a runner who keeps changing course "
                                  f"(technique 6)")
    return q


# ---------------------------------------------------------------- team v1: 2v2, v10 plus a teammate
# The arena whispers "team fight - allies: <ally>; enemies: <e1>, <e2>: ...". The state adds your team (the
# teammate's place, health, what they hold, and - from the team's shared notes - whom they fight and what they
# just did) and the other enemy (and whether they are on your teammate); options switch the fight's focus.

TEAM_RE = re.compile(r"allies?:\s*([^;]*);\s*enemies?:\s*([^:]*?)\s*(?::|$)", re.I)


def parse_team(text):
    m = TEAM_RE.search(text or "")
    if not m:
        return None
    names = lambda s: [x.strip() for x in re.split(r",|\band\b", s) if x.strip()]
    return {"allies": names(m.group(1)), "enemies": names(m.group(2))}


TECHNIQUES_TEAM = TECHNIQUES_V10 + [
    "7. Two against one wins: fight the same enemy as your teammate when you can (focus_<name> switches your fight). "
    "An enemy on your teammate - hit that one. Keep out of your teammate's line of fire; the body holds back an arrow "
    "that would meet them. Low on health, fall back toward your teammate.",
]


def in_fight(obs, name):
    f = cp.fighter(obs, name)
    return f if f and f.get("pos") and (f.get("health") is None or f["health"] > 0) else None


def team_lines(obs, team, shared):
    out = []
    for a in team["allies"]:
        f = in_fight(obs, a)
        if not f:
            out.append(f"{a} (your teammate) is out of the fight.")
            continue
        r = v1.relative(obs, f["pos"])
        held = f.get("holding")
        line = (f"{a} (your teammate) is {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}; health "
                f"{f['health']:.0f}/20; holds {'a ' + cu.item_words(held) if held else 'nothing'}"
                + (", drawing it" if f.get("using") and held == "bow" else ""))
        st = shared.get(a)
        if st and time.time() - st["t"] < 3:
            line += f"; fighting {st['target']}" + (f", just chose: {st['action']}" if st.get("action") else "")
        out.append(line + ".")
    return out


def other_enemy_lines(obs, team, focus):
    out = []
    for e in team["enemies"]:
        if e == focus:
            continue
        f = in_fight(obs, e)
        if not f:
            out.append(f"{e} (the other enemy) is out of the fight.")
            continue
        r = v1.relative(obs, f["pos"])
        held = f.get("holding")
        near = []
        for a in team["allies"]:
            q = in_fight(obs, a)
            if q:
                d = math.dist((q["pos"]["x"], q["pos"]["z"]), (f["pos"]["x"], f["pos"]["z"]))
                if d < 6:
                    near.append(f"{d:.0f} blocks from {a}, your teammate")
        out.append(f"{e} (the other enemy) is {r['dist']:.0f} blocks away, {v1.direction_words(r['angle'])}; health "
                   f"{f['health']:.0f}/20; holds {'a ' + cu.item_words(held) if held else 'nothing'}"
                   + (", drawing it" if f.get("using") and held == "bow" else "") + ("; " + ", ".join(near) if near else "") + ".")
    return out


def step_state_team(obs, mem, cfg, match, team, shared, dt=None):
    focus = mem.instruction["target"].get("name")
    s = step_state10(obs, mem, cfg, match, dt).replace("\n".join(TECHNIQUES_V10), "\n".join(TECHNIQUES_TEAM))
    add = "\n\n".join(x for x in [v2.section("Your team", team_lines(obs, team, shared)),
                                  v2.section("The other enemy", other_enemy_lines(obs, team, focus))] if x)
    return s.replace("\n\n== Now: you ==", "\n\n" + add + "\n\n== Now: you ==", 1) if add else s


def options_team(obs, focus, team, shared, history=()):
    q = options10(obs, focus, history)
    opts = q["criteria"]
    ally_targets = {st["target"] for a, st in shared.items() if a in team["allies"] and time.time() - st["t"] < 3}
    for e in team["enemies"]:
        if e == focus:
            continue
        f = in_fight(obs, e)
        if not f:
            continue
        d = v1.relative(obs, f["pos"])["dist"]
        on_ally = any(in_fight(obs, a) and math.dist((in_fight(obs, a)["pos"]["x"], in_fight(obs, a)["pos"]["z"]),
                                                     (f["pos"]["x"], f["pos"]["z"])) < 6 for a in team["allies"])
        opts[f"focus_{e}"] = (f"switch your fight to {e} ({d:.0f} blocks away, {f['health']:.0f} health)"
                              + (" - your teammate is fighting them" if e in ally_targets else "")
                              + (" - they are on your teammate" if on_ally else "") + " (technique 7)")
    return q
