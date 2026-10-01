"""Everything Jev reads: observation + memory -> state text and questions.

Jev cannot do arithmetic, so every geometric fact is computed here and written as words: the
direction of a player relative to where the bot faces, what lies straight ahead, an egocentric
top-down map. Jev only has to read the page and pick a control.

Conventions (mineflayer): yaw 0 faces north (-z) and grows counter-clockwise seen from above;
forward = (-sin yaw, -cos yaw), right = (cos yaw, -sin yaw); pitch > 0 looks up.
"""
import collections, math, time

AHEAD, BEHIND, SIDE = 6, 2, 4  # extent of the egocentric map, in blocks

PREAMBLE = (
    "You are {name}, a bot playing Minecraft (Java 1.11.2) on a server together with human players. "
    "You control your character like a player at a keyboard and mouse: every step you choose one control "
    "(walk, jump, turn or tilt your view, left-click, right-click). You cannot type in chat. Humans give you "
    "instructions in chat; your job is to carry out the current instruction.")

LEGEND = ("Legend: @ you, P player, . floor, # wall, ^ 1-block step (jump to climb), _ drop (no floor), "
          "~ water, ! lava, D door, b button, L lever, ? not loaded")

COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def action_options(cfg, toward=None, who=None):
    """Control id -> what it does. The ids are what body.js executes.

    toward ("left"/"right"): the side the target is on, marked on the turn options themselves. With "Tester is
    to the LEFT of your crosshair" in the state Jev still picked a right turn in 12 of 16 cases of one run; with
    "- toward Tester" / "- away from Tester" on the options it picked correctly in 15 of 16."""
    step = round(cfg.move_ms / 1000 * 3, 1)  # ~3 blocks/s effective walking speed incl. acceleration
    opts = {
        "forward": f"walk forward (about {step} blocks in the direction you face)",
        "back": f"walk backward (about {step} blocks)",
        "left": "step sideways to your left",
        "right": "step sideways to your right",
        "jump_forward": "jump forward (climbs a 1-block step)",
        "turn_left": f"turn your view {cfg.turn_deg:g}° to the left",
        "turn_right": f"turn your view {cfg.turn_deg:g}° to the right",
        "turn_left_small": f"turn your view slightly, {cfg.fine_turn_deg:g}° to the left (for aiming)",
        "turn_right_small": f"turn your view slightly, {cfg.fine_turn_deg:g}° to the right (for aiming)",
        "look_up": f"tilt your view {cfg.pitch_deg:g}° up",
        "look_down": f"tilt your view {cfg.pitch_deg:g}° down",
        "attack": "left-click: hit the player or mob under the crosshair (it must be within 3.5 blocks)",
        "use": "right-click the block under the crosshair (press a button, pull a lever, open a door)",
        "wait": "do nothing this step",
    }
    if toward in ("left", "right") and who:
        away = "right" if toward == "left" else "left"
        for k in opts:
            if k.startswith(f"turn_{toward}"):
                opts[k] += f" - toward {who}"
            elif k.startswith(f"turn_{away}"):
                opts[k] += f" - away from {who}"
    return opts


# ---------------------------------------------------------------- geometry -> words

def basis(yaw):
    return (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))


def compass(yaw):
    return COMPASS[round((-math.degrees(yaw)) % 360 / 45) % 8]


def pitch_words(pitch):
    deg = math.degrees(pitch)
    if abs(deg) < 5:
        return "looking straight ahead (level)"
    return f"looking {abs(deg):.0f}° {'up' if deg > 0 else 'down'}"


def relative(obs, pos):
    """Where pos is as seen from the bot: distance, signed angle (+ = right), ahead/right components."""
    (fx, fz), (rx, rz) = basis(obs["yaw"])
    dx, dz = pos["x"] - obs["pos"]["x"], pos["z"] - obs["pos"]["z"]
    ahead, right = dx * fx + dz * fz, dx * rx + dz * rz
    return {"dist": math.hypot(dx, dz), "angle": math.degrees(math.atan2(right, ahead)),
            "dy": pos["y"] - obs["pos"]["y"], "ahead": ahead, "right": right}


def direction_words(angle):
    a, side = abs(angle), ("right" if angle > 0 else "left")
    clock = round(angle / 30) % 12 or 12
    if a < 10:
        words = "straight ahead"
    elif a < 45:
        words = f"ahead, slightly to your {side}"
    elif a < 80:
        words = f"ahead-{side}"
    elif a < 110:
        words = f"to your {side}"
    elif a < 160:
        words = f"behind you on the {side}"
    else:
        return f"behind you ({a:.0f}°, {clock} o'clock)"
    return f"{words} ({a:.0f}° {side}, {clock} o'clock)"


def height_words(dy):
    if abs(dy) < 0.5:
        return "same height as you"
    return f"{abs(dy):.0f} block{'s' if abs(dy) >= 1.5 else ''} {'higher' if dy > 0 else 'lower'}"


def aim_words(obs, p, r):
    """Where a player's body is relative to the crosshair; exact degrees, because hitting needs precision."""
    dist = max(r["dist"], 0.3)
    elev = math.degrees(math.atan2(p["pos"]["y"] + 0.9 - (obs["pos"]["y"] + 1.62), dist))
    h, v = r["angle"], elev - math.degrees(obs["pitch"])
    half_w, half_h = math.degrees(math.atan2(0.3, dist)), math.degrees(math.atan2(0.9, dist))
    if abs(h) <= half_w and abs(v) <= half_h:
        return "your crosshair is on them"
    parts = [f"{abs(h):.0f}° to the {'right' if h > 0 else 'left'}" if abs(h) > half_w else None,
             f"{abs(v):.0f}° {'above' if v > 0 else 'below'}" if abs(v) > half_h else None]
    return f"the crosshair misses them: they are {' and '.join(x for x in parts if x)} of it"


def player_line(obs, p, instructor):
    tag = " (gave you the current instruction)" if p["name"] == instructor else ""
    if not p["pos"]:
        return f"{p['name']}{tag}: position unknown (too far away to see)."
    r = relative(obs, p["pos"])
    where = "right next to you" if r["dist"] < 2 else f"{r['dist']:.1f} blocks away"
    line = f"{p['name']}{tag}: {where}, {direction_words(r['angle'])}, {height_words(r['dy'])}."
    if r["dist"] < 2:  # "right next to you" alone read p(done) 0.00-1.00 on 8 such states; with this, 1.00 on all
        line += " You have reached them."
    if r["dist"] <= 6:
        line += f" Aim: {aim_words(obs, p, r)}."
    return line


# ---------------------------------------------------------------- egocentric map

def cell_symbol(col):
    below, feet, head, above = col
    if "?" in (feet, head):
        return "?"
    for block, sym in (("d", "D"), ("b", "b"), ("v", "L")):
        if block in (feet, head):
            return sym
    if head == "s":
        return "#"
    if feet == "s":
        return "#" if above == "s" else "^"
    if "l" in (feet, head, below):
        return "!"
    if "w" in (feet, head, below):
        return "~"
    return "." if below == "s" else "_"


def egocentric_map(obs, show_players=True):
    """Rows from AHEAD blocks in front (top) to BEHIND blocks behind (bottom); columns left -> right."""
    g, R = obs["grid"], obs["grid"]["r"]
    px, pz = obs["pos"]["x"], obs["pos"]["z"]
    bx, bz = math.floor(px), math.floor(pz)
    (fx, fz), (rx, rz) = basis(obs["yaw"])
    cells = []
    for i in range(AHEAD, -BEHIND - 1, -1):
        row = []
        for j in range(-SIDE, SIDE + 1):
            dx = math.floor(px + i * fx + j * rx) - bx
            dz = math.floor(pz + i * fz + j * rz) - bz
            row.append(cell_symbol([layer[dz + R][dx + R] for layer in g["layers"]]))
        cells.append(row)
    for p in obs["players"] if show_players else []:
        if p["pos"]:
            r = relative(obs, p["pos"])
            i, j = round(r["ahead"]), round(r["right"])
            if -BEHIND <= i <= AHEAD and -SIDE <= j <= SIDE:
                cells[AHEAD - i][j + SIDE] = "P"
    cells[AHEAD][SIDE] = "@"
    return cells


def ahead_line(cells):
    names = {"#": "a wall", "^": "a 1-block step up (jump forward to climb it)", "_": "a drop with no floor",
             "~": "water", "!": "lava", "D": "a door", "b": "a button", "L": "a lever", "?": "unloaded terrain"}
    for i in range(1, AHEAD + 1):
        sym = cells[AHEAD - i][SIDE]
        if sym in ".P":
            continue
        where = "right in front of you" if i == 1 else f"{i} blocks in front of you"
        return f"Straight ahead: {names[sym]} {where}."
    return f"Straight ahead: open floor for at least {AHEAD} blocks."


def map_lines(obs, show_players=True):
    cells = egocentric_map(obs, show_players)
    rows = []
    for k, row in enumerate(cells):
        i = AHEAD - k
        label = f"{i} ahead " if i > 0 else ("you     " if i == 0 else f"{-i} behind")
        rows.append(f"{label:>8} | {' '.join(row)}")
    return ["(top-down, rotated so the top row is the direction you face; left column = your left)",
            *rows, LEGEND, ahead_line(cells)]


# ---------------------------------------------------------------- memory -> words

def ago(t_ms, now=None):
    s = max(0, (now or time.time()) - t_ms / 1000)
    return f"{s:.0f}s ago" if s < 120 else f"{s / 60:.0f}min ago"


def chat_lines(chat, me, limit=8):
    if not chat:
        return ["(no messages yet)"]
    return [f"[{ago(m['t'])}] {m['username'] + ' (you)' if m['username'] == me else m['username']}"
            f"{' (whisper)' if m.get('whisper') else ''}{' (via console)' if m.get('via') == 'console' else ''}"
            f": {m['message']}" for m in list(chat)[-limit:]]


def history_line(n, h, opts, show_target=True):
    a, r = h["action"], h["result"]
    if a in ("forward", "back", "left", "right", "jump_forward"):
        effect = "did not move (blocked)" if r["moved"] < 0.2 else f"moved {r['moved']:.1f} blocks"
        if r["dy"] >= 0.5:
            effect += ", climbed up"
        elif r["dy"] <= -0.5:
            effect += ", dropped down"
    elif a.startswith("turn_"):
        effect = f"now facing {compass(r['yaw'])}"
    elif a in ("look_up", "look_down"):
        effect = f"now {pitch_words(r['pitch'])}"
    elif a in ("use", "attack"):
        effect = r["note"]
    else:
        effect = "waited"
    if h.get("target") and show_target:
        name, d0, d1, a0, a1 = h["target"]
        if a.startswith(("turn_", "look_")):
            side = lambda x: f"{abs(x):.0f}° {'right' if x > 0 else 'left'}"
            effect += f"; {name} was {side(a0)}, now {side(a1)}"
        else:
            effect += f"; distance to {name} {d0:.1f} -> {d1:.1f}"
    seen = ""
    if h.get("sees"):
        who, label = h["sees"]
        seen = f"({who} {label}) "
    return f"{n}. {seen}{opts[a].split(' (')[0]} -> {effect}"


# ---------------------------------------------------------------- states

def crosshair_line(obs, show_entity=True):
    ent, cur = obs.get("cursor_entity") if show_entity else None, obs["cursor"]
    if ent:
        what = f"the player {ent['name']}" if ent["type"] == "player" else ent["name"]
        return (f"Crosshair: on {what}, {ent['distance']} blocks away"
                f"{' (within reach: left-click hits it)' if ent['distance'] <= 3.5 else ' (too far to hit)'}.")
    if cur:
        return (f"Crosshair: on {cur['name']} block, {cur['distance']} blocks away"
                f"{' (within reach)' if cur['distance'] <= 4.5 else ' (too far to use)'}.")
    return "Crosshair: nothing within 6 blocks."


def you_lines(obs, show_entity=True):
    p = obs["pos"]
    return [f"Position x={p['x']:.1f} y={p['y']:.1f} z={p['z']:.1f}, "
            f"{'on the ground' if obs['on_ground'] else 'in the air'}, facing {compass(obs['yaw'])}, "
            f"{pitch_words(obs['pitch'])}. Health {obs['health']:.0f}/20.", crosshair_line(obs, show_entity)]


def players_lines(obs, instructor=None, positions=True):
    if not obs["players"]:
        return ["(no other players online)"]
    ordered = sorted(obs["players"], key=lambda p: p["name"] != instructor)
    if not positions:
        names = ", ".join(p["name"] + (" (gave you the current instruction)" if p["name"] == instructor else "")
                          for p in ordered)
        return [f"Online: {names}.",
                "You do not know where they are unless you can see them in Image 1: a player looks like a "
                "person with their name floating above the head.",
                "To find someone who is not in view, turn toward where you last saw them, or if you never "
                "saw them, keep turning the same way to look all around. Once they are in view but not under "
                "the crosshair, turn toward them (only slightly, 10°, if they are close to the crosshair). "
                "When they are under the crosshair, walk forward. Do not walk while they are out of view."]
    return [player_line(obs, p, instructor) for p in ordered]


def progress_line(history):
    """What the steps so far achieved, summed up, so Jev does not have to count through the history."""
    hits, used, walked = collections.Counter(), collections.Counter(), 0.0
    for h in history:
        note = h["result"].get("note", "")
        if note.startswith("hit "):
            hits[note[4:]] += 1
        elif note.startswith("right-clicked "):
            used[note[14:]] += 1
        walked += h["result"].get("moved", 0)
    done = [f"hit {who} {n} time{'s' if n > 1 else ''}" for who, n in hits.items()]
    done += [f"right-clicked {what} {n} time{'s' if n > 1 else ''}" for what, n in used.items()]
    if walked >= 0.5:
        done.append(f"walked {walked:.0f} blocks")
    return "Done so far on this instruction: " + ("; ".join(done) if done else "nothing yet") + "."


def instruction_lines(mem, cfg):
    ins = mem.instruction
    kind = ("This is a one-time request: it is complete as soon as you have done it once."
            if ins["once"] else "This is an ongoing request: keep doing it until you are told to stop.")
    limit = f" (limit {cfg.max_steps})" if ins["once"] else ""
    lines = [f'{ins["from"]} (human) said {ago(ins["t"])}: "{ins["text"]}"', kind,
             f"You have taken {ins['steps']} steps on it so far{limit}.", progress_line(mem.history)]
    for old in list(mem.finished)[-2:]:
        lines.append(f'Earlier: "{old["text"]}" from {old["from"]} - {old["outcome"]} after {old["steps"]} steps.')
    return lines


def section(title, lines):
    return f"== {title} ==\n" + "\n".join(lines)


VIEW = ("Image 1 is your first-person view right now. The white cross in the middle is your crosshair: "
        "whatever is under it is what left-click hits and right-click uses.")


# Pass 1 sees only the image and this. With the full context (history saying "Tester not in view" over and
# over) Jev answered from the text: p(sees)=0.002 with Tester 2 blocks ahead under the crosshair.
LOOK_STATE = ("Image 1 is what you see right now in Minecraft, in first person. The white cross in the middle is your "
              "crosshair. Players look like people with their name floating above the head.")

WHERE = {"far_left": ("near the left edge of the image", -38), "left": ("left of the cross", -18),
         "center": ("on or just next to the cross", 0), "right": ("right of the cross", 18),
         "far_right": ("near the right edge of the image", 38)}
# "How far away does X look" came out biased far (1 block away read "a few blocks"). How much of the image
# a player covers is a question about the picture itself: at eye height 1.62 and 75° vertical fov a player
# 1 block away runs off the bottom edge, 2.3 blocks covers about half the height, 4.6 about 30%, 8 about 17%.
DISTANCE = {"close": ("big: taller than half of the image, or cut off by the bottom edge", 2,
                      "very close, within about 2 blocks"),
            "near": ("medium: between a quarter and a half of the image's height", 4, "a few blocks away"),
            "far": ("small: less than a quarter of the image's height", 10, "far away")}


def look_questions(who):
    """Pass 1 of a vision step: what is on the screen."""
    return {
        "sees": {"type": "boolean", "instructions": f"Can you see {who} in Image 1?"},
        "where": {"type": "choice", "instructions": f"Where in Image 1 is {who}, left to right?",
                  "criteria": {**{k: v[0] for k, v in WHERE.items()}, "none": f"{who} is not visible"}},
        "distance": {"type": "choice", "instructions": f"How big is {who} in Image 1, from feet to head?",
                     "criteria": {**{k: v[0] for k, v in DISTANCE.items()}, "none": f"{who} is not visible"}},
    }


def read_look(look):
    """-> (seen, estimated angle right of the crosshair in degrees, estimated distance in blocks)."""
    best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
    where, dist = best("where"), best("distance")
    if look["sees"]["p_true"] < 0.5 or where == "none":
        return False, None, None
    return True, WHERE[where][1], DISTANCE.get(dist, (None, 5))[1]


def project(obs, angle, dist):
    """World x/z of a point seen `angle` degrees right of where the bot faces, `dist` blocks away."""
    (fx, fz), (rx, rz) = basis(obs["yaw"])
    a = math.radians(angle)
    return {"x": obs["pos"]["x"] + dist * (math.cos(a) * fx + math.sin(a) * rx), "y": obs["pos"]["y"],
            "z": obs["pos"]["z"] + dist * (math.cos(a) * fz + math.sin(a) * rz)}


def view_lines(obs, look, who, sight, scanned, step):
    """Pass 1's answers, plus where the target was last seen (remembered via the bot's own movement)."""
    seen, angle, dist = read_look(look)
    if seen:
        best = lambda q: max(look[q]["probabilities"], key=look[q]["probabilities"].get)
        where = {"far_left": "far to the LEFT, near the edge of your view", "left": "to the LEFT of your crosshair",
                 "center": "right under your crosshair", "right": "to the RIGHT of your crosshair",
                 "far_right": "far to the RIGHT, near the edge of your view"}[best("where")]
        if best("distance") == "close":
            # "very close, within about 2 blocks" read p(done)=0.06-0.14 for "come to me"; saying plainly that
            # they are next to you reads 1.00 (and 0.00 on a far state).
            front = "right in front of you" if best("where") == "center" else where
            return [f"You looked at your screen just now: {who} is right next to you, {front} "
                    f"(within about 2 blocks): you have reached them."]
        dist = DISTANCE[best("distance")][2] if best("distance") in DISTANCE else "at an unclear distance"
        return [f"You looked at your screen just now: {who} is in your view, {where}, {dist}."]
    if sight:
        r = relative(obs, sight["est"])
        ago = step - sight["step"]
        spot = f"{direction_words(r['angle'])}, " + ("right next to you" if r["dist"] < 1.5 else
                                                     f"about {r['dist']:.0f} blocks away")
        return [f"You looked at your screen just now: {who} is NOT in your view.",
                f"You last saw {who} {ago} step{'s' if ago != 1 else ''} ago; judging by how you have moved since, "
                f"that spot is now {spot}."]
    return [f"You looked at your screen just now: {who} is NOT in your view, and you have not seen them yet.",
            f"Since starting to look, you have turned {scanned:.0f}° in total."]


def step_state(obs, mem, cfg, view=None):
    """The state for one control step while an instruction is active.

    cfg.vision: "off" text only; "on" Image 1 plus everything below; "only" Image 1, and nothing derived
    from other players' coordinates (where they are, aim offsets, P on the map, distances in the history)."""
    opts = action_options(cfg)
    hist = list(mem.history)[-cfg.history:]
    coords = cfg.vision != "only"
    return "\n\n".join(x for x in [
        PREAMBLE.format(name=obs["name"]),
        section("What you see", view or [VIEW]) if cfg.vision != "off" else None,
        section("Current instruction", instruction_lines(mem, cfg)),
        section("You", you_lines(obs, show_entity=coords)),
        section("Players", players_lines(obs, mem.instruction["from"], positions=coords)),
        section("Around you", map_lines(obs, show_players=coords)),
        section("Recent chat", chat_lines(mem.chat, obs["name"])),
        section("Your last actions (oldest first)",
                [history_line(k + 1, h, opts, show_target=coords) for k, h in enumerate(hist)] or ["(none yet)"]),
    ] if x)


def target_side(obs, cfg, who, look=None, sight=None):
    """Which way to turn toward the target, from what the bot knows in this mode (None: ahead or unknown)."""
    if cfg.vision == "off":
        p = next((p for p in obs["players"] if p["name"] == who and p["pos"]), None)
        if not p:
            return None
        r = relative(obs, p["pos"])
        tolerance = max(5.0, math.degrees(math.atan2(0.3, max(r["dist"], 0.3))))
        return None if abs(r["angle"]) <= tolerance else ("right" if r["angle"] > 0 else "left")
    if look:
        seen, angle, _ = read_look(look)
        if seen:
            return None if angle == 0 else ("right" if angle > 0 else "left")
    if sight:
        angle = relative(obs, sight["est"])["angle"]
        return None if abs(angle) <= 10 else ("right" if angle > 0 else "left")
    return None


def step_questions(cfg, instructor, toward=None):
    # Order matters: the canvas slots are denoised together. With the bot right next to the player, the
    # same done question read p=0.000 placed after the action slot and p=1.000 placed first (2026-09-25).
    # With vision the step is two passes: look_questions on the image first, then these, reading its answers.
    return {
        "done": {"type": "boolean",
                 "instructions": "Is the current instruction already fully carried out, so you can stop now?"},
        "action": {"type": "choice",
                   "instructions": "Which control should you use now to make progress on the current instruction?",
                   "criteria": action_options(cfg, toward, instructor)},
    }


def message_state(obs, mem, msg):
    """The state for deciding what a new chat message from a human means for the bot."""
    current = (f'Your current instruction: "{mem.instruction["text"]}" from {mem.instruction["from"]}.'
               if mem.instruction else "You have no instruction right now; you are standing idle.")
    return "\n\n".join([
        PREAMBLE.format(name=obs["name"]),
        section("Players", players_lines(obs, msg["username"])),
        section("Recent chat", chat_lines(mem.chat, obs["name"])),
        current,
        f'New message from {msg["username"]} (human): "{msg["message"]}"',
    ])


MESSAGE_QUESTIONS = {
    "is_instruction": {"type": "boolean",
                       "instructions": "Is the new message asking you (the bot) to do something in the game?"},
    # "only ... without asking for any new action": the broader "telling you to stop, wait, or cancel"
    # read "stay away from me" as a stop (0.87); this one reads it as an instruction (stop 0.03).
    "is_stop": {"type": "boolean",
                "instructions": "Does the new message only tell you to stop or cancel what you are doing, "
                                "without asking for any new action?"},
    "once": {"type": "boolean",
             "instructions": "Is it a one-time request that is finished once you have done it, like 'come here', "
                             "'hit me' or 'press the button' (as opposed to something to keep doing until told to "
                             "stop, like 'follow me', 'keep hitting me' or 'stay next to me')?"},
}
