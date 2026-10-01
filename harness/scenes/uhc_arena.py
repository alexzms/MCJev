#!/usr/bin/env python3
"""Block UHC arena: a 1v1 to the death with a light Build UHC kit (a sword, a bow and 10 arrows, a lava and a
water bucket, a stack of cobblestone; no armour, no golden apples) and no natural regeneration. Players place blocks,
pour lava and water and shoot, so every round starts from a rebuilt arena and a clean player.

  scenes/uhc_arena.py build
  scenes/uhc_arena.py reset  --bot UJev --player Steve       one reset, no fight
  scenes/uhc_arena.py referee --bot UJev --player Steve      rounds until Ctrl-C: reset, "fight", watch, score

Layout (flat world, ground y=3), east of the sumo ring: a 61x61 light grey clay field inside an unbreakable barrier box (walls
and a ceiling 30 blocks up), with rises of one block (never more) and piles of rubble (cobblestone, mossy stone,
gravel) to hide behind; a mossy kerb along the edge and lantern pillars in the corners. Mirror-symmetric about the
centre line, so neither spawn is favoured.
The spawns (blue west, red east) are 40 blocks apart. The loser respawns on the spectator stand south of the box.

What a reset has to undo (the order matters):
  - fire on a player: nothing clears it by command in 1.11, so both are first dipped in a water cell;
  - dead players cannot be teleported: wait until whoever died has respawned (health score above 0);
  - the arena: lava, water, fire, placed and broken blocks, cobblestone and obsidian from lava meeting water, dug
    holes, burnt trees - all gone by refilling the whole box from the ground up;
  - entities: dropped kits, arrows, experience orbs, falling blocks - killed inside the box after the refill;
  - the players: inventory, effects (regeneration and absorption from golden apples, the countdown freeze),
    experience levels, health and hunger, survival mode (adventure cannot place blocks), respawn point (the lobby),
    and the sumo ring's respawn tag (so the sumo rule sets the ring again next time they stand on it);
  - the world: naturalRegeneration is a world-wide gamerule; the referee turns it off for the session and puts the
    old value back on exit, together with the player's old game mode.
"""
import random
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from common import server_rcon  # noqa: E402

UX, UZ = 5083, 5007          # arena centre block
H = 30                       # 61x61
X0, X1, Z0, Z1 = UX - H, UX + H, UZ - H, UZ + H
GROUND = 3                   # the floor's top; feet at 4
FLOOR = "stained_hardened_clay 8"   # light grey clay, under the colours (concrete needs 1.12)
FLOOR_TONES = [0, 8, 9, 3]   # the floor's 4x4 tiles: white, light grey, cyan, light blue clay (soft, close)
RISE_TONES = [4, 5, 6, 2]    # a rise (a mirrored pair shares one): yellow, lime, pink, magenta (no orange/red: lava)
TILE = 4
RISES = [(0, 0, 7, 1), (15, -15, 6, 1), (14, 16, 5, 1), (25, -22, 3, 1), (6, 24, 3, 1), (24, 12, 3, 1)]   # dx, dz, r, h
TOP = GROUND + 30            # highest open layer; barrier ceiling above
SPAWN = {"west": (UX - 20 + 0.5, UZ + 0.5, -90), "east": (UX + 20 + 0.5, UZ + 0.5, 90)}   # x, z, facing (MC yaw)
LOBBY = (UX, GROUND + 4, Z1 + 4)                                               # on the spectator stand
RINSE = (X0 + 1, GROUND + 1, Z0 + 1)                                            # the water cell that puts out fire
BOX = f"x={X0},y=0,z={Z0},dx={X1 - X0},dy={TOP + 1},dz={Z1 - Z0}"               # selector volume of the arena

KIT = [  # slot, item, count, data tag - a light kit, not the full Build UHC one: no armour, no golden apples
    ("slot.hotbar.0", "diamond_sword", 1, ""),
    ("slot.hotbar.2", "bow", 1, ""),
    ("slot.hotbar.3", "cobblestone", 64, ""),
    ("slot.hotbar.4", "water_bucket", 1, ""),
    ("slot.hotbar.5", "lava_bucket", 1, ""),
    ("slot.hotbar.8", "arrow", 10, ""),
]


def disc(cx, y, cz, r, block):
    """A rough disc (an octagon from two rectangles)."""
    k = max(0, round(r * 0.6))
    return [f"fill {cx - r} {y} {cz - k} {cx + r} {y} {cz + k} {block}", f"fill {cx - k} {y} {cz - r} {cx + k} {y} {cz + r} {block}"]


def mound(cx, cz, r, h):
    """A stepped hill of the floor's clay: one block per step, so it can be climbed by jumping."""
    out = []
    for i in range(h):
        out += disc(cx, GROUND + 1 + i, cz, max(1, round(r * (1 - i / h))), FLOOR)
    return out


def rubble(x, z, big=False):
    """A small pile of broken stone, with gravel spilled around it."""
    y = GROUND + 1
    c = [f"fill {x - 1} {GROUND} {z - 1} {x + 2} {GROUND} {z + 1} gravel",
         f"fill {x} {y} {z} {x + 1} {y} {z} cobblestone", f"setblock {x} {y} {z + 1} mossy_cobblestone"]
    if big:
        c += [f"setblock {x + 1} {y} {z - 1} stone 0", f"setblock {x} {y + 1} {z} mossy_cobblestone"]
    return c


def mirrored(fn, dx, dz, *args, **kw):
    """The same feature on both halves (x mirrored about the centre), so neither spawn is favoured."""
    return fn(UX - dx, UZ + dz, *args, **kw) + fn(UX + dx, UZ + dz, *args, **kw)


def terrain():
    """The arena's inside, from bedrock up: a clay field with low rises and some rubble."""
    y = GROUND + 1
    c = [f"fill {X0} {y0} {Z0} {X1} {min(TOP, y0 + 7)} {Z1} air" for y0 in range(y, TOP + 1, 8)]   # 32768-block limit
    c += [f"fill {X0} 1 {Z0} {X1} {GROUND} {Z1} {FLOOR}"]   # clay all the way down: a dug hole shows no dirt
    # kerb along the edge, lantern pillars in the corners
    for p, q in (((X0, Z0), (X1, Z0)), ((X0, Z1), (X1, Z1)), ((X0, Z0), (X0, Z1)), ((X1, Z0), (X1, Z1))):
        c.append(f"fill {p[0]} {y} {p[1]} {q[0]} {y} {q[1]} stonebrick 1")
    for x in (X0, X1):
        for z in (Z0, Z1):
            c += [f"fill {x} {y} {z} {x} {y + 2} {z} stonebrick 3", f"setblock {x} {y + 3} {z} sea_lantern"]
    # rises of one block (user: no height difference over one block): one in the middle, pairs on the sides
    for dx, dz, r, h in RISES:
        c += mound(UX, UZ, r, h) if dx == 0 else mirrored(mound, dx, dz, r, h)
    # rubble to hide behind
    for dx, dz, big in ((12, -2, True), (20, -9, False), (8, -19, False), (21, 6, False), (5, 11, True), (27, 22, False)):
        c += mirrored(rubble, dx, dz, big=big)
    for side, color in (("west", 11), ("east", 14)):
        c.append(f"setblock {int(SPAWN[side][0])} {GROUND} {UZ} stained_hardened_clay {color}")
    return c + colours()


def colours(seed=1926, floor_tones=None, rise_tones=None):
    """The floor in 4x4 tiles of soft tones and each rise in a brighter one, painted over the light grey clay (only
    it: the spawn marks and the gravel stay). The tiles are mirror-symmetric in x like the field and come from a
    fixed seed, so every copy and every rebuild looks the same (scenes/uhc_themes.py gives each arena its own)."""
    FLOOR_TONES_, RISE_TONES_ = floor_tones or FLOOR_TONES, rise_tones or RISE_TONES
    rnd, c = random.Random(seed), []
    paint = lambda tone: f"stained_hardened_clay {tone} replace {FLOOR}"
    for tz in range(0, 2 * H + 1, TILE):                        # rows north to south
        z1, z2 = Z0 + tz, min(Z1, Z0 + tz + TILE - 1)
        for t in range(0, H // TILE + 2):                       # tile t spans |dx| in [4t - 2, 4t + 1] (t=0: -1..1)
            tone = rnd.choice(FLOOR_TONES_)
            lo, hi = max(0, TILE * t - 2), min(H, TILE * t + 1)
            if lo > H:
                break
            spans = [(UX - hi, UX + hi)] if lo == 0 else [(UX - hi, UX - lo), (UX + lo, UX + hi)]
            c += [f"fill {a} {GROUND} {z1} {b} {GROUND} {z2} {paint(tone)}" for a, b in spans]
    for i, (dx, dz, r, h) in enumerate(RISES):
        tone = RISE_TONES_[i % len(RISE_TONES_)]
        for k in range(h):
            size = max(1, round(r * (1 - k / h)))
            for x in ([UX] if dx == 0 else [UX - dx, UX + dx]):
                c += disc(x, GROUND + 1 + k, UZ + dz, size, paint(tone))
    return c


def shell():
    """The unbreakable box (walls and ceiling of barrier) and the spectator stand: built once."""
    lx, ly, lz = LOBBY
    t = TOP + 1
    return [f"fill {X0 - 1} 1 {Z0 - 1} {X1 + 1} {t} {Z0 - 1} barrier", f"fill {X0 - 1} 1 {Z1 + 1} {X1 + 1} {t} {Z1 + 1} barrier",
            f"fill {X0 - 1} 1 {Z0 - 1} {X0 - 1} {t} {Z1 + 1} barrier", f"fill {X1 + 1} 1 {Z0 - 1} {X1 + 1} {t} {Z1 + 1} barrier",
            f"fill {X0 - 1} {t} {Z0 - 1} {X1 + 1} {t} {Z1 + 1} barrier",
            f"fill {lx - 5} {GROUND + 1} {lz - 1} {lx + 5} {ly - 1} {lz + 1} stonebrick 0",
            f"fill {lx - 4} {ly - 1} {lz - 1} {lx + 4} {ly - 1} {lz} stonebrick 3",
            f"fill {lx - 5} {ly} {lz - 1} {lx + 5} {ly} {lz - 1} cobblestone_wall 1",
            f"setblock {lx - 5} {ly} {lz} sea_lantern", f"setblock {lx + 5} {ly} {lz} sea_lantern"]


def build(rcon):
    from body import Body
    b = Body("ArenaBuilder")
    try:
        rcon.cmd(f"tp ArenaBuilder {UX}.5 {TOP + 6} {UZ}.5")
        rcon.cmd("effect ArenaBuilder levitation 30 255 true")
        time.sleep(5)
        for cmd in shell() + terrain():
            print(f"/{cmd[:72]:72s} -> {rcon.cmd(cmd)[:50]}")
    finally:
        rcon.cmd("effect ArenaBuilder clear")
        b.close()


# ---------------------------------------------------------------- players and rounds

def found(rcon, selector):
    return "Found" in rcon.cmd(f"testfor {selector}")


def gamemode_of(rcon, name):
    return next((m for m in range(4) if found(rcon, f"@a[name={name},m={m}]")), None)


def setup_scores(rcon):
    rcon.cmd("scoreboard objectives add uhcDeaths deathCount")
    rcon.cmd("scoreboard objectives add uhcHealth health")


def wait_alive(rcon, names, say_to=None, timeout=120):
    """Dead players cannot be teleported: wait for whoever died to respawn (their health score comes back)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        dead = [n for n in names if not found(rcon, f"@a[name={n},score_uhcHealth_min=1]")]
        if not dead:
            return True
        if say_to and say_to in dead:
            rcon.cmd(f'title {say_to} actionbar {{"text":"Respawn for the next round"}}')
        time.sleep(0.5)
    return False


def reset(rcon, bot, player):
    fighters = (bot, player)
    # 1. put out fire: dip both into a water cell inside the box (the refill below removes it)
    rx, ry, rz = RINSE
    rcon.cmd(f"setblock {rx} {ry} {rz} water")
    for n in fighters:
        rcon.cmd(f"tp {n} {rx + 0.5} {ry} {rz + 0.5}")
    time.sleep(0.4)
    # 2. clean the players
    for n in fighters:
        rcon.cmd(f"clear {n}")
        rcon.cmd(f"effect {n} clear")
        rcon.cmd(f"xp -100000L {n}")
        rcon.cmd(f"gamemode 0 {n}")
        rcon.cmd(f"spawnpoint {n} {LOBBY[0]} {LOBBY[1]} {LOBBY[2]}")
        rcon.cmd(f"scoreboard players tag {n} remove sumo")
        rcon.cmd(f"scoreboard players set {n} uhcDeaths 0")
    # 3. rebuild the arena, then clear what is lying around in it
    for cmd in terrain():
        rcon.cmd(cmd)
    rcon.cmd(f"kill @e[type=!player,{BOX}]")
    # 4. heal, feed, kit, places
    for n, side in ((bot, "east"), (player, "west")):
        rcon.cmd(f"effect {n} instant_health 1 10 true")
        rcon.cmd(f"effect {n} saturation 1 10 true")
        for slot, item, count, tag in KIT:
            rcon.cmd(f"replaceitem entity {n} {slot} {item} {count} 0 {tag}".rstrip())
        x, z, yaw = SPAWN[side]
        rcon.cmd(f"tp {n} {x} {GROUND + 1} {z} {yaw} 0")
    time.sleep(0.5)
    rcon.cmd(f"kill @e[type=!player,{BOX}]")  # anything the refill knocked loose


def freeze(rcon, names, on):
    for n in names:
        if on:
            rcon.cmd(f"effect {n} slowness 10 255 true")
            rcon.cmd(f"effect {n} jump_boost 10 250 true")   # 250 is -6 as a byte: no jumping
        else:
            rcon.cmd(f"effect {n} slowness 0")
            rcon.cmd(f"effect {n} jump_boost 0")


def referee(rcon, bot, player, pause=4.0, rounds=0, limit=300):
    setup_scores(rcon)
    regen = rcon.cmd("gamerule naturalRegeneration").split("=")[-1].strip() or "true"
    mode = gamemode_of(rcon, player)
    rcon.cmd("gamerule naturalRegeneration false")
    score = {bot: 0, player: 0}
    rnd, died = 0, []
    try:
        while not rounds or rnd < rounds:
            rnd += 1
            rcon.cmd(f"tell {bot} stop")
            if not wait_alive(rcon, died, say_to=player):
                print("a player did not respawn; stopping")
                break
            reset(rcon, bot, player)
            freeze(rcon, (bot, player), True)
            for n in (3, 2, 1):
                rcon.cmd(f'title {player} subtitle {{"text":"{n}"}}')
                rcon.cmd(f'title {player} title {{"text":"Block UHC - round {rnd}"}}')
                time.sleep(1)
            freeze(rcon, (bot, player), False)
            rcon.cmd(f'title {player} title {{"text":"Fight!"}}')
            rcon.cmd(f"tell {bot} fight {player}: kill {player} (Block UHC)")
            t0, died = time.time(), []
            while not died and time.time() - t0 < limit:
                time.sleep(0.2)
                died = [n for n in (bot, player) if found(rcon, f"@a[name={n},score_uhcDeaths_min=1]")]
            rcon.cmd(f"tell {bot} stop")
            winner = None if len(died) != 1 else (player if died[0] == bot else bot)
            if winner:
                score[winner] += 1
            msg = (f"Round {rnd}: {winner} wins" if winner else f"Round {rnd}: draw ({'both died' if died else 'time'})") \
                + f". {bot} {score[bot]} - {score[player]} {player}"
            rcon.cmd(f"say {msg}")
            print(msg, f"({time.time() - t0:.1f}s)", flush=True)
            time.sleep(pause)
    except KeyboardInterrupt:
        pass
    finally:
        rcon.cmd(f"tell {bot} stop")
        rcon.cmd(f"gamerule naturalRegeneration {regen}")
        if mode is not None:
            if player in died:
                wait_alive(rcon, [player], timeout=30)
            rcon.cmd(f"gamemode {mode} {player}")
        print("final:", score, f"(naturalRegeneration back to {regen})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "reset", "referee"])
    ap.add_argument("--bot", default="UJev")
    ap.add_argument("--player", default="Steve")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0: until Ctrl-C)")
    ap.add_argument("--limit", type=float, default=300, help="seconds before a round is a draw")
    ap.add_argument("--server-dir", default=os.path.dirname(ROOT))
    args = ap.parse_args()
    rcon = server_rcon(args.server_dir)
    if args.what == "build":
        build(rcon)
    elif args.what == "reset":
        setup_scores(rcon)
        reset(rcon, args.bot, args.player)
    else:
        referee(rcon, args.bot, args.player, rounds=args.rounds, limit=args.limit)


if __name__ == "__main__":
    main()
