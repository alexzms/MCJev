#!/usr/bin/env python3
"""Sumo arena: a small raised ring over a pool. Knock the other player off; whoever falls into the water loses.

  scenes/sumo_arena.py build
  scenes/sumo_arena.py rules                                    (re)place the standing rules only
  scenes/sumo_arena.py referee --bot Jev --player Steve     rounds until Ctrl-C: reset, "fight", watch, score

Layout (flat world, ground y=3): a 9x9 ring with cut corners (smooth sandstone, a hay-bale rim, a red clay centre
mark, white wool start marks 4 blocks apart) at y=13, so players stand at y=14, on a stone brick column with sea
lanterns underneath; a 21x21 pool one block deep below breaks every fall. Fists only; both players get
Resistance V so hits do no harm but knock back as usual.

Standing rules: on the Paper server the JevArena plugin keeps them (kill-zone in plugin/resources/config.yml), and
it also runs matches (/arena fight sumo Jev), so the referee here is only for a vanilla server. For vanilla,
`rules` places the same rules as command blocks hidden inside the column: whoever lands in the pool is killed
(survival and adventure players, not creative), and whoever has stood on the ring respawns at its middle.

The referee drives rounds over RCON only: it whispers the bot (/tell <bot> fight <player>), which reaches the bot
wherever its harness runs, and polls who has dropped below the ring (selector y/dy) to score the round.
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from common import server_rcon  # noqa: E402

CX, CZ = 5037, 5007          # ring centre block, next to the creeper arena (x 5000-5014)
RING_Y = 13                  # the ring's top block; feet at RING_Y + 1
R = 4                        # 9x9
START = {"west": (CX - 2 + 0.5, CZ + 0.5, -90), "east": (CX + 2 + 0.5, CZ + 0.5, 90)}  # x, z, facing (MC yaw)
POOL = 10                    # 21x21 pool


def commands():
    c = [
        f"fill {CX - POOL} 4 {CZ - POOL} {CX + POOL} {RING_Y + 6} {CZ + POOL} air",
        f"fill {CX - POOL - 1} 3 {CZ - POOL - 1} {CX + POOL + 1} 3 {CZ + POOL + 1} stonebrick 0",   # rim + floor
        f"fill {CX - POOL} 3 {CZ - POOL} {CX + POOL} 3 {CZ + POOL} water",
        f"fill {CX - POOL} 2 {CZ - POOL} {CX + POOL} 2 {CZ + POOL} sand",
        # column and lights
        f"fill {CX - 1} 3 {CZ - 1} {CX + 1} {RING_Y - 1} {CZ + 1} stonebrick 0",
        f"fill {CX - 1} 3 {CZ - 1} {CX + 1} 4 {CZ + 1} stonebrick 1",
        # the ring
        f"fill {CX - R} {RING_Y} {CZ - R} {CX + R} {RING_Y} {CZ + R} hay_block",
        f"fill {CX - R + 1} {RING_Y} {CZ - R + 1} {CX + R - 1} {RING_Y} {CZ + R - 1} sandstone 2",
    ]
    for sx in (-1, 1):  # cut corners for a rounder ring, sea lanterns under them
        for sz in (-1, 1):
            c += [f"setblock {CX + sx * R} {RING_Y} {CZ + sz * R} air",
                  f"setblock {CX + sx * (R - 1)} {RING_Y - 1} {CZ + sz * (R - 1)} sea_lantern"]
    c += [f"setblock {CX} {RING_Y} {CZ} stained_hardened_clay 14"]
    c += [f"setblock {int(x - 0.5)} {RING_Y} {int(z - 0.5)} wool 0" for x, z, _ in START.values()]
    return c


def rules():
    """Command blocks inside the column (x=CX, z=CZ, y 6-9): a repeating block that sets the respawn point of anyone
    on the ring who has not been set yet, tagging them so it happens once, and two that kill anyone in the pool."""
    ring = f"x={CX - R},y={RING_Y + 1},z={CZ - R},dx={2 * R},dy=2,dz={2 * R}"
    pool = f"x={CX - POOL},y=3,z={CZ - POOL},dx={2 * POOL},dy=1,dz={2 * POOL}"
    blocks = [("repeating_command_block 1", f"spawnpoint @p[{ring},tag=!sumo] {CX} {RING_Y + 1} {CZ}"),  # facing up
              ("chain_command_block 9", f"scoreboard players tag @p[{ring},tag=!sumo] add sumo"),     # up, conditional
              ("repeating_command_block 1", f"kill @a[{pool},m=0]"),
              ("repeating_command_block 1", f"kill @a[{pool},m=2]")]
    return [f'setblock {CX} {6 + i} {CZ} {b} replace {{Command:"{cmd}",auto:1b}}' for i, (b, cmd) in enumerate(blocks)]


def build(rcon):
    from body import Body
    b = Body("ArenaBuilder")
    try:
        rcon.cmd(f"tp ArenaBuilder {CX}.5 {RING_Y + 8} {CZ}.5")
        rcon.cmd("effect ArenaBuilder levitation 20 255 true")
        time.sleep(5)
        for cmd in commands():
            print(f"/{cmd[:72]:72s} -> {rcon.cmd(cmd)[:50]}")
    finally:
        rcon.cmd("effect ArenaBuilder clear")
        b.close()


def reset(rcon, bot, player):
    """Both on their start marks facing each other, full health, can't be hurt (knockback still applies)."""
    for name, side in ((bot, "east"), (player, "west")):
        x, z, yaw = START[side]
        rcon.cmd(f"tp {name} {x} {RING_Y + 1} {z} {yaw} 0")
        rcon.cmd(f"spawnpoint {name} {CX} {RING_Y + 1} {CZ}")
        rcon.cmd(f"effect {name} instant_health 1 10 true")
        rcon.cmd(f"effect {name} resistance 1000000 4 true")
        rcon.cmd(f"effect {name} saturation 1000000 1 true")
        rcon.cmd(f"clear {name}")


def fell(rcon, name):
    """True when the player is below the ring (falling, or in the pool). A volume needs x/z too: without them the
    box sits at the command sender's position, which for RCON is the world origin."""
    box = f"x={CX - 30},y=0,z={CZ - 30},dx=60,dy={RING_Y - 2},dz=60"
    return "Found" in rcon.cmd(f"testfor @a[name={name},{box}]")


def referee(rcon, bot, player, pause=3.0, rounds=0):
    score = {bot: 0, player: 0}
    rnd = 0
    try:
        while not rounds or rnd < rounds:
            rnd += 1
            rcon.cmd(f"tell {bot} stop")
            time.sleep(0.5)
            reset(rcon, bot, player)
            rcon.cmd(f'title {player} title {{"text":"Round {rnd}"}}')
            for n in (3, 2, 1):
                rcon.cmd(f'title {player} subtitle {{"text":"{n}"}}')
                rcon.cmd(f'title {player} title {{"text":"Round {rnd}"}}')
                time.sleep(1)
            rcon.cmd(f'title {player} title {{"text":"Fight!"}}')
            rcon.cmd(f"tell {bot} fight {player}: knock {player} off the ring")
            t0 = time.time()
            loser = None
            while not loser:
                time.sleep(0.2)
                for name in (bot, player):
                    if fell(rcon, name):
                        loser = name
                        break
                if time.time() - t0 > 180:
                    loser = "nobody (3 minutes)"
            winner = player if loser == bot else bot if loser == player else None
            if winner:
                score[winner] += 1
            rcon.cmd(f"tell {bot} stop")
            msg = (f"Round {rnd}: {winner} wins" if winner else f"Round {rnd}: draw, {loser}") + \
                f". {bot} {score[bot]} - {score[player]} {player}"
            rcon.cmd(f"say {msg}")
            print(msg, f"({time.time() - t0:.1f}s)", flush=True)
            time.sleep(pause)
    except KeyboardInterrupt:
        pass
    rcon.cmd(f"tell {bot} stop")
    print("final:", score)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["build", "rules", "unrules", "reset", "referee"])
    ap.add_argument("--bot", default="Jev")
    ap.add_argument("--player", default="Steve")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0: until Ctrl-C)")
    ap.add_argument("--server-dir", default=os.path.dirname(ROOT))
    args = ap.parse_args()
    rcon = server_rcon(args.server_dir)
    if args.what == "build":
        build(rcon)
    elif args.what == "unrules":
        for y in range(6, 10):
            print(rcon.cmd(f"setblock {CX} {y} {CZ} stonebrick 0"))
    elif args.what == "rules":
        for cmd in rules():
            print(f"/{cmd[:90]:90s} -> {rcon.cmd(cmd)[:50]}")
    elif args.what == "reset":
        reset(rcon, args.bot, args.player)
    else:
        referee(rcon, args.bot, args.player, rounds=args.rounds)


if __name__ == "__main__":
    main()
