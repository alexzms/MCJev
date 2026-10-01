# MCJev harness

The harness is everything between a Minecraft server and the decision model: a **body** that plays the game at
20 ticks a second, and a **head** that turns the fight into text and asks [DJev-serve](../serving) which move to make,
dozens of times a second. It is the code that played Block UHC, Sumo and 2v2 against the public in the MCJev release.

```
Minecraft server ──world──▶ body (Node.js, mineflayer)  ──state──▶ agent (Python)  ──POST /api/evaluate──▶ DJev-serve
       ▲                     every 50 ms tick: reads the        writes the fight as text,       one forward pass:
       └──── key presses ─── fight, aims, leads arrows,  ◀──── offers ~16-20 moves,     ◀──── a distribution over
                             dodges, never walks into lava      presses the top one             the moves
```

## Head and hands

A model call, however fast, is not a tick, so the work is split:

- **The body** (`body/body.js`, a [mineflayer](https://github.com/PrismarineJS/mineflayer) client driven over stdio
  JSON lines by `body.py`) runs the reflexes that must happen every 50 ms: aiming, leading arrows with ballistics and
  the measured latency, sidestepping arrows it sees coming (each tick's arrow move checked against your box the way
  the 1.11 server does), never walking into lava or off a ledge, dropping the bow for the sword up close, and turning
  a chosen move ("jump shot", "take cover", "pour lava at their feet", "circle", …) into key presses held over time.
- **The head** (`agent*.py` + `context*.py`) writes what a player could see as text (positions and movement, health
  both ways, blocks around, both hands, arrows in flight, the match so far, the last seconds as timed events), offers
  the moves that make sense now as one multiple-choice question, and sends it to the decision server. It asks again
  as soon as the answer is back, and only at the moments that matter while a draw is in progress (asking every step
  while holding a bow invites letting go early).

## Requirements

- Python 3.10+ (standard library only) and Node.js 22+.
- A Minecraft **Java 1.11.2** server (Paper or Spigot) with `online-mode=false` (the bots log in offline) and
  `pvp=true`. RCON is needed only for the arena referees in `scenes/`.
- A decision server: [DJev-serve](../serving) (or any server with the NanoJev `/api/evaluate` contract).
- Optional: a C compiler for the voxel camera (`--camera vox`), Chrome or Chromium for the browser camera.

```bash
cd harness/body && npm install && cd ..
echo 'JEV_URL=http://127.0.0.1:8765' > .env      # or the gateway; JEV_AUTH=user:key if the server needs a key
```

## Run a Jev

```bash
# Block UHC, the harness of the release (p_uhc_pro_v10)
python3 -u agent.py --name Jev1 --harness p_uhc_pro_v10 --vision off --host 127.0.0.1 --port 25565 --quiet
# Sumo
python3 -u agent.py --name Jev2 --harness p_sumo_v1 --vision off --host 127.0.0.1 --port 25565 --quiet
# 2v2: two of these, from the same checkout (teammates share notes through runs/team/)
python3 -u agent.py --name Jev3 --harness p_uhc_pro_team_v1 --vision off --host 127.0.0.1 --port 25565 --quiet
# a pool of four
./swarm.py 4 --prefix Jev -- --harness p_uhc_pro_v10 --vision off --quiet
# no decision server: random decisions, to check the body and the server
python3 -u agent.py --name Jev1 --harness p_uhc_pro_v10 --vision off --mock
```

Use names like `Jev`, `Jev<n>` or `JevPro<n>`: on a server where everyone is offline, that is how a bot tells other
bots from people (with a proxy that gives people online UUIDs, bots are told apart by their offline UUIDs instead).

### Starting a fight

A Block UHC Jev does not take orders from chat; it fights when the server tells it to, by a whisper from the
console or RCON. Any server can send these:

```text
tell Jev1 fight Steve: kill Steve (Block UHC)
tell Jev1 team fight - allies: Jev2; enemies: Steve, Alex: win the 2v2 (Block UHC)
tell Jev1 stop
tell Jev1 harness p_uhc_pro_v11          # switch the harness version in place
```

The harness also reads `sudden death …` and `… cannot place blocks` whispers (news in the state; block options
removed) and `Round N: X wins (…). A n - m B` lines (the match section of the state). A whisper counts as the
server's only if it comes from the console or RCON. The Sumo Jev also follows chat instructions that it judges to be
meant for it, and fights back when hit.

### A plugin-free arena

The public server ran its own Paper plugin for the lobby, matchmaking, kits and resets. Without it, `scenes/` builds
an arena on a flat world over RCON and referees rounds: it resets the arena and both fighters, gives the kit, sends
the `fight` whisper, watches for a death and keeps the score.

```bash
export RCON_HOST=127.0.0.1 RCON_PORT=25575 RCON_PASSWORD=...   # or --server-dir pointing at the server
python3 scenes/uhc_arena.py build
python3 scenes/uhc_arena.py referee --bot Jev1 --player Steve   # rounds until Ctrl-C
python3 scenes/sumo_arena.py build && python3 scenes/sumo_arena.py referee --bot Jev2 --player Steve
```

### Watching it think

Every agent writes `runs/<start>-<name>.jsonl`, one record per decision with the full text state it read and the
complete answer distribution. `python3 console.py` serves a local viewer of those logs at http://127.0.0.1:8770
(with a command bar that writes to `runs/inbox.jsonl`, which every running agent reads).

## Harness versions

All versions stay selectable with `--harness`; each class docstring in `agent_*.py` says what changed and why.

| Version | What it is |
|---|---|
| `v1`, `v2`, `c1`–`c3` | instruction following and survival: walk, follow, mine, run from mobs; long-term memory (`context*.py`, `memory.py`) |
| `p1`, `p_sumo_v1` | PvP on a sumo ring |
| `p_uhc_v1`, `p_uhc_v2` | first Block UHC harnesses |
| `p_uhc_pro_v1` … `v4` | Block UHC with the body's help: engage styles, the race of sword trades, lava traps, the bow on runners, the sword's recharge timing |
| `p_uhc_pro_v5` … `v7` | techniques distilled from a strong human player, each option naming its technique; full-power shots with lead; the lava guard; the weapon rule |
| `p_uhc_pro_v8`, `v9` | shots between hops, a least-squares lead, the arrow dodge; cover (`take_cover`, `cover_shot`), footwork up close |
| **`p_uhc_pro_v10`** | **the release harness**: a layered state (fixed rules and techniques, the match so far, this round, the last seconds as timed events, "now") and the draw as a task |
| `p_uhc_pro_v11` | v10 plus taking the fight to an opponent who keeps hiding |
| `p_uhc_pro_team_v1` | 2v2: v10 plus a teammate (shared notes, focus on the nearest enemy, no arrows through an ally) |
| `p_uhc_pro_v8_official` | v8 with decisions from TypeSafe's hosted Jev API instead of DJev (needs `TYPESAFE_API_KEY`) |
| `p_uhc_raw_v1` … `v3` | raw controls (keys, mouse, buttons), no aim help: every step Jev picks the keys held, a mouse move and a button |
| `p_uhc_vis_*` | vision: each step a 256×256 picture (`--camera vox --camera-size 256x256`), with or without coordinates |

## Configuration

| | Default | |
|---|---|---|
| `JEV_URL` / `--jev-url` | (required unless `--mock`) | decision server base URL; the agent posts to `{JEV_URL}/api/evaluate` |
| `JEV_AUTH` / `--jev-auth` | none | `user:key`, sent as HTTP basic auth |
| `--harness` | `v1` | the harness version (see above) |
| `--host`, `--port` | `127.0.0.1`, `25565` | the Minecraft server |
| `--vision` | `on` | `off` for the text harnesses; `on` starts a browser camera |
| `--camera`, `--camera-size` | `browser`, unset | `vox` + `256x256` for the vision harnesses (voxcam is compiled with `cc` on first use) |
| `TYPESAFE_API_KEY` | none | only for `p_uhc_pro_v8_official` |

Environment variables can also go in `harness/.env`. `python3 agent.py --help` lists every flag.

## Files

| File | |
|---|---|
| `agent.py` | entry point: the agent loop, run logs, the harness table, the server's harness switch |
| `agent_v2.py`, `agent_survive.py`, `agent_pvp.py`, `agent_uhc.py`, `agent_uhc_pro.py` | harness classes (each version subclasses an earlier one) |
| `context*.py` | the text state and the options of each harness family (`context_uhc_pro.py` for the Pro line) |
| `jev.py` | the `/api/evaluate` client (and a mock) |
| `body.py`, `body/` | the Node.js body: `body.js` (observation, reflexes, actions), `forward.js`, `vis_aim.js`, cameras (`voxcam.js` + `voxcam/`, `camera.js`, `viewer_server.js`) |
| `memory.py`, `knowledge.md` | long-term memory and the Minecraft facts recalled at fight start |
| `swarm.py` | start a pool of agents |
| `console.py`, `console.html` | local viewer for run logs, with a command bar |
| `scenes/uhc_arena.py`, `scenes/sumo_arena.py`, `bench/common.py`, `bench/tester.js` | arena builders and referees over RCON |
