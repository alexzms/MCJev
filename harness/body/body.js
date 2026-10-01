#!/usr/bin/env node
// Jev agent body: a mineflayer bot driven over stdio, one JSON object per line.
//
//   in : {"id": 1, "cmd": "observe", "since": 0}
//        {"id": 2, "cmd": "act", "action": "forward", "ms": 400, "turn_deg": 30, "pitch_deg": 15}
//        {"id": 3, "cmd": "say", "text": "..."}
//        {"id": 4, "cmd": "snapshot"}          -> {"image": "data:image/jpeg;base64,..."} (needs --camera-port)
//   out: {"id": 1, "ok": true, "result": {...}}  /  {"id": 1, "ok": false, "error": "..."}
//        {"event": "ready", ...}  /  {"event": "end", "reason": "..."}
//
// The body only senses and moves; all decisions and all text rendering live in the Python brain.
// stdout is the protocol channel, so everything else goes to stderr.

const mineflayer = require('mineflayer')
const readline = require('readline')
const { Vec3 } = require('vec3')
const { forwardHost } = require('./forward')          // joining behind the online-mode proxy
const { VIS_ACTIONS, visAction } = require('./vis_aim')   // aiming from what was seen (vision "pure" harnesses)

console.log = console.error

const args = Object.fromEntries(process.argv.slice(2).map(a => a.replace(/^--/, '').split('=')))
const GRID_R = 8                  // world grid radius sent with each observation
const GRID_DYS = [-1, 0, 1, 2]    // layers relative to the feet block
const MOVES = {
  forward: ['forward'], back: ['back'], left: ['left'], right: ['right'],
  jump_forward: ['forward', 'jump']
}

const send = obj => process.stdout.write(JSON.stringify(obj) + '\n')
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
const rad = deg => deg * Math.PI / 180
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a))
const round = (v, n = 2) => Math.round(v * 10 ** n) / 10 ** n

const bot = mineflayer.createBot({
  host: args.host || '127.0.0.1',
  port: parseInt(args.port || '25565'),
  username: args.name || 'Jev',
  version: args.version || '1.11.2',
  auth: 'offline',
  fakeHost: forwardHost(args.host || '127.0.0.1', args.name || 'Jev')
})

const chatLog = []
let chatSeq = 0
const logChat = (username, message, whisper, from = {}) => {
  chatLog.push({ seq: ++chatSeq, t: Date.now(), username, message, whisper, self: username === bot.username, ...from })
  if (chatLog.length > 200) chatLog.shift()
}
// Who sent a whisper, from the message itself: a player's name comes with a hover showing their entity (id: their
// UUID) and a click to answer; the console's ("Server", "Rcon") is plain text. So a person named Server is not the
// server (console false), whatever the name says. Public chat is always a player's.
function whisperFrom (msg) {
  const w = msg && msg.json && Array.isArray(msg.json.with) ? msg.json.with[0] : null
  if (!w || typeof w !== 'object') return {}
  const hover = w.hoverEvent && w.hoverEvent.value   // {text: '{name:"UTest",id:"fa0f0e7f-..."}'} (1.11)
  const text = typeof hover === 'string' ? hover : hover && typeof hover.text === 'string' ? hover.text : JSON.stringify(hover || '')
  const id = /id:\\?"?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/.exec(text)
  return { console: !w.hoverEvent && !w.clickEvent && !w.insertion, uuid: id ? id[1] : null }
}
bot.on('chat', (username, message) => logChat(username, message, false, { console: false }))
bot.on('whisper', (username, message, translate, msg) => logChat(username, message, true, whisperFrom(msg)))

let camera = null
bot.once('spawn', async () => {
  let cameraError
  if (args.camera === 'vox') {   // --camera=vox: the CPU ray-caster, no browser (body/voxcam.js)
    try {
      const { startVoxcam } = require('./voxcam')
      const [w, h] = (args['camera-size'] || '256x256').split('x').map(Number)
      camera = await startVoxcam(bot, { width: w, height: h || w })
      camera.url = 'voxcam'
    } catch (err) {
      cameraError = String(err && err.message || err)
    }
  } else if (args['camera-port']) {
    try {
      // loaded only when asked for: its dependencies (node-canvas, a browser) are not needed without vision, and
      // node-canvas's prebuilt libraries do not load on 64K-page aarch64 machines such as the GB200 nodes
      const { startCamera } = require('./camera')
      const [w, h] = (args['camera-size'] || '640x360').split('x').map(Number)   // e.g. --camera-size=256x256
      camera = await startCamera(bot, { port: parseInt(args['camera-port']), browserPath: args.browser, width: w, height: h || w,
        fps: parseFloat(args['camera-fps'] || '0'), gpu: args['camera-gpu'] === '1',
        viewDistance: parseInt(args['camera-view'] || '4') })   // chunks drawn round the bot (scenes/shots.py: 8)
    } catch (err) {
      cameraError = String(err && err.message || err)
    }
  }
  send({ event: 'ready', name: bot.username, version: bot.version, gamemode: bot.game.gameMode,
         camera: camera && camera.url, camera_error: cameraError })
})
bot.on('kicked', reason => send({ event: 'kicked', reason: String(reason) }))
bot.on('error', err => send({ event: 'error', reason: String(err && err.message || err) }))
bot.on('end', async reason => {
  send({ event: 'end', reason: String(reason) })
  if (camera) await camera.close()
  process.exit(0)
})

// One char per block: . empty, s solid, w water, l lava, d door, b button, v lever, ? not loaded
function category (block) {
  if (!block) return '?'
  const n = block.name
  if (n.includes('water')) return 'w'
  if (n.includes('lava')) return 'l'
  if (n === 'fire') return 'f'
  if (n.endsWith('_door') || n === 'wooden_door' || n === 'iron_door') return 'd'
  if (n.includes('button')) return 'b'
  if (n === 'lever') return 'v'
  return block.boundingBox === 'block' ? 's' : '.'
}

function grid () {
  // Feet level: floor(y + 0.25), not floor(y). Standing on a grass path (15/16 high) puts y at 3.94, and floor()
  // took the ground itself for the feet layer: every cell read as a wall. Slabs (y + 0.5) still round down.
  const p = bot.entity.position
  const base = p.floored().offset(0, Math.floor(p.y + 0.25) - Math.floor(p.y), 0)
  return GRID_DYS.map(dy => {
    const rows = []
    for (let dz = -GRID_R; dz <= GRID_R; dz++) {
      let row = ''
      for (let dx = -GRID_R; dx <= GRID_R; dx++) row += category(bot.blockAt(base.offset(dx, dy, dz)))
      rows.push(row)
    }
    return rows
  })
}

// Mobs within 24 blocks: kind and position; a creeper also reports whether its fuse is lit (metadata 12 = 1 in
// 1.11: -1 idle, 1 swelling).
function mobs (range = 24) {
  const me = bot.entity.position
  return Object.values(bot.entities)
    .filter(e => e.type === 'mob' && e.position.distanceTo(me) <= range)
    .map(e => ({
      id: e.id,
      name: e.name,
      pos: { x: round(e.position.x), y: round(e.position.y), z: round(e.position.z) },
      fuse: e.name === 'creeper' ? (e.metadata && e.metadata[12]) === 1 : undefined
    }))
}

let deaths = 0
let lastDeath = null
bot.on('death', () => { deaths++; lastDeath = Date.now(); hold([]); send({ event: 'death' }) })

// Using an item (drawing a bow, eating) slows a player to a fifth and ends the sprint: the vanilla client multiplies
// its movement input by 0.2 while the hand is active (EntityPlayerSP), and a forward input that small cannot sprint.
// prismarine-physics leaves that out, so without this a bot could draw a bow at full sprint.
// Our own record of a drawn bow (set when we press, cleared when we let go or switch items): mineflayer clears
// usingHeldItem on any entity_status packet for us - a hurt, for one - while the server still has the bow drawn.
let bowDrawnAt = null
const bowDrawn = () => bowDrawnAt !== null && bot.heldItem && bot.heldItem.name === 'bow'
bot.once('spawn', () => {
  const simulate = bot.physics.simulatePlayer.bind(bot.physics)
  bot.physics.simulatePlayer = (state, world) => {
    if (bot.usingHeldItem || bowDrawn()) {
      const c = state.control
      state.control = { ...c, forward: c.forward * 0.2, back: c.back * 0.2, left: c.left * 0.2, right: c.right * 0.2, sprint: false }
    }
    return simulate(state, world)
  }
})
function stopSprintForUse () { // pressing the use button: the sprint stops (it has to be pressed again afterwards)
  held.sprint = false
  bot.setControlState('sprint', false)
}

// Keys held in hold mode (continuous control): they stay down until the next decision changes them.
const held = { forward: false, back: false, left: false, right: false, jump: false, sprint: false }
function hold (keys) {
  for (const k of Object.keys(held)) {
    held[k] = keys.includes(k)
    bot.setControlState(k, held[k])
  }
}

// Edge guard (act guard=N turns it on, guard=true is 2): a player does not walk off a ledge by accident. When the
// keys you hold would carry you off the floor you stand on onto a drop of N or more blocks, they are let go, the way
// a player stops at an edge. Only your own keys: a hit can still knock you off.
let edgeGuard = 0
let lastEdgeStop = null
function floorUnder (x, y, z) {
  for (let dy = 1; dy <= edgeGuard; dy++) {
    const b = bot.blockAt(new Vec3(Math.floor(x), Math.floor(y + 0.25) - dy, Math.floor(z)))
    if (!b || b.boundingBox === 'block') return true // unloaded counts as floor: never stop for nothing
  }
  return false
}
// Auto-jump (act autojump=true), the 1.11 client's own option: walking into a one-block step hops up onto it. In
// water, a swimmer pressing into a one-block edge holds jump to climb out (a water pit: running alone never left it).
let autoJump = false
let autoJumping = 0
bot.on('physicsTick', () => {
  const e = bot.entity
  if (!autoJump || !e) return
  if (autoJumping) { if (--autoJumping === 0) bot.setControlState('jump', held.jump); return }
  if (!(e.onGround || e.isInWater) || !(held.forward || held.left || held.right || held.back)) return
  const fwd = (held.forward ? 1 : 0) - (held.back ? 1 : 0), side = (held.right ? 1 : 0) - (held.left ? 1 : 0)
  const sn = Math.sin(e.yaw), cs = Math.cos(e.yaw)
  let dx = -sn * fwd + cs * side, dz = -cs * fwd - sn * side
  const n = Math.hypot(dx, dz)
  if (!n) return
  dx /= n; dz /= n
  const y = feetLevel()
  const at = d => new Vec3(Math.floor(e.position.x + dx * d), y, Math.floor(e.position.z + dz * d))
  const cell = at(0.6)
  if (cell.x === Math.floor(e.position.x) && cell.z === Math.floor(e.position.z)) return
  if (solid(bot.blockAt(cell)) && !solid(bot.blockAt(cell.offset(0, 1, 0))) && !solid(bot.blockAt(cell.offset(0, 2, 0))) &&
      !solid(bot.blockAt(new Vec3(Math.floor(e.position.x), y + 2, Math.floor(e.position.z))))) {
    bot.setControlState('jump', true)
    autoJumping = e.isInWater ? 8 : 1
  }
})

// Lava guard (act lavaguard=true): a player does not walk or jump into lava or fire. When the held keys would carry
// you into a cell with lava or fire at your feet or in the ground under you, they are let go. It looks farther when
// you are about to jump (a sprint-jump carries 3-4 blocks) and, in the air, at where you are coming down (the keys
// go, so the air steering stops). When only the jump would land in it and the walk is clear, only the jump goes: on
// a small platform ringed by lava a sprint-jump's 3.6 blocks reach the lava from most places, and letting go of
// every key there froze you just as you closed in on someone at the edge. Lava that has flowed out counts as lava;
// where it may flow later does not. It runs after every controller (circle, cover, zigzag, dodge) so its word is the
// last in each tick. Only your own keys: a hit can still knock you in.
let lavaGuard = false
let lastLavaStop = null
const hot = b => b && /lava|fire/.test(b.name)
function guardLava () {
  const e = bot.entity
  if (!lavaGuard || !e) return
  const px = Math.floor(e.position.x), pz = Math.floor(e.position.z), y = Math.floor(e.position.y + 0.25)
  const into = (x, z, yy) => !(x === px && z === pz) &&
    (hot(bot.blockAt(new Vec3(x, yy, z))) || hot(bot.blockAt(new Vec3(x, yy - 1, z))))
  const stop = () => { hold([]); lastLavaStop = Date.now() }
  if (!e.onGround) { // in the air: where the fall comes down
    const vx = e.velocity.x, vz = e.velocity.z
    if (Math.hypot(vx, vz) < 0.08 || !(held.forward || held.back || held.left || held.right)) return
    for (const k of [3, 6, 9]) {
      if (into(Math.floor(e.position.x + vx * k), Math.floor(e.position.z + vz * k), y) ||
          into(Math.floor(e.position.x + vx * k), Math.floor(e.position.z + vz * k), y - 1)) return stop()
    }
    return
  }
  const fwd = (held.forward ? 1 : 0) - (held.back ? 1 : 0)
  const side = (held.right ? 1 : 0) - (held.left ? 1 : 0)
  if (!fwd && !side) return
  const sn = Math.sin(e.yaw), cs = Math.cos(e.yaw)
  let dx = -sn * fwd + cs * side, dz = -cs * fwd - sn * side
  const n = Math.hypot(dx, dz)
  dx /= n; dz /= n
  const speed = Math.max(Math.hypot(e.velocity.x, e.velocity.z), held.sprint ? 0.28 : 0.2)
  const jumping = held.jump || bot.getControlState('jump')
  const hotAt = ahead => into(Math.floor(e.position.x + dx * ahead), Math.floor(e.position.z + dz * ahead), y)
  if ([0.4, 2.2 * speed + 0.45].some(hotAt)) return stop()
  if (jumping && [1.2, 2.0, 2.8, 3.6].some(hotAt)) {   // the walk is clear: keep walking, do not jump
    held.jump = false
    bot.setControlState('jump', false)
    lastLavaStop = Date.now()
  }
}

bot.on('physicsTick', () => {
  const e = bot.entity
  if (!edgeGuard || !e || !e.onGround) return
  const fwd = (held.forward ? 1 : 0) - (held.back ? 1 : 0)
  const side = (held.right ? 1 : 0) - (held.left ? 1 : 0)
  if (!fwd && !side) return
  const s = Math.sin(e.yaw), c = Math.cos(e.yaw)
  let dx = -s * fwd + c * side, dz = -c * fwd - s * side  // forward (-sin, -cos), right (cos, -sin)
  const n = Math.hypot(dx, dz)
  dx /= n; dz /= n
  // this tick's move plus the slide once the keys are up (ground friction keeps about half the speed a tick)
  const speed = Math.max(Math.hypot(e.velocity.x, e.velocity.z), held.sprint ? 0.28 : 0.2)
  const ahead = 2.2 * speed + 0.15
  const x = e.position.x + dx * ahead, z = e.position.z + dz * ahead, w = 0.25
  const onFloor = [[-w, -w], [-w, w], [w, -w], [w, w]].some(([ox, oz]) => floorUnder(x + ox, e.position.y, z + oz))
  if (!onFloor) {
    hold([])
    lastEdgeStop = Date.now()
  }
})

// Combat (PvP): attack recharge, a short log of hits given and taken, and the other players' state.
const REACH = 3.0            // vanilla survival/adventure reach, eyes to the target's box
let lastAttack = 0
// 1.9+ attack recharge: only a recharged hit does full damage, sprint knockback and crits. It depends on the item in
// hand, and switching to a different item starts it over.
function rechargeMs () {
  const n = bot.heldItem ? bot.heldItem.name : ''
  if (n.endsWith('_sword')) return 625
  if (n.endsWith('_axe')) return { iron: 1111, stone: 1250, wooden: 1250 }[n.split('_')[0]] || 1000
  if (n.endsWith('_pickaxe')) return 833
  if (n.endsWith('_shovel')) return 1000
  return 250 // hand, or anything that is not a weapon or tool
}
// What a player feels when hit, without seeing who: the health lost, and the knockback (the server pushes you away
// from whoever hit you, so the push says where the hit came from).
const drops = []            // {t, from, to}
let lastHealth = null
bot.on('health', () => {
  if (lastHealth !== null && bot.health < lastHealth - 0.01) {
    drops.push({ t: Date.now(), from: lastHealth, to: bot.health })
    if (drops.length > 10) drops.shift()
  }
  lastHealth = bot.health
})
let knock = null            // {t, vx, vz}: the latest push the server gave you
bot._client.on('entity_velocity', packet => {
  if (bot.entity && packet.entityId === bot.entity.id) {
    const vx = packet.velocityX / 8000, vz = packet.velocityZ / 8000   // blocks per tick
    if (Math.hypot(vx, vz) > 0.05) knock = { t: Date.now(), vx, vz }
  }
})
bot._client.on('explosion', packet => {
  if (Math.hypot(packet.playerMotionX, packet.playerMotionZ) > 0.05) knock = { t: Date.now(), vx: packet.playerMotionX, vz: packet.playerMotionZ }
})
let rechargeStart = 0
let heldName = null
bot.on('heldItemChanged', item => {
  const n = item ? item.name : null
  if (n !== heldName) { heldName = n; rechargeStart = Date.now() }
})
let lastSprintPress = 0      // for W-tap: a hit only gets the sprint knockback if sprint was pressed again since
// Events (p_uhc_pro_v10): what happened in the fight, a line each, kept 20 s - the middle layer of what Jev sees
// (the agent counts them up into the round and the match)
const events = []   // {id, t, kind, text, ...}
let eventSeq = 0
function event (kind, text, extra) {
  events.push({ id: ++eventSeq, t: Date.now(), kind, text, ...(extra || {}) })
  while (events.length && (Date.now() - events[0].t > 20000 || events.length > 80)) events.shift()
}
const combat = []            // {t, kind: 'hit' | 'hurt' | 'swing', who}
const logCombat = (kind, who) => { combat.push({ t: Date.now(), kind, who }); if (combat.length > 20) combat.shift() }
bot.on('entityHurt', ent => {
  if (ent === bot.entity) {
    const near = Object.values(bot.players).filter(p => p.entity && p.username !== bot.username)
      .sort((a, b) => a.entity.position.distanceTo(bot.entity.position) - b.entity.position.distanceTo(bot.entity.position))[0]
    logCombat('hurt', near && near.entity.position.distanceTo(bot.entity.position) < 5 ? near.username : null)
  } else if (ent.type === 'player' && Date.now() - lastAttack < 400) {
    logCombat('hit', ent.username)
  }
})

// when each other player last swung (the server shows everyone's arm swings): their sword recharges from then
const swungAt = {}
bot.on('entitySwingArm', ent => { if (ent && ent.type === 'player' && ent.username) swungAt[ent.username] = Date.now() })

function playerState (p) {
  const e = p.entity
  if (!e) return { name: p.username, pos: null }
  const md = e.metadata || []
  return {
    name: p.username,
    id: e.id,
    pos: { x: round(e.position.x), y: round(e.position.y), z: round(e.position.z) },
    yaw: e.yaw,
    health: typeof md[7] === 'number' ? round(md[7], 1) : null,
    sprinting: typeof md[0] === 'number' ? Boolean(md[0] & 0x08) : null,
    using: typeof md[6] === 'number' ? Boolean(md[6] & 0x01) : null,   // eating, drawing a bow
    burning: typeof md[0] === 'number' ? Boolean(md[0] & 0x01) : null,
    absorption: typeof md[11] === 'number' ? round(md[11], 1) : null,     // golden hearts
    holding: e.heldItem ? e.heldItem.name : null,
    in_water: [0.2, 1.2].some(h => { const b = bot.blockAt(e.position.offset(0, h, 0)); return !!b && /water/.test(b.name) }),
    swung_ago: swungAt[p.username] ? round((Date.now() - swungAt[p.username]) / 1000, 2) : null,
    armor: (e.equipment || []).slice(2).filter(Boolean).map(i => i.name),
    velocity: e.velocity ? { x: round(e.velocity.x), y: round(e.velocity.y), z: round(e.velocity.z) } : null
  }
}

function targetEntity (name) {
  const p = name && bot.players[name]
  return p && p.entity
}

async function strike (target) {
  // look at the middle of the target's body, then hit it if it is within reach, else swing at the air
  await bot.lookAt(target.position.offset(0, 1.2, 0), true)
  lastAttack = rechargeStart = Date.now()
  const eye = bot.entity.position.offset(0, bot.entity.height, 0)
  if (eye.distanceTo(target.position.offset(0, 0.9, 0)) <= REACH + 0.4) {
    bot.attack(target)
    return `hit at ${target.username || target.name}`
  }
  bot.swingArm()
  logCombat('swing', null)
  return `swung at ${target.username || target.name} out of reach`
}

// ---------------------------------------------------------------- engage (p_uhc_pro): the body's hands in a sword fight
// Jev turns it on with a style and keeps choosing where to move; every tick the body keeps the sword on the target
// and swings only when a swing counts: recharged, and the target's box within reach of the eyes (never a swing at the
// air, which would start the recharge over). Styles: 'plain'; 'sprint' - a w-tap before each hit so it carries the
// sprint knockback (sprint let go for one tick, then pressed again: a fast player's w-tap, not a same-tick reset);
// 'crit' - jump when ready and in reach, hit on the way down (1.5x; a crit cannot be a sprint hit, so no sprint).
// It idles while something other than a weapon is in hand (a bow, a bucket: their actions own the view then).
let engage = null            // {target, style, since, swings, wtap, crit, critT}
async function critOnce (t) {
  held.sprint = false
  bot.setControlState('sprint', false)   // a crit cannot be a sprint hit
  bot.setControlState('jump', true)
  const t0 = Date.now()
  while (Date.now() - t0 < 700 && (bot.entity.onGround || bot.entity.velocity.y > -0.1)) await sleep(20)
  await sleep(50)
  bot.setControlState('jump', held.jump)
  if (!engage || bot.entity.onGround || reachTo(t) > REACH) return
  await bot.lookAt(t.position.offset(0, 1.2, 0), true)
  lastAttack = rechargeStart = Date.now()
  bot.attack(t)
  engage.swings++
}
function reachTo (t) {       // eyes to the nearest point of the target's box
  const e = bot.entity.position.offset(0, bot.entity.height, 0), p = t.position
  const dx = Math.max(p.x - 0.3 - e.x, 0, e.x - (p.x + 0.3))
  const dy = Math.max(p.y - e.y, 0, e.y - (p.y + 1.8))
  const dz = Math.max(p.z - 0.3 - e.z, 0, e.z - (p.z + 0.3))
  return Math.hypot(dx, dy, dz)
}
bot.on('physicsTick', () => {
  if (!engage) return
  const t = targetEntity(engage.target), e = bot.entity
  if (!t || !e) return
  const n = bot.heldItem ? bot.heldItem.name : ''
  if (!(n.endsWith('_sword') || n.endsWith('_axe'))) return
  bot.lookAt(t.position.offset(0, 1.2, 0), true).catch(() => {})
  const now = Date.now()
  const ready = (now - rechargeStart) / rechargeMs() >= 0.95
  const r = reachTo(t)
  const swing = () => {
    lastAttack = rechargeStart = now
    bot.attack(t)
    engage.swings++
  }
  if (engage.style === 'crit') {
    // the same sequence as the one-off crit_attack, which the server counts as crits (a swing from inside this tick
    // handler, at the same moment of the fall, was not): jump, wait for the fall, let go of jump, look, hit
    if (engage.crit) return
    if (!e.onGround) {
      // already in the air (a jump Jev holds, a knock): no sprint (a crit cannot be a sprint hit), and the hit on the
      // way down once the server has seen the fall - two falling ticks, so a falling position went out before it
      engage.fall = e.velocity.y < -0.1 ? (engage.fall || 0) + 1 : 0
      if (held.sprint || bot.getControlState('sprint')) { held.sprint = false; bot.setControlState('sprint', false) }
      if (ready && r <= REACH && engage.fall >= 2) swing()
      return
    }
    engage.fall = 0
    if (ready && r <= REACH + 0.8) {
      engage.crit = true
      critOnce(t).finally(() => { if (engage) engage.crit = false })
    }
    return
  }
  if (engage.wtap) { // second tick of the w-tap: sprint again, then the hit carries the knockback
    engage.wtap = false
    if (held.forward) { bot.setControlState('sprint', true); held.sprint = true; lastSprintPress = now }
  }
  if (!ready || r > REACH) return
  if (engage.style === 'sprint' && held.forward && !(lastSprintPress > lastAttack)) {
    bot.setControlState('sprint', false)
    engage.wtap = true
    return
  }
  swing()
})

// Arrow dodge (act dodge=true, p_uhc_pro_v8): an arrow the fight's opponent just shot is flown ahead the way the
// server flies it (EntityArrow: move, x0.99 drag, 0.05 gravity a tick) against where your box will be if you keep
// moving; if it would hit (your box, 0.3 each side, plus the arrow's 0.25), the side keys go the way that ends
// farther from it at the tick it passes you, by a walk model (sideways speed toward 0.216 b/tick: x0.546 friction,
// +0.098 a tick): a shooter who led you expected you to keep going, so a moving target usually reverses. Your own
// side keys come back after its flight.
let dodgeOn = false, dodge = null, lastDodge = null, lastArrow = null
// act arrowwarn=true (raw harnesses): their arrows are seen coming - the 'their_shot' event and obs.incoming (when it
// would hit, which strafe key takes you off its line) - but nothing moves you; dodging is the harness's choice
let arrowWarn = false, incoming = null
function dodgeKey (a, pass) { // startDodge's choice of side, without moving
  const e = bot.entity, sp = Math.hypot(a.velocity.x, a.velocity.z) || 1
  const ux = a.velocity.x / sp, uz = a.velocity.z / sp, px = -uz, pz = ux
  const lat = e.velocity.x * px + e.velocity.z * pz, off = -(pass.dx * px + pass.dz * pz)
  const end = dir => Math.abs(off + sideAfter(lat, dir, pass.tick) - lat * pass.tick)
  const dir = end(1) >= end(-1) ? 1 : -1
  return Math.cos(e.yaw) * px * dir - Math.sin(e.yaw) * pz * dir >= 0 ? 'right' : 'left'
}
function sideAfter (lat, dir, ticks) { // how far sideways you get in `ticks`, starting at `lat` b/tick, keys `dir`
  let y = 0, v = lat
  for (let i = 0; i < ticks; i++) { v = v * 0.546 + 0.098 * dir; y += v }
  return y
}
function blockedStep (p, v) { // does the arrow's move this tick run into a block (it stops there)
  return blockedAt(p, v) !== null
}
function blockedAt (p, v) { // where the arrow's move this tick runs into a block: the share of the move (null: it does not)
  const n = v.norm()
  if (n <= 1e-6) return null
  const hit = bot.world.raycast(p, v.scaled(1 / n), n, b => b && b.boundingBox === 'block')
  if (!hit) return null
  return hit.intersect ? Math.min(1, hit.intersect.distanceTo(p) / n) : 1
}
function segBox (p, v, lo, hi) { // where a move from p by v enters a box: the share of the move (null: it misses)
  let t0 = 0, t1 = 1
  for (const [a, d, l, h] of [[p.x, v.x, lo[0], hi[0]], [p.y, v.y, lo[1], hi[1]], [p.z, v.z, lo[2], hi[2]]]) {
    if (Math.abs(d) < 1e-9) { if (a < l || a > h) return null; continue }
    let u = (l - a) / d, w = (h - a) / d
    if (u > w) [u, w] = [w, u]
    t0 = Math.max(t0, u); t1 = Math.min(t1, w)
    if (t0 > t1) return null
  }
  return t0
}
function arrowPass (a, me, mv) { // the arrow's pass by your box if you keep going: gap 0 when it hits you, else its
  // closest pass (null: a block stops it first). As the server's EntityArrow (1.11): each tick its move runs to the
  // first block in its way, and it hits whoever's box - grown by 0.3 - that move enters before that point. So an arrow
  // at your feet hits the feet before the ground it would land on (only the ends of each 3-block move were looked at
  // before, and a low one passing through your feet between two of them read as landing short - a player's AI found it).
  let p = a.position.clone(), v = a.velocity.clone(), best = null
  for (let tick = 1; tick <= 40; tick++) {
    const bx = me.x + mv.x * tick, bz = me.z + mv.z * tick   // your box this tick, if you keep going
    const stop = blockedAt(p, v)
    const f = segBox(p, v, [bx - 0.6, me.y - 0.3, bz - 0.6], [bx + 0.6, me.y + 2.1, bz + 0.6])   // 0.3 each side, grown 0.3
    if (f !== null && (stop === null || f <= stop)) {
      const h = p.plus(v.scaled(f))
      return { tick, gap: 0, dx: h.x - bx, dz: h.z - bz }
    }
    if (stop !== null) return best && best.gap < 0.55 ? best : null
    p = p.plus(v); v = new Vec3(v.x * 0.99, v.y * 0.99 - 0.05, v.z * 0.99)
    const gap = Math.max(Math.abs(p.x - bx), Math.abs(p.z - bz))
    const inY = p.y > me.y - 0.25 && p.y < me.y + 2.05
    if (inY && (!best || gap < best.gap)) best = { tick, gap, dx: p.x - bx, dz: p.z - bz }
    if ((p.x - bx) * v.x + (p.z - bz) * v.z > 2) break   // gone past you
  }
  return best
}
// A drawn bow holds you to a fifth of your speed, so a sidestep with it drawn barely moves: for an arrow that would
// hit and gives you 4 ticks or more, the draw is dropped (the hotbar slot flicks, which lets go without shooting)
// and shoot_full draws again once the sidestep is over.
let dodgeDropAt = 0
function cancelDraw () {
  const s0 = bot.quickBarSlot
  bot.setQuickBarSlot((s0 + 1) % 9)
  dropUse()
  dodgeDropAt = Date.now()
}
function startDodge (a, pass, why) { // sidestep across the arrow's path, the way that ends farther from it
  if (bowDrawn() && pass.tick >= 4 && pass.gap < 0.4) { cancelDraw(); event('drop_draw', 'you lowered your drawn bow to sidestep an arrow (a drawn bow walks at a fifth)') }
  const e = bot.entity, sp = Math.hypot(a.velocity.x, a.velocity.z) || 1
  const ux = a.velocity.x / sp, uz = a.velocity.z / sp, px = -uz, pz = ux
  const lat = e.velocity.x * px + e.velocity.z * pz, off = -(pass.dx * px + pass.dz * pz)   // your side of its line then
  const end = dir => Math.abs(off + sideAfter(lat, dir, pass.tick) - lat * pass.tick)       // the keys instead of coasting
  const dir = end(1) >= end(-1) ? 1 : -1
  const key = Math.cos(e.yaw) * px * dir - Math.sin(e.yaw) * pz * dir >= 0 ? 'right' : 'left'
  if (!dodge) dodge = { left: held.left, right: held.right }
  dodge.until = Date.now() + pass.tick * 50 + 150
  held.left = key === 'left'; held.right = key === 'right'
  bot.setControlState('left', held.left); bot.setControlState('right', held.right)
  if (zigzag) { zigzag.left = key === 'left'; zigzag.until = dodge.until }
  if (circle) circle.until = dodge.until
  lastDodge = Date.now()
  console.error('[dodge]', why, key, 'tick', pass.tick, 'gap', pass.gap.toFixed(2), 'lat', lat.toFixed(3))
  reflex = { t: lastDodge, what: why === 'aim'
    ? `${fightTarget} was aiming a drawn bow at you: the body stepped ${key}, off their line`
    : `an arrow from ${fightTarget} would have hit you in ${(pass.tick / 20).toFixed(2)} s: the body sidestepped ${key}` }
}
// A team fight (p_uhc_pro_team, act enemies / allies): every enemy's arrows are watched, not only the one you fight;
// an arrow of yours is held back while an ally stands in its path.
let enemyNames = [], allyNames = []
bot.on('entitySpawn', a => {
  if (!(dodgeOn || arrowWarn) || a.name !== 'arrow' || !fightTarget || !bot.entity) return
  const e = bot.entity
  const shooter = (enemyNames.length ? enemyNames : [fightTarget])
    .find(n => { const q = targetEntity(n); return q && a.position.distanceTo(q.position.offset(0, 1.5, 0)) <= 3 })
  if (!shooter) return
  if (Math.hypot(a.velocity.x, a.velocity.z) < 0.3) return
  if (shooter === fightTarget) theirDrawAt = null
  const pass = arrowPass(a, e.position, { x: e.velocity.x, z: e.velocity.z })
  lastArrow = { t: Date.now(), hit: !!(pass && pass.gap < 0.55) }
  const hp0 = bot.health, onLine = lastArrow.hit
  event('their_shot', `${shooter} loosed an arrow at you` + (onLine ? ` (on your line: it would hit in ${(pass.tick / 20).toFixed(2)} s)` : ' (off your line)'), { onLine, who: shooter })
  if (onLine && dodgeOn) startDodge(a, pass, 'arrow')
  else if (onLine) incoming = { t: Date.now(), tick: pass.tick, key: dodgeKey(a, pass) }
  setTimeout(() => {
    const hit = bot.health < hp0
    console.error('[their arrow]', onLine ? 'on line' : 'off line', pass ? `tick ${pass.tick} gap ${pass.gap.toFixed(2)}` : '', hit ? `HIT you (-${(hp0 - bot.health).toFixed(0)})` : 'missed you')
    if (onLine && !hit) event('dodged', `their arrow missed you: the sidestep took you off its line`)
  }, ((pass && pass.tick) || 15) * 50 + 400)
})
// Shot review: an arrow you loosed (spawned at your eyes) is flown on from its spawn as the server flies it (the
// server sends an arrow's position only once a second) against where the target's box really is each tick, until
// it has passed them; a hurt flash on them in that time is a hit. A miss is told against their motion: ahead of
// them / behind them (along how they moved while it flew), high / low. Kept as lastShot, printed for the logs.
let lastShot = null, myArrow = null, lastRelease = null
bot.on('entitySpawn', a => {
  if (a.name !== 'arrow' || !bot.entity || !fightTarget) return
  if (a.position.distanceTo(eyePos()) > 1.5) return
  const t = targetEntity(fightTarget)
  if (!t) return
  const rel = lastRelease && Date.now() - lastRelease.t < 400 ? lastRelease : null
  shotNo++
  event('my_shot', `you loosed an arrow at ${fightTarget}` + (rel ? ` (power ${Math.round(rel.sol.power * 100)}%, ${(rel.sol.ticks / 20).toFixed(2)} s flight)` : ''))
  myArrow = { t, t0: Date.now(), p: a.position.clone(), v: a.velocity.clone(), p0: t.position.clone(), best: null, hit: false, ticks: 0,
    rel, no: shotNo, dist: a.position.distanceTo(t.position), spawnDy: rel ? a.position.y - rel.eye.y : null, speed: a.velocity.norm(),
    dv: rel ? (() => { const e = rel.sol.aim.minus(rel.eye).normalize().scaled(3 * rel.sol.power), d = a.velocity.minus(e); return `${d.x.toFixed(2)},${d.y.toFixed(2)},${d.z.toFixed(2)}` })() : '-' }
})
bot.on('entityHurt', en => { if (myArrow && en === myArrow.t) myArrow.hit = true })
bot.on('physicsTick', () => {
  if (!myArrow) return
  const s = myArrow, t = s.t
  s.p = s.p.plus(s.v); s.v = new Vec3(s.v.x * 0.99, s.v.y * 0.99 - 0.05, s.v.z * 0.99); s.ticks++
  const d = s.p.minus(t.position.offset(0, 0.9, 0))
  const gap = Math.max(Math.abs(d.x) - 0.3, Math.abs(d.z) - 0.3, Math.abs(d.y) - 0.9, 0)
  if (!s.best || gap < s.best.gap) s.best = { gap, d, ticks: s.ticks, tp: t.position.clone() }
  if (s.rel && s.rel.sol.P && !s.atP) { // the arrow at the aimed point's range: how far off the point (the aim), and where they were (the lead)
    const P = s.rel.sol.P, h = Math.hypot(P.x - s.rel.eye.x, P.z - s.rel.eye.z), ha = Math.hypot(s.p.x - s.rel.eye.x, s.p.z - s.rel.eye.z)
    if (ha >= h) s.atP = { ay: s.p.y - P.y, aside: Math.hypot(s.p.x - P.x, s.p.z - P.z), lead: Math.hypot(t.position.x - P.x, t.position.z - P.z), ty: t.position.y + 1.1 - P.y, tick: s.ticks, pticks: s.rel.sol.ticks }
  }
  if (!t.isValid || s.ticks > 60 || d.x * s.v.x + d.z * s.v.z > 3) finishShot()
})
function finishShot () {
  const s = myArrow
  myArrow = null
  if (!s || !s.best) return
  const b = s.best
  const mv = b.tp.minus(s.p0), m = Math.hypot(mv.x, mv.z)
  const along = m > 0.3 ? (b.d.x * mv.x + b.d.z * mv.z) / m : 0
  const miss = s.hit ? 'hit' : [Math.abs(along) > 0.3 ? (along > 0 ? 'ahead of them' : 'behind them') : null,
    Math.abs(b.d.y) > 0.9 ? (b.d.y > 0 ? 'high' : 'low') : null].filter(Boolean).join(', ') || 'beside them'
  lastShot = { t: Date.now(), hit: s.hit, gap: round(b.gap, 2), miss, moved: round(m, 1), flight: round(b.ticks / 20, 2) }
  event('my_result', s.hit ? `your arrow hit ${s.t.username}` : `your arrow missed ${s.t.username} (${miss}, by ${b.gap.toFixed(1)} blocks)`, { hit: s.hit })
  if (leadRotate && s.rel) {
    const how = s.hit ? 'hit' : `${miss} by ${b.gap.toFixed(1)}`
    try { bot.chat(`/tell ${s.t.username} bow#${s.no} ${s.rel.sol.mode} lead ${s.rel.sol.lead.toFixed(1)} flight ${(s.rel.sol.ticks / 20).toFixed(2)}s -> ${how}`) } catch (e) {}
  }
  const P = s.atP
  console.error('[shot]', s.hit ? 'HIT' : 'miss', 'd', s.dist.toFixed(1), 'gap', b.gap.toFixed(2), miss, 'along', along.toFixed(2), 'dy', b.d.y.toFixed(2),
    'moved', m.toFixed(1), 'speed', s.speed.toFixed(2), 'spawnDy', s.spawnDy === null ? '-' : s.spawnDy.toFixed(2),
    s.rel ? `plan ${s.rel.sol.plan || '-'} mode ${s.rel.sol.mode} extra ${s.rel.sol.extra.toFixed(1)}t air ${s.rel.air} power ${s.rel.sol.power} kind ${s.rel.sol.kind} share ${round(s.rel.sol.share, 2)}/${s.rel.sol.guesses} runs ${rhythm.runs.map(Math.round).join(',')}` : '',
    P ? `| at aim pt: arrow dy ${P.ay.toFixed(2)} aside ${P.aside.toFixed(2)} (tick ${P.tick} vs ${P.pticks.toFixed(1)}); target off aim pt ${P.lead.toFixed(2)} dy ${P.ty.toFixed(2)}` : '| no release record')
}

// Health both ways, with the likely cause: an arrow of theirs in the air, their sword swung just now, fire, lava.
let lastHp = null, theirHp = {}
bot.on('physicsTick', () => {
  if (!bot.entity) return
  const now = Date.now(), hp = bot.health
  if (lastHp !== null && hp < lastHp - 0.4) {
    const t = fightTarget && targetEntity(fightTarget), sw = fightTarget && swungAt[fightTarget]
    const cause = lastArrow && now - lastArrow.t < 1500 ? 'their arrow'
      : sw && now - sw < 700 && t && t.position.distanceTo(bot.entity.position) < 5 ? 'their sword'
      : (bot.entity.metadata && bot.entity.metadata[0] & 0x01) ? 'fire' : 'a hit'
    event('hurt', `you lost ${(lastHp - hp).toFixed(0)} health (${cause})`, { dmg: lastHp - hp, cause })
  }
  lastHp = hp
  const t = fightTarget && targetEntity(fightTarget)
  if (t) {
    const md = t.metadata || [], h = typeof md[7] === 'number' ? md[7] : null, h0 = theirHp[fightTarget]
    if (h !== null && h0 !== undefined && h < h0 - 0.4) {
      const cause = myArrow || (lastShot && now - lastShot.t < 300) ? 'your arrow' : now - lastAttack < 400 ? 'your sword'
        : (md[0] & 0x01) ? 'fire' : 'a hit'
      event('they_hurt', `${fightTarget} lost ${(h0 - h).toFixed(0)} health (${cause})`, { dmg: h0 - h, cause })
    }
    if (h !== null) theirHp[fightTarget] = h
  }
})
const allyHp = {}
bot.on('physicsTick', () => {
  for (const n of allyNames) {
    const q = targetEntity(n)
    const h = q && q.metadata && typeof q.metadata[7] === 'number' ? q.metadata[7] : null
    if (h !== null && allyHp[n] !== undefined && h < allyHp[n] - 0.4) {
      const by = enemyNames.find(m => swungAt[m] && Date.now() - swungAt[m] < 700 && targetEntity(m) &&
        targetEntity(m).position.distanceTo(q.position) < 5)
      event('ally_hurt', `${n} (your teammate) lost ${(allyHp[n] - h).toFixed(0)} health` + (by ? ` - ${by}'s sword` : ''), { dmg: allyHp[n] - h, who: n, by })
    }
    if (h !== null) allyHp[n] = h
  }
})
let wasDrawing = false, wasClear = null, theirHeld = null
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget)
  const h = t && t.heldItem ? t.heldItem.name : null
  if (t && h !== theirHeld) {
    if (theirHeld !== null || h) event('they_switch', `${fightTarget} took ${h ? 'a ' + h.replace(/_/g, ' ') : 'nothing'} in hand`, { item: h })
    theirHeld = h
  }
  const drawing = !!(t && theirDrawing(t))
  if (drawing && !wasDrawing) event('they_draw', `${fightTarget} started drawing their bow`)
  wasDrawing = drawing
  if (!bowDrawn() || !t) { wasClear = null; return }
  const clear = clearShot(t)
  if (wasClear !== null && clear !== wasClear) event(clear ? 'exposed' : 'covered', clear ? `${fightTarget} came out of cover: your shot is clear` : `${fightTarget} went behind cover: a block stops your shot`)
  wasClear = clear
})

// Their aim: while the fight's opponent draws a bow, the arrow they would loose now is flown ahead from their eyes
// along their look (power grows with the draw, full at 1 s: (s^2 + 2s) / 3). When it would hit you where you are
// heading, their draw is 0.6 s or more and you are not moving across it already, the body starts across it - so you
// are moving when they let go, and the sidestep on the arrow can reverse that.
let theirDrawAt = null
function theirDrawing (t) {
  const md = t.metadata || []
  return typeof md[6] === 'number' && Boolean(md[6] & 0x01) && !!t.heldItem && t.heldItem.name === 'bow'
}
bot.on('physicsTick', () => {
  if (!dodgeOn || !fightTarget || !bot.entity) return
  const t = targetEntity(fightTarget), e = bot.entity
  if (!t || !theirDrawing(t)) { theirDrawAt = null; return }
  if (!theirDrawAt) theirDrawAt = Date.now()
  const s = (Date.now() - theirDrawAt) / 1000
  if (s < 0.6 || dodge || bowDrawn() || e.position.distanceTo(t.position) < 6) return
  const f = Math.min(1, (s * s + 2 * s) / 3)
  const look = new Vec3(-Math.sin(t.yaw) * Math.cos(t.pitch), Math.sin(t.pitch), -Math.cos(t.yaw) * Math.cos(t.pitch))
  const a = { position: t.position.offset(0, 1.52, 0), velocity: look.scaled(3 * f) }
  const pass = arrowPass(a, e.position, { x: e.velocity.x, z: e.velocity.z })
  if (!pass || pass.gap >= 0.55) return
  const sp = Math.hypot(a.velocity.x, a.velocity.z) || 1
  if (Math.abs(e.velocity.x * -a.velocity.z / sp + e.velocity.z * a.velocity.x / sp) > 0.1) return   // already moving across
  startDodge(a, pass, 'aim')
})
bot.on('physicsTick', () => {
  if (!dodge || Date.now() < dodge.until) return
  if (!zigzag && !circle) { held.left = dodge.left; held.right = dodge.right; bot.setControlState('left', held.left); bot.setControlState('right', held.right) }
  dodge = null
})

// Footwork (action circle, p_uhc_pro_v9): the body moves you around the opponent the way a sword fighter does - in
// to hit when your sword is ready (to 2.7 blocks, eyes to their box), back out of their reach while it recharges
// (3.1-3.6 blocks), and sideways all the time, switching sides every 0.4-0.9 s at random. Any other move ends it.
let circle = null            // {until, left}
bot.on('physicsTick', () => {
  if (!circle || dodge || !bot.entity) return
  const t = targetEntity(fightTarget)
  if (!t) return
  if (!engage) bot.lookAt(t.position.offset(0, 1.2, 0), true).catch(() => {})
  if (Date.now() > circle.until) { circle.left = !circle.left; circle.until = Date.now() + 400 + Math.random() * 500 }
  const r = reachTo(t), ready = (Date.now() - rechargeStart) / rechargeMs() >= 0.85
  const fwd = ready ? r > 2.7 : r > 3.6, back = !ready && r < 3.1
  if (fwd && !held.forward && r > 4) lastSprintPress = Date.now()
  held.forward = fwd; held.back = back; held.left = circle.left; held.right = !circle.left
  if (!fwd) held.sprint = false
  else if (r > 4) held.sprint = true
  for (const k of ['forward', 'back', 'left', 'right']) bot.setControlState(k, held[k])
  if (!(engage && engage.wtap)) bot.setControlState('sprint', held.sprint)
})

// Cover (p_uhc_pro_v9): a spot is out of their line of fire when solid blocks cut the lines from their eyes to your
// chest and head there. coverSpot finds the nearest standable one within 3 blocks (floor under it, room for you, no
// lava or fire); take_cover walks you there (a straight walk, 2.5 s at most); cover_shot shoots over the cover you
// stand behind: the draw behind it, a jump, the release at the top of the jump once the line is clear.
function lineBlocked (from, to) {
  const d = to.minus(from), n = d.norm()
  const hit = bot.world.raycast(from, d.normalize(), n, b => b && b.boundingBox === 'block')
  return !!hit
}
function arcBlocked (from, to) { // a full-power arrow from `from` aimed to meet `to`: does a block stop it on the way
  const dx = to.x - from.x, dz = to.z - from.z, horiz = Math.hypot(dx, dz), h = to.y - from.y
  let lo = -0.6, hi = 0.8, theta = 0
  for (let i = 0; i < 20; i++) { theta = (lo + hi) / 2; const f = flightTo(horiz, theta); if (!f || f.y < h) lo = theta; else hi = theta }
  const ux = dx / (horiz || 1), uz = dz / (horiz || 1)
  let p = from.offset(0, ARROW_DROP0, 0), v = new Vec3(ux * 3 * Math.cos(theta), 3 * Math.sin(theta), uz * 3 * Math.cos(theta))
  for (let tick = 0; tick < 60; tick++) {
    const left = Math.hypot(to.x - p.x, to.z - p.z)
    if (left <= v.norm()) return blockedStep(p, v.scaled(left / (Math.hypot(v.x, v.z) || 1)))
    if (blockedStep(p, v)) return true
    p = p.plus(v); v = new Vec3(v.x * 0.99, v.y * 0.99 - 0.05, v.z * 0.99)
  }
  return false
}
// Covered: an arrow they aim at you - at your head, your chest, and either side of your body - is stopped by a block
// on its arc (an arrow from far comes down over a low wall a straight line would call covering).
function coveredAt (t, feet) {
  const eye = t.position.offset(0, 1.62, 0)
  const dx = feet.x - t.position.x, dz = feet.z - t.position.z, n = Math.hypot(dx, dz) || 1
  const sx = -dz / n * 0.28, sz = dx / n * 0.28
  const pts = [feet.offset(0, 1.6, 0), feet.offset(0, 1.0, 0), feet.offset(sx, 1.2, sz), feet.offset(-sx, 1.2, -sz)]
  return pts.every(q => lineBlocked(eye, q)) && pts.every(q => arcBlocked(eye, q))   // the cheap test first
}
function coverSpot (t) {
  const p = bot.entity.position, y = feetLevel()
  if (coveredAt(t, p)) return { cell: new Vec3(Math.floor(p.x), y, Math.floor(p.z)), d: 0 }
  const cells = []
  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      const c = new Vec3(Math.floor(p.x) + dx, y, Math.floor(p.z) + dz)
      const d = Math.hypot(c.x + 0.5 - p.x, c.z + 0.5 - p.z)
      if (d <= 3.2) cells.push({ c, d })
    }
  }
  for (const { c, d } of cells.sort((a, b) => a.d - b.d)) {   // the nearest covered spot
    const feetB = bot.blockAt(c), headB = bot.blockAt(c.offset(0, 1, 0)), floor = bot.blockAt(c.offset(0, -1, 0))
    if (!feetB || !headB || solid(feetB) || solid(headB) || !solid(floor) || hot(feetB) || hot(floor)) continue
    if (lineBlocked(eyePos(), c.offset(0.5, 1.62, 0.5))) continue   // walled off from you: not a straight walk
    if (coveredAt(t, c.offset(0.5, 0, 0.5))) return { cell: c, d }
  }
  return null
}
let goCover = null           // {cell, until}
bot.on('physicsTick', () => {
  if (!goCover || dodge || !bot.entity) return
  const e = bot.entity, c = goCover.cell
  const dx = c.x + 0.5 - e.position.x, dz = c.z + 0.5 - e.position.z, d = Math.hypot(dx, dz)
  if (d < 0.25 || Date.now() > goCover.until) { goCover = null; hold([]); return }
  // the keys that carry you that way, facing as you are: forward = (-sin yaw, -cos yaw), right = (cos yaw, -sin yaw)
  const f = (-Math.sin(e.yaw) * dx - Math.cos(e.yaw) * dz) / d, r = (Math.cos(e.yaw) * dx - Math.sin(e.yaw) * dz) / d
  held.forward = f > 0.38; held.back = f < -0.38; held.right = r > 0.38; held.left = r < -0.38
  held.sprint = false; held.jump = false
  for (const k of Object.keys(held)) bot.setControlState(k, held[k])
})
// An angle on a hider (p_uhc_pro_v11): the spots within 7 blocks, a straight walk away, where a full arrow from
// your eyes reaches their chest or head past every block, the ones that bring you closer first; find_angle walks
// you to the best one.
function walkable (from, c) { // a straight walk on this level: every cell on the way has room for you and a floor
  const n = Math.ceil(Math.hypot(c.x + 0.5 - from.x, c.z + 0.5 - from.z) * 3)
  for (let i = 1; i <= n; i++) {
    const x = Math.floor(from.x + (c.x + 0.5 - from.x) * i / n), z = Math.floor(from.z + (c.z + 0.5 - from.z) * i / n)
    const q = new Vec3(x, c.y, z)
    if (solid(bot.blockAt(q)) || solid(bot.blockAt(q.offset(0, 1, 0))) || !solid(bot.blockAt(q.offset(0, -1, 0))) || hot(bot.blockAt(q))) return false
  }
  return true
}
function angleSpot (t) {
  const p = bot.entity.position, y = feetLevel(), d0 = p.distanceTo(t.position)
  const cells = []
  for (let dx = -7; dx <= 7; dx++) {
    for (let dz = -7; dz <= 7; dz++) {
      const c = new Vec3(Math.floor(p.x) + dx, y, Math.floor(p.z) + dz)
      const walk = Math.hypot(c.x + 0.5 - p.x, c.z + 0.5 - p.z)
      if (walk > 7.2 || walk < 0.8) continue
      const closer = d0 - Math.hypot(c.x + 0.5 - t.position.x, c.z + 0.5 - t.position.z)
      cells.push({ c, walk, score: walk - 0.5 * closer })
    }
  }
  for (const { c, walk } of cells.sort((a, b) => a.score - b.score)) {
    const feetB = bot.blockAt(c), headB = bot.blockAt(c.offset(0, 1, 0)), floor = bot.blockAt(c.offset(0, -1, 0))
    if (!feetB || !headB || solid(feetB) || solid(headB) || !solid(floor) || hot(feetB) || hot(floor) || /water/.test(feetB.name)) continue
    if (!walkable(p, c)) continue   // around your own wall, not through it
    const eye = c.offset(0.5, 1.62, 0.5)
    if ([0.9, 1.5].some(h => !arcBlocked(eye, t.position.offset(0, h, 0)))) return { cell: c, d: walk }
  }
  return null
}
// A peek (p_uhc_pro_v11): from behind cover, the nearest spot within 2.5 blocks on your level - a lean past your
// wall's edge - where a full arrow from your eyes reaches their chest or head.
function peekSpot (t) {
  const p = bot.entity.position, y = feetLevel()
  const cells = []
  for (let dx = -3; dx <= 3; dx++) {
    for (let dz = -3; dz <= 3; dz++) {
      const c = new Vec3(Math.floor(p.x) + dx, y, Math.floor(p.z) + dz)
      const walk = Math.hypot(c.x + 0.5 - p.x, c.z + 0.5 - p.z)
      if (walk <= 2.6 && walk >= 0.5) cells.push({ c, walk })
    }
  }
  for (const { c, walk } of cells.sort((a, b) => a.walk - b.walk)) {
    const feetB = bot.blockAt(c), headB = bot.blockAt(c.offset(0, 1, 0)), floor = bot.blockAt(c.offset(0, -1, 0))
    if (!feetB || !headB || solid(feetB) || solid(headB) || !solid(floor) || hot(feetB) || hot(floor)) continue
    if (!walkable(p, c)) continue
    const eye = c.offset(0.5, 1.62, 0.5)
    if ([0.9, 1.5].some(h => !arcBlocked(eye, t.position.offset(0, h, 0)))) return { cell: c, d: walk }
  }
  return null
}
let peekCache = { t: 0, v: null }
let theirCoveredSince = null, coverTick = 0   // since when no arc from your eyes reaches them
let angleCache = { t: 0, v: null }
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget)
  if (!t || !bot.entity) { theirCoveredSince = null; return }
  if (++coverTick % 3) return   // every third tick is enough
  theirCoveredSince = clearShot(t) ? null : (theirCoveredSince || Date.now())
})
function coverState (t) { // for the harness: are you covered, where is the nearest cover, can you shoot over it
  const spot = coverSpot(t)
  const covered = !!spot && spot.d === 0
  let over = false
  if (covered) { const e = eyePos(); over = !lineBlocked(e.offset(0, 1.2, 0), t.position.offset(0, 1.1, 0)) }
  return { covered, spot: spot && !covered ? round(spot.d, 1) : null, over }
}

// ---------------------------------------------------------------- items (Block UHC)

const BUILD = ['cobblestone', 'planks', 'dirt', 'stone', 'stonebrick', 'sandstone', 'log', 'wool']
const feetLevel = () => Math.floor(bot.entity.position.y + 0.25)
const solid = b => b && b.boundingBox === 'block'
const replaceable = b => b && (b.boundingBox === 'empty' || /water|lava|fire|tallgrass|snow_layer/.test(b.name))
const eyePos = () => bot.entity.position.offset(0, bot.entity.height, 0)
const withTimeout = (promise, ms) => Promise.race([promise, sleep(ms).then(() => { throw new Error('timed out') })])

// Liquid source blocks (what an empty bucket can take back) within bucket reach of your eyes.
// Lava you poured yourself (cell -> when; a minute): water is for your own fire and their lava, not your traps.
const ownLava = new Map()
const cellKey = c => `${c.x},${c.y},${c.z}`
function isOwnLava (pos) { const t = ownLava.get(cellKey(pos)); return !!t && Date.now() - t < 60000 }
function liquidSources (reach = 4.5) {
  const eye = eyePos()
  const base = eye.floored()
  const out = []
  for (let dx = -4; dx <= 4; dx++) {
    for (let dy = -4; dy <= 3; dy++) {
      for (let dz = -4; dz <= 4; dz++) {
        const b = bot.blockAt(base.offset(dx, dy, dz))
        if (!b || b.metadata !== 0 || !/^(flowing_)?(water|lava)$/.test(b.name)) continue
        const d = eye.distanceTo(b.position.offset(0.5, 0.5, 0.5))
        const kind = b.name.replace('flowing_', '')
        if (d <= reach) out.push({ kind, pos: b.position, dist: round(d, 1), own: kind === 'lava' && isOwnLava(b.position) })
      }
    }
  }
  return out.sort((a, b) => a.dist - b.dist).slice(0, 6)
}

async function equip (pred) {
  if (bot.heldItem && pred(bot.heldItem)) return bot.heldItem
  const hot = bot.inventory.slots.slice(36, 45).find(i => i && pred(i))
  const item = hot || bot.inventory.items().find(pred)
  if (!item) return null
  await bot.equip(item, 'hand')
  return item
}

// The horizontal axis direction you face most (a block goes one cell that way).
function facingCell () {
  const yaw = bot.entity.yaw
  const fx = -Math.sin(yaw), fz = -Math.cos(yaw)
  return Math.abs(fx) > Math.abs(fz) ? new Vec3(Math.sign(fx), 0, 0) : new Vec3(0, 0, Math.sign(fz))
}

function occupied (cell) { // a player's body in the cell: the server refuses a block there
  return Object.values(bot.entities).some(en => en.type === 'player' &&
    Math.abs(en.position.x - (cell.x + 0.5)) < 0.8 && Math.abs(en.position.z - (cell.z + 0.5)) < 0.8 &&
    en.position.y < cell.y + 1 && en.position.y + 1.8 > cell.y)
}

let placeError = ''
function touching (cell) { // your own body right against the cell: the server counts that as in the way
  const p = bot.entity.position
  const ox = Math.max(cell.x - (p.x + 0.3), (p.x - 0.3) - (cell.x + 1))
  const oz = Math.max(cell.z - (p.z + 0.3), (p.z - 0.3) - (cell.z + 1))
  return Math.max(ox, oz) < 0.01 && p.y < cell.y + 1 && p.y + 1.8 > cell.y
}

// Fast building: each block goes down in its own tick by the place packet itself (the reference block's face, the
// cursor on it), with no turn of the head and no wait for the server's answer - a quick player's click rate. The
// server takes them in order, so a block may rest on one sent the tick before.
async function placeFast (plan) {
  for (const q of plan) {
    const fi = FACES.findIndex(f => f.equals(q.face))
    bot._client.write('block_place', { location: q.ref, direction: fi, hand: 0,
      cursorX: 0.5 + q.face.x * 0.5, cursorY: 0.5 + q.face.y * 0.5, cursorZ: 0.5 + q.face.z * 0.5 })
    bot.swingArm()
    await sleep(50)
  }
}

async function place (ref, face) {
  const yaw = bot.entity.yaw, pitch = bot.entity.pitch
  try {
    await withTimeout(bot.placeBlock(ref, face), 900)
    return true
  } catch (err) {
    placeError = String(err && err.message || err)
    return false
  } finally {
    await bot.look(yaw, pitch, true)
  }
}

// Aim with an arrow's drop and the target's lead: full-charge arrows fly about 3 blocks a tick.
function arrowAim (t, track) {
  const eye = eyePos()
  const d = eye.distanceTo(t.position)
  const ticks = d / 3
  let vx = 0, vz = 0
  if (track.length > 1) {
    const [a, b] = [track[0], track[track.length - 1]]
    const dt = Math.max(1, (b.t - a.t) / 50)
    vx = (b.p.x - a.p.x) / dt; vz = (b.p.z - a.p.z) / dt
  }
  return t.position.offset(vx * ticks, 1.0 + 0.025 * ticks * ticks, vz * ticks)
}

// Where to aim a full-power arrow so it meets the target: the arrow's own flight (3 blocks a tick, x0.99 drag and
// 0.05 gravity every tick, as EntityArrow) and the target's lead (its velocity over the last ticks, carried over
// the flight time). The elevation is found by bisection on a simulated flight, the flight time by 3 rounds.
// Measured on a flat range (bench/bow_calib.py): the arrow leaves 0.28 below your eyes, and when you shoot in the air
// the server adds your own up-down speed to it (EntityArrow: motY += the shooter's motY when not on the ground).
const ARROW_DROP0 = -0.28
function flightTo (horiz, theta, speed = 3, vy0 = 0) { // simulate: height at `horiz` blocks out, and the ticks it took (null: falls short)
  let x = 0, y = ARROW_DROP0, vx = speed * Math.cos(theta), vy = speed * Math.sin(theta) + vy0
  for (let tick = 1; tick <= 100; tick++) {
    x += vx; y += vy; vx *= 0.99; vy = vy * 0.99 - 0.05
    if (x >= horiz) { const back = (x - horiz) / Math.max(vx, 1e-6); return { y: y - vy * back, ticks: tick - back } }
  }
  return null
}
function allyInLine (sol) { // the ally an arrow loosed now would meet before its target, or null
  if (!allyNames.length || !sol || !sol.aim) return null
  const dir = sol.aim.minus(sol.eye).normalize()
  let p = sol.eye.offset(0, ARROW_DROP0, 0), v = dir.scaled(3 * (sol.power || 1))
  for (let tick = 0; tick < Math.ceil(sol.ticks) + 1; tick++) {
    p = p.plus(v); v = new Vec3(v.x * 0.99, v.y * 0.99 - 0.05, v.z * 0.99)
    for (const n of allyNames) {
      const q = targetEntity(n)
      if (q && Math.abs(p.x - q.position.x) < 0.55 && Math.abs(p.z - q.position.z) < 0.55 &&
          p.y > q.position.y - 0.25 && p.y < q.position.y + 2.05) return n
    }
  }
  return null
}
function lsVel (track) { // blocks per tick: the least-squares slope of their positions over the track
  const t0 = track[0].t
  const ts = track.map(s => (s.t - t0) / 50)
  const mt = ts.reduce((a, b) => a + b, 0) / ts.length
  const den = ts.reduce((a, t) => a + (t - mt) ** 2, 0) || 1
  const slope = key => {
    const m = track.reduce((a, s) => a + s.p[key], 0) / track.length
    return track.reduce((a, s, i) => a + (ts[i] - mt) * (s.p[key] - m), 0) / den
  }
  return { x: slope('x'), z: slope('z') }
}
// Their lead: a steady mover is led by the last 0.3 s; one who keeps turning back (a weave, left-right) by the drift
// of the last second - the middle of the weave - and one in between by the two halfway. Capped at a sprint-jump.
function leadVelocity (track) {
  if (track.length < 2) return { x: 0, z: 0, kind: 'unknown' }
  const short = lsVel(track.slice(-6)), long = lsVel(track)
  const ss = Math.hypot(short.x, short.z), sl = Math.hypot(long.x, long.z)
  const cos = (short.x * long.x + short.z * long.z) / ((ss * sl) || 1)
  let v, kind
  if (ss < 0.03) { v = short; kind = 'still' } else if (track.length < 8 || cos > 0.8) { v = short; kind = 'steady' } else if (cos < 0) { v = long; kind = 'weave' } else { v = { x: (short.x + long.x) / 2, z: (short.z + long.z) / 2 }; kind = 'turning' }
  let vx = v.x, vz = v.z
  const sp = Math.hypot(vx, vz), top = 0.36
  if (sp > top) { vx *= top / sp; vz *= top / sp }
  return { x: vx, z: vz, kind }
}
function feetAfter (t, track, ticks) { // their feet height after `ticks`: a jump's arc while they are in the air
  const y0 = t.position.y
  if (t.onGround !== false || track.length < 2) return y0
  const a = track[track.length - 2], b = track[track.length - 1]
  let vy = (b.p.y - a.p.y) / Math.max(1, (b.t - a.t) / 50), y = y0
  const ground = Math.min(...track.map(s => s.p.y))
  for (let i = 0; i < Math.round(ticks); i++) { y += vy; vy = (vy - 0.08) * 0.98; if (y <= ground) return ground }
  return y
}
// The bow's power after `s` seconds drawn, as the server counts it (ItemBow: f = ticks/20, (f^2 + 2f)/3, at most 1):
// the arrow leaves at 3f blocks a tick. The release comes about 0.07 s after the aim is solved (the look goes out first).
function drawPower (s) {
  const f = Math.max(0, Math.floor(s * 20)) / 20
  return Math.min(1, (f * f + 2 * f) / 3)
}
function usingItem (t) { const md = t.metadata || []; return typeof md[6] === 'number' && Boolean(md[6] & 0x01) }
let theirUseAt = null   // when the fight's opponent started using an item (a bow drawn, food): they move at a fifth
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget)
  if (!t || !usingItem(t)) { theirUseAt = null; return }
  if (!theirUseAt) theirUseAt = Date.now()
})
// Their motion over the flight: using an item (drawing a bow, eating) holds a player to a fifth of their speed and no
// sprint, so while they do, their lead is the drift since they started (or a fifth of before) - not the run before;
// in the air (a jump) their sideways speed hardly changes, so the last ticks lead them.
function motionOver (t, track) {
  if (usingItem(t) && track.length >= 2) {
    const since = theirUseAt ? track.filter(s => s.t >= theirUseAt) : []
    const v = since.length >= 3 ? lsVel(since) : (() => { const w = leadVelocity(track); return { x: w.x * 0.2, z: w.z * 0.2 } })()
    const sp = Math.hypot(v.x, v.z), top = 0.06
    return sp > top ? { x: v.x * top / sp, z: v.z * top / sp, kind: 'slowed' } : { ...v, kind: 'slowed' }
  }
  if (t.onGround === false && track.length >= 3) return { ...lsVel(track.slice(-3)), kind: 'airborne' }
  return leadVelocity(track)
}
// Their rhythm: the fight's opponent's motion across your line to them is followed every tick; each time it turns
// back, how long that run lasted is remembered (the last 10 runs). A player who weaves turns on a rhythm of their
// own (a runner hardly turns at all), and with how long they have run since the last turn, the rhythm says where
// they may be when an arrow arrives: every remembered run is one guess (turn when it would end, same speed back).
const rhythm = { name: null, hist: [], runs: [], lastTurn: null, sign: 0, turned: false, speed: 0, hurtAt: 0, stillSince: null }
bot.on('entityHurt', en => { const t = fightTarget && targetEntity(fightTarget); if (t && en === t) rhythm.hurtAt = Date.now() })
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget), e = bot.entity
  if (!t || !e) return
  if (rhythm.name !== fightTarget) Object.assign(rhythm, { name: fightTarget, hist: [], runs: [], lastTurn: null, sign: 0, turned: false, speed: 0, stillSince: null })
  const now = Date.now(), h = rhythm.hist
  h.push({ t: now, p: t.position.clone() })
  while (h.length > 80) h.shift()
  if (h.length < 4) return
  const a = h[h.length - 4], b = h[h.length - 1]
  const dx = t.position.x - e.position.x, dz = t.position.z - e.position.z, n = Math.hypot(dx, dz) || 1
  const lat = ((b.p.x - a.p.x) * -dz / n + (b.p.z - a.p.z) * dx / n) / Math.max(1, (b.t - a.t) / 50)
  const sg = Math.abs(lat) < 0.04 ? 0 : Math.sign(lat)
  rhythm.stillSince = sg ? null : (rhythm.stillSince || now)
  if (sg) rhythm.speed = rhythm.speed ? rhythm.speed * 0.9 + Math.abs(lat) * 0.1 : Math.abs(lat)   // their running speed across
  if (sg && sg !== rhythm.sign && now - rhythm.hurtAt > 400) {   // a hit's knockback is not a turn of theirs
    // a run counts from one turn to the next (the first one seen started before we looked); a flicker under 4 ticks is not one
    const run = (now - rhythm.lastTurn) / 50
    if (rhythm.turned && run >= 4) { rhythm.runs.push(run); while (rhythm.runs.length > 10) rhythm.runs.shift() }
    if (rhythm.sign) rhythm.turned = true
    rhythm.lastTurn = now; rhythm.sign = sg
    rhythm.turns = (rhythm.turns || []).filter(x => now - x < 5000).concat([now])
  }
})
function theirMove () { // over the last 3 s: how far they were then and now, how far they went across, turns
  const t = fightTarget && targetEntity(fightTarget), e = bot.entity, h = rhythm.hist
  if (!t || !e || h.length < 10) return null
  const now = Date.now(), old = h.find(q => now - q.t <= 3000) || h[0]
  const secs = (now - old.t) / 1000
  if (secs < 1) return null
  const d0 = old.p.distanceTo(e.position), d1 = t.position.distanceTo(e.position)
  const dx = t.position.x - e.position.x, dz = t.position.z - e.position.z, n = Math.hypot(dx, dz) || 1
  const across = Math.abs((t.position.x - old.p.x) * -dz / n + (t.position.z - old.p.z) * dx / n)
  const went = Math.hypot(t.position.x - old.p.x, t.position.z - old.p.z)
  return { secs: round(secs, 1), d0: round(d0, 1), d1: round(d1, 1), across: round(across, 1), went: round(went, 1),
    turns: (rhythm.turns || []).filter(x => now - x < 3000).length, covered: !clearShot(t) }
}
function lateralForecast (vl, T) { // -> {off: where across the line to aim (blocks), share: the guesses it covers}
  if (rhythm.runs.length < 3 || !rhythm.lastTurn) return { off: vl * T, share: 1, guesses: 0 }
  // stopped (at a wall, or just standing) for more than a moment: they are where they are, not on a run
  if (rhythm.stillSince && Date.now() - rhythm.stillSince > 250) return { off: vl * T, share: 1, guesses: 0 }
  const tau = (Date.now() - rhythm.lastTurn) / 50
  const v = rhythm.sign * rhythm.speed   // their run's speed, this way (at a turn itself they are near still)
  const g = rhythm.runs.map(d => { const r = Math.max(0, d - tau); return r >= T ? v * T : v * r - v * (T - r) })
  let best = v * T, bestN = g.filter(x => Math.abs(x - v * T) <= 0.5).length   // a tie keeps them going
  for (const c of g) { const k = g.filter(x => Math.abs(x - c) <= 0.5).length; if (k > bestN) { bestN = k; best = c } }
  const near = g.filter(x => Math.abs(x - best) <= 0.5)
  return { off: near.reduce((a, b) => a + b, 0) / near.length, share: bestN / g.length, guesses: g.length }
}
// How stale their position is and how late the arrow leaves: what we see of them is half a round trip old, the
// release reaches the server half a round trip later, and it goes out about 60 ms after the aim is solved. All of
// it is time they keep moving, so the lead covers it on top of the flight (the server's measured ping).
// The round trip: the server's own ping in the player list (keep-alive, sent every 30 s, 0 until the first one), and
// until then the body's own probe (a tab-complete request and the answer, every 5 s; it runs about 1.5x high, the
// answer waits for the server's tick), the median of the last five.
const rtts = []
let rttSent = null
function probeRtt () { if (rttSent) return; rttSent = Date.now(); try { bot._client.write('tab_complete', { text: '/jev', assumeCommand: false }) } catch (e) { rttSent = null } }
bot._client.on('tab_complete', () => {
  if (!rttSent) return
  rtts.push(Date.now() - rttSent); rttSent = null
  while (rtts.length > 5) rtts.shift()
})
setInterval(() => { if (rttSent && Date.now() - rttSent > 3000) rttSent = null; probeRtt() }, 5000)
bot.once('spawn', () => setTimeout(probeRtt, 1000))
function rttMs () { // the server's keep-alive ping when it has one (it led a runner best), else the probe, which runs high
  const me = bot.players[bot.username]
  if (me && me.ping > 0) return me.ping
  if (rtts.length) return 0.65 * [...rtts].sort((a, b) => a - b)[Math.floor(rtts.length / 2)]
  return 100
}
// On top of the round trip: the server sends other players' positions every second tick (EntityTracker, players at
// 2), so what we see of them is 50 ms older still on average, and our release waits for the server's next tick
// (25 ms on average) - there even with no network between us. (Through the relay 150 ms fitted better, relay
// buffering included; next to the server that led a sprint-jumper about a block ahead at 15 blocks.)
const TRACK_LAG_MS = 75
function leadExtraTicks () {
  return (Math.min(500, rttMs()) + 60 + TRACK_LAG_MS) / 50
}
// Where on their box to aim, from their feet: shots at standing and moving targets passed 0.3-0.5 above the aimed
// point (the model's arc runs a little low), so the aim sits under the middle of the 1.8-high box.
const AIM_Y = 0.6
// Nobody runs through a wall: the forecast stops at the first block with a collision box (a barrier, a pillar)
// between where they are and where it puts them, their half-width short of it.
function wallClamp (t, P) {
  const from = t.position.offset(0, 0.5, 0), d = new Vec3(P.x - from.x, 0, P.z - from.z), n = Math.hypot(d.x, d.z)
  if (n < 0.3) return P
  const hit = bot.world.raycast(from, d.scaled(1 / n), n + 0.3, b => b && b.boundingBox === 'block')
  if (!hit) return P
  const at = hit.intersect || hit.position.offset(0.5, 0.5, 0.5)
  const k = Math.min(1, Math.max(0, Math.hypot(at.x - from.x, at.z - from.z) - 0.35) / n)
  return new Vec3(from.x + d.x * k, P.y, from.z + d.z * k)
}
// Lead modes, rotated every 4 arrows so they can be compared against a person (act leadrotate=true; else 'body'):
// body - the body's own (slowed / airborne / steady-weave-turning, the rhythm across the line of fire);
// hawkeye - mineflayer HawkEye's: the mean step over the last 10 samples, carried over the flight x1.1 (plus the
// round trip, which it leaves out); damped - a 0.3 s least-squares velocity at 70% (they may turn any time).
// All keep the arrow's calibrated flight, the power, the round trip and the wall.
const LEAD_MODES = ['body', 'hawkeye', 'damped']
let leadRotate = false, shotNo = 0
const leadMode = () => leadRotate ? LEAD_MODES[Math.floor(shotNo / 4) % LEAD_MODES.length] : 'body'
function modeVel (t, track, mode) {
  if (mode === 'hawkeye' && track.length >= 2) {
    const h = track.slice(-10), n = Math.max(1, (h[h.length - 1].t - h[0].t) / 50)
    return { x: (h[h.length - 1].p.x - h[0].p.x) / n, z: (h[h.length - 1].p.z - h[0].p.z) / n, kind: 'hawkeye', flightX: 1.1 }
  }
  if (mode === 'damped' && track.length >= 2) {
    const v = lsVel(track.slice(-6))
    return { x: v.x * 0.7, z: v.z * 0.7, kind: 'damped' }
  }
  return motionOver(t, track)
}
function arrowSolve (t, track, speed, aimY = AIM_Y) { // aimY: the height on them to aim at (the body's own: AIM_Y)
  const mode = leadMode(), eye = eyePos(), v = modeVel(t, track, mode), extra = leadExtraTicks()
  if (speed === undefined) speed = 3 * (bowDrawn() ? drawPower((Date.now() - bowDrawnAt) / 1000 + 0.07) : 1)
  speed = Math.max(0.3, speed)
  const vy0 = bot.entity.onGround ? 0 : bot.entity.velocity.y / 0.98 + 0.08   // the server's motY runs a tick behind ours
  let lastP = null, fc = null
  let ticks = eye.distanceTo(t.position) / speed, aim = null, theta = 0
  for (let round = 0; round < 3; round++) {
    // along your line to them their own velocity; across it the rhythm's best guess (a steady runner: the same)
    const rx = t.position.x - eye.x, rz = t.position.z - eye.z, rn = Math.hypot(rx, rz) || 1
    const ux = rx / rn, uz = rz / rn, px = -uz, pz = ux
    const vr = v.x * ux + v.z * uz, vl = v.x * px + v.z * pz
    const T = ticks * (v.flightX || 1) + extra
    fc = mode !== 'body' || v.kind === 'slowed' || v.kind === 'airborne' ? { off: vl * T, share: 1, guesses: 0 } : lateralForecast(vl, T)
    let P = new Vec3(t.position.x + ux * vr * T + px * fc.off, feetAfter(t, track, T) + aimY,
      t.position.z + uz * vr * T + pz * fc.off)   // the middle of their box then
    P = wallClamp(t, P)
    const ex = aimY === AIM_Y ? exposedPoint(t) : null   // only part of them shows past cover: aim at that part
    if (ex && !ex.center) P = new Vec3(P.x + ex.ox, P.y - AIM_Y + ex.h, P.z + ex.oz)
    lastP = P
    const horiz = Math.hypot(P.x - eye.x, P.z - eye.z), h = P.y - eye.y
    let lo = -0.6, hi = 0.8
    for (let i = 0; i < 24; i++) {
      theta = (lo + hi) / 2
      const f = flightTo(horiz, theta, speed, vy0)
      if (!f || f.y < h) lo = theta; else hi = theta
    }
    const f = flightTo(horiz, theta, speed, vy0)
    if (f) ticks = f.ticks
    const yaw = Math.atan2(P.x - eye.x, P.z - eye.z)
    aim = eye.offset(Math.sin(yaw) * Math.cos(theta) * 10, Math.sin(theta) * 10, Math.cos(yaw) * Math.cos(theta) * 10)
  }
  return { aim, ticks, extra, mode, lead: Math.hypot(v.x, v.z) * (ticks + extra), speed: Math.hypot(v.x, v.z) * 20, kind: v.kind, P: lastP, eye, power: round(speed / 3, 2),
    share: fc ? fc.share : 1, guesses: fc ? fc.guesses : 0 }
}

// ---------------------------------------------------------------- raw controls: the mouse and its buttons, no aim help

const FACES = [new Vec3(0, -1, 0), new Vec3(0, 1, 0), new Vec3(0, 0, -1), new Vec3(0, 0, 1), new Vec3(-1, 0, 0), new Vec3(1, 0, 0)]
let useStart = null   // when the right button went down (held: drawing a bow, eating)

// Predict, then pour. Where a full bucket's liquid lands is the server's own ray (ItemBucket, 1.11): from the eyes
// along the look, up to 5 blocks, through liquids and blocks without a collision box, to the first block with one;
// the liquid goes into the cell on the face it hit. So before pouring we simulate that ray for candidate aims and
// take one whose landing cell is the one we want - and, for lava, at least `minSelf` blocks from us - or pour nothing.
function solidHit (block, iter) {
  if (!block || block.boundingBox !== 'block') return false
  const hit = iter.intersect(block.shapes, block.position)
  if (!hit) return false
  block.face = hit.face
  return true
}
function landingOf (aim) {
  const eye = eyePos()
  const hit = bot.world.raycast(eye, aim.minus(eye).normalize(), 5, solidHit)
  return hit ? hit.position.plus(FACES[hit.face]) : null
}
function planPour (cells, minSelf) { // cells: the landing cells wanted, best first -> {aim, cell} or null
  const me = bot.entity.position
  const seen = new Set()
  for (const c of cells) {
    const key = `${c.x},${c.y},${c.z}`
    if (seen.has(key)) continue
    seen.add(key)
    if (!replaceable(bot.blockAt(c))) continue
    const ground = bot.blockAt(c.offset(0, -1, 0))
    if (!solid(ground)) continue
    if (Math.hypot(c.x + 0.5 - me.x, c.z + 0.5 - me.z) < minSelf) continue
    for (const [ox, oz] of [[0, 0], [0.3, 0], [-0.3, 0], [0, 0.3], [0, -0.3]]) {
      const aim = ground.position.offset(0.5 + ox, 1, 0.5 + oz)
      const L = landingOf(aim)
      if (L && L.equals(c)) return { aim, cell: c }
    }
  }
  return null
}
function cellsAt (t) { // their feet cell, then the cells around it, nearest to them first
  const f = new Vec3(Math.floor(t.position.x), Math.floor(t.position.y + 0.25), Math.floor(t.position.z))
  const around = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]].map(([dx, dz]) => f.offset(dx, 0, dz))
  const d = c => Math.hypot(c.x + 0.5 - t.position.x, c.z + 0.5 - t.position.z)
  return [f, ...around.sort((a, b) => d(a) - d(b))]
}
function cellsBetween (t, ks) { // cells on the line toward them, ks blocks ahead of you
  const p = bot.entity.position, dx = t.position.x - p.x, dz = t.position.z - p.z, d = Math.hypot(dx, dz) || 1
  return ks.map(k => new Vec3(Math.floor(p.x + dx / d * k), feetLevel(), Math.floor(p.z + dz / d * k)))
}
const LAVA_SELF = 2.0   // lava never lands closer to you than this
const TRAP_KS = [2.5, 3, 3.5, 2, 4]
function pourPlans (t) { // what each pour would do now, for the harness (null: no clear, safe aim)
  const me = bot.entity.position
  const describe = pl => pl && { to_them: round(Math.hypot(pl.cell.x + 0.5 - t.position.x, pl.cell.z + 0.5 - t.position.z), 1),
    to_you: round(Math.hypot(pl.cell.x + 0.5 - me.x, pl.cell.z + 0.5 - me.z), 1) }
  const has = n => bot.inventory.items().some(i => i.name === n)
  return {
    lava_at: has('lava_bucket') ? describe(planPour(cellsAt(t), LAVA_SELF)) : null,
    lava_trap: has('lava_bucket') ? describe(planPour(cellsBetween(t, TRAP_KS), LAVA_SELF)) : null,
    water_trap: has('water_bucket') ? describe(planPour(cellsBetween(t, TRAP_KS), 1.5)) : null
  }
}
async function pourAt (plan) { // look, pour, look back
  const yaw = bot.entity.yaw, pitch = bot.entity.pitch
  await bot.lookAt(plan.aim, true)
  await sleep(60)
  bot.activateItem()
  await sleep(60)
  await bot.look(yaw, pitch, true)
}
const FOOD = ['golden_apple', 'cooked_beef', 'bread', 'apple', 'cooked_porkchop', 'cooked_chicken']

function leftClick () { // what the crosshair is on, within the survival reach, gets hit; else a swing
  lastAttack = rechargeStart = Date.now()
  const t = bot.entityAtCursor(REACH)
  if (t) {
    bot.attack(t)
    return `hit ${t.username || t.name}`
  }
  bot.swingArm()
  logCombat('swing', null)
  return 'swung at the air: nobody in reach under your crosshair'
}

async function rightClick () {
  const h = bot.heldItem ? bot.heldItem.name : null
  if (!h) return 'nothing in your hand'
  if (BUILD.includes(h)) {
    const b = bot.blockAtCursor(4.5)
    if (!b || b.face === undefined) return 'no block within reach under your crosshair'
    const face = FACES[b.face]
    const cell = b.position.plus(face)
    if (!replaceable(bot.blockAt(cell))) return 'that spot is taken'
    if (occupied(cell) || touching(cell)) return 'someone is in that spot'
    return (await place(b, face)) ? `placed ${h} at ${cell.x} ${cell.y} ${cell.z}` : `the block did not go down (${placeError})`
  }
  if (h === 'bow') {
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    stopSprintForUse()
    bot.activateItem(); useStart = bowDrawnAt = Date.now()
    return 'drawing the bow: let go of the right button to shoot (fully drawn after 1 s)'
  }
  if (FOOD.includes(h)) {
    stopSprintForUse()
    bot.activateItem(); useStart = Date.now()
    const name = h
    setTimeout(() => { // eating ends by itself after 1.6 s (32 ticks): the hand is no longer in use
      if (bot.usingHeldItem && useStart && bot.heldItem && bot.heldItem.name === name && Date.now() - useStart >= 1650) {
        bot.deactivateItem(); useStart = null
      }
    }, 1700)
    return `eating the ${h}: keep the right button down 1.6 s`
  }
  bot.activateItem()
  await sleep(60)
  if (!/bucket|fishing_rod/.test(h)) bot.deactivateItem()
  return `right-clicked with the ${h.replace(/_/g, ' ')}`
}

const RAW_ACTIONS = new Set(['look', 'click', 'use', 'use_hold', 'use_release', 'slot', 'turn_rate', 'input'])

// A mouse held moving (raw controls): the view keeps turning at `turnRate` degrees a second (+ right) until the next
// decision changes it, the way a player drags the mouse to follow someone running sideways.
let turnRate = 0
bot.on('physicsTick', () => {
  if (!turnRate || !bot.entity) return
  bot.look(wrap(bot.entity.yaw - rad(turnRate) * 0.05), bot.entity.pitch, true).catch(() => {})
})
const ENGAGE_ACTIONS = new Set(['engage', 'disengage', 'zigzag', 'zigzag_forward', 'circle', 'take_cover', 'find_angle'])

// zigzag (a snake walk against arrows): sideways keys switch between left and right every 0.3-0.6 s at random, with
// forward kept if it was held; any other move ends it
let zigzag = null            // {until, left}
bot.on('physicsTick', () => {
  if (!zigzag) return
  if (Date.now() < zigzag.until) return
  zigzag.left = !zigzag.left
  zigzag.until = Date.now() + 300 + Math.random() * 300
  held.left = zigzag.left; held.right = !zigzag.left
  bot.setControlState('left', held.left); bot.setControlState('right', held.right)
})

async function engageAction (action, target, style) {
  if (action === 'circle') {
    const t = targetEntity(target)
    if (!t) return `cannot see ${target}`
    zigzag = null; goCover = null
    circle = { until: 0, left: Math.random() < 0.5 }
    return `circling ${target}: in to hit when your sword is ready, out of their reach while it recharges, sideways all the time`
  }
  if (action === 'find_angle') {
    const t = targetEntity(target)
    if (!t) return `cannot see ${target}`
    if (clearShot(t)) return `you already have a clear shot at ${target}`
    const spot = angleSpot(t)
    if (!spot) return `no spot with a clear shot at ${target} within 7 blocks`
    zigzag = null; circle = null
    goCover = { cell: spot.cell, until: Date.now() + 2500 }
    return `moving ${spot.d.toFixed(1)} blocks to a spot with a clear shot at ${target}`
  }
  if (action === 'take_cover') {
    const t = targetEntity(target)
    if (!t) return `cannot see ${target}`
    const spot = coverSpot(t)
    if (!spot) return `no spot out of ${target}'s line of fire within 3 blocks`
    if (spot.d < 0.3) return `already out of ${target}'s line of fire`
    zigzag = null; circle = null
    goCover = { cell: spot.cell, until: Date.now() + 2500 }
    return `moving ${spot.d.toFixed(1)} blocks to a spot behind cover, out of ${target}'s line of fire`
  }
  if (action === 'zigzag_forward' || action === 'zigzag') { circle = null; goCover = null }
  if (action === 'zigzag_forward') { // run at them weaving: closes in under arrows
    hold(['forward', 'sprint'])
    lastSprintPress = Date.now()
    zigzag = { until: 0, left: Math.random() < 0.5 }
    return 'running at them in a zigzag, left and right'
  }
  if (action === 'zigzag') {
    zigzag = { until: 0, left: Math.random() < 0.5 }
    return held.forward ? 'running forward in a zigzag, left and right' : 'weaving left and right'
  }
  if (action === 'disengage') {
    const was = engage
    engage = null
    if (wsKept.length) { wsStop('all'); hold([]) }   // a Workshop Jev's kept actions end with the fight
    bot.setControlState('jump', held.jump)
    return was ? `stopped fighting (${was.swings} swings)` : 'not fighting'
  }
  if (!targetEntity(target)) return `cannot see ${target}`
  const it = (bot.heldItem && /_sword$|_axe$/.test(bot.heldItem.name)) ? bot.heldItem
    : await equip(i => i.name.endsWith('_sword')) || await equip(i => i.name.endsWith('_axe'))
  if (!it) return 'no sword'
  dropUse()
  const st = ['plain', 'sprint', 'crit'].includes(style) ? style : 'plain'
  if (engage && engage.target === target) { engage.style = st; return `fighting ${target}, now ${st} hits` }
  engage = { target, style: st, since: Date.now(), swings: 0, wtap: false, crit: false }
  return `fighting ${target} with ${it.name.replace(/_/g, ' ')}: ${st} hits whenever they land`
}

async function rawAction (action, { dyaw = 0, dpitch = 0, n, keys, button }) {
  const e = bot.entity
  if (action === 'input') { // one decision's hands at once, the way a player presses keys, moves the mouse and clicks
    // together: keys (the movement keys held from now on; null keeps what is held), then the mouse, then one button
    // ('click', 'use', 'use_release' or 'slot_<n>')
    if (keys) {
      zigzag = null; dodge = null; circle = null; goCover = null
      if (keys.includes('sprint') && !held.sprint) lastSprintPress = Date.now()
      hold(keys.filter(k => k in held))
    }
    if (dyaw || dpitch) await rawAction('look', { dyaw, dpitch })
    if (!button) return ''
    if (button.startsWith('slot_')) return await rawAction('slot', { n: Number(button.slice(5)) })
    return await rawAction(button, {})
  }
  if (action === 'look') { // dyaw: degrees to the right, dpitch: degrees up
    const pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, e.pitch + rad(dpitch)))
    await bot.look(wrap(e.yaw - rad(dyaw)), pitch, true)
    return ''
  }
  if (action === 'turn_rate') {
    turnRate = Math.max(-360, Math.min(360, Number(n) || 0))
    return turnRate ? `turning ${turnRate > 0 ? 'right' : 'left'} at ${Math.abs(turnRate)}°/s` : 'stopped turning'
  }
  if (action === 'click') return leftClick()
  if (action === 'use') return await rightClick()
  if (action === 'use_hold') {
    const note = await rightClick()
    return note
  }
  if (action === 'use_release') {
    const s = useStart ? (Date.now() - useStart) / 1000 : 0
    const h = bot.heldItem ? bot.heldItem.name : ''
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return h === 'bow' ? `let go: shot an arrow (drawn ${s.toFixed(1)} s)` : `let go of the right button after ${s.toFixed(1)} s`
  }
  if (action === 'slot') {
    const i = Math.max(1, Math.min(9, Number(n))) - 1
    if (i !== bot.quickBarSlot) dropUse()
    bot.setQuickBarSlot(i)
    const it = bot.inventory.slots[36 + i]
    return `holding slot ${i + 1}: ${it ? it.name.replace(/_/g, ' ') : 'empty'}`
  }
  throw new Error(`unknown raw action ${action}`)
}

const ITEM_ACTIONS = new Set(['hold_sword', 'eat_gapple', 'shoot_bow', 'throw_rod', 'place_block', 'pillar_up',
  'water_here', 'water_pickup', 'lava_at', 'lava_pickup', 'break_block', 'crit_attack', 'draw_bow', 'release_bow',
  'water_on_lava', 'build_cover', 'lava_trap', 'water_trap', 'shoot_full', 'shoot_hop', 'cover_shot', 'bow_barrage', 'quick_shot', 'water_escape', 'build_window', 'water_block', 'bow_draw', 'bow_release', 'bow_cancel', 'jump_shot', 'peek_shot'])
let hopLeft = false

// draw_bow (assisted): while the bow is drawn the body keeps it on the target, with the arrow's drop and lead,
// until release_bow lets go or a switch to another item drops the arrow
let drawTarget = null
const drawTrack = []
const seenTrack = []          // the fight's opponent: their positions over the last second, drawn or not
let seenOf = null
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget)
  if (!t || seenOf !== fightTarget) { seenTrack.length = 0; seenOf = fightTarget }
  if (!t) return
  seenTrack.push({ t: Date.now(), p: t.position.clone() })
  while (seenTrack.length > 20) seenTrack.shift()
})
bot.on('physicsTick', () => {
  if (!drawTarget) return
  const t = targetEntity(drawTarget)
  if (!bowDrawn() || !t) { drawTarget = null; return }
  drawTrack.push({ t: Date.now(), p: t.position.clone() })
  while (drawTrack.length > 20) drawTrack.shift()   // the last second
  try { bot.lookAt(arrowSolve(t, drawTrack).aim, true).catch(() => {}) } catch (e) { console.error('[aim]', e.message) }
})
// Bow rules (act bowfull / bowswitch, set by p_uhc_pro_v4): a release waits for the full draw (1 s, full power), and
// with the opponent within `switchAt` blocks a bow in hand - drawn or not - is dropped for the sword at once, the
// way a player drops a drawn arrow when someone rushes in. The fight's target is the last act's target.
const bowRules = { full: false, switchAt: 0 }
let fightTarget = null
let reflex = null            // {t, what}: the last thing the body did on its own
let switching = false
let weaponSwitchAt = 0
let itemBusy = false          // an item action is running (pouring, building, drawing): leave the hand alone
bot.on('physicsTick', () => {
  if (!weaponSwitchAt || switching || itemBusy || !fightTarget) return
  const h = bot.heldItem ? bot.heldItem.name : ''
  if (/_sword$|_axe$/.test(h) || (h === 'bow' && bowDrawn())) return   // a weapon, or a drawn bow (the bow rule's)
  const t = targetEntity(fightTarget)
  if (!t || reachTo(t) > weaponSwitchAt) return
  switching = true
  equip(i => i.name.endsWith('_sword')).then(it => {
    if (it) { dropUse(); reflex = { t: Date.now(), what: `${fightTarget} came within ${weaponSwitchAt} blocks: the body put your ${h.replace(/_/g, ' ') || 'empty hand'} away for the sword` }; event('reflex', reflex.what) }
  }).catch(() => {}).finally(() => { switching = false })
})

bot.on('physicsTick', () => {
  if (!bowRules.switchAt || switching || !fightTarget || !bot.heldItem || bot.heldItem.name !== 'bow') return
  const t = targetEntity(fightTarget)
  if (!t || reachTo(t) > bowRules.switchAt) return
  switching = true
  const drawn = bot.usingHeldItem
  equip(i => i.name.endsWith('_sword')).then(it => {
    if (it) {
      dropUse()
      reflex = { t: Date.now(), what: `${fightTarget} came within ${bowRules.switchAt} blocks: the body dropped your bow${drawn ? ' (and the drawn arrow)' : ''} for the sword` }
    }
  }).catch(() => {}).finally(() => { switching = false })
})

function dropUse () { // switching items drops what the hand was doing (a drawn arrow, a half-eaten apple)
  if (bot.usingHeldItem) bot.usingHeldItem = false
  bowDrawnAt = null
  useStart = null
  drawTarget = null
}

// When to let go, so the release is not a beat they can read (in a bow duel only): each shot takes a plan -
// full: at full power (1 s) once where they will be is clear (up to 0.35 s more);
// bait: hold the full draw while they dodge the release they expect, and let go 0.15 s after they turn (their new
//   run is the one to lead) or once they have stood still 0.3 s - 1.2 s past full at most;
// quick: within 20 blocks, let go at 0.6-0.8 s (about 3/4 power: a faster beat; the aim knows the power).
function shotPlan (t) {
  // only in a bow duel (they hold a bow too): a person shooting back reads your beat; against anyone else, the
  // fastest steady fire (full power, back to back) is the pressure
  if (!(t.heldItem && t.heldItem.name === 'bow')) return 'full'
  const d = t.position.distanceTo(bot.entity.position), moving = rhythm.sign !== 0 && !rhythm.stillSince
  const r = Math.random()
  if (d < 20 && r < 0.3) return 'quick'
  return moving && r < 0.65 ? 'bait' : 'full'
}
function settledShot (t) {
  const sol = arrowSolve(t, drawTrack)
  return sol.guesses ? sol.share >= 0.6 : (t.onGround === false || ['still', 'steady', 'slowed'].includes(sol.kind))
}
// Pre-aim on cover: at full draw, an arrow that a block would stop is not loosed - the draw is held with the aim on
// them (the aim follows them behind their cover), and the arrow goes the moment an arc to their chest or head is
// clear, as they come out. After 5 s of it the draw stays held and Jev chooses again.
// What of them a full arrow can reach: nine points of their box - left, middle, right across your line of fire,
// head, chest, legs - the middle and the chest first. Any one clear is a shot (the box is 0.6 wide and 1.8 high:
// a shoulder or a head showing past cover is a hit); the aim goes to the first clear one. Kept 0.1 s.
let exposedCache = { t: 0, who: null, v: null }
const PART_WORDS = { '0,1.5': 'head', '0,0.9': 'chest', '0,0.35': 'legs', '-1': 'left side', '1': 'right side' }
function exposedPoint (t) {
  if (exposedCache.who === t && Date.now() - exposedCache.t < 100) return exposedCache.v
  const eye = eyePos(), dx = t.position.x - eye.x, dz = t.position.z - eye.z, n = Math.hypot(dx, dz) || 1
  const px = -dz / n, pz = dx / n   // across your line of fire (to your left as you face them... sign kept only)
  let v = null
  for (const side of [0, -1, 1]) {
    for (const h of [0.9, 1.5, 0.35]) {
      const q = t.position.offset(px * side * 0.2, h, pz * side * 0.2)
      if (!arcBlocked(eye, q)) {
        v = { ox: px * side * 0.2, oz: pz * side * 0.2, h, center: side === 0 && h === 0.9,
          part: side === 0 ? PART_WORDS[`0,${h}`] : PART_WORDS[String(side)] }
        break
      }
    }
    if (v) break
  }
  exposedCache = { t: Date.now(), who: t, v }
  return v
}
function clearShot (t) {
  return !!exposedPoint(t)
}
async function waitRelease (t, plan) { // -> how it let go, 'cover' when it held on their cover, or null when lost
  const quickAt = 600 + Math.random() * 200, turn0 = rhythm.lastTurn
  let turnSeen = null, coverSince = null
  while (bowDrawn()) {
    const s = Date.now() - bowDrawnAt
    if (s >= 1050 && !clearShot(t)) {   // behind cover: hold the full draw
      coverSince = coverSince || Date.now()
      if (Date.now() - coverSince > 5000) return 'cover'
      await sleep(20)
      continue
    }
    if (coverSince) return `held full on their cover ${((Date.now() - coverSince) / 1000).toFixed(1)} s, let go as they came out`
    if (plan === 'quick' && s >= quickAt && (settledShot(t) || s >= quickAt + 200)) return `a quick shot at ${(s / 1000).toFixed(1)} s`
    if (plan === 'full' && s >= 1050 && (settledShot(t) || s >= 1400)) return 'at full power'
    if (plan === 'bait' && s >= 1050) {
      if (!turnSeen && rhythm.lastTurn !== turn0) turnSeen = Date.now()
      if (turnSeen && Date.now() - turnSeen >= 150) return `held ${((s - 1000) / 1000).toFixed(1)} s past full, let go after their turn`
      if (rhythm.stillSince && Date.now() - rhythm.stillSince > 300) return `held ${((s - 1000) / 1000).toFixed(1)} s past full, let go on their stop`
      if (s >= 2250) return 'held 1.2 s past full: they kept on'
    }
    await sleep(20)
  }
  return null
}

function dryDir () { // the nearest dry cell to stand in, the four ways, 8 blocks at most -> {dx, dz, k} or null
  const p = bot.entity.position, y = feetLevel()
  let best = null
  for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    for (let k = 1; k <= 8; k++) {
      const c = new Vec3(Math.floor(p.x) + dx * k, y, Math.floor(p.z) + dz * k)
      const b = bot.blockAt(c), below = bot.blockAt(c.offset(0, -1, 0)), head = bot.blockAt(c.offset(0, 1, 0))
      if (!b || !below || !head) break
      if (solid(b) && solid(head)) break   // a wall
      if (!solid(b) && !/water/.test(b.name) && !/water/.test(below.name) && solid(below) && !hot(b)) {
        if (!best || k < best.k) best = { dx, dz, k }
        break
      }
    }
  }
  return best
}

async function itemAction (action, target) {
  const t = targetEntity(target)
  if (action === 'hold_sword') {
    const wasDrawing = bot.usingHeldItem && bot.heldItem && bot.heldItem.name === 'bow'
    const it = await equip(i => i.name.endsWith('_sword')) || await equip(i => i.name.endsWith('_axe'))
    if (it) dropUse()
    return it ? `holding ${it.name}` + (wasDrawing ? ' (the drawn arrow dropped)' : '') : 'no sword'
  }
  if (action === 'lava_trap' || action === 'water_trap') { // a liquid on the ground between you, in their path
    if (!t) return `cannot see ${target}`
    const lava = action === 'lava_trap'
    const d = Math.hypot(t.position.x - bot.entity.position.x, t.position.z - bot.entity.position.z)
    if (d < 3) return `${target} is too close for a trap (${d.toFixed(1)} blocks)`
    const plan = planPour(cellsBetween(t, TRAP_KS), lava ? LAVA_SELF : 1.5)
    if (!plan) return `no clear, safe aim for ${lava ? 'lava' : 'water'} between you from here`
    if (!await equip(i => i.name === (lava ? 'lava_bucket' : 'water_bucket'))) return `no ${lava ? 'lava' : 'water'} bucket`
    dropUse()
    await pourAt(plan)
    if (lava) { hold([]); ownLava.set(cellKey(plan.cell), Date.now()) }   // do not run on into your own lava
    const k = Math.hypot(plan.cell.x + 0.5 - bot.entity.position.x, plan.cell.z + 0.5 - bot.entity.position.z)
    return `laid ${lava ? 'lava' : 'water'} ${k.toFixed(1)} blocks ahead of you, between you and ${target} (${d.toFixed(1)} blocks away)` +
      (lava ? '; your keys are let go so you do not run into it' : '')
  }
  if (action === 'water_on_lava') {
    const src = liquidSources().find(x => x.kind === 'lava' && !x.own)
    if (!src) return 'no lava of theirs within reach (your own lava is left alone)'
    const plan = planPour([src.pos], 0)   // the water goes into the lava's own cell
    if (!plan) return `no clear aim at the lava ${src.dist} blocks away`
    if (!await equip(i => i.name === 'water_bucket')) return 'no water bucket'
    dropUse()
    await pourAt(plan)
    return `poured water on the lava ${src.dist} blocks away (it goes out or turns to obsidian)`
  }
  if (action === 'build_cover' || action === 'build_window') {
    // full (build_cover): a wall 3 wide and 2 high that stops their arrows - the middle column first, then the sides;
    // window (build_window, the 凹 shape): the sides 2 high and the middle 1 high - covered below the chest, a gap
    // to shoot through. Across your line to them, one block ahead (two when your own body is in that row); a block
    // a tick, sent straight to the server (placeFast)
    if (!t) return `cannot see ${target}`
    if (!await equip(i => BUILD.includes(i.name))) return 'no blocks'
    dropUse()
    const p = bot.entity.position
    const dx = t.position.x - p.x, dz = t.position.z - p.z
    const dir = Math.abs(dx) > Math.abs(dz) ? new Vec3(Math.sign(dx), 0, 0) : new Vec3(0, 0, Math.sign(dz))
    const side = new Vec3(dir.z, 0, -dir.x)
    const feet = new Vec3(Math.floor(p.x), feetLevel(), Math.floor(p.z))
    const window = action === 'build_window'
    const shape = window ? [[0, 0], [1, 0], [1, 1], [-1, 0], [-1, 1]] : [[0, 0], [0, 1], [1, 0], [-1, 0], [1, 1], [-1, 1]]
    const blocked = c => occupied(c) || touching(c)
    let ahead = 1
    if (shape.some(([k, dy]) => blocked(feet.plus(dir).plus(side.scaled(k)).offset(0, dy, 0)))) ahead = 2
    const plan = [], will = new Set()
    for (const [k, dy] of shape) {
      const cell = feet.plus(dir.scaled(ahead)).plus(side.scaled(k)).offset(0, dy, 0)
      if (!replaceable(bot.blockAt(cell)) || blocked(cell)) continue
      const below = cell.offset(0, -1, 0)
      if (!(solid(bot.blockAt(below)) || will.has(cellKey(below)))) continue
      plan.push({ ref: below, face: new Vec3(0, 1, 0), cell })
      will.add(cellKey(cell))
    }
    if (!plan.length) return 'no room to build cover there'
    await placeFast(plan)
    await sleep(120)
    const placed = plan.filter(q => solid(bot.blockAt(q.cell))).length
    return placed ? `built ${window ? 'a wall with a window (the middle 1 high, a gap to shoot through)' : 'cover'}: ` +
      `${placed} blocks between you and ${target} in ${(plan.length * 0.05).toFixed(2)} s` : 'no room to build cover there'
  }
  if (action === 'cover_shot') { // the draw behind cover, then a jump and the release at its top, over the cover
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (!coveredAt(t, bot.entity.position)) return `not behind cover from ${target} here`
    if (!bowDrawn()) {
      if (!await equip(i => i.name === 'bow')) return 'no bow'
      stopSprintForUse(); hold([])
      bot.activateItem(); useStart = bowDrawnAt = Date.now()
      drawTarget = target; drawTrack.length = 0
    }
    while (Date.now() - bowDrawnAt < 1050 && bowDrawn()) await sleep(20)
    if (!bowDrawn()) return reflex && Date.now() - reflex.t < 1500 ? `the draw was cut short: ${reflex.what}` : 'the draw was dropped'
    if (!bot.entity.onGround) await sleep(150)
    bot.setControlState('jump', true)
    const t0 = Date.now()
    let clear = false
    while (Date.now() - t0 < 600) {
      await sleep(25)
      if (bot.entity.velocity.y <= 0.05 && !bot.entity.onGround) { clear = !lineBlocked(eyePos(), t.position.offset(0, 1.1, 0)); if (clear) break }
    }
    bot.setControlState('jump', false)
    if (!clear) return `no clear line over the cover at the top of the jump: the bow stays drawn`
    const sol = arrowSolve(t, drawTrack)
    await bot.lookAt(sol.aim, true); await sleep(10)
    drawTarget = null
    lastRelease = sol ? { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y } : null
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return `jumped and shot over the cover at ${target} (${(sol.ticks / 20).toFixed(2)} s flight), then dropped back behind it`
  }
  if (action === 'bow_barrage') { // arrow after arrow, a hop between them, until something changes
    // stops when they come within 10 blocks, after 4 arrows, when you have lost 6 health since it began, or when
    // they are out of sight - then Jev chooses again; the body's dodge and weapon rules run all the while
    const hp0 = bot.health, tStart = Date.now()
    let shots = 0, hits = 0, why = 'two arrows loosed'
    const hurt = en => { const tt = targetEntity(target); if (tt && en === tt) hits++ }
    bot.on('entityHurt', hurt)
    try {
      while (shots < 2) {
        const tt = targetEntity(target)
        if (!tt) { why = `lost sight of ${target}`; break }
        const d = tt.position.distanceTo(bot.entity.position)
        if (d < 10) { why = `${target} came within 10 blocks`; break }
        if (hp0 - bot.health >= 6) { why = `you lost ${(hp0 - bot.health).toFixed(0)} health`; break }
        const note = await itemAction('shoot_hop', target)
        if (!note.includes('shot')) { why = note; break }
        shots++
      }
    } finally { bot.removeListener('entityHurt', hurt) }
    return `loosed ${shots} arrow${shots === 1 ? '' : 's'} at ${target} in ${((Date.now() - tStart) / 1000).toFixed(1)} s (${hits} hit); stopped: ${why}`
  }
  if (action === 'shoot_hop') { // a full-power shot, then a hop sideways to a new spot (an archer who does not stand still)
    const note = await itemAction('shoot_full', target)
    if (!note.startsWith('shot')) return note
    // the hop: sideways within 25 blocks; from farther, forward and to the side - closing to where arrows land
    // (at 35-45 blocks an arrow flies 0.7-0.85 s, and the relay adds half a second more they can move in)
    hopLeft = !hopLeft
    const side = hopLeft ? 'left' : 'right'
    // (in a bow duel - they hold a bow - the hop stays sideways: a run straight at an archer is a target they lead)
    const t2 = targetEntity(target), far = t2 && t2.position.distanceTo(bot.entity.position) > 25 &&
      !(t2.heldItem && t2.heldItem.name === 'bow')
    hold(far ? ['forward', 'sprint', side] : [side])
    if (far) lastSprintPress = Date.now()
    bot.setControlState('jump', true)
    await sleep(120)
    bot.setControlState('jump', false)
    await sleep(far ? 480 : 380)
    hold([])
    return `${note}; then hopped ${far ? `forward and ${side}, closer` : side} to a new spot`
  }
  if (action === 'bow_draw') { // the draw as a task (v10): it starts and the call returns; Jev lets go or lowers it later
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (bowDrawn()) return `already drawing (${((Date.now() - bowDrawnAt) / 1000).toFixed(1)} s)`
    if (!await equip(i => i.name === 'bow')) return 'no bow'
    stopSprintForUse()
    bot.activateItem(); useStart = bowDrawnAt = Date.now()
    drawTarget = target; drawTrack.length = 0
    return `drawing the bow at ${target}: the aim follows them, full power at 1 s`
  }
  if (action === 'bow_release') {
    if (!bowDrawn()) return 'not drawing the bow'
    const sol = t ? arrowSolve(t, drawTrack) : null
    const ally = allyInLine(sol)
    if (ally) return `not loosed: ${ally} (your teammate) stands in the arrow's path - the bow stays drawn`
    const clear = t ? clearShot(t) : true
    if (sol) { await bot.lookAt(sol.aim, true); await sleep(50) }   // the look goes out on the next tick
    drawTarget = null
    lastRelease = sol ? { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y } : null
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return sol ? `let go: an arrow at ${target} (power ${Math.round(sol.power * 100)}%, ${(sol.ticks / 20).toFixed(2)} s flight, ` +
      `aimed ${sol.lead.toFixed(1)} blocks ahead of them${clear ? '' : '; a block stood in its way'})` : 'let go'
  }
  if (action === 'bow_cancel') {
    if (!bowDrawn()) return 'not drawing the bow'
    cancelDraw(); dodgeDropAt = 0
    return 'lowered the bow without shooting: you walk at full speed again'
  }
  if (action === 'peek_shot') { // drawn behind cover: lean out to the nearest clear spot, let go, lean back
    if (!t) return `cannot see ${target}`
    if (!bowDrawn()) return 'not drawing the bow'
    const spot = peekSpot(t)
    if (!spot) return `no spot within 2.5 blocks to lean out for a shot at ${target}`
    const home = new Vec3(Math.floor(bot.entity.position.x), feetLevel(), Math.floor(bot.entity.position.z))
    zigzag = null; circle = null
    goCover = { cell: spot.cell, until: Date.now() + 2500 }
    const t0 = Date.now()
    let let_go = null
    while (Date.now() - t0 < 2500 && bowDrawn()) {
      if (clearShot(t)) {
        const sol = arrowSolve(t, drawTrack)
        await bot.lookAt(sol.aim, true); await sleep(50)
        drawTarget = null
        lastRelease = { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y }
        bot.deactivateItem(); useStart = null; bowDrawnAt = null
        let_go = sol
        break
      }
      await sleep(40)
    }
    goCover = { cell: home, until: Date.now() + 1500 }   // back behind the cover
    if (!let_go) return bowDrawn() ? `leaned out but no clear shot came: back behind cover, the bow still drawn` : 'the draw was dropped'
    return `leaned out ${spot.d.toFixed(1)} blocks and let go at ${target} (power ${Math.round(let_go.power * 100)}%, ` +
      `${(let_go.ticks / 20).toFixed(2)} s flight), stepping back behind cover`
  }
  if (action === 'jump_shot') { // drawn behind cover: jump and let go at the top of the jump, over it
    if (!t) return `cannot see ${target}`
    if (!bowDrawn()) return 'not drawing the bow'
    bot.setControlState('jump', true)
    const t0 = Date.now()
    let clear = false
    while (Date.now() - t0 < 600) {
      await sleep(25)
      if (bot.entity.velocity.y <= 0.05 && !bot.entity.onGround) { clear = clearShot(t); if (clear) break }
    }
    bot.setControlState('jump', held.jump)
    if (!clear) return 'no clear shot at the top of the jump: the bow stays drawn'
    const sol = arrowSolve(t, drawTrack)
    const ally = allyInLine(sol)
    if (ally) return `not loosed: ${ally} (your teammate) stands in the arrow's path - the bow stays drawn`
    await bot.lookAt(sol.aim, true); await sleep(10)
    drawTarget = null
    lastRelease = { t: Date.now(), sol, eye: eyePos(), air: true, vy: bot.entity.velocity.y }
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return `jumped and let go over the cover (power ${Math.round(sol.power * 100)}%, ${(sol.ticks / 20).toFixed(2)} s flight)`
  }
  if (action === 'water_block') { // a block into the water's source: the water it feeds drains away
    const src = liquidSources().find(x => x.kind === 'water')
    if (!src) return 'no water source within reach'
    if (!await equip(i => BUILD.includes(i.name))) return 'no blocks'
    // standing in it, the body takes the cell: Jev steps off (or scoops it up) and plugs it next
    if (touching(src.pos) || occupied(src.pos)) return 'you stand in the water source: step off it first, or scoop it up'
    dropUse()
    // what to set it against: the block under it, else a solid neighbour
    const opts = [[0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]]
      .map(([x, y, z]) => ({ ref: src.pos.offset(x, y, z), face: new Vec3(-x, -y, -z) })).filter(q => solid(bot.blockAt(q.ref)))
    if (!opts.length) return 'nothing to set a block against at the water source'
    await placeFast([{ ...opts[0], cell: src.pos }])
    await sleep(120)
    return solid(bot.blockAt(src.pos)) ? `plugged the water source ${src.dist} blocks away with a block: its water drains away`
      : 'the block did not go into the water source'
  }
  if (action === 'water_escape') { // out of water on blocks: run and keep placing under your feet
    // toward the nearest dry cell (8 blocks at most, the four ways): while in the water a block goes into the cell
    // ahead at your feet's level and a jump climbs onto it; standing above the water, a block goes under the next
    // step and you walk on - until the ground ahead is dry, 8 blocks or 3 s
    const inWet = () => bot.entity.isInWater || /water/.test((bot.blockAt(bot.entity.position.floored()) || {}).name || '')
    if (!inWet()) return 'not in water'
    if (!await equip(i => BUILD.includes(i.name))) return 'no blocks'
    dropUse(); zigzag = null; circle = null; goCover = null
    const dd = dryDir() || (() => { const fx = -Math.sin(bot.entity.yaw), fz = -Math.cos(bot.entity.yaw)
      return Math.abs(fx) > Math.abs(fz) ? { dx: Math.sign(fx), dz: 0, k: null } : { dx: 0, dz: Math.sign(fz), k: null } })()
    const dir = new Vec3(dd.dx, 0, dd.dz)
    await bot.look(Math.atan2(-dd.dx, -dd.dz), 0.3, true)   // forward is (-sin yaw, -cos yaw)
    // one step a call (about 0.4 s): Jev chooses again after it - on, or a shot, a sidestep, a plug - as things change
    const t0 = Date.now()
    let placed = 0
    while (Date.now() - t0 < 600 && placed < 1) {
      const p = bot.entity.position, y = feetLevel()
      const here = new Vec3(Math.floor(p.x), y, Math.floor(p.z)), ahead = here.plus(dir), supAhead = ahead.offset(0, -1, 0)
      const bA = bot.blockAt(ahead), bS = bot.blockAt(supAhead), wet = bot.entity.isInWater
      if (!bA || !bS) break
      if (!wet && solid(bS) && !/water/.test(bA.name) && !/water/.test(bS.name)) { // dry ground ahead: step onto it
        hold(['forward']); await sleep(300); hold([])
        break
      }
      let target = null, ref = null, face = null
      if (wet && !solid(bA)) { // in it: a step at your feet's level
        target = ahead
        if (solid(bS)) { ref = bS; face = new Vec3(0, 1, 0) } else { ref = bot.blockAt(here.offset(0, -1, 0)); face = dir }
      } else if (!wet && !solid(bS)) { // above it: the next step under your feet
        target = supAhead; ref = bot.blockAt(here.offset(0, -1, 0)); face = dir
      }
      if (target) {
        if (!ref || !solid(ref)) break
        hold([])
        if (touching(target)) { hold(['back']); await sleep(120); hold([]) }   // your body in the cell: the server refuses it
        if (await place(ref, face)) placed++
        else break
      }
      hold(wet ? ['forward', 'jump'] : ['forward'])
      await sleep(wet ? 350 : 220)
    }
    hold([])
    return inWet() ? `a step out of the water on ${placed} block${placed === 1 ? '' : 's'}` +
      (dd.k ? `; dry ground ${Math.max(0, dd.k - 1)} blocks on` : '') : `out of the water on ${placed} block${placed === 1 ? '' : 's'}`
  }
  if (action === 'quick_shot') { // a short draw (0.6-0.8 s, about 3/4 power; the aim knows the power): at short range
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (!bowDrawn()) {
      if (!await equip(i => i.name === 'bow')) return 'no bow'
      stopSprintForUse()
      bot.activateItem(); useStart = bowDrawnAt = Date.now()
      drawTarget = target; drawTrack.length = 0
    }
    const how = await waitRelease(t, 'quick')
    if (!how) return reflex && Date.now() - reflex.t < 1500 ? `the draw was cut short: ${reflex.what}` : 'the draw was dropped'
    const sol = arrowSolve(t, drawTrack)
    sol.plan = 'quick'
    await bot.lookAt(sol.aim, true); await sleep(60)
    drawTarget = null
    lastRelease = { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y }
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return `shot a quick arrow at ${target} (${how}, power ${sol.power}; ${(sol.ticks / 20).toFixed(2)} s flight)`
  }
  if (action === 'shoot_full') { // draw with the aim on them and their lead, and let go on a rhythm they cannot read
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    const plan = shotPlan(t)
    let redraws = 0, how = null
    for (;;) {
      if (!bowDrawn()) {
        if (!await equip(i => i.name === 'bow')) return 'no bow'
        stopSprintForUse()
        bot.activateItem(); useStart = bowDrawnAt = Date.now()
        drawTarget = target; drawTrack.length = 0
      }
      how = await waitRelease(t, plan)
      if (how === 'cover') return `held the full draw on ${target}'s cover for 5 s; they did not come out (the bow stays drawn, the aim on them)`
      if (how) break
      // the draw was lost: after a sidestep that dropped it, draw again (twice at most)
      if (Date.now() - dodgeDropAt < 1500 && redraws < 2) { redraws++; while (dodge) await sleep(20); continue }
      return reflex && Date.now() - reflex.t < 1500 ? `the draw was cut short: ${reflex.what}` : 'the draw was dropped'
    }
    const sol = arrowSolve(t, drawTrack)
    sol.plan = plan
    await bot.lookAt(sol.aim, true); await sleep(60)
    drawTarget = null
    lastRelease = sol ? { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y } : null
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return `shot a${sol.power >= 1 ? ' full-power' : 'n'} arrow at ${target} (${how}; ${(sol.ticks / 20).toFixed(2)} s flight, aimed ` +
      `${sol.lead.toFixed(1)} blocks ahead of them${redraws ? `; drawn again after ${redraws} sidestep${redraws > 1 ? 's' : ''}` : ''})`
  }
  if (action === 'draw_bow') {
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (bowDrawn()) return 'already drawing'
    if (!await equip(i => i.name === 'bow')) return 'no bow'
    stopSprintForUse()
    bot.activateItem(); useStart = bowDrawnAt = Date.now()
    drawTarget = target; drawTrack.length = 0
    await bot.lookAt(arrowSolve(t, []).aim, true)
    return `drawing the bow at ${target} (full power after 1 s)`
  }
  if (action === 'release_bow') {
    if (!bowDrawn()) return 'not drawing the bow'
    if (bowRules.full && bowDrawnAt) { // finish the draw first: full power at 1 s
      while (Date.now() - bowDrawnAt < 1000 && bowDrawn()) await sleep(20)
      if (!bowDrawn()) return 'the draw was dropped before it was full'
    }
    const s = useStart ? (Date.now() - useStart) / 1000 : 0
    let lead = '', sol = null
    if (t) {
      sol = arrowSolve(t, drawTrack)
      await bot.lookAt(sol.aim, true); await sleep(60)
      lead = `; aimed ${sol.lead.toFixed(1)} blocks ahead of them for a ${(sol.ticks / 20).toFixed(2)} s flight`
    }
    drawTarget = null
    lastRelease = sol ? { t: Date.now(), sol, eye: eyePos(), air: !bot.entity.onGround, vy: bot.entity.velocity.y } : null
    bot.deactivateItem(); useStart = null; bowDrawnAt = null
    return `shot an arrow at ${target} (drawn ${s.toFixed(1)} s${s < 1 ? ': weak' : ''}${lead})`
  }
  if (action === 'eat_gapple') {
    const it = await equip(i => i.name === 'golden_apple')
    if (!it) return 'no golden apples'
    const count = () => bot.inventory.items().filter(i => i.name === 'golden_apple').reduce((a, i) => a + i.count, 0)
    const n0 = count()
    stopSprintForUse()
    bot.activateItem()
    const t0 = Date.now()
    while (Date.now() - t0 < 2200 && count() >= n0) await sleep(50)
    bot.deactivateItem()
    return count() < n0 ? `ate a golden apple (${count()} left)` : 'eating was interrupted'
  }
  if (action === 'shoot_bow') {
    if (!t) return `cannot see ${target}`
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (!await equip(i => i.name === 'bow')) return 'no bow'
    const track = []
    stopSprintForUse()
    bot.activateItem()
    const t0 = Date.now()
    while (Date.now() - t0 < 1050) { // a bow is fully drawn after 1 s
      track.push({ t: Date.now(), p: t.position.clone() })
      while (track.length > 5) track.shift()
      await bot.lookAt(arrowAim(t, track), true)
      await sleep(50)
    }
    await bot.lookAt(arrowAim(t, track), true)
    await sleep(60) // the last look reaches the server before the release
    bot.deactivateItem()
    return `shot an arrow at ${target} (${round(eyePos().distanceTo(t.position), 1)} blocks)`
  }
  if (action === 'throw_rod') {
    if (!t) return `cannot see ${target}`
    if (!await equip(i => i.name === 'fishing_rod')) return 'no fishing rod'
    const d = eyePos().distanceTo(t.position)
    await bot.lookAt(t.position.offset(0, 1.5 + 0.007 * d * d, 0), true)
    await sleep(60)
    bot.activateItem() // cast
    await sleep(Math.min(700, 150 + d * 45))
    bot.activateItem() // reel in
    return `cast the rod at ${target} and reeled in (${round(d, 1)} blocks)`
  }
  if (action === 'place_block' || action === 'pillar_up') {
    const it = await equip(i => BUILD.includes(i.name))
    if (!it) return 'no blocks'
    const feet = new Vec3(Math.floor(bot.entity.position.x), feetLevel(), Math.floor(bot.entity.position.z))
    if (action === 'pillar_up') {
      const below = bot.blockAt(feet.offset(0, -1, 0))
      if (!solid(below)) return 'nothing solid under you to build on'
      if (solid(bot.blockAt(feet.offset(0, 2, 0)))) return 'no room above your head'
      hold([])
      await bot.look(bot.entity.yaw, -Math.PI / 2, true)
      bot.setControlState('jump', true)
      const t0 = Date.now()
      // near the top of the jump, and one more tick so the server has you clear of the cell before the block
      while (Date.now() - t0 < 500 && bot.entity.position.y < feet.y + 1.1) await sleep(20)
      await sleep(50)
      const ok = await place(below, new Vec3(0, 1, 0))
      bot.setControlState('jump', false)
      await sleep(250)
      return ok ? 'pillared up one block; keys released' : 'the block did not go down'
    }
    const dir = facingCell()
    const cell = feet.plus(dir)
    let ref, face, what
    if (replaceable(bot.blockAt(cell))) {
      if (solid(bot.blockAt(cell.offset(0, -1, 0)))) { ref = bot.blockAt(cell.offset(0, -1, 0)); face = new Vec3(0, 1, 0); what = 'a block in front of you, at your feet' } else { ref = bot.blockAt(feet.offset(0, -1, 0)); face = dir; what = 'a block out over the edge in front of you (a bridge)' }
      if (occupied(cell)) return 'someone is standing there'
    } else if (replaceable(bot.blockAt(cell.offset(0, 1, 0)))) {
      ref = bot.blockAt(cell); face = new Vec3(0, 1, 0); what = 'a block in front of you, at head height (a wall two high)'
      if (occupied(cell.offset(0, 1, 0))) return 'someone is standing there'
    } else {
      return 'the way ahead is already blocked'
    }
    if (!solid(ref)) return 'nothing to place it against'
    if (touching(face.y ? ref.position.plus(face) : cell)) return 'you are pressed against that spot: step back first'
    return (await place(ref, face)) ? `placed ${it.name}: ${what}` : `the block did not go down (${placeError})`
  }
  if (action === 'water_here') {
    if (!await equip(i => i.name === 'water_bucket')) return 'no water bucket'
    let ground = null
    for (let dy = 1; dy <= 5 && !ground; dy++) { const b = bot.blockAt(new Vec3(Math.floor(bot.entity.position.x), feetLevel() - dy, Math.floor(bot.entity.position.z))); if (solid(b)) ground = b }
    if (!ground) return 'no ground within reach below you'
    // the pour is checked: the bucket in hand turns empty when the server took it (once, a second try)
    const emptied = async () => { const t0 = Date.now(); while (Date.now() - t0 < 200) { if (bot.heldItem && bot.heldItem.name === 'bucket') return true; await sleep(20) } return false }
    for (const off of [[0.5, 0.5], [0.35, 0.65]]) {
      await bot.lookAt(ground.position.offset(off[0], 1, off[1]), true)
      await sleep(60)
      bot.activateItem()
      if (await emptied()) return 'poured water where you stand'
    }
    return 'the water did not pour (the server did not take it)'
  }
  if (action === 'water_pickup' || action === 'lava_pickup') {
    const kind = action.split('_')[0]
    const src = liquidSources().find(s => s.kind === kind)
    if (!src) return `no ${kind} source within reach`
    if (!await equip(i => i.name === 'bucket')) return 'no empty bucket'
    const yaw = bot.entity.yaw, pitch = bot.entity.pitch
    await bot.lookAt(src.pos.offset(0.5, 0.8, 0.5), true)
    await sleep(60)
    bot.activateItem()
    await sleep(60)
    await bot.look(yaw, pitch, true)
    ownLava.delete(cellKey(src.pos))
    return `scooped up the ${kind} (${src.dist} blocks away)`
  }
  if (action === 'lava_at') {
    if (!t) return `cannot see ${target}`
    const plan = planPour(cellsAt(t), LAVA_SELF)
    if (!plan) return `no clear, safe aim to pour lava at ${target} from here (it would land elsewhere or within ${LAVA_SELF} blocks of you)`
    if (!await equip(i => i.name === 'lava_bucket')) return 'no lava bucket'
    await pourAt(plan)
    ownLava.set(cellKey(plan.cell), Date.now())
    const d = Math.hypot(plan.cell.x + 0.5 - t.position.x, plan.cell.z + 0.5 - t.position.z)
    return `poured lava ${d < 0.8 ? `at ${target}'s feet` : `${d.toFixed(1)} blocks from ${target}`}`
  }
  if (action === 'break_block') {
    const cell = new Vec3(Math.floor(bot.entity.position.x), feetLevel(), Math.floor(bot.entity.position.z)).plus(facingCell())
    const b = [cell.offset(0, 1, 0), cell].map(c => bot.blockAt(c)).find(x => solid(x) && x.name !== 'bedrock' && x.name !== 'barrier')
    if (!b) return 'nothing to break in front of you'
    const wood = /log|planks|leaves/.test(b.name)
    await equip(i => i.name.endsWith(wood ? '_axe' : '_pickaxe'))
    try {
      await withTimeout(bot.dig(b, true), 3000)
      return `broke the ${b.name} in front of you`
    } catch (err) {
      bot.stopDigging()
      return `could not break the ${b.name}`
    }
  }
  if (action === 'crit_attack') {
    if (!t) return `cannot see ${target}`
    if (!bot.entity.onGround) return 'not on the ground'
    held.sprint = false
    bot.setControlState('sprint', false) // a critical hit cannot be a sprint hit
    bot.setControlState('jump', true)
    const t0 = Date.now()
    while (Date.now() - t0 < 700 && (bot.entity.onGround || bot.entity.velocity.y > -0.1)) await sleep(20)
    await sleep(50) // the server counts the fall from the position packets: let one go down first
    bot.setControlState('jump', held.jump)
    const falling = !bot.entity.onGround && bot.entity.velocity.y < 0
    const note = await strike(t)
    return falling ? `jumped and, coming down, ${note} (a critical hit if it landed)` : `${note} (not falling: no critical)`
  }
  throw new Error(`unknown item action ${action}`)
}

function observe ({ since = 0 }) {
  const e = bot.entity
  const eye = e.position.offset(0, e.height, 0)
  const cursor = bot.blockAtCursor(6)
  const target = bot.entityAtCursor(6)
  const players = Object.values(bot.players)
    .filter(p => p.username !== bot.username)
    .map(p => ({
      name: p.username,
      uuid: p.uuid,   // version 3 (offline): our bots, joined directly; 4: a person through the online-mode proxy
      pos: p.entity ? { x: round(p.entity.position.x), y: round(p.entity.position.y), z: round(p.entity.position.z) } : null
    }))
  return {
    t: Date.now(),
    name: bot.username,
    pos: { x: round(e.position.x), y: round(e.position.y), z: round(e.position.z) },
    yaw: e.yaw,
    pitch: e.pitch,
    on_ground: e.onGround,
    health: bot.health,
    food: bot.food,
    gamemode: bot.game.gameMode,
    cursor: cursor && {
      name: cursor.name,
      pos: cursor.position,
      face: cursor.face,
      distance: round(eye.distanceTo(cursor.position.offset(0.5, 0.5, 0.5)), 1)
    },
    cursor_entity: target && {
      name: target.username || target.name,
      type: target.type,
      distance: round(eye.distanceTo(target.position.offset(0, target.height / 2, 0)), 1)
    },
    players,
    fighters: Object.values(bot.players).filter(p => p.username !== bot.username).map(playerState),
    recharge: round(Math.min(1, (Date.now() - rechargeStart) / rechargeMs())),
    recharge_s: round(rechargeMs() / 1000),
    sprint_fresh: lastSprintPress > lastAttack,
    draw: drawTarget && targetEntity(drawTarget) && (() => { const sol = arrowSolve(targetEntity(drawTarget), drawTrack); return { flight_s: round(sol.ticks / 20, 2), lead: round(sol.lead, 1), speed: round(sol.speed, 1), kind: sol.kind } })(),
    reflex: reflex && { ago: round((Date.now() - reflex.t) / 1000, 1), what: reflex.what },
    zigzag: Boolean(zigzag),
    in_water: Boolean(bot.entity.isInWater),
    dry: bot.entity.isInWater ? dryDir() : null,
    circle: Boolean(circle),
    going_cover: Boolean(goCover),
    cover: fightTarget && targetEntity(fightTarget) ? coverState(targetEntity(fightTarget)) : null,
    their_draw_s: theirDrawAt ? round((Date.now() - theirDrawAt) / 1000, 1) : null,
    ping: rttMs(),
    last_shot: lastShot && { ...lastShot, ago: round((Date.now() - lastShot.t) / 1000, 1) },
    events: events.map(e => ({ ...e, ago: round((Date.now() - e.t) / 1000, 1) })),
    task: (() => { // the draw in progress, for Jev to let go of or lower
      if (!bowDrawn()) return null
      const t = (drawTarget && targetEntity(drawTarget)) || (fightTarget && targetEntity(fightTarget)), s = (Date.now() - bowDrawnAt) / 1000
      const trk = drawTarget ? drawTrack : seenTrack   // aimed by the body, or by your own look
      const out = { kind: 'draw', s: round(s, 2), power: round(drawPower(s), 2), to_full: round(Math.max(0, 1.0 - s), 2) }
      if (t) {
        const sol = arrowSolve(t, trk), full = arrowSolve(t, trk, 3)
        const ex = exposedPoint(t)
        Object.assign(out, { clear: !!ex, part: ex && !ex.center ? ex.part : null, flight_s: round(sol.ticks / 20, 2), full_flight_s: round(full.ticks / 20, 2),
          lead: round(sol.lead, 1), path: sol.kind, sure: round(sol.share, 2) })
      }
      return out
    })(),
    their_move: fightTarget && targetEntity(fightTarget) ? theirMove() : null,
    their_hidden_s: theirCoveredSince ? round((Date.now() - theirCoveredSince) / 1000, 1) : null,
    peek: (() => { // drawn and blocked: a spot to lean out to, looked for twice a second at most
      const t = fightTarget && targetEntity(fightTarget)
      if (!t || !bowDrawn() || clearShot(t)) return null
      if (Date.now() - peekCache.t > 500) peekCache = { t: Date.now(), v: peekSpot(t) }
      return peekCache.v && { d: round(peekCache.v.d, 1) }
    })(),
    angle: (() => { // a spot with a clear shot at a hider (3 s+), looked for twice a second at most
      const t = fightTarget && targetEntity(fightTarget)
      if (!t || !theirCoveredSince || Date.now() - theirCoveredSince < 3000) return null
      if (Date.now() - angleCache.t > 500) angleCache = { t: Date.now(), v: angleSpot(t) }
      return angleCache.v && { d: round(angleCache.v.d, 1) }
    })(),
    their_rhythm: rhythm.runs.length >= 3 ? { run_s: round([...rhythm.runs].sort((a, b) => a - b)[Math.floor(rhythm.runs.length / 2)] / 20, 2), runs: rhythm.runs.length } : null,
    their_arrow: lastArrow && { ago: round((Date.now() - lastArrow.t) / 1000, 1), would_hit: lastArrow.hit },
    incoming: incoming && Date.now() < incoming.t + incoming.tick * 50 + 100
      ? { in_s: round(Math.max(0, incoming.t + incoming.tick * 50 - Date.now()) / 1000, 2), key: incoming.key } : null,
    pour: fightTarget && targetEntity(fightTarget) ? pourPlans(targetEntity(fightTarget)) : null,
    engage: engage && { target: engage.target, style: engage.style, s: round((Date.now() - engage.since) / 1000, 1), swings: engage.swings },
    workshop: wsState(),   // a Workshop Jev's kept actions (null for every other harness)
    combat: combat.map(c => ({ ...c, ago: round((Date.now() - c.t) / 1000, 1) })),
    mobs: mobs(),
    deaths,
    last_death: lastDeath,
    held: Object.keys(held).filter(k => held[k]),
    held_item: bot.heldItem ? bot.heldItem.name : null,
    drops: drops.map(d => ({ ago: round((Date.now() - d.t) / 1000, 1), from: round(d.from, 1), to: round(d.to, 1) })),
    knock: knock && { ago: round((Date.now() - knock.t) / 1000, 1), vx: round(knock.vx, 3), vz: round(knock.vz, 3) },
    hotbar: bot.inventory.slots.slice(36, 45).map(i => i && { name: i.name, count: i.count }),
    slot: bot.quickBarSlot,
    turn_rate: turnRate,   // raw: the mouse held moving, degrees a second (+ right)
    using_s: bowDrawn() ? round((Date.now() - bowDrawnAt) / 1000, 1)
      : bot.usingHeldItem && useStart ? round((Date.now() - useStart) / 1000, 1) : null,  // right button held
    items: bot.inventory.items().reduce((m, i) => { m[i.name] = (m[i.name] || 0) + i.count; return m }, {}),
    burning: Boolean((e.metadata && e.metadata[0]) & 0x01),
    absorption: e.metadata && typeof e.metadata[11] === 'number' ? round(e.metadata[11], 1) : 0,
    effects: Object.values(e.effects || {}).map(f => ({ id: f.id, amp: f.amplifier, s: round(f.duration / 20, 1) })),
    falling: !e.onGround && e.velocity.y < -0.1,
    sources: liquidSources(),
    lava_stop: lastLavaStop && round((Date.now() - lastLavaStop) / 1000, 1),   // s since the lava guard let go
    stance: stance,                                                              // keys holding the middle off lava
    edge_stop: lastEdgeStop && round((Date.now() - lastEdgeStop) / 1000, 1),  // s since the edge guard let go
    grid: { r: GRID_R, dys: GRID_DYS, layers: grid() },
    chat: chatLog.filter(m => m.seq > since)
  }
}

// ---------------------------------------------------------------- Jev Workshop programs (act 'program')
// A Workshop Jev (agent_workshop.py) plays a Jev a player wrote on the website: each decision sends the does: of the
// action Jev picked, as workshop/lang.py trees (JSON: ["call", f, [args], {kw}], ["attr", obj, prop], ...). They run
// in order. keep(x) runs x again every tick (never twice at once) until a later decision uses x's channel - look
// (look, turn, face), move (hold, press, release, walk_to, climb_out_of_water), hands (equip, click, the right
// button, place), dodge (dodge_arrows) - or stop() ends it. Conditions read the body's own values where it has them
// (the sword's recharge, reach, falling, the draw...: wsLive) and the ones the agent worked out this step (env) for
// the rest. Nothing here runs for a harness that never sends 'program'.
const WS_CHANNEL = { look: 'look', turn: 'look', face: 'look', hold: 'move', press: 'move', release: 'move', walk_to: 'move',
  climb_out_of_water: 'move', circle: 'move', equip: 'hands', click: 'hands', right_click: 'hands', hold_right_click: 'hands',
  release_right_click: 'hands', place: 'hands', dodge_arrows: 'dodge' }
const WS_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']
const WS_ITEM = { sword: i => /_sword$/.test(i.name), bow: i => i.name === 'bow', blocks: i => BUILD.includes(i.name),
  lava_bucket: i => i.name === 'lava_bucket', water_bucket: i => i.name === 'water_bucket', empty_bucket: i => i.name === 'bucket',
  pickaxe: i => /_pickaxe$/.test(i.name) }
const WS_FACE = { north: 0, west: Math.PI / 2, south: Math.PI, east: -Math.PI / 2 }   // mineflayer yaw: 0 faces -z
let wsKept = []            // {tree, ch: Set, kinds: Set, busy}
let wsEnv = {}
let wsLiveAt = 0, wsLiveVal = null
let wsSwings = 0, wsClickSince = null
let wsLookAt = 0           // when a program last turned the head: a bucket waits for the turn to reach the server

function wsWalk (t, fn) { // every node of a tree
  if (!Array.isArray(t)) return
  fn(t)
  if (t[0] === 'call') { for (const a of t[2] || []) wsWalk(a, fn); for (const a of Object.values(t[3] || {})) wsWalk(a, fn) }
  else if (t[0] === 'fstr') { for (const p of t[1]) if (p[0] === 'expr') wsWalk(p[1], fn) }
  else for (const x of t.slice(1)) if (Array.isArray(x)) wsWalk(x, fn)
}
function wsChannels (t) { const s = new Set(); wsWalk(t, n => { if (n[0] === 'call' && WS_CHANNEL[n[1]]) s.add(WS_CHANNEL[n[1]]) }); return s }
function wsMain (t) { // the channel of what a kept action finally does: keep(when(ready, seq(press(jump), click()))) is hands,
  // so a decision to move leaves a sword fight going, as the official body's engage does
  if (!Array.isArray(t)) return new Set()
  if (t[0] === 'if') return new Set([...wsMain(t[2]), ...wsMain(t[3])])
  if (t[0] !== 'call') return new Set()
  const f = t[1], a = t[2] || []
  if (WS_CHANNEL[f]) return new Set([WS_CHANNEL[f]])
  if (f === 'when') return wsMain(a[1])
  if (f === 'keep') return wsMain(a[0])
  if (f === 'seq') return a.length ? wsMain(a[a.length - 1]) : new Set()
  return new Set()
}
let wsBusy = false        // a decision's actions are running: the round's kept reflexes leave the hands alone meanwhile
const wsGhost = new Map() // cells a block was just sent into (the server has them before this client does)
function wsKinds (t) { const s = new Set(); wsWalk(t, n => { if (n[0] === 'call') s.add(n[1]) }); return s }

function wsLive () { // the values the body knows itself, fresh each tick
  if (wsLiveVal && Date.now() - wsLiveAt < 40) return wsLiveVal
  const e = bot.entity
  const h = bot.heldItem ? bot.heldItem.name : ''
  const weapon = /_sword$|_axe$/.test(h)
  const recharge = Math.min(1, (Date.now() - rechargeStart) / rechargeMs())
  const me = { health: bot.health, held: h ? h.replace(/_/g, ' ').replace(/^bucket$/, 'empty bucket') : 'nothing',
    sword_ready: weapon && recharge >= 1, recharge_left: weapon ? round((1 - recharge) * rechargeMs() / 1000, 2) : 0,
    drawing: bowDrawn(), draw_secs: bowDrawn() ? round((Date.now() - bowDrawnAt) / 1000, 2) : 0,
    falling: !e.onGround && e.velocity.y < -0.1, in_water: Boolean(e.isInWater),
    fighting: wsKept.some(k => k.kinds.has('click')) }
  const t = fightTarget && targetEntity(fightTarget)
  const opponent = t ? { visible: true, dist: round(Math.hypot(t.position.x - e.position.x, t.position.z - e.position.z)),
    in_reach: eyePos().distanceTo(t.position.offset(0, 0.9, 0)) <= REACH + 0.4 } : { visible: false }
  wsLiveAt = Date.now()
  wsLiveVal = { me, opponent }
  return wsLiveVal
}

function wsNum (x) { if (typeof x !== 'number') throw new Error('a number is needed here'); return x }
function wsShow (v, spec) {
  if (spec) return wsNum(v).toFixed(Number(spec[1]))
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Math.round(v * 10) / 10)
  return v === null || v === undefined ? '' : String(v)
}
function wsArgs (t) { // a call's arguments: positional values and name= values, worked out
  return { a: (t[2] || []).map(x => wsValue(x)), kw: Object.fromEntries(Object.entries(t[3] || {}).map(([k, x]) => [k, wsValue(x)])) }
}
function wsValue (t) { // workshop/lang.py value(), for conditions and numbers in a program
  switch (t[0]) {
    case 'num': case 'str': case 'bool': return t[1]
    case 'name': { const v = wsEnv[t[1]]; return v && typeof v === 'object' ? (v.name !== undefined ? v.name : t[1]) : v }
    case 'attr': { const live = wsLive()[t[1]]; if (live && t[2] in live) return live[t[2]]; const o = wsEnv[t[1]] || {}; return o[t[2]] }
    case 'neg': return -wsNum(wsValue(t[1]))
    case 'not': return !wsValue(t[1])
    case 'and': return wsValue(t[1]) && wsValue(t[2])
    case 'or': return wsValue(t[1]) || wsValue(t[2])
    case 'if': return wsValue(t[1]) ? wsValue(t[2]) : wsValue(t[3])
    case 'fstr': return t[1].map(p => p[0] === 'lit' ? p[1] : wsShow(wsValue(p[1]), p[2])).join('')
    case 'bin': {
      const op = t[1], a = wsValue(t[2]), b = wsValue(t[3])
      if (op === '+' && (typeof a === 'string' || typeof b === 'string')) return wsShow(a) + wsShow(b)
      if (op === '==') return a === b
      if (op === '!=') return a !== b
      const x = wsNum(a), y = wsNum(b)
      return { '+': x + y, '-': x - y, '*': x * y, '/': y ? x / y : Infinity, '<': x < y, '>': x > y, '<=': x <= y, '>=': x >= y }[op]
    }
    case 'call': {
      const { a, kw } = wsArgs(t)
      switch (t[1]) {
        case 'abs': return Math.abs(wsNum(a[0]))
        case 'round': { const d = kw.digits !== undefined ? kw.digits : a[1] || 0; return Math.round(wsNum(a[0]) * 10 ** d) / 10 ** d }
        case 'min': return Math.min(...a.map(wsNum))
        case 'max': return Math.max(...a.map(wsNum))
        case 'count': return a.filter(Boolean).length
        case 'join': return a.filter(x => x !== '' && x !== null && x !== false && x !== undefined).map(x => wsShow(x)).join(kw.sep !== undefined ? kw.sep : ', ')
        case 'plural': return !a[0] ? '' : `${wsShow(a[0])} ${a[0] === 1 ? a[1] : (kw.many || a[2] || a[1] + 's')}`
      }
    }
  }
  throw new Error(`cannot work out ${JSON.stringify(t).slice(0, 60)}`)
}

function wsPlace (t) { // a place -> {pos: where to look, cell: its block, entity, cover} or null (not there now)
  const e = bot.entity
  if (t[0] === 'attr' && t[1] === 'opponent') {
    const o = fightTarget && targetEntity(fightTarget)
    if (!o) return null
    if (t[2] === 'feet') {   // the top of the block they stand on, their cell's middle: a bucket poured there fills their cell
      const c = o.position.floored()
      return { pos: new Vec3(c.x + 0.5, Math.floor(o.position.y + 0.01) - 0.02, c.z + 0.5), cell: c, entity: o, aimY: 0.1 }
    }
    return { pos: o.position.offset(0, { head: 1.6, body: 0.9 }[t[2]], 0), cell: o.position.floored(), entity: o,
      aimY: { head: AIM_Y + 0.7, body: AIM_Y }[t[2]] }
  }
  if (t[0] !== 'call') return null
  const { a, kw } = wsArgs(t)
  const n = (i, k, d) => kw[k] !== undefined ? kw[k] : a[i] !== undefined ? a[i] : d
  const feet = new Vec3(Math.floor(e.position.x), feetLevel(), Math.floor(e.position.z))
  const fwd = facingCell(), right = new Vec3(-fwd.z, 0, fwd.x)
  if (t[1] === 'ground') { const c = feet.plus(fwd.scaled(n(0, 'ahead', 1))).offset(0, -1, 0); return { pos: c.offset(0.5, 1, 0.5), cell: c } }
  if (t[1] === 'block') {
    const c = feet.plus(fwd.scaled(n(0, 'ahead', 1))).plus(right.scaled(n(2, 'right', 0) - n(1, 'left', 0))).offset(0, n(3, 'up', 0) - n(4, 'down', 0), 0)
    return { pos: c.offset(0.5, 0.5, 0.5), cell: c }
  }
  if (t[1] === 'cover') {
    const o = fightTarget && targetEntity(fightTarget)
    const s = o && coverSpot(o)
    return s ? { pos: s.cell.offset(0.5, 0, 0.5), cell: s.cell, cover: true } : null
  }
  if (t[1] === 'nearest') {
    const kind = (t[2][0] || [])[1]
    const s = liquidSources().find(x => x.kind === kind)
    return s ? { pos: s.pos.offset(0.5, 0.8, 0.5), cell: s.pos } : null
  }
  return null
}

function wsKeysOf (t) { return (t[2] || []).map(x => x[1]).filter(k => WS_KEYS.includes(k)) }
function wsSetKeys (keys) { // hold: exactly these movement keys down (and sneak)
  zigzag = null; circle = null; goCover = null
  if (keys.includes('sprint') && !held.sprint) lastSprintPress = Date.now()
  hold(keys.filter(k => k in held))
  bot.setControlState('sneak', keys.includes('sneak'))
}

async function wsDo (t) { // one action of a program; resolves when it is done -> a note, or ''
  if (!Array.isArray(t)) return ''
  if (t[0] === 'if') return wsDo(wsValue(t[1]) ? t[2] : t[3])
  if (t[0] !== 'call') return ''
  const f = t[1], args = t[2] || [], kw = t[3] || {}
  const v = (i, k, d) => kw[k] !== undefined ? wsValue(kw[k]) : args[i] !== undefined ? wsValue(args[i]) : d
  switch (f) {
    case 'nothing': return ''
    case 'keep': wsKeep(args[0], wsPersisting); return ''
    case 'stop': { const k = args[0] ? args[0][1] : 'all'; wsStop(k, k !== 'all'); return k === 'all' ? 'stopped every kept action but start:\'s' : `stopped the kept ${k}` }
    case 'when': return wsValue(args[0]) ? wsDo(args[1]) : ''
    case 'seq': { const notes = []; for (const x of args) { const n = await wsDo(x); if (n) notes.push(n) } return notes.join('; ') }
    case 'wait': await sleep(Math.max(0, Math.min(2000, v(0, 'ms', 0)))); return ''
    case 'wait_until': {
      const t0 = Date.now(), max = Math.max(0, Math.min(2000, v(1, 'max', 1000)))
      while (Date.now() - t0 < max && !wsValue(args[0])) await sleep(25)
      return ''
    }
    case 'look': {
      const p = wsPlace(args[0])
      if (!p) return 'nothing to look at there now'
      drawTarget = null
      if (p.entity && v(1, 'predict_arrow', false)) {   // the arrow's aim: where it meets them, over its fall - drawn or not
        await bot.lookAt(arrowSolve(p.entity, seenTrack, undefined, p.aimY).aim, true)
      } else await bot.lookAt(p.pos, true)
      wsLookAt = Date.now()
      return ''
    }
    case 'turn': {
      const e = bot.entity
      drawTarget = null
      const pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, e.pitch + rad(v(2, 'up', 0) - v(3, 'down', 0))))
      await bot.look(wrap(e.yaw + rad(v(0, 'left', 0) - v(1, 'right', 0))), pitch, true)
      wsLookAt = Date.now()
      return ''
    }
    case 'face': drawTarget = null; await bot.look(WS_FACE[args[0][1]], bot.entity.pitch, true); wsLookAt = Date.now(); return ''
    case 'hold': wsSetKeys(wsKeysOf(t)); return ''
    case 'press': {
      const keys = wsKeysOf(t)
      for (const k of keys) bot.setControlState(k, true)
      if (keys.includes('sprint')) lastSprintPress = Date.now()
      await sleep(Math.max(0, Math.min(2000, kw.ms !== undefined ? wsValue(kw.ms) : 100)))   // ms comes by name: the keys take the rest
      for (const k of keys) bot.setControlState(k, k === 'sneak' ? false : Boolean(held[k]))
      return ''
    }
    case 'release': {
      const keys = wsKeysOf(t)
      if (!keys.length) { wsSetKeys([]); return '' }
      for (const k of keys) { if (k in held) held[k] = false; bot.setControlState(k, false) }
      return ''
    }
    case 'circle': return await engageAction('circle', fightTarget)
    case 'walk_to': {
      const p = wsPlace(args[0])
      if (!p) return 'no such place now'
      if (p.cover) return await engageAction('take_cover', fightTarget)
      zigzag = null; circle = null
      goCover = { cell: p.cell, until: Date.now() + 2500 }
      return `walking to ${p.cell.x} ${p.cell.y} ${p.cell.z}`
    }
    case 'climb_out_of_water': itemBusy = true; try { return await itemAction('water_escape', fightTarget) } finally { itemBusy = false }
    case 'dodge_arrows': dodgeOn = true; return ''
    case 'equip': {
      const pred = WS_ITEM[args[0][1]]
      if (bot.heldItem && pred(bot.heldItem)) return ''
      dropUse()
      return (await equip(pred)) ? '' : `no ${args[0][1].replace(/_/g, ' ')}`
    }
    case 'click': { if (!wsClickSince) wsClickSince = Date.now(); wsSwings++; return leftClick() }
    case 'right_click': {
      const since = Date.now() - wsLookAt   // the server pours where it last heard you look: give the turn a tick
      if (bot.heldItem && /bucket$/.test(bot.heldItem.name) && since < 60) await sleep(60 - since)
      const lava = bot.heldItem && bot.heldItem.name === 'lava_bucket' && bot.blockAtCursor(5)
      const note = await rightClick()
      if (lava && lava.face !== undefined) ownLava.set(cellKey(lava.position.plus(FACES[lava.face])), Date.now())
      return note
    }
    case 'hold_right_click': return bowDrawn() ? '' : await rightClick()
    case 'release_right_click': return bowDrawn() || bot.usingHeldItem ? await rawAction('use_release', {}) : ''
    case 'place': {   // the quick way, as the official body builds: the place packet itself, no turn of the head, one a tick
      const p = wsPlace(args[0])
      if (!p) return 'no such place now'
      const c = p.cell
      const ghost = q => (wsGhost.get(cellKey(q)) || 0) > Date.now() - 1500
      if (!replaceable(bot.blockAt(c)) || ghost(c)) return ''       // there already
      if (occupied(c) || touching(c)) return 'someone is in that spot'
      if (!await equip(i => BUILD.includes(i.name))) return 'no blocks'
      for (const face of [new Vec3(0, 1, 0), new Vec3(1, 0, 0), new Vec3(-1, 0, 0), new Vec3(0, 0, 1), new Vec3(0, 0, -1), new Vec3(0, -1, 0)]) {
        const r = c.minus(face)
        if ((solid(bot.blockAt(r)) || ghost(r)) && eyePos().distanceTo(r.offset(0.5, 0.5, 0.5)) <= 4.5) {
          await placeFast([{ ref: r, face }])
          wsGhost.set(cellKey(c), Date.now())
          return ''
        }
      }
      return 'nothing to place it against'
    }
  }
  throw new Error(`unknown workshop action ${f}`)
}

let wsPersisting = false
function wsKeep (tree, persist) { // once: the same action kept again (a decision repeated, a keep inside a keep) is not doubled
  const key = JSON.stringify(tree)
  if (!wsKept.some(k => k.key === key)) wsKept.push({ tree, key, ch: wsMain(tree), kinds: wsKinds(tree), busy: false, persist: !!persist })
}
function wsStop (kind = 'all', round = true) { // round=false: what start: keeps (the round's reflexes) stays
  const gone = wsKept.filter(k => (kind === 'all' || k.kinds.has(kind)) && (round || !k.persist))
  wsKept = wsKept.filter(k => !gone.includes(k))
  if (gone.some(k => k.kinds.has('dodge_arrows')) && !wsKept.some(k => k.kinds.has('dodge_arrows'))) dodgeOn = false
  if (!wsKept.some(k => k.kinds.has('click'))) { wsClickSince = null; wsSwings = 0 }
}
async function wsProgram (steps, env, persist) { // one decision: its actions take over their channels, then run in order.
  // persist (start:): what it keeps lasts the round - the player's reflexes - and no decision takes it over
  wsEnv = env || {}
  const chans = new Set()
  for (const s of steps || []) for (const c of (s[0] === 'call' && s[1] === 'keep' ? wsMain(s) : wsChannels(s))) chans.add(c)
  const gone = wsKept.filter(k => !k.persist && [...k.ch].some(c => chans.has(c)))
  wsKept = wsKept.filter(k => !gone.includes(k))
  if (chans.has('look')) drawTarget = null
  if (chans.has('move')) { zigzag = null; circle = null; goCover = null }   // the body's own footwork ends with a new move
  if (gone.some(k => k.kinds.has('dodge_arrows'))) dodgeOn = false
  if (!wsKept.some(k => k.kinds.has('click'))) { wsClickSince = null; wsSwings = 0 }
  const notes = []
  wsBusy = true; wsPersisting = !!persist
  try {
    for (const s of steps || []) {
      try { const n = await wsDo(s); if (n) notes.push(n) } catch (err) { notes.push(`${err.message}`) }
    }
  } finally { wsBusy = false; wsPersisting = false }
  return notes.join('; ')
}
bot.on('physicsTick', () => {
  if (!wsKept.length || !bot.entity) return
  for (const k of wsKept) {
    if (k.busy || (k.persist && wsBusy && k.ch.has('hands'))) continue
    k.busy = true
    wsDo(k.tree).catch(() => {}).finally(() => { k.busy = false })
  }
})
const wsState = () => wsKept.length || wsClickSince
  ? { kept: wsKept.map(k => [...k.kinds].join(' ')), clicking: wsKept.some(k => k.kinds.has('click')),
      s: wsClickSince ? round((Date.now() - wsClickSince) / 1000, 1) : 0, swings: wsSwings } : null

// hold=true (continuous control): movement keys stay pressed after the call returns, until the next decision;
// turning keeps them pressed (you run in a curve); "wait" keeps whatever is held; "stop" releases everything.
// hold=false: each move is a short key press and the bot comes to rest, as before.
const HOLD_MOVES = { ...MOVES, sprint: ['forward', 'sprint'], sprint_jump: ['forward', 'sprint', 'jump'] }

async function act ({ action, ms = 400, turn_deg = 30, fine_deg = 10, pitch_deg = 15, hold: holdMode = false, target, guard, autojump, dyaw, dpitch, n, keys, button, dist, style, bowfull, bowswitch, lavaguard, weaponswitch, dodge: dodgeParam, leadrotate, arrowwarn, enemies, allies, stance: stanceParam, steps, env, persist }) {
  if (bowfull !== undefined) bowRules.full = !!bowfull
  if (bowswitch !== undefined) bowRules.switchAt = Number(bowswitch) || 0
  if (weaponswitch !== undefined) weaponSwitchAt = Number(weaponswitch) || 0
  if (Array.isArray(enemies)) enemyNames = enemies
  if (Array.isArray(allies)) allyNames = allies
  if (target && target !== fightTarget && bowDrawn() && drawTarget) { drawTarget = target; drawTrack.length = 0 }   // a new focus: the drawn bow turns to it
  if (target) fightTarget = target
  if (guard !== undefined) edgeGuard = guard === true ? 2 : Number(guard) || 0
  if (autojump !== undefined) autoJump = !!autojump
  if (lavaguard !== undefined) lavaGuard = !!lavaguard
  if (stanceParam !== undefined) stanceOn = !!stanceParam
  if (dodgeParam !== undefined) dodgeOn = !!dodgeParam
  if (arrowwarn !== undefined) arrowWarn = !!arrowwarn
  if (leadrotate !== undefined) leadRotate = !!leadrotate
  const e = bot.entity
  const p0 = e.position.clone()
  let note = ''
  if (holdMode) {
    // keys take effect on the next physics tick; nothing to wait for here (a fixed 60 ms wait used to slow every step)
    if (HOLD_MOVES[action] || action === 'stop') { zigzag = null; dodge = null; circle = null; goCover = null }
    if (HOLD_MOVES[action]) {
      if (HOLD_MOVES[action].includes('sprint') && !held.sprint) lastSprintPress = Date.now()
      hold(HOLD_MOVES[action])
    } else if (action === 'stop') {
      hold([])
    } else if (action === 'turn_around') {
      await bot.look(wrap(e.yaw + Math.PI), e.pitch, true)
    } else if (action.startsWith('sprint_turn_')) { // turn, then sprint the new way: one decision, as a player does
      const turn = { sprint_turn_around: Math.PI, sprint_turn_left: Math.PI / 2, sprint_turn_right: -Math.PI / 2 }[action]
      await bot.look(wrap(e.yaw + turn), e.pitch, true)
      hold(['forward', 'sprint'])
    } else if (action === 'wait') {
      // keep doing what you are doing
    } else if (['aim', 'aim_attack', 'wtap_attack'].includes(action)) {
      const t = targetEntity(target)
      if (!t) {
        note = `cannot see ${target}`
      } else if (action === 'aim') {
        await bot.lookAt(t.position.offset(0, 1.2, 0), true)
        note = `aimed at ${target}`
      } else {
        if (action === 'wtap_attack') { // let go of forward/sprint for a moment and press again: the hit gets sprint knockback
          bot.setControlState('sprint', false)
          bot.setControlState('forward', false)
          await sleep(60)
          hold(['forward', 'sprint'])
          lastSprintPress = Date.now()
          await sleep(40)
        }
        note = await strike(t)
      }
    } else if (action === 'jump') {
      bot.setControlState('jump', true)
      await sleep(120)
      bot.setControlState('jump', held.jump)
    } else if (ITEM_ACTIONS.has(action)) {
      itemBusy = true
      try { note = await itemAction(action, target) } finally { itemBusy = false }
    } else if (RAW_ACTIONS.has(action)) {
      note = await rawAction(action, { dyaw, dpitch, n, keys, button })
    } else if (VIS_ACTIONS.has(action)) {
      note = await visAction(bot, action, { dyaw, dist }, {
        leftClick, equip, stopSprintForUse, dropUse, hold, held,
        setUseStart: v => { useStart = v }, useStart: () => useStart })
    } else if (ENGAGE_ACTIONS.has(action)) {
      note = await engageAction(action, target, style)
    } else if (action === 'program') {   // a Workshop Jev's decision (see wsProgram)
      note = await wsProgram(steps, env, persist)
    }
    if (['aim', 'aim_attack', 'wtap_attack', 'jump', 'program'].includes(action) || ITEM_ACTIONS.has(action) || RAW_ACTIONS.has(action) ||
        ENGAGE_ACTIONS.has(action) || VIS_ACTIONS.has(action)) {
      const p1 = e.position
      return { moved: round(Math.hypot(p1.x - p0.x, p1.z - p0.z)), dy: round(p1.y - p0.y),
               pos: { x: round(p1.x), y: round(p1.y), z: round(p1.z) }, yaw: e.yaw, pitch: e.pitch,
               note, held: Object.keys(held).filter(k => held[k]) }
    }
    if (HOLD_MOVES[action] || ['stop', 'turn_around', 'wait'].includes(action) || action.startsWith('sprint_turn_')) {
      const p1 = e.position
      return { moved: round(Math.hypot(p1.x - p0.x, p1.z - p0.z)), dy: round(p1.y - p0.y),
               pos: { x: round(p1.x), y: round(p1.y), z: round(p1.z) }, yaw: e.yaw, pitch: e.pitch,
               note, held: Object.keys(held).filter(k => held[k]) }
    }
  } else {
    bot.clearControlStates()
  }
  if (MOVES[action]) {
    for (const c of MOVES[action]) bot.setControlState(c, true)
    await sleep(ms)
    bot.clearControlStates()
    await sleep(150) // let the bot come to rest so the next observation is stable
  } else if (/^turn_(left|right)(_small|_wide)?$/.test(action)) {
    const sign = action.startsWith('turn_left') ? 1 : -1 // mineflayer yaw grows counter-clockwise seen from above
    const deg = action.endsWith('_small') ? fine_deg : action.endsWith('_wide') ? 90 : turn_deg
    await bot.look(wrap(e.yaw + sign * rad(deg)), e.pitch, true)
  } else if (action === 'look_up' || action === 'look_down') {
    const sign = action === 'look_up' ? 1 : -1
    const pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, e.pitch + sign * rad(pitch_deg)))
    await bot.look(e.yaw, pitch, true)
  } else if (action === 'use') {
    const b = bot.blockAtCursor(5)
    if (b) {
      await bot.activateBlock(b)
      note = `right-clicked ${b.name}`
    } else {
      note = 'nothing within reach under the crosshair'
    }
    await sleep(150)
  } else if (action === 'attack') {
    const target = bot.entityAtCursor(3.5) // vanilla survival reach
    if (target) {
      bot.attack(target)
      note = `hit ${target.username || target.name}`
    } else {
      bot.swingArm()
      note = 'swung at the air: no player or mob within reach under the crosshair'
    }
    await sleep(150)
  } else if (action === 'wait') {
    await sleep(ms)
  } else {
    throw new Error(`unknown action ${action}`)
  }
  const p1 = e.position
  return {
    moved: round(Math.hypot(p1.x - p0.x, p1.z - p0.z)),
    dy: round(p1.y - p0.y),
    pos: { x: round(p1.x), y: round(p1.y), z: round(p1.z) },
    yaw: e.yaw,
    pitch: e.pitch,
    note
  }
}

// Landmarks: clusters of uncommon blocks around the bot (a diamond tower, a gold floor, a line of torches).
// Blocks that make up much of the scanned volume (grass, dirt, stone, air) are left out as background;
// what is left is grouped into 6-connected clusters of the same block type.
function landmarks ({ radius = 32, below = 6, above = 24, background = 0.02, max = 12 }) {
  const base = bot.entity.position.floored()
  const counts = new Map()
  const cells = new Map() // "x,y,z" -> name
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dy = -below; dy <= above; dy++) {
        const b = bot.blockAt(base.offset(dx, dy, dz))
        if (!b || b.boundingBox === 'empty' || b.name === 'air') continue
        counts.set(b.name, (counts.get(b.name) || 0) + 1)
        cells.set(`${base.x + dx},${base.y + dy},${base.z + dz}`, b.name)
      }
    }
  }
  const total = [...counts.values()].reduce((a, b) => a + b, 0)
  const rare = new Set([...counts].filter(([, n]) => n / total < background).map(([name]) => name))
  const seen = new Set()
  const clusters = []
  for (const [key, name] of cells) {
    if (!rare.has(name) || seen.has(key)) continue
    const stack = [key]
    const members = []
    seen.add(key)
    while (stack.length) {
      const k = stack.pop()
      members.push(k.split(',').map(Number))
      const [x, y, z] = members[members.length - 1]
      for (const [ox, oy, oz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
        const n = `${x + ox},${y + oy},${z + oz}`
        if (!seen.has(n) && cells.get(n) === name) { seen.add(n); stack.push(n) }
      }
    }
    const lo = [0, 1, 2].map(i => Math.min(...members.map(m => m[i])))
    const hi = [0, 1, 2].map(i => Math.max(...members.map(m => m[i])))
    clusters.push({
      name,
      count: members.length,
      min: { x: lo[0], y: lo[1], z: lo[2] },
      max: { x: hi[0] + 1, y: hi[1] + 1, z: hi[2] + 1 },
      center: { x: (lo[0] + hi[0] + 1) / 2, y: lo[1], z: (lo[2] + hi[2] + 1) / 2 }
    })
  }
  const d = c => Math.hypot(c.center.x - bot.entity.position.x, c.center.z - bot.entity.position.z)
  return { clusters: clusters.sort((a, b) => b.count - a.count || d(a) - d(b)).slice(0, max) }
}

const handlers = {
  observe,
  act,
  landmarks,
  say: ({ text }) => { bot.chat(String(text).slice(0, 100)); return {} },
  camera_focus: ({ names = null }) => { // draw only these players in the camera (null: everyone)
    if (!camera) throw new Error('no camera')
    camera.focus = names && names.length ? names : null
    return { focus: camera.focus }
  },
  snapshot: async ({ yaw_offset: yawOffset = 0, zoom = 0 }) => {
    if (!camera) throw new Error('no camera (start the body with --camera-port)')
    if (zoom) { // and a zoomed view of the middle (field of view `zoom` degrees), from the same moment
      if (!camera.snapshotPair) throw new Error('a zoomed view needs the vox camera')
      const [image, zoomed] = await camera.snapshotPair(zoom)
      return { image, zoom: zoomed, age_ms: camera.age() }
    }
    const image = await camera.snapshot(yawOffset)
    return { image, age_ms: camera.age() }
  }
}

readline.createInterface({ input: process.stdin }).on('line', async line => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
    const h = handlers[msg.cmd]
    if (!h) throw new Error(`unknown cmd ${msg.cmd}`)
    send({ id: msg.id, ok: true, result: await h(msg) })
  } catch (err) {
    send({ id: msg && msg.id, ok: false, error: String(err && err.message || err) })
  }
}).on('close', async () => {
  bot.quit()
  if (camera) await camera.close()
  setTimeout(() => process.exit(0), 300)
})

// Lava stance (act stance=true): in a sword fight near lava (the final duel's platform in its ring of lava), a hit
// sends you away from the one who swings, two or three blocks for a sprint hit. Circling round them did not help
// (tested: bench/lava_stance_test.py): they follow. So hold the middle: with lava within 4 blocks, the keys that take
// you away from it (the pull of every lava cell near, the nearer the stronger), whatever way you face - so that a hit
// from any side has the most ground to cross. Only while the sword is out and no arrow is being dodged; after the
// controllers, before the lava guard.
let stanceOn = false
let stance = null            // the way it steers, for observe
function lavaStance () {
  const e = bot.entity
  if (!stanceOn || !e || !engage || dodge || !e.onGround) { stance = null; return }
  const x0 = e.position.x, z0 = e.position.z, y = Math.floor(e.position.y + 0.25)
  let px = 0, pz = 0, near = 9
  for (let dx = -4; dx <= 4; dx++) {
    for (let dz = -4; dz <= 4; dz++) {
      const cx = Math.floor(x0) + dx, cz = Math.floor(z0) + dz
      if (!hot(bot.blockAt(new Vec3(cx, y, cz))) && !hot(bot.blockAt(new Vec3(cx, y - 1, cz)))) continue
      const vx = cx + 0.5 - x0, vz = cz + 0.5 - z0, d = Math.max(0.5, Math.hypot(vx, vz))
      if (d > 4) continue
      near = Math.min(near, d)
      px += vx / (d * d * d); pz += vz / (d * d * d)
    }
  }
  if (near > 4 || Math.hypot(px, pz) < 1e-6) { stance = null; return }
  const n = Math.hypot(px, pz), mx = -px / n, mz = -pz / n        // away from the lava
  const fwd = mx * -Math.sin(e.yaw) + mz * -Math.cos(e.yaw)       // forward is (-sin yaw, -cos yaw)
  const rgt = mx * Math.cos(e.yaw) + mz * -Math.sin(e.yaw)        // right is (cos yaw, -sin yaw)
  held.forward = fwd > 0.35; held.back = fwd < -0.35; held.right = rgt > 0.35; held.left = rgt < -0.35
  if (held.back) held.sprint = false
  for (const k of ['forward', 'back', 'left', 'right', 'sprint']) bot.setControlState(k, held[k])
  stance = (held.forward ? 'f' : '') + (held.back ? 'b' : '') + (held.left ? 'l' : '') + (held.right ? 'r' : '')
}
bot.on('physicsTick', lavaStance)

// the lava guard has the last word in each tick: registered after every controller
bot.on('physicsTick', guardLava)

// Track log: every tick of a fight, where the opponent is (as we see them) and where we are, to runs/tracks-<name>
// .jsonl - the ground truth for choosing how to lead a person (bench/lead_eval.py replays it)
const trackBuf = []
bot.on('physicsTick', () => {
  const t = fightTarget && targetEntity(fightTarget), e = bot.entity
  if (!t || !e) return
  trackBuf.push({ t: Date.now(), who: fightTarget, x: +t.position.x.toFixed(3), y: +t.position.y.toFixed(3), z: +t.position.z.toFixed(3),
    g: t.onGround !== false, u: usingItem(t), mx: +e.position.x.toFixed(2), my: +e.position.y.toFixed(2), mz: +e.position.z.toFixed(2), rtt: Math.round(rttMs()) })
})
setInterval(() => {
  if (!trackBuf.length) return
  try {
    require('fs').appendFileSync(require('path').join(__dirname, '..', 'runs', `tracks-${bot.username}.jsonl`),
      trackBuf.map(r => JSON.stringify(r)).join('\n') + '\n')
  } catch (e) {}
  trackBuf.length = 0
}, 1000)
