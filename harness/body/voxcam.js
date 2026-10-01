// A camera without a browser: voxcam (body/voxcam/voxcam.c) ray-casts the bot's view on the CPU in a few ms, from the
// world as it is at the moment of the snapshot (no frame is ever old). Same interface as camera.js.
//
// The C side keeps a box of blocks around the bot (96 x 48 x 96); this side fills it from the bot's world, sends each
// block change, refills it when the bot has moved far from its middle or new chunks have arrived, and asks for a
// frame with the eye, the view and the other players.
const { spawn, execFileSync } = require('child_process')
const fs = require('fs')
const os = require('os')
const path = require('path')

const DIR = path.join(__dirname, 'voxcam')
const TEXTURES = path.join(__dirname, 'node_modules', 'prismarine-viewer', 'public', 'textures', '1.11.2')
const BOX = { sx: 96, sy: 48, sz: 96, below: 16 }   // blocks around the bot; `below` of the height is under its feet
const RECENTER = 24                                  // refill when the bot is this far from the box's middle
const FOV = 70                                       // the camera's field of view, degrees (voxcam.c's default)

function binary () { // build it on first use, for this machine
  const bin = path.join(DIR, `voxcam-${process.platform}-${process.arch}`)
  const src = path.join(DIR, 'voxcam.c')
  if (!fs.existsSync(bin) || fs.statSync(bin).mtimeMs < fs.statSync(src).mtimeMs) {
    const omp = process.platform === 'linux' ? ['-fopenmp', '-march=native'] : []
    execFileSync('cc', ['-O3', '-std=gnu11', ...omp, '-o', bin, src, '-lm'], { cwd: DIR, stdio: 'inherit' })
  }
  return bin
}

async function startVoxcam (bot, { width = 256, height = 256, threads = 16 } = {}) {
  const env = { ...process.env }
  if (threads) env.OMP_NUM_THREADS = String(threads)
  const proc = spawn(binary(), [TEXTURES, String(width), String(height)], { stdio: ['pipe', 'pipe', 'inherit'], env })
  const file = path.join(os.tmpdir(), `voxcam-${process.pid}-${bot.username}.bin`)
  let box = null        // {x0, y0, z0}
  let dirty = true      // new chunks arrived: refill before the next frame

  const fill = () => {
    const p = bot.entity.position
    const x0 = Math.floor(p.x) - BOX.sx / 2, y0 = Math.max(0, Math.floor(p.y) - BOX.below), z0 = Math.floor(p.z) - BOX.sz / 2
    const buf = new Uint16Array(BOX.sx * BOX.sy * BOX.sz)
    const { Vec3 } = require('vec3')
    const v = new Vec3(0, 0, 0)
    for (let cx = Math.floor(x0 / 16); cx <= Math.floor((x0 + BOX.sx - 1) / 16); cx++) {
      for (let cz = Math.floor(z0 / 16); cz <= Math.floor((z0 + BOX.sz - 1) / 16); cz++) {
        const col = bot.world.getColumn(cx, cz)
        if (!col) continue
        for (let lx = 0; lx < 16; lx++) {
          const x = cx * 16 + lx - x0
          if (x < 0 || x >= BOX.sx) continue
          for (let lz = 0; lz < 16; lz++) {
            const z = cz * 16 + lz - z0
            if (z < 0 || z >= BOX.sz) continue
            v.x = lx; v.z = lz
            for (let y = 0; y < BOX.sy; y++) {
              v.y = y0 + y
              if (v.y > 255) break
              const type = col.getBlockType(v)
              if (type) buf[(y * BOX.sz + z) * BOX.sx + x] = (type << 4) | col.getBlockData(v)
            }
          }
        }
      }
    }
    fs.writeFileSync(file, Buffer.from(buf.buffer))
    proc.stdin.write(`R ${x0} ${y0} ${z0} ${BOX.sx} ${BOX.sy} ${BOX.sz} ${file}\n`)
    box = { x0, y0, z0 }
    dirty = false
  }

  const onBlock = (oldBlock, newBlock) => {
    if (!box || !newBlock) return
    const b = newBlock.position
    proc.stdin.write(`B ${b.x} ${b.y} ${b.z} ${(newBlock.type << 4) | newBlock.metadata}\n`)
  }
  const onChunk = () => { dirty = true }
  bot.on('blockUpdate', onBlock)
  bot.on('chunkColumnLoad', onChunk)

  // frames come back in order: a queue of waiting snapshots, filled from stdout
  const waiting = []
  let pending = Buffer.alloc(0)
  proc.stdout.on('data', chunk => {
    pending = Buffer.concat([pending, chunk])
    while (pending.length >= 4) {
      const len = pending.readUInt32LE(0)
      if (pending.length < 4 + len) break
      const jpg = pending.subarray(4, 4 + len)
      pending = pending.subarray(4 + len)
      const w = waiting.shift()
      if (w) w('data:image/jpeg;base64,' + jpg.toString('base64'))
    }
  })
  proc.on('exit', code => { for (const w of waiting.splice(0)) w(null); if (code) console.error('voxcam exited', code) })

  fill()
  let lastFrameMs = null
  const camera = {
    focus: null,   // names to draw (null: everyone)
    url: null,
    async snapshot (yawOffset = 0) { return (await camera.frames(yawOffset, [null]))[0] },
    // the view and a zoomed view of its middle (field of view `zoom` degrees, a scope), of the same moment
    async snapshotPair (zoom) { return camera.frames(0, [null, zoom]) },
    async frames (yawOffset, fovs) {
      const t0 = Date.now()
      const e = bot.entity
      const mid = box && { x: box.x0 + BOX.sx / 2, z: box.z0 + BOX.sz / 2 }
      if (dirty || !box || Math.hypot(e.position.x - mid.x, e.position.z - mid.z) > RECENTER ||
          e.position.y - box.y0 < 4 || box.y0 + BOX.sy - e.position.y < 8) fill()
      // only the players a player would tell apart as the ones that matter (a name tag shows who is who); others,
      // e.g. bots in the next arena seen through its invisible barrier wall, are left out when a focus is set
      const players = Object.values(bot.players).filter(p => p.entity && p.username !== bot.username &&
        (!camera.focus || camera.focus.includes(p.username)))
        .map(p => `${p.entity.position.x.toFixed(3)} ${p.entity.position.y.toFixed(3)} ${p.entity.position.z.toFixed(3)} ${(p.entity.yaw || 0).toFixed(4)}`)
      const line = `F ${e.position.x.toFixed(3)} ${(e.position.y + e.height).toFixed(3)} ${e.position.z.toFixed(3)} ` +
        `${(e.yaw + yawOffset).toFixed(5)} ${e.pitch.toFixed(5)} ${players.length} ${players.join(' ')}\n`
      const out = fovs.map(fov => {
        if (fov) proc.stdin.write(`V ${fov}\n`)
        const uri = new Promise(resolve => { waiting.push(resolve); proc.stdin.write(line) })
        if (fov) proc.stdin.write(`V ${FOV}\n`)
        return uri
      })
      const uris = await Promise.all(out)
      lastFrameMs = Date.now() - t0
      return uris
    },
    age () { return 0 },            // rendered on demand from the current state
    renderMs () { return lastFrameMs },
    async close () {
      bot.removeListener('blockUpdate', onBlock)
      bot.removeListener('chunkColumnLoad', onChunk)
      try { proc.stdin.write('Q\n') } catch {}
      setTimeout(() => proc.kill(), 200)
      try { fs.unlinkSync(file) } catch {}
    }
  }
  return camera
}

module.exports = { startVoxcam }
