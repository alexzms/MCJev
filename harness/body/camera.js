// First-person camera: prismarine-viewer renders the bot's view (three.js) in a headless Chromium page,
// snapshot() returns it as a JPEG data URI with a crosshair drawn in the middle.
const startViewer = require('./viewer_server')
const puppeteer = require('puppeteer-core')
const fs = require('fs')
const os = require('os')
const path = require('path')

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

// The viewer redraws in a requestAnimationFrame loop, 60 frames a second whether anyone looks or not (one
// software-rendered camera = ~2 CPU cores). Its callbacks are queued instead and run by us: a few frames for an
// on-demand snapshot, or a steady loop at `fps` so that a snapshot is only an encode of the latest frame.
// WebGL canvases keep their drawing buffer, so the latest frame can be read at any time; __snap copies it to a 2D
// canvas, draws the crosshair (inverted against the background) and encodes a JPEG, all inside the page.
const GATE_FRAMES = `(() => {
  let queue = []
  window.requestAnimationFrame = cb => queue.push(cb)
  window.cancelAnimationFrame = () => {}
  const tick = () => { const cbs = queue; queue = []; const t = performance.now(); for (const cb of cbs) cb(t) }
  window.__renderFrames = n => new Promise(resolve => {
    const step = () => { tick(); if (--n > 0) setTimeout(step, 0); else resolve() }
    step()
  })
  // steady loop: after each frame, copy it off the GL canvas and encode it without blocking this thread
  // (createImageBitmap / convertToBlob are async), so a snapshot just hands over the latest JPEG
  let loop = null, latest = null, busy = false
  const cross = (g, w, h) => {
    const cx = w / 2, cy = h / 2, r = Math.max(4, Math.round(w / 32))
    g.globalCompositeOperation = 'difference'
    g.fillStyle = '#fff'
    g.fillRect(cx - r, cy - 1, 2 * r, 2); g.fillRect(cx - 1, cy - r, 2, r - 1); g.fillRect(cx - 1, cy + 1, 2, r - 1)
    g.globalCompositeOperation = 'source-over'
  }
  let oc = null
  const capture = async () => {
    if (busy) return
    busy = true
    try {
      const gl = document.querySelector('canvas')
      const bmp = await createImageBitmap(gl)
      if (!oc || oc.width !== bmp.width || oc.height !== bmp.height) oc = new OffscreenCanvas(bmp.width, bmp.height)
      const g = oc.getContext('2d')
      g.drawImage(bmp, 0, 0)
      bmp.close()
      cross(g, oc.width, oc.height)
      const blob = await oc.convertToBlob({ type: 'image/jpeg', quality: 0.75 })
      latest = await new Promise(resolve => { const fr = new FileReader(); fr.onload = () => resolve(fr.result); fr.readAsDataURL(blob) })
      if (window.__push) window.__push(latest)   // hand it to the Node side, which answers snapshots without the page
    } finally { busy = false }
  }
  window.__loop = fps => { clearInterval(loop); loop = fps > 0 ? setInterval(() => { tick(); capture() }, 1000 / fps) : null }
  window.__latest = () => latest
  const getContext = HTMLCanvasElement.prototype.getContext
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    if (/webgl/.test(type)) attrs = Object.assign({}, attrs, { preserveDrawingBuffer: true })
    return getContext.call(this, type, attrs)
  }
  let c2 = null
  window.__snap = quality => {
    const gl = document.querySelector('canvas')
    if (!c2) c2 = document.createElement('canvas')
    c2.width = gl.width; c2.height = gl.height
    const g = c2.getContext('2d')
    g.drawImage(gl, 0, 0)
    cross(g, gl.width, gl.height)
    return c2.toDataURL('image/jpeg', quality)
  }
})()`

const BROWSERS = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium'
]

// white cross, inverted against the background so it shows on sky and grass alike
const CROSSHAIR = `body::after { content: ''; position: fixed; left: 50%; top: 50%; width: 16px; height: 16px;
  margin: -8px 0 0 -8px; pointer-events: none; mix-blend-mode: difference;
  background: linear-gradient(#fff, #fff) center / 2px 16px no-repeat, linear-gradient(#fff, #fff) center / 16px 2px no-repeat; }`

// fps > 0: render continuously and let snapshot() return the latest frame (a few ms, at most 1/fps old);
// fps = 0: render only when a snapshot is asked for (cheaper when pictures are rare, but ~100 ms each).
async function startCamera (bot, { port, width = 640, height = 360, viewDistance = 4, browserPath, fps = 0, gpu = false }) {
  const executablePath = browserPath || BROWSERS.find(p => p && fs.existsSync(p))
  if (!executablePath) throw new Error('no Chromium-based browser found; set CHROME_PATH')
  const viewer = startViewer(bot, { port, viewDistance })
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jev-camera-'))
  const browser = await puppeteer.launch({
    executablePath,
    headless: true,
    userDataDir,
    // Linux (the GPU node): no sandbox (no user namespaces in the container) and no /dev/shm size worries
    args: [...(gpu ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist', '--enable-gpu-rasterization']
      : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']), '--no-first-run', '--mute-audio',
      ...(process.platform === 'linux' ? ['--no-sandbox', '--disable-dev-shm-usage'] : [])]
  })
  const page = await browser.newPage()
  await page.setViewport({ width, height })
  await page.evaluateOnNewDocument(GATE_FRAMES)
  let pushed = null, pushedAt = 0   // the latest frame the page encoded (continuous mode)
  if (fps) await page.exposeFunction('__push', uri => { pushed = uri; pushedAt = Date.now() })
  await page.goto(`http://127.0.0.1:${port}`, { waitUntil: 'load' })
  if (!fps) await page.addStyleTag({ content: CROSSHAIR })
  await sleep(1000)
  bot.emit('move')
  await sleep(2500) // first chunks are meshed in web workers
  let mover = null
  if (fps) { // keep the camera on the bot (the viewer moves it only on 'move') and keep rendering
    mover = setInterval(() => bot.emit('move'), Math.round(1000 / fps))
    await page.evaluate(f => window.__loop(f), fps)
  }

  return {
    url: `http://127.0.0.1:${port}`,
    // yawOffset: turn the rendered camera this many radians left of where the bot looks (Math.PI: behind it)
    async snapshot (yawOffset = 0) {
      if (fps && !yawOffset && pushed) return pushed   // the latest frame, already encoded and here
      bot.emit('move') // the viewer moves its camera only on 'move', which a bot standing still never fires
      if (yawOffset) viewer.aim(yawOffset)
      await sleep(30)
      await page.evaluate(() => window.__renderFrames(3))
      const uri = fps ? await page.evaluate(q => window.__snap(q), 0.75)
        : 'data:image/jpeg;base64,' + await page.screenshot({ type: 'jpeg', quality: 75, encoding: 'base64' })
      if (yawOffset) viewer.aim(0)
      return uri
    },
    age () { return fps && pushed ? Date.now() - pushedAt : null },   // ms since the latest frame arrived
    async close () {
      if (mover) clearInterval(mover)
      viewer.close()
      await browser.close().catch(() => {})
      try { // the browser may still be flushing its profile
        fs.rmSync(userDataDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })
      } catch {}
    }
  }
}

module.exports = { startCamera }
