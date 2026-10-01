// First-person viewer server: prismarine-viewer's mineflayer plugin (lib/mineflayer.js), trimmed to first person,
// bound to 127.0.0.1, and with aim(yawOffset): point the rendered camera some angle away from where the bot looks
// without turning the bot (a rear-view mirror), then back with aim(0).
const { WorldView } = require('prismarine-viewer/viewer/lib/worldView')   // not the index: it loads node-canvas, which does not load on 64K-page aarch64
const { setupRoutes } = require('prismarine-viewer/lib/common')

// What the rendered view leaves out: barrier blocks (id 166 before 1.13) are invisible in the real client, and the
// arenas are walled with them. Chunks are copied with those blocks turned to air before they reach the browser;
// the bot's own world keeps them.
const HIDDEN = new Set([166])

function viewWorld (bot) {
  const Chunk = require('prismarine-chunk')(bot.version)
  const { Vec3 } = require('vec3')
  const hide = column => {
    const copy = Chunk.fromJson(column.toJson())
    const p = new Vec3(0, 0, 0)
    for (p.y = 0; p.y < 256; p.y++) for (p.z = 0; p.z < 16; p.z++) for (p.x = 0; p.x < 16; p.x++) {
      if (HIDDEN.has(copy.getBlockType(p))) copy.setBlockType(p, 0)
    }
    return copy
  }
  return new Proxy(bot.world, {
    get (target, prop) {
      if (prop === 'getColumnAt') return async pos => { const c = await target.getColumnAt(pos); return c && hide(c) }
      const v = target[prop]
      return typeof v === 'function' ? v.bind(target) : v
    }
  })
}

module.exports = (bot, { viewDistance = 4, port }) => {
  const express = require('express')
  const app = express()
  const http = require('http').createServer(app)
  const io = require('socket.io')(http, { path: '/socket.io' })
  setupRoutes(app, '')
  const sockets = []
  let yawOffset = 0

  const packet = () => ({ pos: bot.entity.position, yaw: bot.entity.yaw + yawOffset, pitch: bot.entity.pitch, addMesh: true })

  io.on('connection', socket => {
    socket.emit('version', bot.version)
    sockets.push(socket)
    const worldView = new WorldView(viewWorld(bot), viewDistance, bot.entity.position, socket)
    worldView.init(bot.entity.position)
    const botPosition = () => {
      socket.emit('position', packet())
      worldView.updatePosition(bot.entity.position)
    }
    bot.on('move', botPosition)
    worldView.listenToBot(bot)
    socket.on('disconnect', () => {
      bot.removeListener('move', botPosition)
      worldView.removeListenersFromBot(bot)
      sockets.splice(sockets.indexOf(socket), 1)
    })
  })
  http.listen(port, '127.0.0.1', () => console.log(`viewer on 127.0.0.1:${port}`))

  return {
    aim (offset) {
      yawOffset = offset
      for (const s of sockets) s.emit('position', packet())
    },
    close () {
      http.close()
      for (const s of sockets) s.disconnect()
    }
  }
}
