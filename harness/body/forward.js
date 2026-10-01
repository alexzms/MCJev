// Joining the backend directly while people come in through the online-mode proxy (Velocity, BungeeCord-style
// "legacy" forwarding): the handshake's host carries the address and the player's offline UUID. A backend with
// spigot.yml bungeecord: true requires it; one without ignores it. Pass forwardHost(host, name) as createBot's fakeHost.
const crypto = require('crypto')

const offlineUuid = name => {                        // what an offline-mode server gives this name (a version 3 UUID)
  const h = crypto.createHash('md5').update('OfflinePlayer:' + name).digest()
  h[6] = h[6] & 0x0f | 0x30
  h[8] = h[8] & 0x3f | 0x80
  return h.toString('hex')
}

const forwardHost = (host, name) => [host, '127.0.0.1', offlineUuid(name)].join('\u0000')

module.exports = { offlineUuid, forwardHost }
