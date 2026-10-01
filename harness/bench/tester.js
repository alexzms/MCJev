// Fake human for benchmarks: joins as the given name, stands still, and sends every stdin line as chat.
// Run from bench/ so that require() finds ../body/node_modules (NODE_PATH is set by come_to_me.py).
const mineflayer = require('mineflayer')

const [name = 'Tester', host = '127.0.0.1', port = '25565'] = process.argv.slice(2)
const { forwardHost } = require('../body/forward')
const bot = mineflayer.createBot({ host, port: parseInt(port), username: name, version: '1.11.2', auth: 'offline',
  fakeHost: forwardHost(host, name) })
bot.once('spawn', () => console.log('tester spawned'))
bot.on('end', reason => { console.log('tester end', reason); process.exit(0) })
require('readline').createInterface({ input: process.stdin })
  .on('line', line => line.trim() && bot.chat(line.trim()))
  .on('close', () => { bot.quit(); setTimeout(() => process.exit(0), 300) })
