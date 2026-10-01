// Aiming from what was seen, not from where the opponent really is: the vision+body "pure" harnesses tell the body
// how far off the cross they saw the player (degrees right) and how far away they judged them (blocks, from the
// size in the picture); the body turns there, the way a player flicks the mouse onto someone they see, and then
// acts. The hit lands only if the crosshair ends up on the body within reach, as in the game.
const rad = d => d * Math.PI / 180
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a))
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

const VIS_ACTIONS = new Set(['vis_aim', 'vis_hit', 'vis_wtap', 'vis_crit', 'vis_draw', 'vis_release', 'vis_lava'])

// pitch (degrees, + up) that puts the cross on the middle of a player's body `dist` blocks away on level ground
function bodyPitch (dist) { return -Math.atan2(0.62, Math.max(dist, 0.5)) * 180 / Math.PI }
// how much higher a fully drawn arrow must go at that distance (about 3 blocks a tick, drop 0.025 t^2)
function arrowLift (dist) { const t = dist / 3; return Math.atan2(0.025 * t * t, Math.max(dist, 0.5)) * 180 / Math.PI }

async function turnTo (bot, dyaw, pitchDeg) {
  const e = bot.entity
  await bot.look(wrap(e.yaw - rad(dyaw)), Math.max(-Math.PI / 2, Math.min(Math.PI / 2, rad(pitchDeg))), true)
}

// h: helpers from body.js (leftClick, equip, stopSprintForUse, dropUse, setUseStart, held, useStart)
async function visAction (bot, action, { dyaw = 0, dist = 3 }, h) {
  if (action === 'vis_aim') {
    await turnTo(bot, dyaw, bodyPitch(dist))
    return `turned ${Math.abs(dyaw).toFixed(0)}° ${dyaw >= 0 ? 'right' : 'left'} to where you saw them`
  }
  if (action === 'vis_hit' || action === 'vis_wtap') {
    if (action === 'vis_wtap') { // let go of the sprint and press it again: the hit gets sprint knockback
      bot.setControlState('sprint', false); bot.setControlState('forward', false)
      await sleep(50)
      h.hold(['forward', 'sprint'])
    }
    await turnTo(bot, dyaw, bodyPitch(dist))
    return h.leftClick()
  }
  if (action === 'vis_crit') {
    if (!bot.entity.onGround) return 'not on the ground'
    h.held.sprint = false
    bot.setControlState('sprint', false)
    bot.setControlState('jump', true)
    const t0 = Date.now()
    while (Date.now() - t0 < 700 && (bot.entity.onGround || bot.entity.velocity.y > -0.1)) await sleep(20)
    await sleep(50)
    bot.setControlState('jump', h.held.jump)
    await turnTo(bot, dyaw, bodyPitch(dist))
    return 'jumped and, coming down, ' + h.leftClick()
  }
  if (action === 'vis_draw') {
    if (!bot.inventory.items().some(i => i.name === 'arrow')) return 'no arrows'
    if (!(bot.usingHeldItem && bot.heldItem && bot.heldItem.name === 'bow')) {
      if (!await h.equip(i => i.name === 'bow')) return 'no bow'
      h.stopSprintForUse()
      bot.activateItem(); h.setUseStart(Date.now())
    }
    await turnTo(bot, dyaw, bodyPitch(dist) + arrowLift(dist))
    return 'drawing the bow where you saw them'
  }
  if (action === 'vis_release') {
    if (!(bot.usingHeldItem && bot.heldItem && bot.heldItem.name === 'bow')) return 'not drawing the bow'
    const s = h.useStart() ? (Date.now() - h.useStart()) / 1000 : 0
    await turnTo(bot, dyaw, bodyPitch(dist) + arrowLift(dist))
    await sleep(60) // the last look reaches the server before the release
    bot.deactivateItem(); h.setUseStart(null)
    return `shot an arrow where you saw them (drawn ${s.toFixed(1)} s${s < 1 ? ': weak' : ''})`
  }
  if (action === 'vis_lava') {
    if (dist > 4.8) return 'too far for lava'
    if (!await h.equip(i => i.name === 'lava_bucket')) return 'no lava bucket'
    const yaw = bot.entity.yaw, pitch = bot.entity.pitch
    await turnTo(bot, dyaw, -Math.atan2(1.62, Math.max(dist, 0.5)) * 180 / Math.PI)   // the ground at their feet
    await sleep(60)
    bot.activateItem()
    await sleep(60)
    await bot.look(yaw, pitch, true)
    return 'poured lava where you saw them standing'
  }
  throw new Error(`unknown vision action ${action}`)
}

module.exports = { VIS_ACTIONS, visAction }
