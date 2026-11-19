const {test} = require('node:test');
const assert = require('node:assert/strict');
const {playbackTiming} = require('../../src/omega_playground/ops/web/video.js');
test('half-second fragments do not cause periodic seek jumps', () => {
  for (let t = 1; t < 20; t += .02) {
    const end = Math.floor(t * 2) / 2 + 1.4;
    assert.equal(playbackTiming(t, 0, end, true).seek, null);
  }
});
test('join and long outage recover with a buffer, ordinary jitter uses bounded speed', () => {
  assert.deepEqual(playbackTiming(0, 100, 101, false), {seek:100.1, rate:1});
  assert.equal(playbackTiming(20, 0, 25, true).seek, 24.1);
  assert.deepEqual(playbackTiming(20, 0, 21.7, true), {seek:null, rate:1.03});
  assert.deepEqual(playbackTiming(20, 0, 20.4, true), {seek:null, rate:.98});
});
