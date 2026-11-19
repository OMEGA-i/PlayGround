const {test} = require('node:test');
const assert = require('node:assert/strict');
const {ReceivedFrameRate, MP4FrameCounter, JPEGFrameReader} = require('../../src/omega_playground/ops/web/video.js');
function box(type, payload) { const b = Buffer.alloc(8 + payload.length); b.writeUInt32BE(b.length); b.write(type,4); payload.copy(b,8); return b; }
function fragment(n) { const run=Buffer.alloc(8);run.writeUInt32BE(n,4);return Buffer.concat([box('moof',box('traf',box('trun',run))),box('mdat',Buffer.from([1,2,3]))]); }
test('counts samples only after complete media payload, across arbitrary network chunks', () => {
  const got=[], parser=new MP4FrameCounter(n=>got.push(n));
  parser.feed(box('ftyp',Buffer.alloc(4)));parser.feed(box('moov',Buffer.alloc(4)));
  const bytes=Buffer.concat([fragment(12),fragment(13)]);
  for (const byte of bytes) parser.feed(Uint8Array.of(byte));
  assert.deepEqual(got,[12,13]);
});
test('MJPEG counts complete images, not chunks or header boundaries', () => {
  const got=[], parser=new JPEGFrameReader(b=>got.push([...b]));
  const bytes=Buffer.concat([Buffer.from('--omega-frame\r\nContent-Type: image/jpeg\r\nContent-Length: 3\r\n\r\n'),Buffer.from([1,2,3]),Buffer.from('\r\n--omega-frame\r\nContent-Length: 2\r\n\r\n'),Buffer.from([4,5])]);
  for(const b of bytes) parser.feed(Uint8Array.of(b));
  assert.deepEqual(got,[[1,2,3],[4,5]]);
});
test('rate uses elapsed reception time and falls to zero after a stall',()=>{
  let now=0;const rate=new ReceivedFrameRate(()=>now);
  now=500;rate.add(12);now=1000;rate.add(13);
  assert.equal(rate.value(),25);
  now=3100;assert.equal(rate.value(),0);
  rate.reset();assert.equal(rate.value(),0);
});
