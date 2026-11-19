// Count complete received frames, independently of video playback/decode cadence.
class ReceivedFrameRate {
  constructor(clock = () => performance.now()) { this.clock = clock; this.events = []; this.since = clock(); }
  add(count = 1) { if (count > 0) this.events.push([this.clock(), count]); }
  reset() { this.events = []; this.since = this.clock(); }
  value() {
    const now = this.clock();
    this.events = this.events.filter(([at]) => at > now - 2000);
    const elapsed = Math.min(2000, now - this.since);
    return elapsed > 0 ? this.events.reduce((n, [, count]) => n + count, 0) * 1000 / elapsed : 0;
  }
}

function joinBytes(a, b) {
  const result = new Uint8Array(a.length + b.length);
  result.set(a); result.set(b, a.length); return result;
}
// A fragment may contain many video samples. WS messages / TCP chunks are not frames.
class MP4FrameCounter {
  constructor(received) { this.received = received; this.pending = new Uint8Array(); this.samples = 0; }
  feed(bytes) {
    this.pending = joinBytes(this.pending, new Uint8Array(bytes));
    while (this.pending.length >= 8) {
      const view = new DataView(this.pending.buffer, this.pending.byteOffset, this.pending.byteLength);
      let size = view.getUint32(0), header = 8;
      if (size === 1) {
        if (this.pending.length < 16) return;
        size = Number(view.getBigUint64(8)); header = 16;
      }
      if (size < header || size > 64 * 1024 * 1024) throw Error('Invalid MP4 box');
      if (this.pending.length < size) return;
      const kind = String.fromCharCode(...this.pending.subarray(4, 8));
      if (kind === 'moof') this.samples = this.sampleCount(this.pending.subarray(header, size));
      if (kind === 'mdat') { this.received(this.samples); this.samples = 0; }
      this.pending = this.pending.slice(size);
    }
  }
  sampleCount(bytes) {
    let count = 0;
    for (let offset = 0; offset + 8 <= bytes.length;) {
      const view = new DataView(bytes.buffer, bytes.byteOffset + offset, bytes.length - offset);
      const size = view.getUint32(0);
      if (size < 8 || offset + size > bytes.length) throw Error('Invalid MP4 fragment');
      const kind = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
      if (kind === 'traf') count += this.sampleCount(bytes.subarray(offset + 8, offset + size));
      if (kind === 'trun') { if (size < 16) throw Error('Invalid MP4 track run'); count += view.getUint32(12); }
      offset += size;
    }
    return count;
  }
}

class JPEGFrameReader {
  constructor(received) { this.received = received; this.pending = new Uint8Array(); this.length = null; }
  feed(bytes) {
    this.pending = joinBytes(this.pending, bytes);
    while (true) {
      if (this.length === null) {
        let end = -1;
        for (let i = 0; i + 3 < this.pending.length; i++) {
          if (this.pending[i] === 13 && this.pending[i+1] === 10 && this.pending[i+2] === 13 && this.pending[i+3] === 10) { end = i; break; }
        }
        if (end < 0) { if (this.pending.length > 4096) throw Error('Invalid JPEG headers'); return; }
        const match = new TextDecoder().decode(this.pending.subarray(0, end)).match(/Content-Length:\s*(\d+)/i);
        if (!match || +match[1] > 16 * 1024 * 1024 || +match[1] < 1) throw Error('Invalid JPEG length');
        this.length = +match[1]; this.pending = this.pending.slice(end + 4);
      }
      if (this.pending.length < this.length) return;
      this.received(this.pending.slice(0, this.length));
      this.pending = this.pending.slice(this.length); this.length = null;
    }
  }
}

const receivedRates = typeof window === 'undefined' ? null : {main: new ReceivedFrameRate(), ego: new ReceivedFrameRate()};
let renderedFps = null;
const jpegViews = new Map();
function setJPEGView(id, path, rate, streaming) {
  if (jpegViews.get(id)?.path === path) return;
  const old = jpegViews.get(id);
  old?.controller?.abort(); if (old?.blob) URL.revokeObjectURL(old.blob);
  const state = {path, controller: new AbortController(), blob: null}; jpegViews.set(id, state);
  rate.reset();
  const img = document.getElementById(id);
  if (!streaming) { img.src = path; return; }
  (async () => {
    while (!state.controller.signal.aborted) {
      let reader;
      try {
        const response = await fetch(path, {signal: state.controller.signal});
        if (!response.ok) throw Error('Waiting for camera');
        reader = response.body.getReader();
        const parser = new JPEGFrameReader(frame => {
          rate.add();
          const blob = URL.createObjectURL(new Blob([frame], {type: 'image/jpeg'}));
          img.src = blob;
          if (state.blob) URL.revokeObjectURL(state.blob);
          state.blob = blob;
        });
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          if (state.controller.signal.aborted) return;
          parser.feed(value);
        }
      } catch (_) { /* Pause/rebuild cancels the old stream. */ }
      finally { if (reader) { await reader.cancel().catch(() => {}); reader.releaseLock(); } }
      if (!state.controller.signal.aborted) await new Promise(resolve => setTimeout(resolve, 1000));
    }
  })();
}
function updateEgoVideo(paused) {
  setJPEGView('ego-image', paused ? '/api/ego-viewer' : '/api/ego-stream', receivedRates.ego, !paused);
}
function updateVideoRates(state) {
  renderedFps = state.paused ? 0 : (state.performance?.wall_fps ?? null);
}
function paintVideoRates() {
  for (const [view, rate] of Object.entries(receivedRates)) {
    const badge = document.getElementById(`${view}-fps`);
    if (badge) badge.textContent = `视频 ${rate.value().toFixed(1)} / ${renderedFps === null ? '—' : renderedFps.toFixed(1)} FPS`;
  }
}

// HEVC/AVC share the recorder's real frames; JPEG remains the paused preview/fallback.
let liveVideoReady = false, liveVideoPaused = true, liveVideoFallback = false, previewAt = 0;
if (typeof window !== "undefined") window.liveVideoFps = null;

// Keep enough buffered video for the half-second MP4 fragments. Ordinary
// jitter adjusts playback gently; only joining or a long outage needs a seek.
function playbackTiming(current, start, end, joined) {
  const lag = end - current;
  if (!joined || current < start || lag > 4)
    return {seek: Math.max(start, end - .9), rate: 1};
  return {seek: null, rate: lag > 1.6 ? 1.03 : lag < .55 ? .98 : 1};
}
if (typeof module !== "undefined") module.exports = {playbackTiming, ReceivedFrameRate, MP4FrameCounter, JPEGFrameReader};
function updateLiveVideo(paused) {
  liveVideoPaused = paused;
  const video = document.getElementById('live-video');
  const preview = document.getElementById('viewer');
  if (liveVideoFallback) {
    setJPEGView('viewer', paused ? '/api/viewer' : '/api/stream', receivedRates.main, !paused);
    return;
  }
  video.hidden = paused || !liveVideoReady;
  preview.hidden = !video.hidden;
  if (preview.hidden || Date.now() - previewAt < 2000) return;
  previewAt = Date.now();
  preview.src = '/api/viewer?t=' + previewAt;
}

async function startLiveVideo() {
  const video = document.getElementById('live-video');
  setInterval(paintVideoRates, 250);
  paintVideoRates();
  if (!window.MediaSource) {
    liveVideoFallback = true;
    updateLiveVideo(liveVideoPaused);
    return;
  }
  let sample = null;
  if (video.requestVideoFrameCallback) {
    const count = (now, meta) => {
      if (!sample) sample = [now, meta.presentedFrames];
      if (now - sample[0] >= 1500) {
        window.liveVideoFps = 1000 * (meta.presentedFrames - sample[1]) / (now - sample[0]);
        sample = [now, meta.presentedFrames];
      }
      video.requestVideoFrameCallback(count);
    };
    video.requestVideoFrameCallback(count);
  }
  const waitUpdate = buffer => new Promise((resolve, reject) => {
    const clear = () => { buffer.removeEventListener('updateend', done); buffer.removeEventListener('error', fail); };
    const done = () => { clear(); resolve(); };
    const fail = () => { clear(); reject(Error('Video decode failed')); };
    buffer.addEventListener('updateend', done, {once:true});
    buffer.addEventListener('error', fail, {once:true});
  });
  while (true) {
    let socket;
    let url;
    try {
      const bootstrap = await (await fetch('/api/bootstrap')).json();
      let receive, mime;
      const frameCounter = new MP4FrameCounter(count => receivedRates.main.add(count));
      if (bootstrap.video_transport !== 'websocket') {
        // Allow an existing simulation process to finish during a rolling deployment.
        const controller = new AbortController();
        const response = await fetch('/api/video', {signal: controller.signal});
        if (!response.ok) throw Error('Waiting for live video');
        const reader = response.body.getReader();
        socket = {send() {}, close() { controller.abort(); }};
        mime = 'video/mp4; codecs="avc1.42E01F"';
        receive = async () => {
          const {value, done} = await reader.read();
          if (done) throw Error('Video ended');
          frameCounter.feed(value);
          return value;
        };
      } else {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/video-ws`,
                             ['omega-video', 'token.' + bootstrap.token]);
      socket.binaryType = 'arraybuffer';
      const queue = [];
      let wake, closed = false;
      socket.onmessage = event => {
        try { if (event.data instanceof ArrayBuffer) frameCounter.feed(event.data); }
        catch (_) { socket.close(); return; }
        queue.push(event.data); if (wake) wake();
      };
      socket.onclose = socket.onerror = () => { closed = true; if (wake) wake(); };
      receive = async () => {
        while (!queue.length && !closed) await new Promise(resolve => { wake = resolve; });
        wake = null;
        if (!queue.length) throw Error('Video connection closed');
        return queue.shift();
      };
      ({mime} = JSON.parse(await receive()));
      }
      if (!MediaSource.isTypeSupported(mime)) {
        liveVideoFallback = true;
        video.hidden = true;
        const preview = document.getElementById('viewer');
        preview.hidden = false;
        updateLiveVideo(liveVideoPaused);
        preview.title = '浏览器不支持 H.265 视频解码，当前显示 JPEG 预览';
        return;
      }
      const source = new MediaSource();
      url = URL.createObjectURL(source);
      video.src = url;
      await new Promise(resolve => source.addEventListener('sourceopen', resolve, {once:true}));
      const buffer = source.addSourceBuffer(mime);
      let joined = false;
      while (true) {
        const value = await receive();
        const appended = waitUpdate(buffer);
        buffer.appendBuffer(value);
        await appended;
        socket.send("ack");
        if (!buffer.buffered.length) continue;
        const start = buffer.buffered.start(0), end = buffer.buffered.end(buffer.buffered.length - 1);
        if (!joined && end - start < .9) continue;
        const timing = playbackTiming(video.currentTime, start, end, joined);
        if (timing.seek !== null) video.currentTime = timing.seek;
        video.playbackRate = timing.rate;
        joined = true;
        liveVideoReady = true;
        updateLiveVideo(liveVideoPaused);
        if (video.paused) await video.play();
        if (video.currentTime - start > 10) {
          const removed = waitUpdate(buffer);
          buffer.remove(start, video.currentTime - 5);
          await removed;
        }
      }
    } catch (_) {
      // Paused initialization has no recorded fragment yet. Retry without
      // turning a normal pause into an operation-panel error.
    } finally {
      if (socket) socket.close();
      if (url) URL.revokeObjectURL(url);
      liveVideoReady = false;
      updateLiveVideo(liveVideoPaused);
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
