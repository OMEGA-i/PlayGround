// Isolated browser test: pass the private test server URL, never a user's active task.
const {chromium} = require('playwright');
const fs = require('node:fs');
(async () => {
  const url = process.argv[2], output = process.argv[3];
  if (!url || !output) throw Error('Usage: node qc_video_ws.cjs URL OUTPUT');
  fs.mkdirSync(output, {recursive:true});
  const replay = process.argv.includes("--replay");
  const peer = replay ? null : process.argv[4];
  const initial = peer ? await (await fetch(peer+'/api/state')).json() : null;
  const peerCommand = async kind => {
    const {token}=await(await fetch(peer+'/api/bootstrap')).json();
    const response=await fetch(peer+'/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Omega-Token':token},body:JSON.stringify({id:'video-qc-'+kind+'-'+Date.now(),kind})});
    if (!response.ok) throw Error('Peer command failed');
  };
  if (initial && !initial.paused) {
    await peerCommand('pause');
    for(let i=0;i<100;i++) {
      if((await(await fetch(peer+'/api/state')).json()).paused) break;
      await new Promise(r=>setTimeout(r,100));
    }
    if(!(await(await fetch(peer+'/api/state')).json()).paused) throw Error('Peer did not pause');
  }
  const browser = await chromium.launch({channel:'chrome', headless:false});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    let messages=0, bytes=0;
    page.on('console', m => console.log('console',m.text()));
    page.on('pageerror', e => console.log('pageerror',e.message));
    page.on('websocket', ws => { console.log('WS',ws.url());ws.on('socketerror', e => console.log('WSerror',e));ws.on('close', () => console.log('WSclosed')); });
    page.on('websocket', ws => ws.on('framereceived', ({payload}) => {messages++;bytes+=payload.length;}));
    if(replay) {
      await page.goto(url+'/api/state');
      await page.setContent('<img id="viewer"><video id="live-video" autoplay muted playsinline style="width:1280px"></video>');
      await page.addScriptTag({url:url+'/video.js'});
      await page.evaluate(() => {updateLiveVideo(false);startLiveVideo();});
    } else {
      await page.goto(url);
      await page.locator(".simulation-panel").scrollIntoViewIfNeeded();
    }
    const supported = await page.evaluate(() => MediaSource.isTypeSupported('video/mp4; codecs="hvc1.1.6.L123.B0"'));
    if (!supported) throw Error('Chrome cannot decode HEVC on this host');
    await page.evaluate(async () => {
      const {token} = await (await fetch('/api/bootstrap')).json();
      const r = await fetch('/api/command', {method:'POST',headers:{'Content-Type':'application/json','X-Omega-Token':token},
        body:JSON.stringify({id:'video-qc-resume-'+Date.now(),kind:'resume'})});
      if (!r.ok) throw Error(await r.text());
    });
    try { await page.waitForFunction(() => document.querySelector('#live-video').currentTime > 2, null, {timeout:45000}); } catch(e) { console.log('debug', messages,bytes,await page.evaluate(() => {const v=document.querySelector('#live-video');return {time:v.currentTime,ready:v.readyState,error:v.error?.message,src:v.src};}));await page.screenshot({path:output+'/failure.png'});throw e; }
    const samples=[];
    for (let i=0;i<10;i++) {
      await page.waitForTimeout(1000);
      samples.push(await page.evaluate(() => {
        const v=document.querySelector('#live-video'),q=v.getVideoPlaybackQuality();
        return {wall:performance.now(),time:v.currentTime,total:q.totalVideoFrames,dropped:q.droppedVideoFrames,ready:v.readyState,
          width:v.videoWidth,height:v.videoHeight,hidden:v.hidden};
      }));
    }
    await page.locator('#live-video').screenshot({path:output+'/video.png'});
    fs.writeFileSync(output+'/report.json',JSON.stringify({supported,messages,bytes,samples},null,2));
    if(samples.at(-1).total-samples[0].total<100 || samples.at(-1).hidden) throw Error('Video did not sustain playback');
    console.log(JSON.stringify({supported,messages,bytes,first:samples[0],last:samples.at(-1)}));
    await page.evaluate(async () => {
      const {token}=await(await fetch('/api/bootstrap')).json();
      await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Omega-Token':token},body:JSON.stringify({id:'video-qc-pause-'+Date.now(),kind:'pause'})});
    });
  } finally {
    await browser.close();
    if(initial && !initial.paused) {
      const current=await(await fetch(peer+'/api/state')).json();
      if(current.session_id===initial.session_id && current.paused) await peerCommand('resume');
    }
  }
})().catch(e => { console.error(e);process.exitCode=1; });
