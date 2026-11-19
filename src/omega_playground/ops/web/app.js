const el = id => document.getElementById(id);
const canvas = el('map'), ctx = canvas.getContext('2d'), image = new Image(), occupancy = new Image();
let bootstrap, state, bounds, connected = false, pending = null;
let draft = {points: [], submitted: false};
const embodimentOf = avatar => ['ue4','ue5'].includes(avatar)?'sih2300':avatar;
let workflowPending=false;
function syncAvatars(value=el('avatar').value, selectedBody=null) {
  const options=OmegaWorkflow.robotOptions(state?.wbc_policy || el('wbc-policy').value,bootstrap?.prepared_options);
  const bodies=[...new Set(options.map(o=>o.embodiment))];
  const body=selectedBody || embodimentOf(value);
  el('embodiment').replaceChildren(...bodies.map(id=>new Option(id.toUpperCase(),id)));
  el('embodiment').value=bodies.includes(body)?body:bodies[0];
  const choices=options.filter(o=>o.embodiment===el('embodiment').value);
  el('avatar').replaceChildren(...choices.map(o=>new Option(o.label,o.avatar)));
  el('avatar').value=choices.some(o=>o.avatar===value)?value:choices[0]?.avatar;
  el('appearance-label').hidden=choices.length<2;
}
let populationEditing=false, populationKey=null, rebuilding=false;
let motionEditing = false, lastMotion = null, lastCamera = null;
const colors = ['#e66740', '#5689eb', '#78d3ac'];
const phases = {idle:'站立待命',decelerate:'减速停稳', turn:'转向', accelerate:'加速', track:'行进中', settle:'到点停稳', complete:'已完成'};
const preview = OmegaWorkflow.createPreview((body,signal)=>request('/api/drawn-path',body,signal),()=>render());
const commandNames = {control_mode:'控制模式',drawn_path:'手绘轨迹',select_scene:'切换场景',configure_robot:'机器人配置',pause:'暂停', resume:'继续', locomotion:'运动参数', reset:'恢复', viewer:'视角'};

function msg(text) { el('message').textContent = text; }
async function request(path, body, signal) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json', 'X-Omega-Token':bootstrap.token}, body:JSON.stringify(body),signal} : {signal});
  const data = await response.json();
  if (!response.ok) { const error = Error(data.error || response.status); error.rejected = true; throw error; }
  return data;
}
function team() { return state?.teams?.[0]; }
function motionValue() { return {mode:el('sonic-mode').value, speed:Number(el('sonic-speed').value)}; }
function rules() {
  const value = motionValue(), current = team()?.locomotion;
  return OmegaWorkflow.controls({state, connected, pending:pending || workflowPending, draft,
    motionChanged: Boolean(current && (current.mode !== value.mode || current.speed !== value.speed)),
    motionValid: el('sonic-speed').checkValidity(), cameraAvailable:state?.viewer, previewReady:preview.value.status==='ready' && preview.value.plan?.validation?.valid===true});
}
function legColor(i) { return `hsl(${140+(25+i*73)%200} 72% 65%)`; }
function pixel(p) { return [(p[0]-bounds[0])/(bounds[2]-bounds[0])*canvas.width, (bounds[3]-p[1])/(bounds[3]-bounds[1])*canvas.height]; }
let actualTrail=[], trailClock=-1;
function draw() {
  if (!bounds) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (image.complete && image.naturalWidth) ctx.drawImage(image,0,0,canvas.width,canvas.height);
  if (occupancy.complete && occupancy.naturalWidth) ctx.drawImage(occupancy,0,0,canvas.width,canvas.height);
  for (const [i,t] of (state?.teams || []).entries()) {
    if (!t.reference?.length) continue;
    ctx.strokeStyle=colors[i%colors.length]; ctx.lineWidth=3; ctx.beginPath();
    for (const route of [t.reference]) {
      ctx.beginPath(); route.forEach((p,j)=>{ const [x,y]=pixel(p); j?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
    }
  }
  const position=state?.agents?.[0]?.position;
  if(position){
    if(state.sim_time<trailClock)actualTrail=[];
    trailClock=state.sim_time;
    const last=actualTrail.at(-1);
    if(!last || Math.hypot(position[0]-last[0],position[1]-last[1])>.1){
      actualTrail.push(position.slice(0,2));if(actualTrail.length>4000)actualTrail.shift();
    }
    ctx.strokeStyle='#ffa15c';ctx.lineWidth=2;ctx.beginPath();
    actualTrail.forEach((p,i)=>i?ctx.lineTo(...pixel(p)):ctx.moveTo(...pixel(p)));ctx.stroke();
  }
  if (preview.value.plan) {
    ctx.save(); ctx.strokeStyle='#79dec7'; ctx.lineWidth=4; ctx.setLineDash([12,7]); ctx.beginPath();
    for (const [index,route] of (preview.value.plan.legs || [preview.value.plan.route]).entries()) {
      ctx.strokeStyle=legColor(index);
      ctx.beginPath(); route.forEach((p,i)=>{ const [x,y]=pixel(p); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }); ctx.stroke();
    }
    ctx.restore();
    ctx.fillStyle='#ff5555';
    for (const p of preview.value.plan.validation?.invalid_points || []) {
      ctx.beginPath(); ctx.arc(...pixel(p),4,0,Math.PI*2); ctx.fill();
    }
  }
  draft.points.forEach((p,i)=>{
    const [x,y]=pixel(p); ctx.fillStyle=legColor(i); ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2); ctx.fill();
    ctx.fillStyle='white'; ctx.font='16px system-ui'; ctx.fillText(i+1,x+9,y-5);
  });
  for (const a of state?.agents || []) {
    const [x,y]=pixel(a.position); ctx.fillStyle=colors[(state.teams.findIndex(t=>t.agents.includes(a.id))+colors.length)%colors.length];
    ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.fill(); ctx.strokeStyle='white'; ctx.stroke();
    ctx.fillStyle='white'; ctx.fillText(a.id,x+10,y-5);
  }
}
function clearDraft() {
  draft.points=[]; draft.origin=null; draft.submitted=false; preview.clear();
}
function updatePreview() {
  const selected=team();
  if (draft.submitted || !selected || !selected.agents.length || !draft.points.length) {
    preview.clear(); render(); return;
  }
  draft.origin ??= state.agents[selected.agents[0]].position.slice(0,2);
  preview.update({keypoints:[draft.origin,...draft.points]});
}
function syncMotion(force=false) {
  el('policy-quality').textContent=state?.policy_quality_note || '';
  el('policy-quality').hidden=!state?.policy_quality_note;
  const value=team()?.locomotion;
  if (!value) return;
  const key=JSON.stringify([0,value]);
  if (force || (!motionEditing && key!==lastMotion)) {
    el('sonic-mode').value=value.mode; el('sonic-speed').max=state.wbc_policy==='supermimic'?2:value.mode==='run'?3:2; el('sonic-speed').value=value.speed;
    lastMotion=key;
  }
  el('locomotion-status').textContent=`当前 ${state.wbc_policy==='supermimic'?'':(value.mode==='run'?'Run · ':'Walk · ')}${value.speed.toFixed(2)} m/s`;
}
function syncCamera() {
  const value=state?.viewer;
  if (!value || pending?.kind==='viewer') return;
  const key=JSON.stringify(value);
  if (key===lastCamera) return;
  el('camera').value=value.camera;
  for (const field of ['yaw','pitch','distance']) el(field).value=value[field];
  lastCamera=key;
}
function updateOptions() {
  const key=[state.wbc_policy,state.avatar,state.armed].join(':');
  if (!populationEditing && populationKey!==key) {
    el('wbc-policy').value=state.wbc_policy; syncAvatars(state.avatar);
    el('armed').value=String(!!state.armed); populationKey=key;
  }
  el('team-summary').textContent=`当前：${state.wbc_policy==='supermimic'?'SuperMimic':'SONIC'} · ${embodimentOf(state.avatar).toUpperCase()}`;
  el('pipeline-summary').textContent=`${state.control_mode==='manual'?'键盘':'FFPD Player'} → ${embodimentOf(state.avatar).toUpperCase()} Joystick → ${embodimentOf(state.avatar).toUpperCase()} Tracker${state.tracker_iterations?' · '+state.tracker_iterations/1000+'k':''}${['ue4','ue5'].includes(state.avatar)?' · 外观 '+state.avatar.toUpperCase():''}`;
  el('reset').textContent='原地恢复';
}
function render() {
  if (state?.ready) { updateOptions(); syncMotion(); syncCamera(); }
  el('speed-value').value=Number(el('sonic-speed').value).toFixed(2);
  el('speed-max').textContent=Number(el('sonic-speed').max).toFixed(2);
  el('sonic-speed').setAttribute('aria-valuetext',`${el('speed-value').value} m/s`);
  const scene=el('scene');
  scene.disabled=!state?.ready || !bootstrap?.scenes?.length || !!pending;
  el('scene-apply').disabled=scene.disabled || scene.value===bootstrap?.scene;
  const manual=state?.control_mode==='manual';
  el('control-mode').disabled=!state?.ready || state?.agents?.length!==1 || !!pending;
  el('control-mode').value=manual?'manual':'auto';
  el('keyboard-help').hidden=!manual;
  document.querySelector('.task-card').hidden=manual;
  const r=rules(), t=team()?.task;
  const same=el('wbc-policy').value===state?.wbc_policy && el('avatar').value===state?.avatar && (el('armed').value==='true')===!!state?.armed;
  for (const id of ['wbc-policy','embodiment','avatar','armed']) el(id).disabled=r.locked || rebuilding;
  const workflowChanged=el('wbc-policy').value!==state?.wbc_policy;
  el('workflow-apply').disabled=r.locked || rebuilding || !state?.paused || !workflowChanged;
  for (const id of ['embodiment','avatar','armed']) el(id).disabled ||= workflowChanged;
  el('gait-label').hidden=state?.wbc_policy==='supermimic';
  el('motion-help').textContent=(manual?'速度作为键盘前后移动的幅值；转向由 A / D 控制。':'速度作为自动循迹的目标；方向和转弯由 FFPD 控制。')+(state?.wbc_policy==='supermimic'?' SuperMimic 无独立步态输入。':'');
  if(bootstrap?.prepared_options){
    const armed=el('armed');
    for(const opt of armed.options) opt.disabled=!bootstrap.prepared_options.some(row=>row.avatar===el('avatar').value && String(row.armed)===opt.value);
    if(armed.selectedOptions[0]?.disabled) armed.value=[...armed.options].find(opt=>!opt.disabled)?.value || '';
  }
  el('team-config-apply').disabled=r.locked || rebuilding || workflowChanged || same;
  el('status').textContent=!connected?'连接中断':!state?.ready?'加载中':state.hold?'计算异常':state.paused?'已暂停':'执行中';
  el('task-status').textContent=!state?.ready?'等待状态':manual?(state.paused?'手动控制 · 已暂停':'手动控制 · 键盘就绪'):`${t?.mode==='drawn_path'?'手绘轨迹':t?.mode==='idle'?'待命':'初始路线'} · ${state.hold?'计算异常':t?.phase==='complete'?'已完成':state.paused?'已暂停':phases[t?.phase]||'行进中'}`;
  el('task-toggle').textContent=r.toggleLabel+(state?.teams?.length>1?'全部':'');
  el('task-toggle').disabled=!r.canToggle;
  el('task-hint').textContent=state?.hold?`已触发 ${state.hold.code}，仿真状态无法继续计算，请在“恢复”中处理。`:state?.paused && r.hasDraft?'执行新任务将开始运行；清空草稿后可继续原任务。':'';
  el('execute-hint').textContent='从当前位置出发，替换当前任务。';
  el('task-submit').disabled=!r.canExecute || !bootstrap?.task_start;
  el('task-submit').textContent=pending?.kind==='drawn_path'?'规划并执行中…':'执行任务';
  el('draft-clear').disabled=!r.canClear;
  el('draft-undo').disabled=!r.canClear;

  el('plan-info').textContent=draft.points.length?`${draft.submitted?'已提交':'已选'} ${draft.points.length} 个点 · 按编号顺序执行`:'在地图上绘制轨迹，拖动关键点修改，右键删除。';

  el('preview-status').textContent=preview.value.status==='loading'?'正在校验轨迹…':preview.value.status==='ready' && preview.value.plan.validation?.valid===false?preview.value.plan.validation.message:preview.value.status==='ready'?`预览 ${preview.value.plan.length_m.toFixed(1)} m · 点击“执行任务”开始`:
    preview.value.status==='error'?`无法生成轨迹：${preview.value.error}`:'';
  el('preview-status').dataset.status=preview.value.plan?.validation?.valid===false?'error':preview.value.status;
  el('map-help').textContent='点击或拖画添加关键点，拖动修改，右键删除。虚线为 B 样条轨迹。红色：障碍/无支撑；橙色：不足 0.6 m 净空；轨迹红点：不可通行。';
  canvas.style.cursor=r.canEdit?'crosshair':'default';
  el('sonic-mode').disabled=el('sonic-speed').disabled=r.locked||Boolean(state?.hold)||!team()?.locomotion;
  el('reset').disabled=!r.canReset;
  el('reset-origin').disabled=!r.canReset;

  el('reset-hint').textContent='原地恢复保留任务进度；原点恢复从任务首段重新开始。'+(state?.paused?'恢复后点击“继续”。':'恢复后继续运行。');
  for (const id of ['camera','yaw','pitch','distance']) el(id).disabled=!r.canCamera;
  el('camera-orbit').hidden=el('camera').value!=='free';
  el('camera-hint').textContent=state?.viewer?'选择后直接生效。':'等待渲染器就绪。';
  if (state?.ready) {
    el('clock').textContent=`${state.sim_time.toFixed(2)} s`;
    el('metrics').textContent=`物理 ${state.rates.physics_hz} Hz · Player ${state.rates.player_hz} Hz · WBC ${state.rates.wbc_hz} Hz\n`+
      (state.performance?`服务端 ${state.performance.wall_fps.toFixed(1)} FPS · 浏览器 ${window.liveVideoFps?.toFixed(1)??'—'} FPS\n实时倍率 ${state.performance.rtf.toFixed(2)}×\n`:'')+
      state.agents.map(a=>`机器人 ${a.id} · 速度 ${(a.speed_m_s??0).toFixed(2)} m/s${state.error_m?` · 局部路线偏差 ${state.error_m[a.id].toFixed(2)} m`:''}`).join('\n');
    updateLiveVideo(state.paused);
    updateEgoVideo(state.paused);
    updateVideoRates(state);
  }
  draw();
}
async function command(kind, fields={}, accepted=()=>{}) {
  if (pending || !connected || !state?.ready) return;
  const id=globalThis.crypto?.randomUUID?.() || ('command-'+Date.now()+'-'+Math.random());
  pending={id,kind,accepted}; render(); msg('正在提交，等待回执…');
  try { await request(kind==='viewer'?'/api/viewer':'/api/command',{id,kind,team:0,...fields}); }
  catch(error) {
    if (error.rejected) { if (kind==='viewer') lastCamera=null; pending=null; msg(error.message); render(); }
    else { connected=false; msg('连接中断，正在确认命令结果，请勿重复提交。'); render(); }
  }
}
function describeEvent(event) {
  const time=`${Number(event.sim_time??0).toFixed(1)} s`;
  if (event.kind==='ack') return `${time} · ${event.command==='reset'?(event.detail?.recovery_mode==='task_start'?'原点恢复':'原地恢复'):(commandNames[event.command]||event.command)} ${event.accepted?'成功':`失败：${event.error}`}`;
  if (event.kind==='task_phase') return `${time} · 队伍 ${(event.team??0)+1} · ${phases[event.phase]||event.phase}${event.segment?` · 第 ${event.segment}/${event.segments} 段`:''}`;
  if (event.kind==='route_replanned') return `${time} · 路线已更新 · ${Number(event.plan?.length_m??0).toFixed(1)} m`;
  if (event.kind==='locomotion_changed') return `${time} · ${event.mode==='run'?'Run':'Walk'} · ${event.speed} m/s`;
  if (event.kind==='warning') return `${time} · 告警（继续运行）：${event.code}${event.error_m?.length?' · 局部偏差 '+Math.max(...event.error_m).toFixed(2)+' m':''}`;
  if (event.kind==='warning_cleared') return `${time} · 告警已解除：${event.code}`;
  if (event.kind==='hold') return `${time} · 计算异常：${event.code}`;
  if (event.kind==='reset_audit') return `${time} · 已${event.recovery_mode==='task_start'?'原点':'原地'}恢复，原任务已保留`;
  return `${time} · ${event.kind}`;
}
async function poll() {
  try {
    let snapshot=await request('/api/state');
    if (snapshot.session_id && bootstrap.session_id && snapshot.session_id!==bootstrap.session_id) {
      location.reload(); return;
    }
    rebuilding ||= Boolean(snapshot.reconfiguring);
    const events=await request('/api/events');
    const ack=pending && events.find(e=>e.kind==='ack' && e.id===pending.id);
    if (ack) {
      snapshot=await request('/api/state');
      const completed=pending; state=snapshot;
      if (ack.accepted) completed.accepted();
      if (completed.kind==='viewer') lastCamera=null;
      pending=null;
      msg(ack.accepted?`${commandNames[completed.kind]}已生效`:ack.error);
    }
    state=snapshot; connected=true;
    if (rebuilding) msg('正在重建仿真，完成后自动连接，请稍候…');
    el('events').textContent=events.slice(-30).reverse().map(describeEvent).join('\n');
    render();
  } catch(error) { connected=false; msg(rebuilding?'正在重建仿真，完成后自动连接，请稍候…':pending?'连接中断，等待命令回执。':'连接中断，正在重连。'); render(); }
  finally { setTimeout(poll,500); }
}
let stroke=null;
function pointerXY(event) {
  const box=canvas.getBoundingClientRect();
  return [bounds[0]+(event.clientX-box.left)/box.width*(bounds[2]-bounds[0]),
          bounds[3]-(event.clientY-box.top)/box.height*(bounds[3]-bounds[1])];
}
function hitPoint(event) {
  const box=canvas.getBoundingClientRect(), point=pointerXY(event);
  const distance=p=>Math.hypot((p[0]-point[0])*box.width/(bounds[2]-bounds[0]),
                              (p[1]-point[1])*box.height/(bounds[3]-bounds[1]));
  return draft.points.findIndex(p=>distance(p)<10);
}
canvas.onpointerdown=event=>{
  if(state?.control_mode==='manual' || !bounds || !rules().canEdit || event.button!==0)return;
  const i=hitPoint(event), p=pointerXY(event);
  if(i<0 && draft.points.length>=255)return;
  draft.submitted=false;
  let index=i, freehand=false;
  if(i<0){
    const route=preview.value.plan?.route || [];
    const box=canvas.getBoundingClientRect();
    const distance=q=>Math.hypot((q[0]-p[0])*box.width/(bounds[2]-bounds[0]),
                                 (q[1]-p[1])*box.height/(bounds[3]-bounds[1]));
    let closest=-1, best=8;
    route.forEach((q,j)=>{const d=distance(q);if(d<best){best=d;closest=j;}});
    index=draft.points.length;
    if(closest>=0){
      let next=0;
      for(const q of route.slice(1,closest+1)){
        const target=draft.points[next];
        if(target && Math.hypot(q[0]-target[0],q[1]-target[1])<1e-6)next++;
      }
      index=next;
    }else freehand=true;
    draft.points.splice(index,0,p);
  }
  stroke={index,draw:freehand};
  canvas.setPointerCapture(event.pointerId);updatePreview();
};
canvas.onpointermove=event=>{
  if(!stroke)return;
  const p=pointerXY(event);
  if(stroke.draw){
    if(draft.points.length>=255 || Math.hypot(...p.map((v,i)=>v-draft.points.at(-1)[i]))<1)return;
    draft.points.push(p);
  }else draft.points[stroke.index]=p;
  updatePreview();
};
canvas.onpointerup=canvas.onpointercancel=()=>{stroke=null;};
canvas.oncontextmenu=event=>{
  event.preventDefault();
  if(state?.control_mode==='manual' || !bounds || !rules().canEdit)return;
  const i=hitPoint(event);
  if(i>=0){draft.points.splice(i,1);draft.submitted=false;updatePreview();}
};
for (const id of ['embodiment','avatar','armed']) el(id).onchange=()=>{
  populationEditing=true;
  if(id==='embodiment')syncAvatars(el('avatar').value,el('embodiment').value);
  render();
};
el('wbc-policy').onchange=()=>{populationEditing=true;render();};
el('workflow-apply').onclick=async()=>{
  if(el('workflow-apply').disabled)return;
  workflowPending=true;render();
  try{await request('/api/workflow',{workflow:el('wbc-policy').value});location.reload();}
  catch(error){workflowPending=false;msg(error.message);render();}
};
el('team-config-apply').onclick=()=>{
  if (el('team-config-apply').disabled) return;
  command('configure_robot',{avatar:el('avatar').value,armed:el('armed').value==='true'},()=>{rebuilding=true;clearDraft();});
};

el('draft-clear').onclick=()=>{ if (!rules().canClear) return; clearDraft(); msg('草稿已清空，当前任务保持不变。'); render(); };
el('draft-undo').onclick=()=>{ if (!rules().canClear) return; draft.points.pop(); draft.submitted=false; msg('已撤销末点，当前任务保持不变。'); updatePreview(); };

el('task-submit').onclick=()=>{
  if (!rules().canExecute || !bootstrap.task_start) return;
  command('drawn_path',{keypoints:[draft.origin,...draft.points],start:true},()=>{
    draft.submitted=true;
  });
};
el('task-toggle').onclick=()=>{ if (rules().canToggle) command(state.paused?'resume':'pause'); };
let motionTimer, motionRevision=0;
function queueMotion() {
  motionEditing=true; render();
  const revision=++motionRevision;
  clearTimeout(motionTimer);
  function submit() {
    if(revision!==motionRevision)return;
    if(pending || !connected || !state?.ready){motionTimer=setTimeout(submit,120);return;}
    if(!rules().canApplyMotion)return;
    command('locomotion',motionValue(),()=>{
      if(revision===motionRevision){motionEditing=false;syncMotion(true);}
    });
  }
  motionTimer=setTimeout(submit,120);
}
el('sonic-mode').onchange=()=>{ const run=el('sonic-mode').value==='run'; el('sonic-speed').max=run?3:2; el('sonic-speed').value=run?1.5:.65; queueMotion(); };
el('sonic-speed').oninput=queueMotion;
function recover(recovery_mode) { if (!rules().canReset) return; command('reset',{recovery_mode},updatePreview); }
el('reset').onclick=()=>recover('in_place');
el('reset-origin').onclick=()=>recover('task_start');
function cameraChanged() {
  if (!rules().canCamera) return;
  const value={camera:el('camera').value,agent:0,yaw:Number(el('yaw').value),pitch:Number(el('pitch').value),distance:Number(el('distance').value)};
  if (JSON.stringify(value)===JSON.stringify(state.viewer)) return;
  command('viewer',value,()=>{lastCamera=null;});
}
for (const id of ['camera','yaw','pitch','distance']) el(id).onchange=cameraChanged;
(async()=>{
  try {
    bootstrap=await request('/api/bootstrap'); el('wbc-policy').querySelector('[value="supermimic"]').disabled=false; el('scene').value=bootstrap.scene || ''; document.querySelector('.map-panel h1').textContent=el('scene').selectedOptions[0]?.text || '2D 地图';  bounds=bootstrap.map.bounds;

    canvas.width=1200; canvas.height=Math.round(1200*(bounds[3]-bounds[1])/(bounds[2]-bounds[0])); fitMapViewport();
    const ego=el('ego-image');
    ego.onload=()=>{ego.hidden=false;el('ego-wait').hidden=true;};
    occupancy.onload=draw; occupancy.src='/occupancy.png';
    image.onload=draw; image.src='/map.png'; poll(); startLiveVideo();
  } catch(error) { msg('初始化失败，请刷新页面重试。'); render(); }
})();

// Fit the actual canvas box, so click coordinates remain accurate without letterbox offsets.
function fitMapViewport() {
  const host = canvas.parentElement;
  if (window.innerWidth < 1000) {
    canvas.style.width = '100%'; canvas.style.height = 'auto'; return;
  }
  const scale = Math.min(host.clientWidth / canvas.width, host.clientHeight / canvas.height);
  canvas.style.width = `${Math.max(0, canvas.width * scale)}px`;
  canvas.style.height = `${Math.max(0, canvas.height * scale)}px`;
}
new ResizeObserver(fitMapViewport).observe(canvas.parentElement);
window.addEventListener('resize', fitMapViewport);

const pressed=new Set();
let sendingKeys=false;
async function sendKeys(){
  if(sendingKeys || state?.control_mode!=='manual' || !connected)return;
  sendingKeys=true;
  try {
    await request('/api/command',{kind:'keyboard',id:'key-'+Date.now()+'-'+Math.random(),
      forward:Number(pressed.has('KeyW'))-Number(pressed.has('KeyS')),
      turn:Number(pressed.has('KeyA'))-Number(pressed.has('KeyD'))});
  }catch(error){msg(error.message);}finally{sendingKeys=false;}
}
function releaseKeys(){pressed.clear();sendKeys();}
el('control-mode').onchange=()=>{
  const mode=el('control-mode').value;
  releaseKeys();
  command('control_mode',{mode},()=>{
    if(mode==='auto' && draft.points.length){draft.origin=null;draft.submitted=false;updatePreview();}
    render();
  });
};
window.addEventListener('keydown',event=>{
  if(state?.control_mode!=='manual' || !['KeyW','KeyS','KeyA','KeyD'].includes(event.code)
     || event.target.matches('input,textarea'))return;
  event.preventDefault();if(!event.repeat){pressed.add(event.code);sendKeys();}
});
window.addEventListener('keyup',event=>{if(pressed.delete(event.code))sendKeys();});
window.addEventListener('blur',releaseKeys);
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeys();});
setInterval(()=>{if(pressed.size)sendKeys();},120);

el('scene').onchange=render;
el('scene-apply').onclick=()=>command('select_scene',{scene:el('scene').value},()=>{rebuilding=true;});
