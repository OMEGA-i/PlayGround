const {test} = require('node:test');
const assert = require('node:assert/strict');
const {controls,createPreview} = require('../../src/omega_playground/ops/web/workflow.js');
const base = () => ({
  state: {ready:true,paused:true,hold:null,teams:[{id:0,agents:[0],task:{phase:'track'}}]},
  connected:true,pending:null,draft:{team:0,mode:'navigate',points:[],patrol:''},
  motionChanged:false,motionValid:true,cameraAvailable:true,previewReady:true,
});
test('draft -> execute -> retained selection -> pause -> continue is unambiguous', () => {
  const input=base();
  assert.equal(controls(input).canExecute,false);
  assert.equal(controls(input).canToggle,true);
  input.draft.points=[[1,2],[3,4]];
  assert.equal(controls(input).canExecute,true);
  assert.equal(controls(input).canToggle,false); // cannot accidentally continue old task
  input.pending={kind:'navigate'};
  for (const [key,value] of Object.entries(controls(input))) if (key.startsWith('can')) assert.equal(value,false,key);
  input.pending=null; input.state.paused=false; input.draft.submitted=true;
  assert.equal(controls(input).canExecute,false); // cannot submit the same draft twice
  assert.equal(controls(input).toggleLabel,'暂停');
  assert.equal(controls(input).canToggle,true);
  input.state.paused=true;
  assert.equal(controls(input).canToggle,true); // retained points must not block resume
  assert.equal(controls(input).canClear,true);
  assert.equal(input.draft.points.length,2);
  input.state.paused=false; input.draft.submitted=false;
  input.draft.points.push([5,6]);
  assert.equal(controls(input).canExecute,true);
  assert.equal(controls(input).canToggle,true); // pause remains available while drafting
  input.state.paused=true;
  assert.equal(controls(input).canToggle,false);
  input.draft.points=[];
  assert.equal(controls(input).toggleLabel,'继续');
  assert.equal(controls(input).canToggle,true);
});
test('recovery works while running; computational failure permits recovery/viewing', () => {
  const input=base(); input.state.paused=false;
  assert.equal(controls(input).canReset,true);
  input.state.paused=true; input.state.hold={code:'fall'};
  input.draft.points=[[1,2]]; input.motionChanged=true;
  const r=controls(input);
  for (const key of ['canExecute','canToggle','canEdit','canApplyMotion']) assert.equal(r[key],false,key);
  assert.equal(r.canReset,true); assert.equal(r.canCamera,true);
  input.connected=false;
  for (const [key,value] of Object.entries(controls(input))) if (key.startsWith('can')) assert.equal(value,false,key);
});
test('motion requires a valid change, completed tasks cannot resume', () => {
  const input=base();
  assert.equal(controls(input).canApplyMotion,false);
  input.motionChanged=true;
  assert.equal(controls(input).canApplyMotion,true);
  input.motionValid=false;
  assert.equal(controls(input).canApplyMotion,false);
  input.state.teams[0].task.phase='complete';
  assert.equal(controls(input).canToggle,false);
  input.state.teams.push({id:1,agents:[1],task:{phase:'track'}});
  assert.equal(controls(input).canToggle,true); // global resume still useful for other team
});

test('execution waits for a valid preview, while points remain editable', () => {
  const input=base(); input.draft.points=[[1,2]]; input.previewReady=false;
  assert.equal(controls(input).canExecute,false);
  assert.equal(controls(input).canEdit,true);
  input.previewReady=true;
  assert.equal(controls(input).canExecute,true);
});
const tick = () => new Promise(resolve=>setTimeout(resolve,5));
test('newer preview wins even if the cancelled request returns late', async () => {
  const requests=[];
  const preview=createPreview((body,signal)=>new Promise((resolve,reject)=>requests.push({body,signal,resolve,reject})),()=>{},0);
  preview.update({goals:[[1,2]]}); await tick();
  preview.update({goals:[[1,2],[3,4]]}); await tick();
  assert.equal(requests[0].signal.aborted,true);
  assert.equal(preview.value.plan,null);
  requests[1].resolve({route:[[0,0],[1,2],[3,4]],length_m:8}); await tick();
  requests[0].resolve({route:[[0,0],[1,2]],length_m:3}); await tick();
  assert.equal(preview.value.status,'ready');
  assert.equal(preview.value.plan.length_m,8);
  preview.clear();
  assert.equal(preview.value.status,'idle');
});
test('clearing a draft discards in-flight success and errors', async () => {
  for (const failed of [false,true]) {
    let complete;
    const preview=createPreview(()=>new Promise((resolve,reject)=>{complete=failed?reject:resolve;}),()=>{},0);
    preview.update({patrol_id:'north'}); await tick(); preview.clear();
    complete(failed?Error('unreachable'):{route:[[0,0],[1,2]]}); await tick();
    assert.equal(preview.value.status,'idle'); assert.equal(preview.value.plan,null);
  }
});
test('failure removes the previous route and a corrected draft can recover', async () => {
  const preview=createPreview(async body=>{
    if (body.invalid) throw Error('unreachable');
    return {route:[[0,0],[2,3]],length_m:4};
  },()=>{},0);
  preview.update({goals:[[2,3]]}); await tick();
  assert.equal(preview.value.status,'ready');
  preview.update({invalid:true});
  assert.equal(preview.value.plan,null);
  await tick(); assert.equal(preview.value.status,'error');
  assert.equal(preview.value.error,'unreachable');
  preview.update({goals:[[2,3]]}); await tick();
  assert.equal(preview.value.status,'ready');
});
test('manual simulation can resume with a retained auto draft but cannot submit it', () => {
  const input=base(); input.state.control_mode='manual';
  input.state.teams[0].task.phase='idle'; input.draft.points=[[1,2]];
  assert.equal(controls(input).canToggle,true);
  assert.equal(controls(input).canExecute,false);
  assert.equal(controls(input).canEdit,false);
  input.state.control_mode='auto';input.draft.points=[];
  assert.equal(controls(input).canToggle,false);
});
test('embodiments and skins follow the workflow and prepared bundles', () => {
  const {robotOptions}=require('../../src/omega_playground/ops/web/workflow.js');
  const prepared=['g1','a3','sih2300','ue4','ue5'].map(avatar=>({avatar}));
  assert.deepEqual(robotOptions('sonic',prepared).map(x=>x.avatar),['g1']);
  const options=robotOptions('supermimic',prepared);
  assert.deepEqual(options.filter(x=>x.embodiment==='sih2300').map(x=>x.avatar),['sih2300','ue4','ue5']);
  assert.equal(options.some(x=>x.embodiment==='ue4'),false);
  assert.equal(options.some(x=>x.avatar==='h2'),false);
});
