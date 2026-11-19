/* Shared state rules for the workbench and its transition tests. */
(function(root) {
  function robotOptions(workflow, prepared) {
    const all=workflow==='sonic'?[['g1','g1','G1 原生']]:[
      ['g1','g1','G1 原生'],['h2','h2','H2 原生'],['a3','a3','A3 原生'],
      ['sih2300','sih2300','SIH2300 原生'],['sih2300','ue4','UE4-Mann'],['sih2300','ue5','UE5-Mann']];
    return all.filter(([,avatar])=>!prepared || prepared.some(p=>p.avatar===avatar))
      .map(([embodiment,avatar,label])=>({embodiment,avatar,label}));
  }
  function controls({state, connected, pending, draft, motionChanged, motionValid, cameraAvailable, previewReady=false}) {
    const ready = Boolean(connected && state?.ready);
    const locked = !ready || Boolean(pending);
    const hold = Boolean(state?.hold);
    const paused = Boolean(state?.paused);
    const team = state?.teams?.[0];
    const hasSelection = draft.points.length > 0;
    const hasDraft = hasSelection && !draft.submitted;
    const manual=state?.control_mode==='manual';
    const runnable = manual || Boolean(state?.teams?.some(t => t.task && !['complete','idle'].includes(t.task.phase)));
    return {
      locked, hasDraft,
      canExecute: !locked && !hold && !manual && hasDraft && previewReady && team?.agents.length > 0,
      canToggle: !locked && !hold && (!paused || ((manual || !hasDraft) && runnable)),
      toggleLabel: paused ? '继续' : '暂停',
      canClear: !locked && hasSelection,
      canEdit: !locked && !hold && !manual && team?.agents.length > 0,
      canApplyMotion: !locked && !hold && motionChanged && motionValid,
      canReset: !locked && Boolean(team),
      canCamera: !locked && Boolean(cameraAvailable),
    };
  }
  function createPreview(request, changed, delay=120) {
    let revision=0, timer, controller;
    const preview = {
      value: {status:'idle', plan:null, error:''},
      clear() {
        revision++; clearTimeout(timer); controller?.abort();
        preview.value={status:'idle', plan:null, error:''};
      },
      update(body) {
        preview.clear();
        const current=revision;
        preview.value={status:'loading', plan:null, error:''}; changed();
        controller=new AbortController();
        const signal=controller.signal;
        timer=setTimeout(async()=>{
          try {
            const plan=await request(body,signal);
            if (current!==revision) return;
            preview.value={status:'ready',plan,error:''};
          } catch(error) {
            if (current!==revision) return;
            preview.value={status:'error',plan:null,error:error.message};
          }
          changed();
        },delay);
      },
    };
    return preview;
  }
  if (typeof module !== 'undefined') module.exports = {controls, robotOptions, createPreview};
  else root.OmegaWorkflow = {controls, robotOptions, createPreview};
})(typeof window === 'undefined' ? globalThis : window);
