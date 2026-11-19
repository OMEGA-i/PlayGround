"""Execute-then-plan timing, causal history, and chunk coordinate regression tests."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

from omega_playground.policy.supermimic.chunk import ProprioHistory, streaming_window
from omega_playground.policy.supermimic.heading import place_from_proprio
from omega_playground.policy.supermimic.rollout import TrackingSession


def test_history_samples_executed_states_in_physics_time():
    state=torch.zeros(1,17)
    history=ProprioHistory(state,100)
    snapshot=history.snapshot()
    assert snapshot['ticks'].tolist()==[100]*4
    assert snapshot['valid'].tolist()==[False,False,False,True]
    for tick in range(101,109):
        state.fill_(tick)
        history.add(state,tick)
    state.fill_(-999)  # owned snapshots, not views of mutable simulator storage
    snapshot=history.snapshot()
    assert snapshot['ticks'].tolist()==[102,104,106,108]
    np.testing.assert_array_equal(snapshot['states'][:,0],[102,104,106,108])
    np.testing.assert_allclose(snapshot['timestamp'],snapshot['physics_steps']/200)
    with pytest.raises(ValueError):
        history.add(state,110)


def test_streaming_consumes_all_intervals_without_reset_or_extra_pd_steps():
    joints=2
    session=TrackingSession.__new__(TrackingSession)
    session.names=(['a','b'],[])
    # 60 new 25 FPS frames + current anchor -> 121 samples /120 controls.
    reference=torch.zeros(121,17)
    reference[:,0]=torch.arange(121)
    reference[:,7:13]=2
    reference[:,15:17]=3
    session.queue=reference.clone()
    session.history=torch.zeros(1,10,1)
    calls=[]
    windows=[]
    def actor(history,goals):
        windows.append(goals.clone())
        return torch.zeros(1,joints)
    session.actor=actor
    def step(action):
        calls.append(action)
        return torch.ones(1,17)*len(calls)
    session.env=SimpleNamespace(step=step,view=lambda _:torch.zeros(1,joints),default=0,
        layout=SimpleNamespace(done=lambda state,target:torch.tensor([False]),
                               obs=lambda *a,**kw:torch.ones(1,1)*len(calls)))
    for _ in range(120):
        session.step(streaming=True)
    assert len(calls)==120 and len(session.queue)==1
    assert session.history[0,:,0].tolist()==list(range(111,121))
    assert session.lookahead_valid.tolist()==[True,True]+[False]*9
    assert windows[-1][0,:,0].tolist()==[119]+[120]*10
    assert torch.all(windows[-1][0,2:,7:13]==0)
    assert torch.all(windows[-1][0,2:,15:17]==0)
    assert torch.equal(session.queue,reference[-1:])  # padding never enters queue
    with pytest.raises(RuntimeError,match='exhausted'):
        session.step(streaming=True)
    assert len(calls)==120


def quat(angle):
    q=Rotation.from_euler('z',angle).as_quat()
    return q[[3,0,1,2]]


def test_chunk_transform_uses_actual_heading_and_position_not_previous_plan():
    q=np.zeros((3,9))
    q[:,:3]=[[2,3,1],[3,3,1.1],[4,3,1.2]]
    q[:,3:7]=quat(.2)
    q[:,7:]=[[.1,.2],[.2,.3],[.3,.4]]
    ref=dict(qpos=q,root_linear_velocity=np.tile([1.,0,0],(3,1)),
             root_angular_velocity_world=np.tile([0.,0,.5],(3,1)),
             joint_velocity=np.ones((3,2)))
    state=np.r_[10.,20.,2.,quat(.2+np.pi/2)]
    world=place_from_proprio(ref,state)
    np.testing.assert_allclose(world['qpos'][:,:3],[[10,20,2],[10,21,2.1],[10,22,2.2]])
    np.testing.assert_allclose(world['qpos'][0,3:7],state[3:7])
    np.testing.assert_allclose(world['qpos'][:,7:],q[:,7:])
    np.testing.assert_allclose(world['root_linear_velocity'],np.tile([0,1,0],(3,1)),atol=1e-7)
    recovered=(np.c_[world['qpos'][:,:3],np.ones(3)] @ np.linalg.inv(world['world_from_chunk']).T)[:,:3]
    np.testing.assert_allclose(recovered,q[:,:3])
    np.testing.assert_array_equal(ref['qpos'],q)


def test_live_loop_plans_only_after_execution_and_records_observed_boundary(tmp_path, monkeypatch):
    import importlib
    import json
    import queue
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools/supermimic'))
    live=importlib.import_module('serve_supermimic_live')
    monkeypatch.setattr(live.time,'sleep',lambda _:None)
    world=live.World.__new__(live.World)
    world.state=torch.zeros(1,17)
    world.state[0,3]=1
    world.tick=world.chunk_index=world.chunk_remaining=0
    world.chunk_trace=world.next_config=None
    world.paused=world.last_warning=False
    world.robot='test'
    world.vx=world.vy=world.facing=0.
    world.trace_root=tmp_path
    world.proprio_history=ProprioHistory(world.state,0)
    world.server=SimpleNamespace(commands=queue.Queue())
    world.select_command=lambda:None
    world.publish=lambda:None
    world.event=lambda *a,**kw:None
    world.samples=[]
    planned=[]
    rendered=[]
    def generate(vx,vy,observation):
        planned.append((world.tick,observation['states'].copy()))
        q=np.zeros((5,9));q[:,3]=1
        return dict(qpos=q,joint_names=['a','b'],fps=50,embodiment='test',
                    root_linear_velocity=np.zeros((5,3)),root_angular_velocity_world=np.zeros((5,3)),
                    joint_velocity=np.zeros((5,2)),timestamps=np.arange(5)/50,
                    includes_current_anchor=True,planner_future_frames=2,observation_tick=world.tick)
    world.planner=SimpleNamespace(generate=generate)
    tracker=SimpleNamespace(names=(['a','b'],[]),checkpoint_sha256='test',queue=None)
    tracker.append=lambda ref:setattr(tracker,'queue',ref['qpos'])
    def step(**kwargs):
        assert kwargs==dict(measured_heading=True,streaming=True)
        tracker.queue=tracker.queue[1:]
        tracker.last_action=torch.zeros(1,2)
        tracker.last_pd_target=torch.ones(1,2)
        tracker.lookahead_valid=torch.ones(11,dtype=torch.bool)
        actual=world.state.clone()
        actual[0,0]+=1  # Deliberately differs from stationary reference.
        return actual,torch.tensor([False])
    tracker.step=step
    world.tracker=tracker
    def snapshot(session,tick):
        rendered.append((tick,float(world.state[0,0])))
        return dict(physics_step=tick*4,timestamp=tick/50,ego_jpeg=b'ego',main_jpeg=b'main')
    world.view=SimpleNamespace(snapshot=snapshot,show=lambda *a:None,offset=torch.zeros(3),
                               renderer=SimpleNamespace(frames=0))
    def finish(status):
        live.World.finish_chunk(world,status)
        if world.chunk_index==2:world.next_config='finished'
    world.finish_chunk=finish
    assert world.run()=='finished'
    assert [tick for tick,_ in planned]==[0,4]
    assert rendered==[(0,0.),(4,4.)]
    assert planned[1][1][-1,0]==4  # Actual execution, not predicted chunk tail.
    assert world.tick==8
    with np.load(tmp_path/'chunk-000001/reference.npz') as ref:
        assert ref['qpos'][0,0]==4
        np.testing.assert_allclose(ref['timestamps'],np.arange(4,9)/50)
    with np.load(tmp_path/'chunk-000000/execution.npz') as execution:
        assert execution['actual'].shape==(5,17)
        np.testing.assert_array_equal(execution['state_physics_steps'],[0,4,8,12,16])
    manifest=json.loads((tmp_path/'chunk-000001/manifest.json').read_text())
    assert manifest['executed_control_steps']==4 and manifest['status']=='complete'
    world.chunk_remaining=1
    with pytest.raises(RuntimeError,match='must finish'):
        world.plan_chunk()


def test_decision_snapshot_rejects_stale_or_missing_camera():
    from omega_playground.policy.supermimic.live_view import LiveView
    view=LiveView.__new__(LiveView)
    view.renderer=SimpleNamespace(latest_sim_time=0.,ego_jpeg=b'ego',latest_jpeg=b'main')
    view.show=lambda *a,**kw:None
    with pytest.raises(RuntimeError,match='timestamps differ'):
        view.snapshot(None,10)
    view.renderer.latest_sim_time=.2
    assert view.snapshot(None,10)['physics_step']==40
    view.renderer.ego_jpeg=None
    with pytest.raises(RuntimeError,match='ego camera'):
        view.snapshot(None,10)
