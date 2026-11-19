"""Single-agent SuperMimic live workbench; the GPU Tracker owns the sole world."""
import argparse
import json
import hashlib
from collections import deque
import numpy as np
import math
import os
from pathlib import Path
import queue
import time
from types import SimpleNamespace

import torch

from run_supermimic import ROOT, SHARED, STEPS, tracker_checkpoint
from omega_playground.policy.supermimic.release import tracker_release_id
from omega_playground.control.drawn_path import drawn_path
from omega_playground.control.tensor_web_player import TensorWebPlayer
from omega_playground.ops.server import OperationsServer
from omega_playground.policy.base import RobotBatchView
from omega_playground.policy.supermimic.heading import HeadingFrame, place_from_proprio
from omega_playground.policy.supermimic.chunk import ProprioHistory
from omega_playground.policy.supermimic.live_client import JoystickClient, JoystickTimeout
from omega_playground.policy.supermimic.live_view import LiveView
from omega_playground.policy.supermimic.rollout import TrackingSession

CONTACT_SHA={'campus':'a1f8f8087b1462f1cc977b3c0084df7946a296890f5fff4291bad44815965e4f',
             'obstacle':'754213e8a512ddffb988202916a58f6b2e9257ef9c5671f711e3ba48a409a8af'}
from omega_playground.ops.scenes import SCENES

ROBOTS=STEPS
AVATARS={r:r for r in ROBOTS}|{'ue4':'sih2300','ue5':'sih2300'}


def variants(scene):
    return [dict(avatar=avatar,armed=armed) for avatar in AVATARS for armed in (False,True)
            if (ROOT/'artifacts/mjwarp'/f'supermimic-{avatar}-{scene}{"-armed" if armed else ""}'/'SHA256SUMS.json').is_file()]


class World:
    def __init__(self, avatar, scene, port, armed=False):
        robot=AVATARS[avatar]
        self.avatar,self.armed=avatar,armed
        self.robot,self.scene=robot,scene
        self.tracker_release_id=tracker_release_id()
        _,path,config=SCENES[scene]
        nav=SimpleNamespace(directory=ROOT/'artifacts/maps'/path)
        nav.meta=json.loads((nav.directory/'map.json').read_text())
        with np.load(nav.directory/'layers.npz') as layers:
            self.support=layers['support_height']
        self.samples=deque(maxlen=40)
        self.spec=json.loads((ROOT/'configs/scenes'/config).read_text())
        self.spawn=self.spec['route'][0]
        video=ROOT/'artifacts/runs/supermimic/live'/f'{robot}-{scene}-{time.time_ns()}.mp4'
        video.parent.mkdir(exist_ok=True,parents=True)
        self.server=OperationsServer(nav,nav.meta['scene_sha256'],port,video,'video/mp4; codecs="hvc1.1.6.L120.B0"')
        self.server.active_scene,self.server.scenes=scene,list(SCENES)
        self.server.prepared_options=variants(scene)
        self.server.start()
        self.next_config=None
        self.mode,self.paused,self.player,self.route='auto',True,None,[]
        self.tick,self.keys,self.lease=0,(0.,0.),0.
        self.speed,self.gait=.65,'walk'
        self.vx=self.vy=self.facing=0.
        self.last_warning=False
        self.chunk_index=0
        self.chunk_remaining=0
        self.chunk_trace=None
        self.trace_root=video.with_suffix('')
        self.trace_root.mkdir(parents=True,exist_ok=True)
        self.view=self.planner=None
        try:
            xml=(SHARED/'joystick/assets/sih2300/robot.xml' if robot=='sih2300' else SHARED/'supermimic/assets/robots'/robot/'robot.xml')
            ck=tracker_checkpoint(robot, SHARED)
            contact=ROOT.parent/f'{scene}-full-contacts-v2.xml'
            if hashlib.sha256(contact.read_bytes()).hexdigest()!=CONTACT_SHA[scene]:
                raise ValueError('SuperMimic scene collision digest mismatch')
            os.environ['OMEGA_SUPERMIMIC_CONTACTS']=str(contact)
            self.planner=JoystickClient(robot,xml)
            self.tracker=TrackingSession(robot,ck,xml)
            self.heading=HeadingFrame([0,0])
            self.tracker.append(self.heading.place(self.planner.generate(0,0),0.))
            self.view=LiveView(self.tracker,xml,robot,scene,nav.meta['scene_sha256'],self.spawn,video,self.frame,avatar=avatar,armed=armed)
            self.view.show(self.tracker,0,False)
            self.state=self.tracker.env.state()
            self.proprio_history=ProprioHistory(self.state,self.tick)
            self.publish()
        except BaseException:
            self.close()
            raise

    def frame(self,jpeg,*_):
        if self.view:
            self.server.ego_jpeg=self.view.renderer.ego_jpeg
        self.server.publish_frame(jpeg)

    def event(self,kind,**fields):
        self.server.event(dict(kind=kind,sim_time=self.tick/50,**fields))

    def physical(self):
        s=self.state
        pos=s[:,:3]+self.view.offset
        q=s[:,[4,5,6,3]]
        n=len(self.tracker.names[0])
        return RobotBatchView(pos[:,None],q[:,None],s[:,None,7:10],s[:,None,10:13],
                              s[:,13:13+n],s[:,13+n:13+2*n],self.view.offset[2:3])

    def publish(self):
        v=self.physical()
        q=v.root_rot[0].tolist()
        yaw=math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
        r=self.view.renderer
        task=self.player.status() if self.mode=='auto' and self.player else dict(mode='idle',phase='idle')
        self.server.publish(dict(session_id=self.server.session_id,physics_backend="mujoco_warp",ready=True,paused=self.paused,
            control_mode=self.mode,wbc_policy='supermimic',avatar=self.avatar,armed=self.armed,
            tracker_release=self.tracker_release_id,tracker_iterations=self.tracker.iterations,tracker_sha256=self.tracker.checkpoint_sha256,
            policy_quality_note=('H2 Joystick 候选权重：已验证物理步进，零速停稳仍待验收。' if self.robot=='h2' else ''),wbc_choices=['supermimic','sonic'],
            sim_time=self.tick/50,physics_step=self.tick*4,rates=dict(physics_hz=200,wbc_hz=50,player_hz=1),
            performance=(dict(wall_fps=(self.samples[-1][1]-self.samples[0][1])/(self.samples[-1][0]-self.samples[0][0]),
                              rtf=(self.samples[-1][2]-self.samples[0][2])/50/(self.samples[-1][0]-self.samples[0][0]))
                         if len(self.samples)>1 else None),
            agents=[dict(id=0,position=v.root_pos[0].tolist(),heading_deg=math.degrees(yaw),speed_m_s=float(v.root_vel[0,:2].norm()))],
            teams=[dict(id=0,agents=[0],reference=self.route,task=task,locomotion=dict(mode=self.gait,speed=self.speed))],
            viewer=dict(camera=r.camera_mode,agent=0,yaw=r.orbit_yaw,pitch=r.orbit_pitch,distance=r.orbit_distance)))

    def select_command(self):
        if self.mode=='manual':
            forward,turn=self.keys if time.monotonic()<self.lease else (0.,0.)
            q=self.state[0,3:7].detach().cpu().numpy()
            heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
            self.facing=heading+.45*turn
            self.vx,self.vy=self.speed*forward,0.
        elif self.player and self.tick>=self.next_player:
            v=self.physical()
            cmd=self.player.act(v,None,self.tick/50)
            self.next_player=self.tick+50
            self.facing=math.atan2(float(cmd.facing[0,1]),float(cmd.facing[0,0]))
            speed=max(0.,float(cmd.speed[0]))
            dx,dy=(cmd.movement[0,:2]*speed).tolist()
            c,s=math.cos(self.facing),math.sin(self.facing)
            self.vx,self.vy=c*dx+s*dy,-s*dx+c*dy

    def ground(self, point):
        meta=self.server.navigation.meta
        x,y=np.floor((np.array(point)-meta['bounds'][:2])/meta['resolution']).astype(int)
        if not (0<=x<self.support.shape[1] and 0<=y<self.support.shape[0] and np.isfinite(self.support[y,x])):
            raise ValueError('手绘轨迹超出场景地面')
        return float(self.support[y,x])

    def command(self,c):
        kind=c['kind']
        if kind=='control_mode':
            if c['mode'] not in ('auto','manual'): raise ValueError('Unknown mode')
            self.mode=c['mode'];self.keys=(0.,0.);self.vx=self.vy=0.
            self.paused=self.mode=='auto'
            self.next_player=self.tick
        elif kind=='keyboard':
            if self.mode!='manual': raise ValueError('Manual mode required')
            keys=c['forward'],c['turn']
            if any(v not in (-1,0,1) for v in keys): raise ValueError('Invalid keyboard axis')
            self.keys,self.lease=keys,time.monotonic()+.4
        elif kind=='drawn_path':
            if self.mode!='auto': raise ValueError('Auto mode required')
            plan=drawn_path([self.physical().root_pos[0,:2].tolist(),*c['keypoints'][1:]])
            from omega_playground.map.preview import inspect_path
            validation=inspect_path(self.server.preview_navigation(),plan['route'],plan['clearance_m'])
            if not validation['valid']:raise ValueError(validation['message'])
            for p in plan['route']:self.ground(p)
            spec={**self.spec,**plan,'task_mode':'drawn_path','reference':{**self.spec['reference'],'preserve_path':True}}
            self.player=TensorWebPlayer(spec,'cuda:0')
            self.player.reset(torch.tensor([0],device='cuda:0'),self.physical())
            self.player.configure_locomotion(self.gait,self.speed,self.tick/50)
            self.route,self.next_player,self.paused=plan['route'],self.tick,False
        elif kind=='pause': self.paused=True;self.keys=(0.,0.);self.samples.clear()
        elif kind=='reset':
            mode=c.get('recovery_mode','in_place')
            if mode not in ('in_place','task_start'):raise ValueError('Unknown recovery mode')
            xy=(self.route[0] if self.route else self.spawn) if mode=='task_start' else self.physical().root_pos[0,:2].tolist()
            z=self.ground(xy)-float(self.view.offset[2])
            self.finish_chunk('reset')
            self.heading=HeadingFrame(np.asarray(xy)-self.spawn, self.facing)
            ref=self.heading.place(self.planner.generate(0,0,reset=True),self.facing)
            self.height_offset=z
            ref['qpos'][:,2]+=z
            self.tracker.queue=self.tracker.history=None
            self.tracker.append(ref)
            self.state=self.tracker.env.state()
            self.proprio_history=ProprioHistory(self.state,self.tick)
            self.chunk_remaining=0
            if self.player:
                ids=torch.tensor([0],device='cuda:0')
                if mode=='task_start': self.player.reset(ids,self.physical())
                else: self.player.recover_task(ids,self.physical(),self.tick/50)
            self.keys=(0.,0.);self.vx=self.vy=0.;self.next_player=self.tick
            self.view.show(self.tracker,self.tick,False)
            self.event('reset_audit',recovery_mode=mode)
        elif kind=='resume': self.paused=False
        elif kind=='locomotion':
            if c['mode'] not in ('walk','run') or not .1<=float(c['speed'])<=2: raise ValueError('Speed outside native range')
            if self.player:self.player.configure_locomotion(c['mode'],float(c['speed']),self.tick/50)
            self.speed,self.gait=float(c['speed']),c['mode']
        elif kind=='viewer':
            r=self.view.renderer
            if c['camera'] not in ('chase','qc','free','top','squad'):raise ValueError('Unknown camera')
            r.camera_mode=c['camera']
            for key in ('yaw','pitch','distance'):
                if key in c:setattr(r,'orbit_'+key,float(c[key]))
            self.view.show(self.tracker,self.tick,False)
        elif kind in ('configure_robot','select_scene'):
            avatar,scene=c.get('avatar',self.avatar),c.get('scene',self.scene)
            armed=c.get('armed',self.armed)
            if scene not in SCENES or type(armed)!=bool or dict(avatar=avatar,armed=armed) not in variants(scene): raise ValueError('Unsupported SuperMimic configuration')
            self.next_config=avatar,scene,armed
        else: raise ValueError('Unsupported live command: '+kind)

    def plan_chunk(self):
        """No physics or next-plan prefetch occurs while this snapshot is inferred."""
        if self.chunk_remaining or self.chunk_trace is not None:
            raise RuntimeError('Current chunk must finish before planning')
        snapshot=self.view.snapshot(self.tracker,self.tick)
        observation=self.proprio_history.snapshot()
        observation['joint_names']=list(self.tracker.names[0])
        q=self.state[0,3:7].detach().cpu().numpy()
        actual_heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        observation['command_heading_delta']=math.atan2(
            math.sin(self.facing-actual_heading),math.cos(self.facing-actual_heading))
        ref=self.planner.generate(self.vx,self.vy,observation=observation)
        if int(ref.get('observation_tick',-1)) != self.tick or not ref.get('includes_current_anchor'):
            raise ValueError('Planner result does not match decision physics timestamp')
        local_ref=ref
        ref=place_from_proprio(ref,self.state[0].detach().cpu().numpy())
        steps=len(ref['qpos'])-1
        if steps != 2*int(ref['planner_future_frames']) or steps <= 0:
            raise ValueError('Planner frames and control intervals disagree')
        # Preserve Tracker observation history and physics, replace only exhausted reference.
        self.tracker.queue=None
        self.tracker.append(ref)
        self.chunk_remaining=steps
        folder=self.trace_root/f'chunk-{self.chunk_index:06d}'
        folder.mkdir(exist_ok=False)
        (folder/'ego.jpg').write_bytes(snapshot['ego_jpeg'])
        (folder/'main.jpg').write_bytes(snapshot['main_jpeg'])
        np.savez_compressed(folder/'input.npz',**observation,
                            camera_physics_step=snapshot['physics_step'],
                            camera_timestamp=snapshot['timestamp'])
        ref['timestamps']=self.tick/50+np.arange(len(ref['qpos']))/50
        np.savez_compressed(folder/'reference.npz',**ref)
        np.savez_compressed(folder/'reference_local.npz',**local_ref)
        scene_from_tracker=np.eye(4)
        scene_from_tracker[:3,3]=self.view.offset.detach().cpu().numpy()
        metadata=dict(schema='omega.playground.executed-chunk/1',chunk_index=self.chunk_index,
                      start_tick=self.tick,physics_start_step=self.tick*4,
                      timestamp=self.tick/50,planned_control_steps=steps,
                      planner_future_frames=int(ref['planner_future_frames']),
                      embodiment=self.robot,joint_names=list(self.tracker.names[0]),
                      scene_from_tracker=scene_from_tracker.tolist(),
                      tracker_sha256=self.tracker.checkpoint_sha256,
                      camera_timestamp=snapshot['timestamp'],context_source='executed_proprio',
                      command=dict(vx=self.vx,vy=self.vy,facing=self.facing),
                      status='executing',lookahead_policy='terminal_hold_zero_velocity',
                      video_conditioned=False)
        (folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        self.chunk_trace=dict(folder=folder,metadata=metadata,
                              actual=[self.state[0].detach().cpu().numpy().copy()],
                              policy_action=[],pd_target=[],lookahead_valid=[])
        self.chunk_index+=1
        self.event('chunk_started',chunk_index=self.chunk_index-1,control_steps=steps)

    def finish_chunk(self, status):
        trace=self.chunk_trace
        if trace is None:
            return
        metadata=trace['metadata']
        start=metadata['start_tick']
        count=len(trace['policy_action'])
        np.savez_compressed(trace['folder']/'execution.npz',
                            actual=np.asarray(trace['actual']),
                            policy_action=np.asarray(trace['policy_action']),
                            pd_target=np.asarray(trace['pd_target']),
                            lookahead_valid=np.asarray(trace['lookahead_valid']),
                            state_ticks=np.arange(start,start+count+1),
                            state_physics_steps=np.arange(start,start+count+1)*4,
                            state_timestamps=np.arange(start,start+count+1)/50)
        metadata.update(status=status,executed_control_steps=count,
                        end_tick=self.tick,physics_end_step=self.tick*4,end_timestamp=self.tick/50)
        (trace['folder']/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        self.chunk_trace=None

    def run(self):
        deadline=time.monotonic()
        while not self.next_config:
            try:c=self.server.commands.get(timeout=.04 if self.paused else 0)
            except queue.Empty:c=None
            if c:
                try:
                    self.command(c);self.publish()
                    if c['kind']!='keyboard': self.event('ack',id=c['id'],command=c['kind'],accepted=True)
                except (ValueError,KeyError,TypeError) as exc:
                    self.event('ack',id=c['id'],command=c['kind'],accepted=False,error=str(exc))
            if self.paused or self.next_config:
                deadline=time.monotonic();continue
            self.select_command()
            if self.chunk_remaining == 0:
                self.plan_chunk()
                # Inference/render wall time must not cause catch-up physics steps.
                deadline=time.monotonic()
            self.state,done=self.tracker.step(measured_heading=True,streaming=True)
            self.tick+=1
            self.chunk_remaining-=1
            self.proprio_history.add(self.state,self.tick)
            self.chunk_trace['actual'].append(self.state[0].detach().cpu().numpy().copy())
            self.chunk_trace['policy_action'].append(self.tracker.last_action[0].cpu().numpy())
            self.chunk_trace['pd_target'].append(self.tracker.last_pd_target[0].cpu().numpy())
            self.chunk_trace['lookahead_valid'].append(self.tracker.lookahead_valid.cpu().numpy())
            if self.chunk_remaining == 0:
                self.finish_chunk('complete')
            failed=bool(done.item())
            if failed!=self.last_warning:
                self.event('warning' if failed else 'warning_cleared',code='tracking_divergence')
                self.last_warning=failed
            if self.tick%2==0:self.view.show(self.tracker,self.tick)
            if self.tick%5==0:
                self.samples.append((time.monotonic(),self.view.renderer.frames,self.tick))
                self.publish()
            deadline+=.02
            time.sleep(max(0,deadline-time.monotonic()))
            deadline=max(deadline,time.monotonic()-.02)
        return self.next_config

    def close(self):
        self.finish_chunk('closed')
        try:
            if self.view:self.view.renderer.close()
        finally:
            if self.planner:self.planner.close()
            self.server.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--robot',choices=AVATARS,default='g1')
    cli.add_argument('--armed',action='store_true')
    cli.add_argument('--scene',choices=SCENES,default='campus')
    cli.add_argument('--port',type=int,default=18783)
    args=cli.parse_args()
    import sys
    import traceback

    def restart(config):
        avatar, scene, armed = config
        os.execv(sys.executable,[sys.executable,__file__,'--robot',avatar,'--scene',scene,
                                 '--port',str(args.port),*(['--armed'] if armed else [])])

    previous = os.environ.pop('OMEGA_SM_PREVIOUS_CONFIG', None)
    try:
        world = World(args.robot,args.scene,args.port,args.armed)
    except Exception as exc:
        if previous is None:
            raise
        traceback.print_exc()
        os.environ['OMEGA_SM_REBUILD_ERROR'] = f'{args.robot}/{args.scene}: {exc}'
        restart(json.loads(previous))

    error = os.environ.pop('OMEGA_SM_REBUILD_ERROR', None)
    if error:
        world.event('warning',code='configuration_restored',error=error)
    recovery = os.environ.pop('OMEGA_SM_RECOVERY', None)
    if recovery:
        world.event('warning', **json.loads(recovery))
    from supermimic.simulation_health import NonfiniteSimulationState
    try:
        next_config = world.run()
    except (NonfiniteSimulationState, JoystickTimeout) as exc:
        traceback.print_exc()
        next_config = (args.robot, args.scene, args.armed)
        reason = 'Joystick 响应超时' if isinstance(exc, JoystickTimeout) else '物理状态异常'
        os.environ['OMEGA_SM_RECOVERY'] = json.dumps({
            'code': 'joystick_recovered' if isinstance(exc, JoystickTimeout) else 'physics_recovered',
            'error': f'{args.robot}/{args.scene} 第 {world.tick + 1} 步{reason}，仿真已重置，暂停待命。',
        })
    finally:
        world.close()
    if next_config != (args.robot, args.scene, args.armed):
        os.environ['OMEGA_SM_PREVIOUS_CONFIG'] = json.dumps([args.robot,args.scene,args.armed])
    restart(next_config)
