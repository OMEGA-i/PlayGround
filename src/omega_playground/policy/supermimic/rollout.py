"""Run one upstream Tracker and GPU MuJoCo Warp environment on a released reference."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class TrackingSession:
    """One native MJWarp world; appending a planner release preserves physical state and actor history."""
    @torch.inference_mode()
    def __init__(self, robot, checkpoint, xml, *, num_envs=1):
        import mujoco
        from supermimic.core import Actor, validate_checkpoint
        from supermimic.kinematics import Kinematics
        from supermimic.mjwarp import MJWarp
        import supermimic.mjwarp as tracker_runtime
        from .provenance import repository
        self.tracker_code = repository(Path(tracker_runtime.__file__).resolve().parent.parent)

        device = 'cuda:0'
        from omega_playground.policy.supermimic.release import tracker_release
        entry = tracker_release()[robot]
        with Path(checkpoint).open('rb') as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != entry['sha256']:
                raise ValueError('Tracker checkpoint differs from accepted delivery')
        ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        validate_checkpoint(ck)
        if (ck['iteration'] + 1 != entry['completed_iterations']
                or list(ck['joint_names']) != entry['joint_names']
                or list(ck['body_names']) != entry['body_names']):
            raise ValueError('Tracker iteration or named schema differs from pinned release')
        if Path(checkpoint).name != f"ckpt_{ck['iteration']+1}.pt":
            raise ValueError("Checkpoint filename and completed iterations disagree")
        cfg = ck['config']
        ids = {'sih2300':'sih2300','g1':'unitree_g1_29dof','a3':'agibot_a3','h2':'unitree_h2'}
        if (cfg['robot'] != ids[robot] or ck['training_contract']['backend'] != 'mujoco-warp'
                or hashlib.sha256(Path(xml).read_bytes()).hexdigest() != ck['training_contract']['asset_sha256']):
            raise ValueError('SuperMimic Tracker asset/backend mismatch')
        os.environ[cfg['asset_env']] = str(Path(xml).resolve())
        if robot == 'sih2300':
            os.environ.setdefault(cfg['embodiment_contract']['yaml_env'], str(Path(__file__).resolve().parents[4]/'configs/supermimic/sih2300.yaml'))
        names = (ck['joint_names'], ck['body_names'])
        fk = Kinematics(mujoco.MjModel.from_xml_path(str(xml)), *names).to(device)
        scene_xml = os.environ.get("OMEGA_SUPERMIMIC_CONTACTS")
        env = MJWarp(cfg, names, num_envs, device, scene_xml=scene_xml) if scene_xml else MJWarp(cfg, names, num_envs, device)
        actor = Actor(env.layout, ck['args']['small']).to(device).eval()
        actor.load_state_dict(ck['actor'], strict=True)
        self.robot, self.names, self.device = robot, names, device
        self.fk, self.env, self.actor = fk, env, actor
        self.checkpoint_sha256 = entry['sha256']
        checkpoint_path = Path(checkpoint).resolve()
        checkpoint_stat = checkpoint_path.stat()
        self.checkpoint_identity = dict(path=str(checkpoint_path), sha256=entry['sha256'],
                                        size_bytes=checkpoint_stat.st_size)
        self.checkpoint_mtime_ns = checkpoint_stat.st_mtime_ns
        self.iterations, self.scene_xml = ck['iteration']+1, scene_xml
        self.queue = self.history = None

    @torch.inference_mode()
    def append(self, ref):
        names, robot, device = self.names, self.robot, self.device
        joint_names = list(ref['joint_names'])
        if (str(ref['embodiment']) != robot or int(ref['fps']) != 50
                or len(set(joint_names)) != len(joint_names) or set(joint_names) != set(names[0])):
            raise ValueError('Tracker reference schema mismatch')
        order = [joint_names.index(n) for n in names[0]]
        q = ref['qpos']
        root = np.c_[q[:,:7], ref['root_linear_velocity'], ref['root_angular_velocity_world']]
        tensors = [torch.as_tensor(x,device=device,dtype=torch.float32) for x in
                   (root, q[:,7:][:,order], ref['joint_velocity'][:,order])]
        clip = torch.cat([*tensors, self.fk(*tensors).flatten(1)], dim=1)
        self.queue = clip if self.queue is None else torch.cat((self.queue, clip))
        if self.history is None:
            self.env.reset(torch.zeros(1, device=device, dtype=torch.long), clip[:1])
            prev = clip.new_zeros((1,len(names[0])))
            first = self.env.layout.obs(self.env.state(),prev,self.env.default,False,goal=clip[:1])
            self.history = first[:,None].repeat(1,10,1)
        return clip

    def physical_pose(self):
        xyz = self.env.view('xpos')[0,1:]
        wxyz = self.env.view('xquat')[0,1:]
        return torch.cat((xyz,wxyz[:,[1,2,3,0]]),-1)

    @torch.inference_mode()
    def step(self, *, measured_heading=False, streaming=False):
        if streaming:
            from .chunk import streaming_window
            window,self.lookahead_valid=streaming_window(self.queue,len(self.names[0]))
        else:
            if self.queue is None or len(self.queue) < 11:
                raise RuntimeError('Tracker needs eleven reference frames')
            window=self.queue[:11]
            self.lookahead_valid=torch.ones(11,dtype=torch.bool,device=window.device)
        if measured_heading:
            # A-course goals express ten future orientations relative to frame zero.
            # Re-anchor only its yaw to the measured root so persistent heading error
            # remains visible to the actor after the reference stops rotating.
            from supermimic.core import yawq, qconj, qmul
            window=window.clone()
            actual=self.env.view('xquat')[0,self.env.root_bid]
            target=window[:1,3:7]
            window[:1,3:7]=qmul(yawq(actual[None]),qmul(qconj(yawq(target)),target))
        action = self.actor(self.history,window[None])
        state = self.env.step(action)
        self.last_action=action.detach().clone()
        self.last_pd_target=self.env.view('ctrl').detach().clone()
        self.queue = self.queue[1:]
        target = self.queue[:1]
        failed = self.env.layout.done(state,target)
        obs = self.env.layout.obs(state,action,self.env.default,False,goal=target)
        self.history = torch.cat((self.history[:,1:],obs[:,None]),1)
        return state, failed


@torch.inference_mode()
def run(robot, checkpoint, xml, reference, output, max_steps=None):
    session = TrackingSession(robot,checkpoint,xml)
    env, names = session.env, session.names
    with np.load(reference, allow_pickle=False) as ref:
        clip = session.append(ref)
    poses, actual = [session.physical_pose()], [env.state().clone()]
    limit = len(clip)-10 if max_steps is None else min(max_steps,len(clip)-10)
    failed = False
    for _ in range(limit):
        state, done = session.step()
        actual.append(state.clone())
        poses.append(session.physical_pose())
        failed = bool(done.item())
        if failed:
            break
    actual = torch.cat(actual).cpu().numpy()
    output = Path(output)
    np.savez_compressed(output/'body_states.npz', body_q=torch.stack(poses).cpu().numpy(),
                        sim_time=np.arange(len(poses)) / 50,
                        body_names=[env.cpu_model.body(i).name for i in range(1,env.cpu_model.nbody)])
    np.savez_compressed(output/'rollout.npz', actual=actual, reference=clip[:len(actual)].cpu().numpy(),
                        joint_names=names[0],body_names=names[1],fps=50)
    result = dict(workflow='SuperMimic',robot=robot,checkpoint=str(checkpoint),completed_iterations=session.iterations,
                  checkpoint_sha256=session.checkpoint_sha256,physics='mujoco-warp',collision_layer=session.scene_xml,device=session.device,tracker_hz=50,steps=len(actual)-1,failed=failed)
    (output/'trajectory.json').write_text(json.dumps(dict(actual=actual[:,:2].tolist(),
        reference=clip[:len(actual),:2].cpu().tolist()))+'\n')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
