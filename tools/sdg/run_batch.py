"""Record independent task episodes with one native multi-world physics step.

Coroutine recorders reuse the single-task contract. They never duplicate a recorded
trajectory: each owns a distinct environment, actual feedback and planner context.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch
import mujoco

from run_episode import record_task, collision_asset
from omega_playground.map.navigation import CampusNavigation
from omega_playground.ops.scenes import SCENES
from omega_playground.policy.supermimic.batched import BatchedTrackingSession
from omega_playground.policy.supermimic.live_client import JoystickClient
from omega_playground.policy.supermimic.release import tracker_checkpoint
from omega_playground.recording.execution import Cancellation


class EnvironmentView:
    def __init__(self, shared, index):
        self.shared, self.index = shared, index
        self.cpu_model = shared.env.cpu_model

    def state(self):
        return self.shared.env.state()[self.index:self.index+1]

    def view(self, name):
        return self.shared.env.view(name)[self.index:self.index+1]


class World:
    def __init__(self, shared, index):
        self.shared, self.index = shared, index
        self.env = EnvironmentView(shared, index)
        self.names, self.device = shared.names, shared.device
        self.checkpoint_sha256 = shared.checkpoint_sha256
        self.tracker_code = shared.tracker_code
        self.checkpoint_identity = shared.checkpoint_identity
        self.checkpoint_mtime_ns = shared.checkpoint_mtime_ns
        self.model_path = shared.model_path
        self.queue = None
        self.initialized = False

    def append(self, reference):
        result = self.shared.append_one(self.index, reference, reset=not self.initialized)
        self.initialized = True
        return result

    def physical_pose(self):
        return self.shared.physical_poses()[self.index]

    @property
    def last_action(self):
        return self.shared.last_action[self.index:self.index+1]

    @property
    def last_pd_target(self):
        return self.shared.last_pd_target[self.index:self.index+1]

    @property
    def lookahead_valid(self):
        return self.shared.lookahead_valid[self.index]


class Planner:
    def __init__(self, client, index, seed):
        self.client, self.index, self.seed = client, index, seed
        self.provenance = dict(client.provenance, production_env_index=index, production_seed=seed,
            sampling_seed_policy='SHA256 of ordered per-request seeds; actual batch seed and IDs stored in each reference')
        self.generation = 0

    def request(self, vx, vy, observation=None):
        request = dict(env_id=self.index, velocity=[vx, vy], seed=self.seed+self.generation,
                       reset=self.generation == 0, observation=observation)
        self.generation += 1
        return request

    def close(self):
        pass  # Shared owner closes the resident worker after every recorder is sealed.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--waves', type=int, default=1)
    parser.add_argument('--target-seconds', type=float)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error('num-envs must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    task = json.loads(args.task.read_text())
    robot, scene = task['embodiment'], task['scene']
    root = Path(__file__).resolve().parents[2]
    shared = Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets'))
    xml = (shared/'joystick/assets/sih2300/robot.xml' if robot == 'sih2300'
           else shared/'supermimic/assets/robots'/robot/'robot.xml')
    contact = collision_asset(task, shared)
    os.environ['OMEGA_SUPERMIMIC_CONTACTS'] = str(contact)
    navigation = CampusNavigation(root/'artifacts/maps'/SCENES[scene][1])
    started = time.perf_counter()
    session = BatchedTrackingSession(robot, tracker_checkpoint(robot, shared), xml, args.num_envs)
    session.model_path = args.output/'physics.mjb'
    mujoco.mj_saveModel(session.env.cpu_model, str(session.model_path), None)
    client = JoystickClient(robot, xml)
    recorders, results, requests = {}, {}, {}
    with Cancellation().installed() as cancellation:
        try:
            loop_started = None
            successful_seconds = 0.
            empty_waves = 0.
            for wave in range(args.waves):
                before_wave = successful_seconds
                for index in range(args.num_envs):
                    episode = wave * args.num_envs + index
                    specific = dict(task, seed=int(task['seed'])+episode,
                                    speed=float(task.get('speed', .55))*(.9+.2*index/max(1,args.num_envs-1)))
                    world = World(session, index)
                    planner = Planner(client, index, specific['seed'])
                    recorder = record_task(specific, args.output/f'env-{episode:06d}', cancellation,
                        planner_factory=lambda *_, p=planner: p, session_factory=lambda *_, w=world: w,
                        navigation=navigation)
                    requests[index] = next(recorder)
                    recorders[index] = recorder
                while recorders:
                    pending = [i for i in recorders if isinstance(requests[i], tuple)]
                    if pending:
                        commands = []
                        for i in pending:
                            _, planner, vx, vy, observation = requests[i]
                            commands.append(planner.request(vx, vy, observation))
                        references = client.generate_many(commands)
                        for i, reference in zip(pending, references):
                            try:
                                requests[i] = recorders[i].send(reference)
                            except StopIteration as finished:
                                results[wave*args.num_envs+i] = finished.value
                                if finished.value['status'] == 'complete':
                                    successful_seconds += finished.value['control_steps']/50
                                del recorders[i]
                                del requests[i]
                        continue
                    if loop_started is None:
                        torch.cuda.synchronize()
                        initialized = time.perf_counter() - started
                        loop_started = time.perf_counter()
                    state, failed = session.step_batch(active=set(recorders),
                        measured_heading=task.get('measured_heading', True))
                    for index, recorder in list(recorders.items()):
                        try:
                            if bool(session.invalid[index]):
                                requests[index] = recorder.throw(FloatingPointError('Nonfinite physical transition; episode excluded'))
                            else:
                                requests[index] = recorder.send((state[index:index+1], failed[index:index+1]))
                        except StopIteration as finished:
                            results[wave*args.num_envs+index] = finished.value
                            if finished.value['status'] == 'complete':
                                successful_seconds += finished.value['control_steps']/50
                            del recorders[index]
                            del requests[index]
                torch.cuda.synchronize()
                if loop_started is None:
                    initialized = time.perf_counter() - started
                    loop_started = time.perf_counter()
                duration = time.perf_counter() - loop_started
                seconds = sum(r['control_steps']/50 for r in results.values())
                report = dict(num_envs=args.num_envs, robot=robot, scene=scene, completed_waves=wave+1,
                    initialization_seconds=initialized, recording_wall_seconds=duration,
                    recorded_simulated_seconds=seconds, successful_simulated_seconds=successful_seconds,
                    simulation_recording_throughput=seconds/duration,
                    end_to_end_capture_throughput=seconds/(time.perf_counter()-started),
                    successful_episodes=sum(r['status']=='complete' for r in results.values()),
                    scope='Closed-loop Joystick+batched Tracker+physics+recording; RGB-D/export pending')
                temporary = args.output/'batch.tmp'
                temporary.write_text(json.dumps(report, indent=2)+'\n')
                temporary.replace(args.output/'batch.json')
                print(json.dumps(report), flush=True)
                empty_waves = empty_waves + 1 if successful_seconds == before_wave else 0
                if empty_waves >= 3:
                    raise RuntimeError('Three consecutive waves produced no successful task; inspect recordings')
                if cancellation.reason or (args.target_seconds and successful_seconds >= args.target_seconds):
                    break
        finally:
            for recorder in recorders.values():
                recorder.close()
            client.close()


if __name__ == '__main__':
    main()
