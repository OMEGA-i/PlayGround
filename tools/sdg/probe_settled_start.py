"""Diagnose task startup after an explicitly saved physical standing initialization.

Initialization is a separate phase, never included in episode duration or quotas.
The ordinary task spawn/alignment, contact, trajectory and sensor gates still apply.
"""
import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from run_episode import record_task, sha
from omega_playground.policy.supermimic.rollout import TrackingSession
from omega_playground.recording.execution import Cancellation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error('Initialization must fit inside the standing reference (1..100 controls)')
    task = json.loads(args.task.read_text())
    task['initialization_control_steps'] = args.steps

    class SettledSession(TrackingSession):
        def append(self, reference):
            first = self.history is None
            clip = super().append(reference)
            if not first:
                return clip
            destination = args.output/'initialization'
            destination.mkdir()
            np.savez_compressed(destination/'reference.npz', **reference)
            mujoco.mj_saveModel(self.env.cpu_model, str(destination/'physics.mjb'), None)
            positions, velocities, bodies = [], [], []
            def capture():
                positions.append(self.env.view('qpos')[0].cpu().numpy().copy())
                velocities.append(self.env.view('qvel')[0].cpu().numpy().copy())
                bodies.append(self.physical_pose().cpu().numpy().copy())
            capture()
            try:
                for _ in range(args.steps):
                    self.step(measured_heading=task.get('measured_heading', True), streaming=True)
                    capture()
            finally:
                ticks = np.arange(len(positions))
                np.savez_compressed(destination/'physics_states.npz', qpos=positions,
                    qvel=velocities, physics_step=ticks*4)
                np.savez_compressed(destination/'body_states.npz', body_q=bodies,
                    control_tick=ticks, physics_step=ticks*4, timestamp=ticks/50,
                    body_names=[self.env.cpu_model.body(i).name
                                for i in range(1, self.env.cpu_model.nbody)])
                report = dict(phase='physical_standing_initialization', excluded_from_dataset=True,
                    recording_body_hz=50, control_steps=len(positions)-1,
                    embodiment=dict(id=task['embodiment']),
                    scene=dict(id=task['scene'], physical_collision_sha256=sha(Path(os.environ['OMEGA_SUPERMIMIC_CONTACTS']))))
                (destination/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
            return clip

    with Cancellation().installed() as cancellation:
        recorder = record_task(task, args.output, cancellation, session_factory=SettledSession)
        try:
            request = next(recorder)
            while True:
                if isinstance(request, tuple):
                    _, planner, vx, vy, observation = request
                    result = planner.generate(vx, vy, observation=observation)
                else:
                    result = request.step(measured_heading=task.get('measured_heading', True), streaming=True)
                request = recorder.send(result)
        except StopIteration as finished:
            print(json.dumps(finished.value['task']), flush=True)
        finally:
            recorder.close()


if __name__ == '__main__':
    main()
