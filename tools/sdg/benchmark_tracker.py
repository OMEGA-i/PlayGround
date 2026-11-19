"""Measure actual multi-world physical throughput on a recorded reference chunk.

This is a benchmark, never a source of production-hour credit.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from omega_playground.policy.supermimic.batched import BatchedTrackingSession
from omega_playground.policy.supermimic.release import tracker_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--num-envs', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.execution/'manifest.json').read_text())
    robot, scene = manifest['embodiment']['id'], manifest['scene']['id']
    shared = Path(os.environ.get('OMEGA_SHARED_ROOT', Path.home()/'omega-assets'))
    xml = (shared/'joystick/assets/sih2300/robot.xml' if robot == 'sih2300'
           else shared/'supermimic/assets/robots'/robot/'robot.xml')
    root = Path(__file__).resolve().parents[2]
    os.environ['OMEGA_SUPERMIMIC_CONTACTS'] = str(root.parent/f'{scene}-full-contacts-v2.xml')
    with np.load(args.execution/'chunk-000000/reference.npz', allow_pickle=False) as data:
        reference = {key: data[key].copy() for key in data.files}
    start = time.perf_counter()
    session = BatchedTrackingSession(robot, tracker_checkpoint(robot, shared), xml, args.num_envs)
    session.append_batch([reference for _ in range(args.num_envs)], reset=True)
    torch.cuda.synchronize()
    initialization = time.perf_counter() - start
    # Warm up actor kernels, then reset the actual worlds before timing.
    session.step_batch()
    session.append_batch([reference for _ in range(args.num_envs)], reset=True)
    torch.cuda.synchronize()
    steps = len(reference['qpos']) - 1
    start = time.perf_counter()
    warnings = torch.zeros(args.num_envs, dtype=torch.int64, device=session.device)
    for _ in range(steps):
        state, failed = session.step_batch()
        warnings += failed
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = dict(robot=robot, scene=scene, num_envs=args.num_envs,
                  control_steps_per_env=steps, initialization_seconds=initialization,
                  wall_seconds=elapsed, simulated_seconds=args.num_envs*steps/50,
                  simulated_seconds_per_wall_second=args.num_envs*steps/50/elapsed,
                  finite=bool(torch.isfinite(state).all()), tracking_warnings=warnings.cpu().tolist(),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  scope='Tracker+physics only; excludes Joystick, recording, rendering and export')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
