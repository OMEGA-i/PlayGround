"""Consume sealed successful raw episodes and reuse one GPU scene at a time."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback

from render_episode import render, _RENDER_CACHE


def render_shard(uid, count):
    if count < 1:
        raise ValueError('Shard count must be positive')
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], 'big') % count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--robots', nargs='+', choices=('g1', 'a3', 'h2', 'sih2300'),
                        help='Disjoint robot partition when running multiple GPU render workers')
    parser.add_argument('--scenes', nargs='+', choices=('campus', 'obstacle'),
                        help='Optional disjoint scene partition within the selected robots')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--worker-id', default='worker')
    parser.add_argument('--execution-python', type=Path,
                        help='Physics runtime for contact QC before spending GPU render time')
    parser.add_argument('--quota-ledger', type=Path,
                        help='Skip cells already at the verified-seconds target in the production ledger')
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('shard-index must be in [0, num-shards)')
    if shutil.which('ffmpeg') is None:
        parser.error('ffmpeg must be available on PATH before starting the render queue')
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started, frames, episodes = time.perf_counter(), 0, 0
    rejected = set()
    try:
        while True:
            pending = []
            for path in args.source.glob('*/env-*/manifest.json'):
                try:
                    manifest = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if args.robots and manifest['embodiment']['id'] not in args.robots:
                    continue
                if args.scenes and manifest['scene']['id'] not in args.scenes:
                    continue
                if render_shard(manifest['sdg_task_uid'], args.num_shards) != args.shard_index:
                    continue
                if (manifest.get('status') == 'complete' and not manifest.get('execution_error')
                        and not manifest.get('recording_complete') and not (path.parent/'render_error.json').exists()):
                    pending.append((manifest['embodiment']['id'], manifest['scene']['id'], path.parent))
            for robot, scene, execution in sorted(pending):
                if stopping:
                    break
                if execution in rejected:
                    continue
                if args.quota_ledger and args.quota_ledger.exists():
                    ledger = json.loads(args.quota_ledger.read_text())
                    cell = ledger.get('cells', {}).get(f'{robot}-{scene}', {})
                    if cell.get('verified_seconds', 0) >= ledger['target_seconds_per_cell']:
                        continue
                if args.execution_python:
                    # Use the capture runtime: renderer MuJoCo may be a different version.
                    contact = execution/'contact_report.json'
                    if not contact.exists():
                        with (execution/'contact-preflight.log').open('w') as log:
                            result = subprocess.run([str(args.execution_python.absolute()),
                                str(Path(__file__).with_name('check_contacts.py')),
                                '--execution', str(execution)], stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode not in (0, 2):
                            print(json.dumps(dict(execution=str(execution), contact_error=result.returncode)), flush=True)
                            continue
                    if not json.loads(contact.read_text())['passed']:
                        rejected.add(execution)
                        print(json.dumps(dict(execution=str(execution), skipped='contact_qc_failed')), flush=True)
                        continue
                visual_suffix = f'supermimic-{robot}-{scene}'
                for key in list(_RENDER_CACHE):
                    if not key[0].endswith(visual_suffix):
                        _, viewer = _RENDER_CACHE.pop(key)
                        viewer.close()
                        del viewer
                        gc.collect()
                before = time.perf_counter()
                try:
                    render(execution, reuse=True, replay_stride=5)
                    report = json.loads((execution/'render_report.json').read_text())
                    frames += report['frames']
                    episodes += 1
                    report['wall_seconds'] = time.perf_counter()-before
                    report['rgbd_frames_per_wall_second'] = report['frames']/report['wall_seconds']
                    (execution/'render_report.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(json.dumps(dict(execution=str(execution), render=report)), flush=True)
                except Exception:
                    (execution/'render_error.json').write_text(json.dumps(dict(error=traceback.format_exc()))+'\n')
                    traceback.print_exc()
            status = dict(pid=os.getpid(), episodes=episodes, paired_frames=frames,
                          wall_seconds=time.perf_counter()-started, pending_at_scan=len(pending),
                          contact_rejected=len(rejected))
            temporary = args.source/f'render-{args.worker_id}.tmp'
            temporary.write_text(json.dumps(status, indent=2)+'\n')
            temporary.replace(args.source/f'render-{args.worker_id}.json')
            if args.once or stopping:
                break
            time.sleep(5)
    finally:
        for _, viewer in _RENDER_CACHE.values():
            viewer.close()


if __name__ == '__main__':
    main()
