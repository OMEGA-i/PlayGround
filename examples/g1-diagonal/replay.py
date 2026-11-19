"""Project a Joystick single-window reference to native G1 joints, then run the pinned tracker."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Joystick custom_single.npz")
    parser.add_argument("--xml", required=True, type=Path, help="Matching G1 robot.xml")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Pinned G1 ckpt_45000.pt")
    parser.add_argument("--tracker-python", required=True, type=Path, help="CUDA/MJWarp tracker interpreter")
    parser.add_argument("--tracker-repo", required=True, type=Path, help="Directory containing the upstream supermimic package")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the physical rollout")
    args = parser.parse_args()
    required = [args.reference, args.xml, args.checkpoint, args.tracker_python, args.tracker_repo / "supermimic/__init__.py"]
    for path in required:
        if not path.is_file():
            parser.error(f"Missing local prerequisite: {path}")

    import numpy as np
    from omega_playground.policy.supermimic.reference import native_reference

    reference = native_reference(args.reference, args.xml, "g1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "reference50.npz", **reference)
    env = os.environ.copy()
    env.pop("OMEGA_SUPERMIMIC_CONTACTS", None)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(args.tracker_repo.resolve())])
    subprocess.run([
        str(args.tracker_python.absolute()), str(ROOT / "tools/supermimic/run_supermimic.py"),
        "track", "g1", str(output), str(args.xml.resolve()), str(args.checkpoint.resolve()),
    ], env=env, check=True)


if __name__ == "__main__":
    main()
