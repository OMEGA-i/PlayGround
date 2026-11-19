"""Compile a policy-native robot and authored static collisions for GPU simulation."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from bake_supermimic_contacts import bake

from omega_playground.policy.package import load
from omega_playground.scene.package import load as load_scene


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--scene", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--center", type=float, nargs=2, required=True)
    args = cli.parse_args()
    package, scene = load(args.package), load_scene(args.scene)
    spec = mujoco.MjSpec.from_file(str(package.asset_path))
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    # The training release specifies explicit torque PD with Euler integration.
    spec.option.timestep = 1 / package.physics_hz
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    with TemporaryDirectory() as tmp:
        contact = Path(tmp) / "collisions.xml"
        bake(args.scene, np.array(args.center), float("inf"), contact)
        child = mujoco.MjSpec.from_file(str(contact))
        # Bake normalizes support to zero. Restore original scene height and XY.
        # Preserve all authored geometry coordinates by undoing only the bake translation.
        for geom in child.worldbody.geoms:
            geom.pos[:2] += np.array(args.center)
            geom.pos[2] = 0.0
        spec.attach(child, prefix="scene_", frame=spec.worldbody.add_frame())
        model = spec.compile()
    args.output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(model, str(args.output / "physics.mjb"))
    meta = {
        "schema": "omega.mjwarp-physics/1",
        "policy_sha256": package.sha256sum(),
        "scene_sha256": scene.scene_sha256,
        "collision_sha256": scene.collision_sha256,
        "physics_hz": package.physics_hz,
        "joint_physics": package.manifest["pd"],
        "entry": "physics.mjb",
        "mujoco_version": mujoco.__version__,
    }
    (args.output / "asset.json").write_text(json.dumps(meta, indent=2) + "\n")
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}
    (args.output / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")


if __name__ == "__main__":
    main()
