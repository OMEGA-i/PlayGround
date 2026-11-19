"""Prepare a checksummed Campus+robot visual model offline; no physics stepping."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from omega_playground.policy.package import load
from omega_playground.scene.package import load as load_scene
from omega_playground.viewer.usd_scene import add_usd_scene


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    robot = parser.add_mutually_exclusive_group(required=True)
    robot.add_argument('--package', type=Path)
    robot.add_argument('--supermimic-xml', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--armed', action='store_true')
    parser.add_argument('--agents', type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    scene = load_scene(args.scene)
    package = load(args.package) if args.package else None
    if not package and args.agents != 1:
        parser.error('SuperMimic visual bundles use one native robot')
    xml = package.asset_path if package else args.supermimic_xml
    args.output.mkdir(parents=True, exist_ok=False)
    spec = mujoco.MjSpec.from_file(str(xml))
    # Visual-only artifact: no contacts are computed here. MJWarp rejects
    # native CCD with nonzero robot margins even for pose-only uploads.
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    # Robot packages may contain a demonstration floor; the Campus owns all static visuals.
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    if package:
        body_names = list(package.body_names)
    else:
        original = mujoco.MjModel.from_xml_path(str(xml))
        body_names = [original.body(i).name for i in range(1, original.nbody)]
    physical_names=body_names.copy()
    for i in range(1, args.agents):
        child = mujoco.MjSpec.from_file(str(xml))
        for item in [*child.worldbody.geoms, *child.worldbody.lights, *child.worldbody.cameras]:
            child.delete(item)
        for texture in list(child.textures):
            if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                child.delete(texture)
        prefix = f'agent_{i:03d}_'
        spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame())
        body_names.extend(prefix+name for name in package.body_names)
    spec.worldbody.add_camera(name='qc', pos=[0, -4, 1.25], fovy=np.degrees(2*np.arctan(1.25/4)))
    report = add_usd_scene(spec, scene.entry, args.output/'textures')
    from omega_playground.viewer.usd_lighting import add_usd_lighting
    report['lighting'] = add_usd_lighting(spec, scene.entry)
    report['visual_gaps'].extend(report['lighting']['visual_gaps'])
    (args.output/'conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'instance_transforms'}), flush=True)
    armed = None
    if args.armed:
        from omega_playground.viewer.mjwarp_armed import add_equipment
        if package:armed = add_equipment(spec, package, args.agents)
        else:
            from omega_playground.policy.supermimic.native_armed import bake
            armed=bake(spec,original)
        body_names.extend(armed['body_names'])
    model = spec.compile()
    if args.armed and package:
        from omega_playground.viewer.mjwarp_armed import apply_prepared_hands
        apply_prepared_hands(model, package, args.agents)
    mujoco.mj_saveModel(model, str(args.output/'visual.mjb'))
    report.update(schema='omega.mjwarp-visual/1', armed=armed, scene_sha256=scene.scene_sha256,
                  mujoco_version=mujoco.__version__, body_names=body_names, num_agents=args.agents,
                  policy_body_names=list(package.body_names) if package else physical_names,
                  policy_sha256=package.sha256sum() if package else None, entry='visual.mjb')
    if not package:
        report.update(schema='omega.supermimic-visual/1', xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest())
    (args.output/'asset.json').write_text(json.dumps(report, indent=2)+'\n')
    paths = sorted(p for p in args.output.rglob('*') if p.is_file())
    sums = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output/'SHA256SUMS.json').write_text(json.dumps(sums, indent=2)+'\n')


if __name__ == '__main__':
    main()
