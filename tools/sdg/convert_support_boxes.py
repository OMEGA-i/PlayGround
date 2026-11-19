"""Build an audited, geometrically equivalent native-box support collision candidate."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--collision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geoms', nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tree = ET.parse(args.collision)
    changes = []
    for name in args.geoms:
        geom = model.geom(name).id
        if model.geom_bodyid[geom] != 0 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f'{name}: expected a static world mesh')
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        if count != 8:
            raise ValueError(f'{name}: not an eight-vertex box')
        vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
        low, high = vertices.min(0), vertices.max(0)
        expected = np.array(list(itertools.product(*zip(low, high))))
        distances = np.linalg.norm(vertices[:,None]-expected[None], axis=-1)
        error = max(float(distances.min(0).max()), float(distances.min(1).max()))
        if error > 1e-7 or np.any(high-low <= 0):
            raise ValueError(f'{name}: mesh is not an axis-aligned solid box ({error=})')
        elements = [g for g in tree.findall('worldbody/geom') if g.get('name') == name]
        if len(elements) != 1 or elements[0].get('type') != 'mesh':
            raise ValueError(f'{name}: collision source mismatch')
        element = elements[0]
        before = dict(element.attrib)
        for key in ('mesh', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'):
            element.attrib.pop(key, None)
        element.set('type', 'box')
        element.set('pos', ' '.join(format(x, '.17g') for x in (low+high)/2))
        element.set('size', ' '.join(format(x, '.17g') for x in (high-low)/2))
        changes.append(dict(name=name, before=before, after=dict(element.attrib),
                            corner_error_m=error, bounds=[low.tolist(), high.tolist()]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding='unicode')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(source_sha256=digest(args.collision), candidate_sha256=digest(args.output),
                  changed_geoms=changes, claim='Same convex support volume; native collision algorithm differs')
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
