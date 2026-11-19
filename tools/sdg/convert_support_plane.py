"""Create an audited diagnostic plane or finite flat heightfield support.

This is not a globally equivalent scene: it extends support outside the original
box. The audit retains that boundary for explicit trajectory-domain checks.
All non-support collision elements remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collision', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--representation', choices=('plane', 'heightfield'), default='plane')
    parser.add_argument('--grid-spacing-m', type=float)
    args = parser.parse_args()
    if args.grid_spacing_m is not None and not .1 <= args.grid_spacing_m <= 5:
        parser.error('grid-spacing-m must be between 0.1 and 5 metres')
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = json.loads(args.audit.read_text())
    if digest(args.collision) != audit['candidate_sha256']:
        raise ValueError('Box collision file does not match its geometric audit')
    records = {row['name']: row for row in audit['changed_geoms']}
    low, high = np.asarray(records[args.support]['bounds'])
    tree = ET.parse(args.collision)
    world = tree.find('worldbody')
    removed = []
    for name, record in records.items():
        if name == args.support:
            continue
        other_low, other_high = np.asarray(record['bounds'])
        if (np.any(other_low < low-1e-7) or np.any(other_high > high+1e-7)
                or abs(other_high[2]-high[2]) > 1e-7):
            raise ValueError('Secondary support is not redundant within the primary volume')
        removed.append(name)
    changed = {args.support, *removed}
    def retained():
        return hashlib.sha256(b''.join(ET.tostring(g) for g in world if g.get('name') not in changed)).hexdigest()
    unchanged = retained()
    for geom in list(world):
        if geom.get('name') in removed:
            world.remove(geom)
        elif geom.get('name') == args.support:
            if geom.get('type') != 'box':
                raise ValueError('Audited support must be a native box')
            if args.representation == 'plane':
                geom.set('type', 'plane')
                geom.set('pos', f'0 0 {high[2]:.17g}')
                geom.set('size', '0 0 .1')
            else:
                assets = tree.find('asset')
                if assets is None:
                    assets = ET.SubElement(tree.getroot(), 'asset')
                name = args.support + '_flat_heightfield'
                half = (high-low)/2
                rows = int(np.ceil((high[1]-low[1])/args.grid_spacing_m))+1 if args.grid_spacing_m else 2
                cols = int(np.ceil((high[0]-low[0])/args.grid_spacing_m))+1 if args.grid_spacing_m else 2
                ET.SubElement(assets, 'hfield', name=name, nrow=str(rows), ncol=str(cols),
                    size=f'{half[0]:.17g} {half[1]:.17g} .1 {high[2]-low[2]:.17g}', elevation=' '.join(['0']*(rows*cols)))
                geom.set('type', 'hfield')
                geom.set('hfield', name)
                geom.set('pos', f'{(low[0]+high[0])/2:.17g} {(low[1]+high[1])/2:.17g} {high[2]:.17g}')
                geom.attrib.pop('size', None)
    if retained() != unchanged:
        raise AssertionError('Non-support collision elements changed')
    tree.write(args.output, encoding='unicode')
    report = dict(source_sha256=digest(args.collision), candidate_sha256=digest(args.output),
        support_geom=args.support, removed_redundant_support_geoms=removed,
        finite_support_bounds_tracker_xyz=[low.tolist(), high.tolist()],
        non_support_elements_sha256=unchanged, excluded_from_production=True,
        representation=args.representation, grid_spacing_m=args.grid_spacing_m,
        globally_equivalent_support_union=args.representation == 'heightfield',
        limitation=('Analytic plane extends beyond original finite support; trajectory-domain audit required'
                    if args.representation == 'plane' else 'Finite flat heightfield preserves the audited support union; runtime validation pending'))
    args.output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
