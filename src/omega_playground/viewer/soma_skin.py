# Adapted from ProtoMotions ADD SOMA23 visual binding.
"""ASE-style authored-bind palette, driven by SOMA23 rotations.

Preserve the display hierarchy and bone lengths. Physical positions drive the root
and a common sole-height translation. These visuals do not certify full collision alignment.
"""
import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation


def align(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    angle = np.arccos(np.clip(a @ b, -1, 1))
    if np.linalg.norm(axis) < 1e-8:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()


class Soma23Palette:
    def __init__(self, data, body_names, robot_xml=None):
        self.armed_mounts = data.get('armed_mounts')
        self.names = data['names'].tolist()
        self.bind = data['bind'].astype(float)
        self.inverse = np.linalg.inv(self.bind)
        self.parents = data['parents'].astype(int)
        self.soma_names = data['soma_names'].tolist()
        self.rest = dict(zip(self.soma_names, data['soma_rest']))
        self.indices = [body_names.index(n) for n in self.soma_names]
        self.mapping = {bone: body for body, bone in zip(self.soma_names, data['mapping'].tolist()) if bone}
        self.calibration = self.bind[:, :3, :3].copy()
        self.root = self.names.index('pelvis')
        ends = {}
        for side in ['Left', 'Right']:
            ends.update({side+'Arm': side+'ForeArm', side+'ForeArm': side+'Hand',
                         side+'Leg': side+'Shin', side+'Shin': side+'Foot'})
        reverse = {v:k for k,v in self.mapping.items()}
        deltas = {}
        for bone, body in self.mapping.items():
            i = self.names.index(bone)
            # Native robot joint offsets need not lie on a straight limb axis.
            # Aligning hip->knee and knee->ankle destroys those lateral offsets.
            native_leg=bool(data.get('native_leg_frames',False)) and body.endswith(('Leg','Shin'))
            if body in ends and not native_leg:
                child = self.names.index(reverse[ends[body]])
                deltas[body] = align(self.bind[child,:3,3]-self.bind[i,:3,3],
                                     self.rest[ends[body]]-self.rest[body])
                self.calibration[i] = deltas[body] @ self.bind[i,:3,:3]
        # Preserve the native elbow's bend plane when opening the authored arm
        # into SOMA's T-pose. Independent upper/lower direction alignments add
        # an arbitrary axial roll (about 70 degrees on A3/H2).
        for side in ['Left', 'Right']:
            forearm = side+'ForeArm'
            i = self.names.index(reverse[forearm])
            child = self.names.index(reverse[side+'Hand'])
            upper_delta = deltas[side+'Arm']
            source = self.bind[child, :3, 3]-self.bind[i, :3, 3]
            target = self.rest[side+'Hand']-self.rest[forearm]
            deltas[forearm] = align(upper_delta @ source, target) @ upper_delta
            self.calibration[i] = deltas[forearm] @ self.bind[i, :3, :3]
        for side in ['Left','Right']:
            body = side+'Hand'
            self.calibration[self.names.index(reverse[body])] = deltas[side+'ForeArm'] @ self.bind[self.names.index(reverse[body]),:3,:3]
        # Explicit anatomical frames constrain hand roll as well as finger direction.
        if 'semantic_frames' in data:
            palm_frames = dict(zip(data['semantic_names'].tolist(), data['semantic_frames']))
        else:
            # Manny has the same landmarks as native hands, in its authored bind.
            # Forearm alignment alone leaves UE4/UE5 palms in different roll frames.
            palm_frames = {}
            for side, suffix in [('Left', 'l'), ('Right', 'r')]:
                name = reverse[side+'Hand']
                wrist = self.bind[self.names.index(name), :3, 3]
                knuckles = [self.bind[self.names.index(f'{finger}_01_{suffix}'), :3, 3]
                            for finger in ('middle', 'index', 'pinky')]
                distal = knuckles[0]-wrist
                distal /= np.linalg.norm(distal)
                across = knuckles[1]-knuckles[2]
                across -= distal * (distal @ across)
                across /= np.linalg.norm(across)
                palm_frames[name] = np.column_stack([distal, across, np.cross(distal, across)])
        self.palm_frames = palm_frames
        for side, sign in [('Left', 1.), ('Right', -1.)]:
            name = reverse[side+'Hand']
            distal = np.array([sign, 0., 0.])
            across = np.array([0., -1., 0.])
            target = np.column_stack([distal, across, np.cross(distal, across)])
            i = self.names.index(name)
            self.calibration[i] = target @ palm_frames[name].T @ self.bind[i, :3, :3]
        # Extra UE5 spine bones use Chest; retain their authored translations.
        for bone in self.names:
            if bone.startswith('spine_') and bone not in self.mapping:
                self.mapping[bone] = 'Chest'
        for i,parent in enumerate(self.parents):
            if parent >= i:
                raise ValueError('Display skeleton must be parent-first')
        self.root_offset = np.zeros(3)
        rest_targets = self.targets(np.array([self.rest[n] for n in self.soma_names]),
                                    np.tile([0.,0.,0.,1.],(len(self.soma_names),1)), ordered=True)
        # Match mean ankle height in the calibrated rest pose without resizing.
        self.root_offset[2] = np.mean([self.rest[s+'Foot'][2]-rest_targets[self.names.index(reverse[s+'Foot']),2,3]
                                      for s in ['Left','Right']])
        self.sole = SoleSupport(data, body_names, robot_xml) if robot_xml is not None else None

    def targets(self, positions, quaternions, ordered=False):
        if not ordered:
            positions,quaternions = positions[self.indices],quaternions[self.indices]
        rotations=Rotation.from_quat(quaternions).as_matrix()
        state=dict(zip(self.soma_names,rotations))
        root_pos=positions[self.soma_names.index('Hips')]
        root_rot=state['Hips']
        targets=np.tile(np.eye(4),(len(self.names),1,1))
        for i,name in enumerate(self.names):
            parent=self.parents[i]
            if parent>=0:
                targets[i]=targets[parent] @ self.inverse[parent] @ self.bind[i]
            else:
                targets[i,:3,:3]=root_rot @ self.bind[i,:3,:3]
                targets[i,:3,3]=root_pos+root_rot@(self.bind[i,:3,3]-self.bind[self.root,:3,3]+self.root_offset)
            if name in self.mapping:
                targets[i,:3,:3]=state[self.mapping[name]] @ self.calibration[i]
        return targets

    def matrices(self, positions, quaternions):
        matrices = self.targets(positions,quaternions) @ self.inverse
        if self.sole is not None:
            # One translation preserves every authored segment and joint rotation.
            # Match the physical support surface, not ground height: jumping and
            # penetration remain visible, and this never changes simulation poses.
            matrices[:, 2, 3] += self.sole.physical_height(positions, quaternions) - self.sole.visual_height(matrices)
        return matrices


class SoleSupport:
    """Exact lowest sole vertex and physical foot-box support, cached at load time."""

    def __init__(self, data, body_names, robot_xml, *, bones=None, support_bodies=None, basis=None):
        names = data['names'].tolist()
        if bones is None:
            mapping = dict(zip(data['soma_names'].tolist(), data['mapping'].tolist()))
            bones = [names.index(mapping[n]) for n in mapping
                     if n.endswith(('Foot', 'ToeBase')) and mapping[n]]
        vertices, joints, weights = [], [], []
        for slot in range(int(data['slots'])):
            j, w = data[f's{slot}_joints'], data[f's{slot}_weights']
            mask = (w * np.isin(j, bones)).sum(axis=1) > .8
            vertices.append(data[f's{slot}_points'][mask])
            joints.append(j[mask])
            weights.append(w[mask])
        points = np.concatenate(vertices)
        if basis is not None:
            points = points @ basis.T
        if not len(points):
            raise ValueError('SOMA skin has no foot/toe vertices')
        self.joints = np.concatenate(joints)
        weights = np.concatenate(weights)
        if not np.allclose(weights.sum(axis=1), 1., atol=1e-5):
            raise ValueError('SOMA sole skin weights must sum to one')
        self.weighted_points = np.c_[points, np.ones(len(points))][:, None, :] * weights[:, :, None]

        model = ET.parse(robot_xml)
        corners, owners = [], []
        for name in (support_bodies or ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')):
            node = model.find(f'.//body[@name="{name}"]')
            if node is None:
                raise ValueError(f'SOMA support body missing: {name}')
            geoms = node.findall('geom')
            if not geoms:
                raise ValueError(f'SOMA support geometry missing: {name}')
            for geom in geoms:
                # Explicit contract for the four boxes in sih23_var00.
                if geom.get('type') != 'box' or any(k in geom.attrib for k in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
                    raise ValueError(f'Unsupported SOMA support geometry: {name}')
                size = np.fromstring(geom.attrib['size'], sep=' ')
                quat = np.fromstring(geom.get('quat', '1 0 0 0'), sep=' ')
                rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                origin = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
                points = np.array(list(product((-1., 1.), repeat=3))) * size
                corners.extend(points @ rotation.T + origin)
                owners.extend([body_names.index(name)] * 8)
        self.corners = np.asarray(corners)
        self.owners = np.asarray(owners)

    def visual_height(self, matrices):
        return float(np.einsum('vki,vki->v', self.weighted_points, matrices[self.joints, 2, :]).min())

    def physical_height(self, positions, quaternions):
        rotations = Rotation.from_quat(quaternions).as_matrix()
        return float((np.einsum('vi,vi->v', rotations[self.owners, 2, :], self.corners)
                      + positions[self.owners, 2]).min())
