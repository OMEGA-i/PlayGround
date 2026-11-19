"""Bake static convex collision pieces around a bounded SuperMimic rollout."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from omega_playground.scene.package import load


def bake(scene_dir, center, radius, output):
    scene = load(scene_dir)
    stage = Usd.Stage.Open(str(scene.collision_entry))
    transforms = UsdGeom.XformCache()
    root = ET.Element('mujoco')
    assets, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        name = 'pg_' + str(len(world))
        mat = np.asarray(transforms.GetLocalToWorldTransform(prim))
        if prim.IsA(UsdGeom.Mesh):
            vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        elif prim.IsA(UsdGeom.Cube):
            from itertools import product
            vertices = np.array(list(product([-1, 1], repeat=3))) * UsdGeom.Cube(prim).GetSizeAttr().Get()/2
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            axis = 'XYZ'.index(str(cylinder.GetAxisAttr().Get()))
            angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
            ring = np.array([np.cos(angles),np.sin(angles)]).T*cylinder.GetRadiusAttr().Get()
            vertices = np.array([np.insert(p,axis,z) for z in [-.5,.5] for p in ring])
            vertices[:,axis] *= cylinder.GetHeightAttr().Get()
        else:
            raise ValueError(str(prim.GetPath()))
        vertices = (np.c_[vertices,np.ones(len(vertices))] @ mat)[:,:3]
        lo, hi = vertices[:,:2].min(0), vertices[:,:2].max(0)
        if np.any(lo > center+radius) or np.any(hi < center-radius):
            continue
        vertices[:,:2] -= center
        ET.SubElement(assets,'mesh',name=name,vertex=' '.join(f'{v:.8g}' for v in vertices.flat))
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,friction='1 .005 .0001',contype='1',conaffinity='1')
    if not len(world):
        raise ValueError('Empty collision window')
    # Express the scene in the training clip's local frame, with support at z=0.
    import mujoco
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hit = np.full(1, -1, dtype=np.int32)
    above = 1.
    distance = mujoco.mj_ray(model, data, np.array([0., 0., above]),
                             np.array([0., 0., -1.]), None, True, -1, hit)
    if distance < 0:
        raise ValueError('No support under spawn')
    for geom in world:
        geom.set('pos', f'0 0 {distance-above:.9g}')
    ET.ElementTree(root).write(output)
    print(scene.name, len(world), 'collision pieces')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('scene',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--center',type=float,nargs=2,required=True)
    p.add_argument('--radius',type=float,default=8)
    a = p.parse_args()
    bake(a.scene,np.array(a.center),a.radius,a.output)
