import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdLux

from omega_playground.viewer.usd_lighting import add_usd_lighting


def test_light_ratios_world_direction_and_no_duplicate_defaults(tmp_path):
    path = tmp_path/'lights.usda'
    stage = Usd.Stage.CreateNew(str(path))
    parent = UsdGeom.Xform.Define(stage, '/Lighting')
    parent.AddRotateYOp().Set(90)
    for name in ('Sun1', 'Sun2'):
        sun = UsdLux.DistantLight.Define(stage, f'/Lighting/{name}')
        sun.CreateIntensityAttr(1000)
        sun.CreateExposureAttr(1)
    sky = UsdLux.DomeLight.Define(stage, '/Sky')
    sky.CreateIntensityAttr(500)
    hidden = UsdLux.DistantLight.Define(stage, '/Hidden')
    hidden.CreateIntensityAttr(1e6)
    UsdGeom.Imageable(hidden).CreateVisibilityAttr('invisible')
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(name='old')
    report = add_usd_lighting(spec, path)
    model = spec.compile()
    assert model.nlight == 2
    assert model.vis.headlight.active == 0
    assert len(report['lights']) == 3
    np.testing.assert_allclose(model.light_diffuse, np.ones((2, 3)))
    np.testing.assert_allclose(model.light_ambient.sum(axis=0), [.25]*3)
    np.testing.assert_allclose(model.light_dir, [[-1, 0, 0]]*2, atol=1e-6)
    assert report['intensity_scale'] == 1/2000


def test_unlit_asset_keeps_fallback(tmp_path):
    path = tmp_path/'empty.usda'
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(name='fallback')
    assert not add_usd_lighting(spec, path)['imported']
    assert spec.compile().nlight == 1
