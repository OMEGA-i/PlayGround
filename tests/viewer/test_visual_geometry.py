import mujoco
import numpy as np

from omega_playground.viewer.visual_geometry import render_model


def test_hide_duplicate_proxies_without_mutating_source_or_hiding_ground():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="ground" type="plane" size="1 1 .1"/>
      <geom name="scenery" type="sphere" size=".1" contype="0" conaffinity="0"/>
      <body name="hand"><geom name="visual" type="sphere" size=".1" group="1" contype="0" conaffinity="0"/>
        <geom name="proxy" type="sphere" size=".1"/></body>
      <body name="collision_only"><geom name="sole" type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    groups = model.geom_group.copy()
    rendered = render_model(model)
    assert rendered.geom('proxy').group == 5
    for name in ('ground', 'scenery', 'visual', 'sole'):
        assert rendered.geom(name).group == model.geom(name).group
    np.testing.assert_array_equal(model.geom_group, groups)
    np.testing.assert_array_equal(rendered.geom_contype, model.geom_contype)
