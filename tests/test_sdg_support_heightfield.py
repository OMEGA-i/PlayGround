"""Finite support conversion preserves compiled bounds and all other obstacles."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('support_surface', Path(__file__).parents[1]/'tools/sdg/convert_support_plane.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('secondary_x', [0, 4])
def test_finite_heightfield_union_and_obstacles(tmp_path, monkeypatch, secondary_x):
    source = tmp_path/'boxes.xml'
    source.write_text(f'<mujoco><worldbody><geom name="primary" type="box" pos="0 0 -.125" size="2 3 .125"/><geom name="secondary" type="box" pos="{secondary_x} 0 -.125" size="1 1 .125"/><geom name="obstacle" type="box" pos="1 1 .3" size=".2 .4 .3"/></worldbody></mujoco>')
    audit = tmp_path/'boxes.audit.json'
    audit.write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), changed_geoms=[dict(name='primary',bounds=[[-2,-3,-.25],[2,3,0]]),dict(name='secondary',bounds=[[secondary_x-1,-1,-.25],[secondary_x+1,1,0]])])))
    output = tmp_path/'heightfield.xml'
    monkeypatch.setattr(sys,'argv',['convert','--collision',str(source),'--audit',str(audit),'--output',str(output),'--support','primary','--representation','heightfield','--grid-spacing-m','.5'])
    if secondary_x:
        with pytest.raises(ValueError, match='not redundant'):
            MODULE.main()
        assert not output.exists()
        return
    MODULE.main()
    original = mujoco.MjModel.from_xml_path(str(source))
    result = mujoco.MjModel.from_xml_path(str(output))
    assert result.nhfield == 1 and result.ngeom == original.ngeom-1
    assert result.hfield_nrow[0] == 13 and result.hfield_ncol[0] == 9
    np.testing.assert_array_equal(result.hfield_data, 0)
    np.testing.assert_allclose(result.hfield_size[0], [2,3,.1,.25])
    np.testing.assert_allclose(result.geom('primary').pos, [0,0,0])
    for field in ('pos','size','type','friction'):
        np.testing.assert_array_equal(getattr(result.geom('obstacle'),field),getattr(original.geom('obstacle'),field))
    assert json.loads(output.with_suffix('.audit.json').read_text())['globally_equivalent_support_union']
