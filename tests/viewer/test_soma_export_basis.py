import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('soma_export', Path(__file__).parents[2]/'tools/assets/prepare_soma_skin.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.mark.parametrize('yaw', [0, 90, -90, 180])
def test_native_and_precalibrated_sources_share_one_anatomical_frame(yaw):
    names = ['left_upper_arm', 'right_upper_arm']
    bind = np.tile(np.eye(4), (2, 1, 1))
    source = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
    bind[:, :3, 3] = np.array([[.15, 0, 1.4], [-.15, 0, 1.4]]) @ source.T
    rest = {'LeftArm': np.array([.165, 0, 1.4]), 'RightArm': np.array([-.165, 0, 1.4])}
    basis = export.native_basis(bind, names, rest)
    np.testing.assert_allclose(basis @ source, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(basis @ (bind[0,:3,3]-bind[1,:3,3]), [.3,0,0], atol=1e-12)
