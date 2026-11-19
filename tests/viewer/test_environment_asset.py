import hashlib
import json
import numpy as np
import pytest
from omega_playground.viewer.environment_asset import load, validate
from omega_playground.viewer.environment_importance import prepare_distribution


def fixture_arrays():
    rgb = np.ones((4, 8, 3), np.float32)*2
    d = prepare_distribution(rgb, 32, 'cpu')
    return dict(radiance=rgb, pdf=d.pdf.numpy(), samples=d.samples.numpy(),
                world_to_dome=np.eye(3), gain=np.ones(3))


def write_bundle(root, arrays):
    np.savez(root/'environment.npz', **arrays)
    (root/'asset.json').write_text(json.dumps({'schema':'omega.mjwarp-environment/1',
        'scene_sha256':'scene', 'entry':'environment.npz', 'samples_per_strategy':32}))
    (root/'SHA256SUMS.json').write_text(json.dumps({k:hashlib.sha256((root/k).read_bytes()).hexdigest()
        for k in ('asset.json','environment.npz')}))


def test_load_preserves_arrays_and_rejects_wrong_scene_or_tampering(tmp_path):
    arrays = fixture_arrays()
    write_bundle(tmp_path, arrays)
    loaded, meta = load(tmp_path, 'scene')
    for key in arrays:
        np.testing.assert_array_equal(loaded[key], arrays[key])
    with pytest.raises(ValueError, match='scene mismatch'):
        load(tmp_path, 'other')
    with (tmp_path/'environment.npz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load(tmp_path, 'scene')


@pytest.mark.parametrize('field', ['pdf', 'samples', 'world_to_dome', 'gain'])
def test_invalid_prepared_arrays_rejected(field):
    a = fixture_arrays()
    if field == 'pdf':
        a[field] *= 2
    elif field == 'samples':
        a[field][:, 3] *= 2
    elif field == 'world_to_dome':
        a[field][0, 0] = -1
    else:
        a[field][0] = -1
    with pytest.raises(ValueError):
        validate(a, 32)
