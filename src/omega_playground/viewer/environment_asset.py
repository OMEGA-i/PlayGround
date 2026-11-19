"""Checksummed HDR/MIS assets; runtime loads arrays without USD or CDF building."""
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = {'radiance', 'pdf', 'samples', 'world_to_dome', 'gain'}


def prepare(scene, output, samples=32):
    from .environment_source import load_dome
    from .environment_importance import prepare_distribution
    rgb, rotation, gain, source = load_dome(scene.entry)
    distribution = prepare_distribution(rgb, samples, 'cpu')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    np.savez(root/'environment.npz', radiance=np.asarray(rgb, np.float32),
             pdf=distribution.pdf.numpy(), samples=distribution.samples.numpy(),
             world_to_dome=rotation, gain=gain)
    meta = {'schema': 'omega.mjwarp-environment/1', 'scene_sha256': scene.scene_sha256,
            'entry': 'environment.npz', 'samples_per_strategy': samples, 'source': source,
            'integration': 'GGX + HDR balance MIS', 'occlusion': False}
    (root/'asset.json').write_text(json.dumps(meta, indent=2)+'\n')
    sums = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('asset.json', 'environment.npz')}
    (root/'SHA256SUMS.json').write_text(json.dumps(sums, indent=2)+'\n')
    return meta


def load(path, scene_sha256):
    root = Path(path)
    sums = json.loads((root/'SHA256SUMS.json').read_text())
    if set(sums) != {'asset.json', 'environment.npz'}:
        raise ValueError('Environment asset has an invalid checksum manifest')
    for name, expected in sums.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Environment checksum mismatch: {name}')
    meta = json.loads((root/'asset.json').read_text())
    if (meta.get('schema') != 'omega.mjwarp-environment/1' or meta.get('scene_sha256') != scene_sha256
            or meta.get('entry') != 'environment.npz'):
        raise ValueError('Environment schema or scene mismatch')
    with np.load(root/'environment.npz', allow_pickle=False) as data:
        if set(data.files) != FIELDS:
            raise ValueError('Environment array fields mismatch')
        arrays = {key: data[key] for key in FIELDS}
    validate(arrays, meta['samples_per_strategy'])
    return arrays, meta


def validate(a, count):
    rgb, pdf, samples, rotation, gain = [a[k] for k in ('radiance', 'pdf', 'samples', 'world_to_dome', 'gain')]
    if (type(count) is not int or not 1 <= count <= 1024 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 1 or pdf.shape != rgb.shape[:2] or samples.shape != (count, 4)
            or rotation.shape != (3, 3) or gain.shape != (3,)):
        raise ValueError('Environment array dimensions mismatch')
    if (any(not np.isfinite(x).all() for x in a.values()) or (rgb < 0).any()
            or (pdf <= 0).any() or (gain < 0).any() or (samples[:, 3] <= 0).any()):
        raise ValueError('Environment arrays contain invalid values')
    if (not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)
            or not np.allclose(np.linalg.norm(samples[:, :3], axis=-1), 1., atol=1e-5)):
        raise ValueError('Environment rotation or sample directions invalid')
    h, w = pdf.shape
    area = (np.cos(np.arange(h)*np.pi/h)-np.cos((np.arange(h)+1)*np.pi/h))*2*np.pi/w
    if not np.isclose((pdf*area[:, None]).sum(), 1., atol=1e-5):
        raise ValueError('Environment PDF is not normalized')
    x = np.floor((np.pi-np.arctan2(samples[:, 0], samples[:, 2]))/(2*np.pi)*w).astype(int)%w
    y = np.minimum((np.arccos(np.clip(samples[:, 1], -1, 1))/np.pi*h).astype(int), h-1)
    if not np.allclose(pdf[y, x], samples[:, 3], rtol=1e-5, atol=1e-7):
        raise ValueError('Environment sample PDFs disagree with distribution')
