"""Collision experiments must pin bytes without replacing the scene release."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('sdg_episode', Path(__file__).parents[1]/'tools/sdg/run_episode.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pinned_candidate_and_baseline_validation(tmp_path, monkeypatch):
    baseline = tmp_path/'campus-full-contacts-v2.xml'
    baseline.write_text('<mujoco/>')
    candidate = tmp_path/'candidate.xml'
    candidate.write_text('<mujoco model="candidate"/>')
    monkeypatch.setattr(MODULE, 'ROOT', tmp_path/'playground')
    monkeypatch.setattr(MODULE, 'CONTACT_SHA', {'campus': MODULE.sha(baseline)})
    task = {'scene': 'campus'}
    assert MODULE.collision_asset(task, tmp_path) == baseline
    task['physical_collision'] = {'path': 'candidate.xml', 'sha256': MODULE.sha(candidate)}
    assert MODULE.collision_asset(task, tmp_path) == candidate
    candidate.write_text('modified')
    with pytest.raises(ValueError, match='hash mismatch'):
        MODULE.collision_asset(task, tmp_path)
    task['physical_collision']['path'] = '../candidate.xml'
    with pytest.raises(ValueError, match='relative'):
        MODULE.collision_asset(task, tmp_path)
    baseline.write_text('modified')
    with pytest.raises(ValueError, match='accepted release'):
        MODULE.collision_asset({'scene': 'campus'}, tmp_path)
