import pytest
import torch
from omega_playground.policy.runner import StageTimer


@pytest.mark.parametrize('trace', [False, True])
def test_stage_annotations_are_opt_in_and_preserve_exception_timing(monkeypatch, trace):
    if trace:
        monkeypatch.setenv('OMEGA_PHYSX_CUDA_TRACE', 'diagnostic.json')
    else:
        monkeypatch.delenv('OMEGA_PHYSX_CUDA_TRACE', raising=False)
    timer = StageTimer(False)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with pytest.raises(RuntimeError, match='probe'):
            with timer('write_targets'):
                raise RuntimeError('probe')
    names = [event.key for event in profiler.key_averages()]
    assert ('playground::write_targets' in names) == trace
    assert timer.report()['write_targets']['count'] == 1
