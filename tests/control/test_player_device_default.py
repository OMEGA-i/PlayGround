import pytest

from omega_playground.control.player_factory import device_player_enabled


@pytest.mark.parametrize('device,expected',[('cuda',True),('cuda:0',True),('cuda:1',True),('cpu',False)])
def test_default_tracks_execution_device(monkeypatch,device,expected):
    monkeypatch.delenv('OMEGA_GPU_PLAYER',raising=False)
    assert device_player_enabled(device) is expected


@pytest.mark.parametrize('override,expected',[('0',False),('1',True)])
def test_explicit_comparison_override(monkeypatch,override,expected):
    monkeypatch.setenv('OMEGA_GPU_PLAYER',override)
    assert device_player_enabled('cuda:0') is expected
    assert device_player_enabled('cpu') is expected


def test_invalid_override_rejected(monkeypatch):
    monkeypatch.setenv('OMEGA_GPU_PLAYER','yes')
    with pytest.raises(ValueError,match='0 or 1'):device_player_enabled('cuda:0')
