import signal
import numpy as np
import pytest
from omega_playground.recording.execution import PhysicalTrajectory, camera_samples, Cancellation


@pytest.mark.parametrize('controls', [1, 2, 3, 120, 121])
def test_terminal_is_saved_but_not_encoded(tmp_path, controls):
    trace = PhysicalTrajectory(['root'])
    pose = np.array([[0., 0., 1., 0., 0., 0., 1.]])
    for tick in range(controls+1):
        pose[0, 0] = tick
        trace.append(tick, pose, np.array([tick]), np.array([0.]))
    pose[:] = 999  # source buffers may be reused by physics
    trace.save(tmp_path)
    with np.load(tmp_path/'body_states.npz') as data:
        assert data['body_q'].shape == (controls+1, 1, 7)
        np.testing.assert_array_equal(data['body_q'][:,0,0], np.arange(controls+1))
        np.testing.assert_array_equal(data['physics_step'], np.arange(controls+1)*4)
        indexes = camera_samples(data['control_tick'], controls)
        assert len(indexes) == (controls+1)//2
        assert indexes[-1] < controls
    with np.load(tmp_path/'physics_states.npz') as data:
        assert data['qpos'][-1,0] == controls


def test_missing_camera_boundary_is_rejected():
    with pytest.raises(ValueError, match='incomplete'):
        camera_samples([0, 1, 3, 4], 4)


def test_capture_rejects_gap_and_nonfinite():
    trace = PhysicalTrajectory(['root'])
    with pytest.raises(ValueError, match='tick'):
        trace.append(1, np.zeros((1,7)), [0], [0])
    with pytest.raises(ValueError, match='snapshot'):
        trace.append(0, np.full((1,7), np.nan), [0], [0])


def test_cancellation_preserves_signal_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    with Cancellation().installed() as cancellation:
        cancellation.request(signal.SIGTERM)
        assert cancellation.reason == 'sigterm'
    assert signal.getsignal(signal.SIGTERM) == previous
