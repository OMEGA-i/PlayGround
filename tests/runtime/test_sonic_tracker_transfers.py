import numpy as np
import pytest
import torch

from omega_playground.policy.sonic.tracker import SonicTracker


@pytest.mark.parametrize('return_tokens', [False, True])
def test_action_only_path_does_not_transfer_unused_tokens(return_tokens):
    transferred = []

    class Token:
        def cpu(self):
            transferred.append('token')
            return torch.tensor([[3., 4.]])

    tracker = SonicTracker.__new__(SonicTracker)
    tracker.gpu, tracker.device = True, torch.device('cpu')
    tracker.batch = lambda *inputs: (Token(), torch.tensor([[.1, -.2]]))
    token, action = tracker.infer(np.zeros((1, 2)), np.zeros((1, 2)), return_tokens=return_tokens)
    np.testing.assert_array_equal(action, np.array([[.1, -.2]], dtype=np.float32))
    assert transferred == (['token'] if return_tokens else [])
    if return_tokens:
        np.testing.assert_array_equal(token, [[3., 4.]])
    else:
        assert token is None
