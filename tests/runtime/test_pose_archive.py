import pytest
import torch

from omega_playground.eval.pose_archive import DevicePoseArchive


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_archive_orders_cross_stream_frames_and_owns_snapshots():
    archive = DevicePoseArchive('cuda:0', capacity=2)
    producer = torch.cuda.Stream()
    expected = []
    for i in range(9):
        with torch.cuda.stream(producer):
            # Noncontiguous input, produced on a stream other than default.
            poses = (torch.arange(42, device='cuda').reshape(7, 6) + i).T
            archive.append(poses, i / 30)
        expected.append(torch.arange(42).reshape(7, 6).T.numpy() + i)
        del poses
        assert len(archive.pending) <= 2
    frames, times = archive.finish()
    import numpy as np
    np.testing.assert_array_equal(frames, expected)
    assert times == [i / 30 for i in range(9)]
    assert not archive.pending and not archive.free
    frames[0][0, 0] = -1
    assert frames[1][0, 0] == 1
    assert archive.finish()[0] is frames
