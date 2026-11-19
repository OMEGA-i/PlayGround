import numpy as np
import pytest

from omega_playground.viewer.process import RenderProcess


class FakeViewer:
    frames = 0
    latest_jpeg = b''

    def show(self, poses, time, record):
        if time < 0:
            raise ValueError('invalid time')
        self.frames += int(record)
        self.latest_jpeg = str((poses[0, 0], time, self.camera_mode)).encode()

    def close(self):
        return self.frames


def test_process_owns_snapshots_and_drains_in_order():
    import weakref
    frames = []
    process = RenderProcess(FakeViewer, lambda jpeg, time, record, wall: frames.append((jpeg, time, wall)))
    queues = [weakref.ref(process._requests), weakref.ref(process._responses)]
    poses = np.zeros((1, 7))
    for i in range(6):
        poses[0, 0] = i
        process.show(poses, i)
    process.camera_mode = 'chase'
    process.show(poses, 6, record=False)
    assert process.close() == 6
    assert all(reference() is None for reference in queues)
    assert [f[1] for f in frames] == list(range(7))
    assert b'chase' in frames[-1][0]
    assert all(frames[i][2] <= frames[i + 1][2] for i in range(6))


def test_process_propagates_failure_without_hanging():
    process = RenderProcess(FakeViewer, lambda *args: None)
    with pytest.raises(RuntimeError, match='invalid time'):
        process.show(np.zeros((1, 7)), -1, record=False)
    with pytest.raises(RuntimeError):
        process.close()


class EncodingViewer(FakeViewer):
    stream_video = True

    def show(self, poses, time, record):
        self.frame_sink(poses.copy(), record)
        self.frames += int(record)

    def encode_frame(self, rgb, record):
        import time
        time.sleep(.01)
        if rgb[0, 0] < 0:
            raise ValueError('encoding failed')
        return str(rgb[0, 0]).encode()


def test_encoding_pipeline_keeps_frame_order_and_drains():
    frames = []
    process = RenderProcess(EncodingViewer, lambda jpeg, time, record, wall: frames.append((jpeg, time)))
    poses = np.zeros((1, 7))
    for i in range(10):
        poses[0, 0] = i
        process.show(poses, i)
    process.show(poses, 10, record=False)
    assert process.close() == 10
    assert frames == [(str(float(i)).encode(), i) for i in range(10)]+[(b'9.0', 10)]


def test_encoding_failure_wakes_waiting_preview():
    process = RenderProcess(EncodingViewer, lambda *args: None)
    with pytest.raises(RuntimeError, match='encoding failed'):
        process.show(-np.ones((1, 7)), 0, record=False)
    with pytest.raises(RuntimeError):
        process.close()


class CudaViewer(FakeViewer):
    def show(self, poses, time, record):
        import time as clock
        assert poses.is_cuda
        clock.sleep(.01)  # Let producer race ahead; its slots must stay owned.
        self.latest_jpeg = str(int(poses[0, 0].item())).encode()
        self.frames += int(record)


def test_cuda_slots_preserve_snapshots():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    frames = []
    process = RenderProcess(CudaViewer, lambda jpeg, *args: frames.append(jpeg))
    try:
        poses = torch.zeros((66, 7), device='cuda')
        for i in range(12):
            poses.fill_(i)
            process.show(poses, i)
        poses.fill_(99)
        assert process.close() == 12
    finally:
        if process._process.is_alive():
            process._process.terminate()
            process._process.join(timeout=10)
    assert frames == [str(i).encode() for i in range(12)]
