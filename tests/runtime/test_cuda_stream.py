import pytest
import torch
import warp as wp

from omega_playground.ops.serve import configure_simulation_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA stream interoperability')
def test_live_shared_stream_supports_graph_capture_and_torch_reads():
    torch_previous = torch.cuda.current_stream()
    warp_previous = wp.get_stream('cuda:0')
    try:
        configure_simulation_stream(torch.device('cuda:0'))
        values = wp.zeros(4, dtype=wp.float32, device='cuda:0')
        values.fill_(7.)
        with wp.ScopedCapture() as capture:
            values.fill_(7.)
        values.zero_()
        wp.capture_launch(capture.graph)
        assert wp.to_torch(values).cpu().tolist() == [7.] * 4
    finally:
        torch.cuda.set_stream(torch_previous)
        wp.set_stream(warp_previous)
