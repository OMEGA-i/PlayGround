"""Semantic regression tests for full-graph operators and request batching."""

import torch

from omega_playground.policy.sonic.planner_graph import TorchSPlanner, execute


def test_flatten_rank_boundary():
    x = torch.arange(12).reshape(3, 4)
    assert execute("Flatten", [x], {"axis": 2}).shape == (12, 1)
    assert execute("Flatten", [x], {"axis": 0}).shape == (1, 12)


def test_scatter_nd_under_vmap():
    data = torch.zeros(2, 3, 4)
    idx = torch.tensor([[[0], [2]], [[1], [2]]])
    values = torch.arange(16).reshape(2, 2, 4).float()
    out = torch.vmap(lambda a, b, c: execute("ScatterND", [a, b, c], {}))(data, idx, values)
    assert torch.equal(out[0, 0], values[0, 0])
    assert torch.equal(out[1, 1], values[1, 0])


def test_max_microbatch_tail_is_not_padded(monkeypatch):
    sizes = []
    original_vmap = torch.vmap

    def recording_vmap(fn):
        mapped = original_vmap(fn)

        def invoke(*args):
            sizes.append(len(args[0]))
            return mapped(*args)

        return invoke

    monkeypatch.setattr(torch, "vmap", recording_vmap)

    class Probe:
        microbatch = 4
        inputs = ("x",)
        outputs = ("y",)

        def forward(self, x):
            return (x.square(),)

    probe = Probe()
    values = {"x": torch.arange(7).float()[:, None]}
    all_rows = TorchSPlanner.plan_batch(probe, values)["y"]
    assert sizes == [4, 3]
    singles = torch.cat(
        [TorchSPlanner.plan_batch(probe, {"x": values["x"][i : i + 1]})["y"] for i in range(7)]
    )
    assert torch.equal(all_rows, singles)


def test_native_convolution_matches_dilation_stride_and_padding():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 3, 17, generator=generator)
    weight = torch.randn(5, 3, 3, generator=generator)
    bias = torch.randn(5, generator=generator)
    actual = execute("Conv", [x, weight, bias], {"pads": [2, 2], "strides": [2], "dilations": [2]})
    expected = torch.nn.functional.conv1d(x, weight, bias, stride=2, padding=2, dilation=2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_cuda_graph_replay_updates_inputs_and_keeps_live_rows():
    import pytest
    from omega_playground.policy.sonic.cuda_batch import CudaBatch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    batch = CudaBatch(lambda x: (x.square(),), 4, 'cuda:0')
    first = torch.arange(12, device='cuda').float().reshape(4, 3)
    torch.testing.assert_close(batch(first)[0], first.square())
    second = torch.tensor([[9., 8., 7.], [2., 3., 4.]], device='cuda')
    torch.testing.assert_close(batch(second)[0], second.square())
    with pytest.raises(ValueError):
        batch(torch.ones(2, 1, device='cuda'))
