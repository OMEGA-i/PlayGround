"""Pinned SONIC ONNX operator semantics for batched Torch inference."""

# Adapted from ProtoMotions sonic_pp at 7337905fe4a1dd77ae998e1bf086b0b61110e751.
import numpy as np
import torch
from onnx import helper
from torch.nn import functional as F


def ints(value):
    return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]


def run_op(op, x, a):
    binary = {
        "Add": torch.add,
        "Sub": torch.sub,
        "Mul": torch.mul,
        "MatMul": torch.matmul,
        "Less": torch.lt,
    }
    unary = {"Relu": torch.relu, "Sqrt": torch.sqrt, "Not": torch.logical_not}
    if op in binary:
        return binary[op](x[0], x[1])
    if op in unary:
        return unary[op](x[0])
    if op == "Div":
        return torch.div(x[0], x[1], rounding_mode=None if x[0].is_floating_point() else "trunc")
    if op == "Cast":
        dtype = torch.from_numpy(np.empty((), dtype=helper.tensor_dtype_to_np_dtype(a["to"]))).dtype
        return x[0].to(dtype)
    if op == "Shape":
        return torch.tensor(
            list(x[0].shape)[a.get("start", 0) : a.get("end", x[0].ndim)],
            device=x[0].device,
        )
    if op == "Reshape":
        shape = ints(x[1])
        if not a.get("allowzero", 0):
            shape = [x[0].shape[i] if v == 0 else v for i, v in enumerate(shape)]
        return x[0].reshape(shape)
    if op == "Transpose":
        return x[0].permute(a.get("perm", list(reversed(range(x[0].ndim)))))
    if op == "Gather":
        axis = a.get("axis", 0) % x[0].ndim
        index = x[1].long().remainder(x[0].shape[axis])
        return (
            torch.index_select(x[0], axis, index.reshape(-1)).reshape(
                *x[0].shape[:axis], *index.shape, *x[0].shape[axis + 1 :]
            )
            if x[0].ndim + index.ndim > 1
            else x[0][index]
        )
    if op == "Slice":
        result = x[0]
        starts, ends = ints(x[1]), ints(x[2])
        axes = ints(x[3]) if len(x) > 3 else list(range(len(starts)))
        steps = ints(x[4]) if len(x) > 4 else [1] * len(starts)
        for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
            begin, stop, stride = slice(start, end, step).indices(result.shape[axis])
            index = torch.arange(begin, stop, stride, device=result.device)
            result = result.index_select(axis, index)
        return result
    if op == "Unsqueeze":
        result = x[0]
        axes = ints(x[1]) if len(x) > 1 else a["axes"]
        for axis in sorted(v % (result.ndim + len(axes)) for v in axes):
            result = result.unsqueeze(axis)
        return result
    if op == "Squeeze":
        result = x[0]
        axes = ints(x[1]) if len(x) > 1 else a.get("axes")
        if axes is None:
            return result.squeeze()
        for axis in sorted((v % result.ndim for v in axes), reverse=True):
            if result.shape[axis] != 1:
                raise ValueError("Cannot squeeze non-singleton dimension")
            result = result.squeeze(axis)
        return result
    if op == "Concat":
        return torch.cat(x, dim=a["axis"])
    if op == "Softmax":
        return torch.softmax(x[0], dim=a.get("axis", -1))
    if op == "LayerNormalization":
        axis = a.get("axis", -1) % x[0].ndim
        return F.layer_norm(
            x[0],
            x[0].shape[axis:],
            x[1],
            x[2] if len(x) > 2 else None,
            a.get("epsilon", 1e-5),
        )
    if op == "Gemm":
        left = x[0].T if a.get("transA", 0) else x[0]
        right = x[1].T if a.get("transB", 0) else x[1]
        result = a.get("alpha", 1.0) * (left @ right)
        return result + a.get("beta", 1.0) * x[2] if len(x) > 2 else result
    if op == "Where":
        return torch.where(x[0], x[1], x[2])
    if op == "Expand":
        return x[0].expand(torch.broadcast_shapes(x[0].shape, tuple(ints(x[1]))))
    if op == "Tile":
        return x[0].repeat(ints(x[1]))
    if op == "Flatten":
        axis = a.get("axis", 1) % x[0].ndim
        return x[0].reshape(int(np.prod(x[0].shape[:axis])), int(np.prod(x[0].shape[axis:])))
    if op == "ArgMax":
        if a.get("select_last_index", 0):
            raise NotImplementedError("Last-index tie breaking")
        return x[0].argmax(dim=a.get("axis", 0), keepdim=bool(a.get("keepdims", 1)))
    if op == "ScatterElements":
        if a.get("reduction", b"none") != b"none":
            raise NotImplementedError("Scatter reduction")
        axis = a.get("axis", 0)
        return x[0].scatter(axis, x[1].long().remainder(x[0].shape[axis]), x[2])
    if op == "ScatterND":
        if a.get("reduction", b"none") != b"none":
            raise NotImplementedError("Scatter reduction")
        result = x[0].clone()
        result[tuple(x[1].long().unbind(-1))] = x[2]
        return result
    if op == "Conv":
        spatial = x[0].ndim - 2
        pads = a.get("pads", [0] * (2 * spatial))
        if pads[:spatial] != pads[spatial:] or a.get("auto_pad", b"NOTSET") != b"NOTSET":
            raise NotImplementedError("Asymmetric/automatic padding")
        fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[spatial]
        return fn(
            x[0],
            x[1],
            x[2] if len(x) > 2 else None,
            a.get("strides", [1] * spatial),
            pads[:spatial],
            a.get("dilations", [1] * spatial),
            a.get("group", 1),
        )
    if op == "Resize":
        if (
            a.get("mode"),
            a.get("coordinate_transformation_mode"),
            a.get("nearest_mode"),
        ) != (b"nearest", b"asymmetric", b"floor"):
            raise NotImplementedError("Only pinned nearest/asymmetric/floor resize supported")
        scales = x[2].detach().cpu().tolist()
        if scales[:2] != [1.0, 1.0]:
            raise NotImplementedError("Batch/channel resize")
        return F.interpolate(x[0], scale_factor=scales[2:], mode="nearest")
    raise NotImplementedError(op)
