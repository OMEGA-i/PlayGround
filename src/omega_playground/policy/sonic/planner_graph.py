# Adapted from ProtoMotions sonic_pp at 7337905fe4a1dd77ae998e1bf086b0b61110e751.
"""Torch execution of the pinned SONIC graph, including its target controller.

All model parameters and behavior constants come from ONNX. No MotionBricks
checkpoint, library, or training code is required. Shape metadata stays on host;
runtime numerical tensors stay on the requested device.
"""

from functools import reduce

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn
from torch.nn import functional as F

from .graph_ops import run_op

META = {
    "Reshape": (1,),
    "Slice": (1, 2, 3, 4),
    "Unsqueeze": (1,),
    "Squeeze": (1,),
    "Expand": (1,),
    "Tile": (1,),
    "Resize": (1, 2, 3),
    "ConstantOfShape": (0,),
    "Range": (0, 1, 2),
    "Split": (1,),
    "Pad": (1, 2, 3),
    "TopK": (1,),
    "ReduceSum": (1,),
    "ReduceL2": (1,),
    "CumSum": (1,),
    "OneHot": (1,),
}


def integers(value):
    """Read CPU shape metadata; never synchronize model data to the host."""
    if value.device.type != "cpu":
        raise ValueError("Data-dependent GPU shape is not supported")
    return [int(v) for v in value.reshape(-1).tolist()]


def execute(op, x, a):
    """Implement the release operators not used by the earlier neural slices."""
    if op == "Conv":
        # User accepts small batch-dependent rounding differences. Use the
        # native CUDA/cuDNN path rather than manually materializing windows.
        if x[0].ndim != 3 or a.get("group", 1) != 1:
            raise NotImplementedError("Release Conv1d only")
        pads = a.get("pads", [0, 0])
        stride = a.get("strides", [1])[0]
        dilation = a.get("dilations", [1])[0]
        bias = x[2] if len(x) > 2 else None
        if pads[0] == pads[1]:
            return F.conv1d(x[0], x[1], bias, stride=stride, padding=pads[0], dilation=dilation)
        return F.conv1d(F.pad(x[0], pads), x[1], bias, stride=stride, dilation=dilation)
    if op == "Shape":
        return torch.tensor(tuple(x[0].shape)[a.get("start", 0) : a.get("end", x[0].ndim)])
    binary = {
        "Equal": torch.eq,
        "Greater": torch.gt,
        "LessOrEqual": torch.le,
        "GreaterOrEqual": torch.ge,
        "Pow": torch.pow,
        "And": torch.logical_and,
        "Or": torch.logical_or,
    }
    unary = {
        "Floor": torch.floor,
        "Atan": torch.atan,
        "Sin": torch.sin,
        "Cos": torch.cos,
        "Neg": torch.neg,
        "Abs": torch.abs,
        "IsNaN": torch.isnan,
        "Reciprocal": torch.reciprocal,
    }
    if op in binary:
        return binary[op](*x)
    if op in unary:
        return unary[op](x[0])
    if op == "Identity":
        return x[0]
    if op == "Flatten":
        axis = a.get("axis", 1)
        if axis < 0:
            axis += x[0].ndim
        return x[0].reshape(int(np.prod(x[0].shape[:axis])), int(np.prod(x[0].shape[axis:])))
    if op == "Mod":
        return (torch.fmod if a.get("fmod", 0) else torch.remainder)(*x)
    if op in ("Max", "Min"):
        return reduce(torch.maximum if op == "Max" else torch.minimum, x)
    if op == "ConstantOfShape":
        value = numpy_helper.to_array(a["value"]) if "value" in a else np.array([0.0], np.float32)
        return torch.full(integers(x[0]), value.item(), dtype=torch.from_numpy(value.copy()).dtype)
    if op == "Range":
        vals = [integers(v)[0] if not v.is_floating_point() else float(v.item()) for v in x]
        return torch.arange(*vals, dtype=x[0].dtype)
    if op in ("ReduceSum", "ReduceL2"):
        axes = tuple(integers(x[1])) if len(x) > 1 else tuple(a.get("axes", range(x[0].ndim)))
        if not axes and a.get("noop_with_empty_axes", 0):
            return x[0]
        axes = axes or tuple(range(x[0].ndim))
        y = x[0].square() if op == "ReduceL2" else x[0]
        y = y.sum(axes, keepdim=bool(a.get("keepdims", 1)))
        return y.sqrt() if op == "ReduceL2" else y
    if op == "ArgMin":
        return x[0].argmin(a.get("axis", 0), keepdim=bool(a.get("keepdims", 1)))
    if op == "OneHot":
        depth = integers(x[1])[0]
        idx = x[0].long()
        hot = F.one_hot(idx.remainder(depth), depth).to(x[2].dtype)
        hot = hot * ((idx >= -depth) & (idx < depth)).unsqueeze(-1)
        y = hot * (x[2][1] - x[2][0]) + x[2][0]
        return y.movedim(-1, a.get("axis", -1))
    if op == "Split":
        sizes = integers(x[1]) if len(x) > 1 else a.get("split")
        return torch.split(x[0], sizes, dim=a.get("axis", 0))
    if op == "LeakyRelu":
        return F.leaky_relu(x[0], a.get("alpha", 0.01))
    if op == "GatherElements":
        return torch.gather(x[0], a.get("axis", 0), x[1].long().remainder(x[0].shape[a.get("axis", 0)]))
    if op == "CumSum":
        axis = integers(x[1])[0]
        y = x[0].flip([axis]) if a.get("reverse", 0) else x[0]
        out = y.cumsum(axis)
        if a.get("exclusive", 0):
            out = out - y
        return out.flip([axis]) if a.get("reverse", 0) else out
    if op == "Clip":
        y = x[0]
        if len(x) > 1 and x[1] is not None:
            y = torch.maximum(y, x[1])
        if len(x) > 2 and x[2] is not None:
            y = torch.minimum(y, x[2])
        return y
    if op == "TopK":
        return torch.topk(
            x[0],
            integers(x[1])[0],
            dim=a.get("axis", -1),
            largest=bool(a.get("largest", 1)),
            sorted=bool(a.get("sorted", 1)),
        )
    if op == "Einsum":
        return torch.einsum(a["equation"].decode(), *x)
    if op == "Pad":
        pads = integers(x[1])
        n = x[0].ndim
        mode = a.get("mode", b"constant").decode()
        if mode != "constant":
            raise NotImplementedError("Nonconstant pad")
        value = 0.0 if len(x) < 3 or x[2] is None else float(x[2].item())
        return F.pad(x[0], [v for i in range(n - 1, -1, -1) for v in (pads[i], pads[n + i])], value=value)
    if op == "ScatterND":
        shape = x[0].shape
        depth = x[1].shape[-1]
        strides = [int(np.prod(shape[i + 1 : depth])) for i in range(depth)]
        index = sum(x[1][..., i].long().remainder(shape[i]) * strides[i] for i in range(depth))
        tail = shape[depth:]
        flat = x[0].reshape(int(np.prod(shape[:depth])), *tail)
        idx = index.reshape(-1, *([1] * len(tail))).expand(-1, *tail)
        return flat.scatter(0, idx, x[2].reshape(-1, *tail)).reshape(shape)
    return run_op(op, x, a)


class TorchSPlanner(nn.Module):
    """Frozen ONNX-derived Torch graph with capped vmap request batching."""

    def __init__(self, path, device="cuda", seed=1234, microbatch=32):
        super().__init__()
        model = onnx.load(path)
        self.seed = seed
        self.device = torch.device(device)
        self.microbatch = microbatch
        self.batched = True
        if microbatch < 1:
            raise ValueError("Positive fixed planner microbatch required")
        self.inputs = tuple(v.name for v in model.graph.input)
        self.outputs = tuple(v.name for v in model.graph.output)
        constants = {
            v.name: torch.from_numpy(numpy_helper.to_array(v).copy()) for v in model.graph.initializer
        }
        nodes = []
        for n in model.graph.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "Constant":
                constants[n.output[0]] = torch.from_numpy(numpy_helper.to_array(attrs["value"]).copy())
                continue
            if all(not k or k in constants for k in n.input):
                values = [constants[k] if k else None for k in n.input]
                result = execute(n.op_type, values, attrs)
                result = result if isinstance(result, tuple) else (result,)
                constants.update(zip(n.output, result, strict=True))
                continue
            nodes.append((n.name, n.op_type, tuple(n.input), tuple(n.output), attrs))
        needed = {k for _, _, inputs, _, _ in nodes for k in inputs if k in constants}
        self.host_constants = {k: constants[k] for k in needed}
        self.device_keys = {}
        self.device_metadata = {}
        for i, k in enumerate(sorted(needed)):
            key = f"constant_{i}"
            self.register_buffer(key, constants[k].to(self.device))
            self.device_keys[k] = key
        self.nodes = nodes
        uses = {}
        for i, (_, _, inputs, _, _) in enumerate(nodes):
            for k in inputs:
                uses[k] = i
        self.release = [[] for _ in nodes]
        for k, i in uses.items():
            if k not in needed and k not in self.outputs:
                self.release[i].append(k)
        self.eval()
        self.requires_grad_(False)

    def device_value(self, name, value, device):
        key = (name, str(device))
        if key not in self.device_metadata:
            self.device_metadata[key] = value.to(device)
        return self.device_metadata[key]

    def forward(self, *args):
        """Run one logical release request; vmap supplies the outer batch axis."""
        values = dict(zip(self.inputs, args, strict=True))
        values[""] = None
        for i, (name, op, inputs, outputs, attrs) in enumerate(self.nodes):
            x = [self.host_constants[k] if k in self.host_constants else values[k] for k in inputs]
            metadata = META.get(op, ())
            # Promoted teacher weights must participate in autograd, including CPU tests.
            for j, k in enumerate(inputs):
                if j not in metadata and k in self.device_keys and self.device_keys[k] in self._parameters:
                    x[j] = getattr(self, self.device_keys[k])
            for j in metadata:
                if j < len(x) and x[j] is not None and x[j].device.type != "cpu":
                    raise ValueError(f"{name}: GPU metadata argument {j}")
            device = next(
                (
                    v.device
                    for j, v in enumerate(x)
                    if j not in metadata and isinstance(v, torch.Tensor) and v.device.type != "cpu"
                ),
                None,
            )
            if device is not None:
                for j, k in enumerate(inputs):
                    if j not in metadata and x[j] is not None and x[j].device != device:
                        x[j] = (
                            getattr(self, self.device_keys[k])
                            if k in self.device_keys
                            else self.device_value(k, x[j], device)
                        )
            try:
                result = execute(op, x, attrs)
                result = result if isinstance(result, tuple) else (result,)
                values.update(zip(outputs, result, strict=True))
            except Exception as error:
                raise RuntimeError(f"{name} ({op}): {error}") from error
            for k in self.release[i]:
                values.pop(k, None)
        return tuple(values[k] if k in values else self.host_constants[k] for k in self.outputs)

    @torch.no_grad()
    def plan_batch(self, values):
        """Execute up to microbatch real requests per call without padding the tail."""
        count = len(values[self.inputs[0]])
        chunks = {k: [] for k in self.outputs}
        if count < 1:
            raise ValueError("Empty planner batch")
        if any(len(values[k]) != count for k in self.inputs):
            raise ValueError("Planner batch lengths differ")
        for begin in range(0, count, self.microbatch):
            n = min(self.microbatch, count - begin)
            args = []
            for k in self.inputs:
                value = values[k][begin : begin + n]
                args.append(value.unsqueeze(1))
            result = torch.vmap(self.forward)(*args)
            for k, v in zip(self.outputs, result, strict=True):
                chunks[k].append(v[:n])
        return {k: torch.cat(v) for k, v in chunks.items()}
