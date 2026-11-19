# Adapted from ProtoMotions sonic_pp at 7337905fe4a1dd77ae998e1bf086b0b61110e751.
"""Batched CUDA interpretation of the exact released tracker ONNX weights.

vmap keeps the original sample's fixed batch-one dimensions. No exported graph
shape rewriting or learned replacement. Parity against ORT is required before use.
"""

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn

from .graph_ops import ints, run_op


class TrackerGraph(nn.Module):
    def __init__(self, path):
        super().__init__()
        model = onnx.load(path)
        self.input = model.graph.input[0].name
        self.output = model.graph.output[0].name
        arrays = {t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer}
        self.nodes = []
        for n in model.graph.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "Constant":
                arrays[n.output[0]] = numpy_helper.to_array(attrs["value"]).copy()
            else:
                self.nodes.append((n.name, n.op_type, list(n.input), list(n.output), attrs))
        self.device_metadata = {}
        self.keys = {}
        self.cpu_constants = {}
        for i, (name, value) in enumerate(arrays.items()):
            if value.dtype.kind in "iub":
                self.cpu_constants[name] = torch.from_numpy(value)
            else:
                key = f"value_{i}"
                self.register_buffer(key, torch.from_numpy(value))
                self.keys[name] = key

    def device_value(self, name, value, device):
        # Host values are constants or shape metadata; populate on warmup, not capture.
        if value.device == device:
            return value
        key = (name, str(device))
        if key not in self.device_metadata:
            self.device_metadata[key] = value.to(device)
        return self.device_metadata[key]

    def forward(self, obs):
        values = {k: getattr(self, v) for k, v in self.keys.items()}
        values.update(self.cpu_constants)
        values[self.input] = obs
        values[""] = None
        for name, op, inputs, outputs, a in self.nodes:
            x = [values[k] for k in inputs]
            try:
                if op in ("Add", "Sub", "Mul", "Div", "Equal", "Where", "Concat"):
                    devices = [v.device for v in x if isinstance(v, torch.Tensor) and v.is_cuda]
                    if devices:
                        x = [
                            self.device_value(inputs[j], v, devices[0]) if isinstance(v, torch.Tensor) else v
                            for j, v in enumerate(x)
                        ]
                if op == "Shape":
                    result = torch.tensor(list(x[0].shape)[a.get("start", 0) : a.get("end", x[0].ndim)])
                elif op == "Sigmoid":
                    result = x[0].sigmoid()
                elif op == "Tanh":
                    result = x[0].tanh()
                elif op == "Round":
                    result = x[0].round()
                elif op == "Equal":
                    result = x[0] == self.device_value(inputs[1], x[1], x[0].device)
                elif op == "Gather":
                    result = run_op(op, [x[0], self.device_value(inputs[1], x[1], x[0].device)], a)
                elif op == "ReduceSum":
                    axes = ints(x[1]) if len(x) > 1 else a.get("axes")
                    result = x[0].sum(
                        dim=tuple(axes) if axes is not None else tuple(range(x[0].ndim)),
                        keepdim=bool(a.get("keepdims", 1)),
                    )
                elif op == "ConstantOfShape":
                    value = (
                        numpy_helper.to_array(a["value"]).copy()
                        if "value" in a
                        else np.array([0.0], np.float32)
                    )
                    result = torch.full(
                        tuple(ints(x[0])),
                        value.item(),
                        dtype=torch.from_numpy(value).dtype,
                        device="cpu" if value.dtype.kind in "iub" else obs.device,
                    )
                elif op == "ScatterND":
                    result = torch.index_put(x[0], tuple(x[1].long().to(x[0].device).unbind(-1)), x[2])
                else:
                    result = run_op(op, x, a)
                if len(outputs) != 1:
                    raise NotImplementedError("Multiple tracker outputs")
                values[outputs[0]] = result
            except Exception as e:
                raise RuntimeError(f"{name} ({op}): {e}") from e
        return values[self.output]

    def batched(self, obs):
        """Evaluate live rows up to the capacity, without padding dynamic batches."""
        run = torch.vmap(lambda row: self(row[None])[0])
        capacity = getattr(self, "max_batch", None)
        if capacity is not None:
            if len(obs) <= capacity:
                return run(obs)
            return torch.cat([run(chunk) for chunk in obs.split(capacity)])
        size = getattr(self, "microbatch", 0)
        if not size:
            return run(obs)
        # Keep GEMM shapes fixed across world counts, including the final chunk.
        # vmap preserves sample independence; padded rows never enter a controller.
        outputs = []
        for chunk in obs.split(size):
            count = len(chunk)
            if count < size:
                chunk = torch.cat([chunk, chunk.new_zeros((size - count, *chunk.shape[1:]))])
            outputs.append(run(chunk)[:count])
        return torch.cat(outputs)
