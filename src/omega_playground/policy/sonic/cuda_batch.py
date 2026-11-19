"""Fixed-capacity CUDA graph replay around true vmap batches of release weights.

The original ONNX files are never rewritten. Padding only stabilizes GPU shapes;
callers receive exactly the live rows and retain control of per-agent histories.
"""

import torch


class CudaBatch:
    def __init__(self, function, capacity, device):
        self.function, self.capacity = function, capacity
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SONIC torch_cuda_batch requires CUDA; no CPU fallback")
        self.graph = None

    @torch.no_grad()
    def __call__(self, *values):
        count = len(values[0])
        if not 0 < count <= self.capacity or any(len(v) != count for v in values):
            raise ValueError("Invalid CUDA batch length")
        if self.graph is None:
            self.inputs = [
                torch.zeros((self.capacity, *v.shape[1:]), dtype=v.dtype, device=self.device) for v in values
            ]
        for target, value in zip(self.inputs, values, strict=True):
            if target.shape[1:] != value.shape[1:] or target.dtype != value.dtype:
                raise ValueError('CUDA replay input shape/dtype changed')
            target[:count].copy_(value)
            if count < self.capacity:
                target[count:].copy_(value[:1].expand(self.capacity - count, *value.shape[1:]))
        if self.graph is None:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.function(*self.inputs)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = self.function(*self.inputs)
        self.graph.replay()
        return tuple(value[:count] for value in self.outputs)
