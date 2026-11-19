"""Derive an explicit CUDA execution package without modifying release weights."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from omega_playground.policy.package import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package = load(args.source)
    if package.family != "sonic":
        raise ValueError("Expected SONIC package")
    shutil.copytree(args.source, args.output)
    spec = json.loads((args.output / "policy.json").read_text())
    spec["name"] += "-cuda-batch"
    spec["quality_accepted"] = False
    spec["execution_source"] = {
        "package_sha256sums": package.sha256sum(),
        "implementation": "omega_playground.policy.sonic",
        "backend": "torch_cuda_batch",
        "tf32": False,
    }
    for module in spec["modules"].values():
        module["runtime"] = {"backend": "torch_cuda_batch", "batch": "world_capacity", "cuda_graph": True}
    (args.output / "policy.json").write_text(json.dumps(spec, indent=2) + "\n")
    sums = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (args.output / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    print(load(args.output).sha256sum())


if __name__ == "__main__":
    main()
