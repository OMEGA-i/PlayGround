"""Prepare an immutable HDR/MIS asset before simulation startup."""
import argparse
import json
from pathlib import Path
from omega_playground.scene.package import load
from omega_playground.viewer.environment_asset import prepare

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, choices=range(1, 1025), default=32)
a = p.parse_args()
print(json.dumps(prepare(load(a.scene), a.output, a.samples), indent=2))
