"""Policy package loading (``omega.policy/1``); any mismatch refuses to load."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from omega_playground.policy.schedule import Schedule, physics_steps

SCHEMA = "omega.policy/1"
REQUIRED = ("name", "family", "robot", "coordinates", "physics", "layers", "modules", "sensors",
            "action", "pd", "solver", "source", "parity", "quality_accepted")


class PackageError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PolicyPackage:
    root: Path
    manifest: dict

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def family(self) -> str:
        return self.manifest["family"]

    @property
    def physics_hz(self) -> int:
        return int(self.manifest["physics"]["hz"])

    @property
    def robot(self) -> dict:
        return self.manifest["robot"]

    @property
    def asset_path(self) -> Path:
        return self.root / self.robot["asset"]

    @property
    def dof_names(self) -> list[str]:
        return self.robot["dof_names"]

    @property
    def body_names(self) -> list[str]:
        return self.robot["body_names"]

    def module_path(self, name: str) -> Path:
        return self.root / self.manifest["modules"][name]["file"]

    def layer_schedule(self, layer: str) -> Schedule | None:
        spec = self.manifest["layers"][layer]
        return None if spec is None else Schedule.from_json(spec["schedule"])

    def sha256sum(self) -> str:
        return sha256(self.root / "SHA256SUMS.json")


def load(root: str | Path) -> PolicyPackage:
    root = Path(root)
    manifest = json.loads((root / "policy.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise PackageError(f"Unsupported schema {manifest.get('schema')!r}; expected {SCHEMA}")
    missing = [k for k in REQUIRED if k not in manifest]
    if missing:
        raise PackageError(f"policy.json is missing {missing}")

    sums = json.loads((root / "SHA256SUMS.json").read_text())
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS.json"}
    if set(sums) != files:
        raise PackageError(f"SHA256SUMS.json does not list exactly the package files: {sorted(set(sums) ^ files)}")
    for rel, digest in sums.items():
        if sha256(root / rel) != digest:
            raise PackageError(f"sha256 mismatch: {rel}")

    pkg = PolicyPackage(root, manifest)
    if sha256(pkg.asset_path) != pkg.robot["asset_sha256"]:
        raise PackageError("robot asset sha256 mismatch")
    for name, spec in manifest["modules"].items():
        if sha256(root / spec["file"]) != spec["sha256"]:
            raise PackageError(f"module {name} sha256 mismatch")
        physics_steps(spec["period_s"], pkg.physics_hz)
    if sha256(root / manifest["parity"]["file"]) != manifest["parity"]["sha256"]:
        raise PackageError("parity fixtures sha256 mismatch")

    dof = len(pkg.dof_names)
    for key in ("offset", "scale"):
        if len(manifest["action"][key]) != dof:
            raise PackageError(f"action.{key} has wrong length")
    for key in ("kp", "kd", "armature", "effort_limit"):
        if len(manifest["pd"][key]) != dof:
            raise PackageError(f"pd.{key} has wrong length")
    if manifest["action"]["transform"] not in ("clamp", "none", "tanh"):
        raise PackageError(f"Unknown action transform {manifest['action']['transform']!r}")
    if manifest["coordinates"]["quat"] not in ("xyzw", "wxyz"):
        raise PackageError("coordinates.quat must be xyzw or wxyz")
    for layer in ("joystick", "wbc"):
        schedule = pkg.layer_schedule(layer)
        if schedule is not None:
            schedule.physics_steps(pkg.physics_hz)
    if manifest["layers"]["wbc"] is None:
        raise PackageError("layers.wbc is required")
    return pkg
