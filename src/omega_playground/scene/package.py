"""Scene packages (contracts §4): ``scene.usdc`` single entry, ``meta.json``, ``SHA256SUMS.json``."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ScenePackage:
    root: Path
    meta: dict
    scene_sha256: str

    @property
    def name(self) -> str:
        return self.meta["sceneName"]

    @property
    def entry(self) -> Path:
        return self.root / self.meta["entry"]

    @property
    def collision_entry(self) -> Path:
        return self.root / self.meta["collision"]["entry"]

    @property
    def collision_sha256(self) -> str:
        return self.meta["collision"]["sha256"]

    def describe(self) -> dict:
        return {"name": self.name, "root": str(self.root), "revision": self.meta.get("revision"),
                "scene_sha256": self.scene_sha256, "collision": self.meta.get("collision")}


def load(root: str | Path, *, allow_legacy: bool = False) -> ScenePackage:
    """Load and verify every file listed in ``SHA256SUMS.json``."""
    root = Path(root).resolve()
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("entry") != "scene.usdc":
        raise ValueError(f"{root}: scene packages have the single entry scene.usdc, got {meta.get('entry')!r}")
    if meta.get("upAxis") != "Z" or meta.get("metersPerUnit") != 1:
        raise ValueError(f"{root}: scene must be Z-up in metres (upAxis={meta.get('upAxis')!r}, "
                         f"metersPerUnit={meta.get('metersPerUnit')!r})")
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    if "scene.usdc" not in sums:
        raise ValueError(f"{root}: SHA256SUMS.json does not list scene.usdc")
    for name in sums:
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"Scene checksum path escapes package: {name}")
    collision = meta.get("collision")
    if collision is None and not allow_legacy:
        raise ValueError("Scene has no standard collision layer; republish it before starting a new run")
    if collision is not None:
        if collision.get("schema") != "omega.scene-collision/1":
            raise ValueError("Unsupported scene collision schema")
        entry = collision.get("entry")
        if not isinstance(entry, str) or entry not in sums or entry == meta["entry"]:
            raise ValueError("Collision layer must be a separate checksummed file")
        if collision.get("sha256") != sums[entry] or "meta.json" not in sums:
            raise ValueError("Collision layer hash/metadata must be covered by the manifest")
    bad = [name for name, digest in sums.items() if not (root / name).is_file() or _sha256(root / name) != digest]
    if bad:
        raise ValueError(f"{root}: missing or modified files: {bad[:10]}{' ...' if len(bad) > 10 else ''}")
    if collision is not None:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics

        layer_stage = Usd.Stage.Open(str(root / collision["entry"]))
        scene_stage = Usd.Stage.Open(str(root / meta["entry"]))
        for stage in (layer_stage, scene_stage):
            if UsdGeom.GetStageUpAxis(stage) != "Z" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
                raise ValueError("Scene entry and collision layer must both author Z-up metres")
        def colliders(stage):
            return {str(p.GetPath()): p for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False}
        authored, composed = colliders(layer_stage), colliders(scene_stage)
        if not authored or authored.keys() != composed.keys():
            raise ValueError("Scene must compose exactly the colliders from its one collision layer")
        a_cache, b_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
        for path, prim in composed.items():
            if not any(Path(spec.layer.realPath).resolve() == (root / collision["entry"]).resolve()
                       for spec in prim.GetPrimStack() if spec.layer.realPath):
                raise ValueError(f"Collider {path} does not reference the declared collision layer")
            if not np.allclose(a_cache.GetLocalToWorldTransform(authored[path]),
                               b_cache.GetLocalToWorldTransform(prim), atol=1e-8, rtol=0):
                raise ValueError(f"Scene overrides collision transform at {path}")
    identity = (hashlib.sha256(json.dumps(sums, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if collision is not None else sums["scene.usdc"])
    return ScenePackage(root, meta, identity)
