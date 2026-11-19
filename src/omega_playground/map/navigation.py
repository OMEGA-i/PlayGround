"""Bake and query support and clearance for hand-drawn Playground paths."""

import itertools
import json
import math
from pathlib import Path

import numpy as np


def resample(points, spacing=1.0):
    out = []
    for a, b in itertools.pairwise(points):
        n = max(1, math.ceil(math.dist(a, b) / spacing))
        out.extend((np.asarray(a) + (np.asarray(b) - a) * t / n).tolist() for t in range(n))
    return out + [list(points[-1])]


class CampusNavigation:
    def __init__(self, directory, scene_sha256=None, collision_sha256=None):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "map.json").read_text())
        if self.meta.get("schema") != "omega.map/3":
            raise ValueError("Expected omega.map/3; rebuild legacy maps")
        if scene_sha256 is not None and self.meta["scene_sha256"] != scene_sha256:
            raise ValueError("Map scene_sha256 does not match the running scene")
        if collision_sha256 is not None and self.meta.get("collision_sha256") != collision_sha256:
            raise ValueError("Map collision_sha256 does not match the running scene; rebuild map")
        arrays = np.load(self.directory / "layers.npz")
        self.height = arrays["support_height"]
        self.clearance = arrays["clearance"]
        self.road = arrays["road_mask"]
        self.resolution = self.meta["resolution"]
        self.bounds = self.meta["bounds"]
        self.shape = self.height.shape

    def support_continuous(self, points, heights, *, grid=False):
        from omega_playground.map.terrain import support_continuous
        return support_continuous(points, heights, self.meta.get('terrain_zones', []),
                                  grid_resolution=self.resolution if grid else 0.)

    def cells(self, points):
        return np.floor((np.asarray(points)[..., :2] - self.bounds[:2]) / self.resolution).astype(int)

    def world(self, cell):
        return (np.asarray(cell) + 0.5) * self.resolution + np.asarray(self.bounds[:2])

    def valid(self, points, radius=1.25):
        cells = self.cells(points)
        x, y = cells[..., 0], cells[..., 1]
        inside = (x >= 0) & (y >= 0) & (x < self.shape[1]) & (y < self.shape[0])
        xx, yy = np.clip(x, 0, self.shape[1] - 1), np.clip(y, 0, self.shape[0] - 1)
        return inside & (self.clearance[yy, xx] >= radius)

    def swept(self, start, paths, radius=1.25):
        """Conservative fine sampling of every predicted segment, vectorized by candidate."""
        previous = np.concatenate((np.broadcast_to(start, (len(paths), 1, 2)), paths[:, :-1]), axis=1)
        n = max(2, int(np.ceil(np.linalg.norm(paths - previous, axis=-1).max() / (self.resolution * 0.45))))
        legal = np.ones(len(paths), bool)
        for t in np.linspace(0, 1, n + 1):
            legal &= self.valid(previous + (paths - previous) * t, radius).all(axis=1)
        return legal

    def line_valid(self, a, b, radius):
        return bool(self.valid(resample([a, b], self.resolution * 0.4), radius).all())

def campus_roads(layout, scene_sha256):
    """Only apply surveyed Campus centerlines to their exact frozen scene."""
    roads = [{"id": r["id"], "width": r["width"], "points": [[x, -z] for x, z in r["points"]]}
             for r in layout["routes"]]
    lo_x, hi_x, lo_z, hi_z = layout["site_bounds"]
    bounds = [lo_x, -hi_z, hi_x, -lo_z]
    supplement = json.loads(Path(__file__).with_name('campus_roads.json').read_text())
    if scene_sha256 in supplement.get('scene_sha256s', [supplement['scene_sha256']]):
        roads += supplement['roads']
        bounds = [min(bounds[i], supplement['bounds'][i]) if i < 2 else
                  max(bounds[i], supplement['bounds'][i]) for i in range(4)]
    return roads, bounds, float(supplement["goal_snap_limit_m"])


def build(scene, output, resolution=0.5, *, terrain_ground=None):
    """Bake invisible collision meshes separately from visible top-down cartography."""
    from PIL import Image, ImageDraw
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

    from omega_playground.scene.package import load

    scene = Path(scene)
    scene_package = load(scene.parent)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if scene_package.name.startswith('VibeObstacle'):
        layer = Usd.Stage.Open(str(scene_package.collision_entry))
        cache = UsdGeom.BBoxCache(0, ['default','proxy','guide','render'], False, True)
        extent = cache.ComputeWorldBound(layer.GetDefaultPrim()).ComputeAlignedRange()
        bounds = [*extent.GetMin()[:2], *extent.GetMax()[:2]]
        layout, roads, snap_limit = {'buildings': []}, [], 0.
    else:
        layout = json.loads((scene.parent / "layout.json").read_text())
        roads, bounds, snap_limit = campus_roads(layout, scene_package.scene_sha256)
    width, height = math.ceil((bounds[2]-bounds[0])/resolution), math.ceil((bounds[3]-bounds[1])/resolution)
    support = np.full((height, width), -np.inf, np.float32)
    blocked = Image.new("L", (width, height))
    bd = ImageDraw.Draw(blocked)
    stage = Usd.Stage.Open(str(scene))
    transforms = UsdGeom.XformCache()
    faces = []
    collider_count = 0
    materials = {}

    def slab(poly, threshold, above):
        result = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            ia = (a[2] >= threshold) if above else (a[2] <= threshold)
            ib = (b[2] >= threshold) if above else (b[2] <= threshold)
            if ia:
                result.append(a)
            if ia != ib:
                result.append(a + (b - a) * (threshold - a[2]) / (b[2] - a[2]))
        return np.asarray(result)

    def pixel(p):
        return (p - np.array(bounds[:2])) / resolution - 0.5

    primitives = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Cube):
            if (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
            ):
                ext = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), 0)
                if ext is None:
                    raise ValueError(f"Unsupported collider {prim.GetPath()}")
                import itertools

                corners = np.array(list(itertools.product(*zip(ext[0], ext[1]))))
                xyz = (np.c_[corners, np.ones(8)] @ np.asarray(transforms.GetLocalToWorldTransform(prim)))[
                    :, :3
                ]
                low, high = xyz.min(0), xyz.max(0)
                collider_count += 1
                if low[2] < 2.3 and high[2] > 0.35:
                    a, b = pixel(low[:2]), pixel(high[:2])
                    bd.rectangle([*a, *b], fill=255)
                    primitives.append(
                        ((low[:2] + high[:2]) / 2, max(0.7, float(np.max(high[:2] - low[:2])) / 2))
                    )
            continue
        mesh = UsdGeom.Gprim(prim)
        if prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            points = np.asarray([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                 [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float) * half
            indices = np.asarray([0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7])
            counts = [4] * 6
        else:
            usd_mesh = UsdGeom.Mesh(prim)
            points = np.asarray(usd_mesh.GetPointsAttr().Get(), float)
            indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
            counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        if not len(points):
            continue
        points = (
            np.c_[points, np.ones(len(points))] @ np.asarray(transforms.GetLocalToWorldTransform(prim))
        )[:, :3]
        low, high = points.min(0), points.max(0)
        if np.any(low[:2] > bounds[2:]) or np.any(high[:2] < bounds[:2]):
            continue
        collision = (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        )
        visible = mesh.ComputeVisibility() != UsdGeom.Tokens.invisible
        if collision:
            collider_count += 1
        rgb = (0.38, 0.46, 0.41)
        if visible:
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                key = str(material.GetPath())
                if key not in materials:
                    shader, _, _ = material.ComputeSurfaceSource()
                    value = shader.GetInput("diffuseColor").Get() if shader else None
                    materials[key] = tuple(value) if value is not None else rgb
                rgb = materials[key]
        tint = tuple(int(255 * v) for v in np.clip(0.08 + 0.60 * np.asarray(rgb), 0.06, 0.75))
        offset = 0
        for count in counts:
            face = points[indices[offset : offset + count]]
            offset += count
            if count < 3:
                continue
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            area = abs(normal[2])
            if collision:
                clipped = slab(face, 0.35, True)
                if len(clipped):
                    clipped = slab(clipped, 2.3, False)
                if len(clipped) >= 2:
                    q = [tuple(p) for p in pixel(clipped[:, :2])]
                    bd.polygon(q, fill=255) if len(q) > 2 else bd.line(q, fill=255, width=1)
            if area < 1e-8:
                continue
            if visible and area > 0.5 * np.linalg.norm(normal):
                z = float(face[:, 2].mean())
                name = str(prim.GetPath()).lower()
                color = "#425652" if z < 0.35 else "#82938f"
                if any(k in name for k in ("road", "asphalt")):
                    color = "#4b6067"
                elif any(k in name for k in ("grass", "field", "lawn")):
                    color = "#385b4b"
                elif "water" in name:
                    color = "#2e596b"
                elif any(k in name for k in ("track", "brick", "court")):
                    color = "#785c54"
                elif z > 3:
                    color = "#86938c"
                faces.append((z, face[:, :2], tint))
            if not collision or face[:, 2].max() > 0.35 or face[:, 2].min() < -0.15:
                continue
            for j in range(1, count - 1):
                a, b, c = face[[0, j, j + 1]]
                u, v = b - a, c - a
                det = u[0] * v[1] - u[1] * v[0]
                if abs(det) < 1e-9:
                    continue
                lower = np.maximum(np.floor(pixel(np.minimum(np.minimum(a, b), c)[:2])).astype(int), 0)
                upper = np.minimum(
                    np.ceil(pixel(np.maximum(np.maximum(a, b), c)[:2])).astype(int) + 1, [width, height]
                )
                x0, y0 = lower
                x1, y1 = upper
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dx = bounds[0] + (xx + 0.5) * resolution - a[0]
                dy = bounds[1] + (yy + 0.5) * resolution - a[1]
                s = (dx * v[1] - dy * v[0]) / det
                t = (u[0] * dy - u[1] * dx) / det
                inside = (s >= -1e-5) & (t >= -1e-5) & (s + t <= 1 + 1e-5)
                z = np.where(inside, a[2] + s * u[2] + t * v[2], -np.inf)
                support[y0:y1, x0:x1] = np.maximum(support[y0:y1, x0:x1], z)
    finite = np.isfinite(support)
    h = np.where(finite, support, 0.0)
    jumps = maximum_filter(h, 3) - minimum_filter(h, 3) > 0.05001
    free = finite & (np.asarray(blocked) == 0) & ~jumps
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * resolution - resolution * 0.71
    from omega_playground.map.terrain import bake_bridge
    terrain_zones = bake_bridge(scene_package, support, clearance, bounds, resolution, ground=terrain_ground)
    road_img = Image.new("L", (width, height))
    rd = ImageDraw.Draw(road_img)
    for r in roads:
        pts = r["points"]
        rd.line([tuple(pixel(p)) for p in pts], fill=255, width=max(1, round(r["width"] / resolution)))
    np.savez_compressed(
        output / "layers.npz",
        support_height=support,
        collision_blocked=np.asarray(blocked) > 0,
        clearance=clearance.astype(np.float32),
        road_mask=np.asarray(road_img) > 0,
    )
    # The full map is rendered once; no frame-time USD or scene edits.
    scale = 4
    image = Image.new("RGB", (width * scale, height * scale), "#213431")
    draw = ImageDraw.Draw(image)
    for _, xy, color in sorted(faces, key=lambda f: f[0]):
        p = pixel(xy) * scale
        p[:, 1] = height * scale - p[:, 1]
        draw.polygon([tuple(v) for v in p], fill=color)
    for center, radius in primitives:
        p = pixel(center) * scale
        p[1] = height * scale - p[1]
        r = radius / resolution * scale
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill="#315545", outline="#50745d")
    image.save(output / "campus.png")
    for radius in [0.5, 1.25, 3.9]:
        overlay = np.zeros((height, width, 4), np.uint8)
        overlay[clearance >= radius] = [28, 216, 181, 90]
        Image.fromarray(overlay[::-1]).save(output / f"navigable_{radius:g}.png")
    labels = [
        {"name": b.get("displayName", b["id"]), "position": [b["origin"][0], -b["origin"][2]]}
        for b in layout["buildings"]
    ]
    meta = {
        "schema": "omega.map/3",
        "clearance_radii": [0.5, 1.25, 3.9],
        "bounds": bounds,
        "resolution": resolution,
        "scene_sha256": scene_package.scene_sha256,
        "collision_sha256": scene_package.collision_sha256,
        "scene": str(scene),
        "roads": roads,
        "goal_snap_limit_m": snap_limit,
        "labels": labels,
        "colliders": collider_count,
        "clearance_m": 0.5,
        "grid_shape": [height, width],
        "north": "+Y",
        "height_step_limit_m": 0.05,
        "terrain_zones": terrain_zones,
        "method": "authored road preference + collision support triangles / conservative obstacle footprints",
        "limitations": "Campus support is restricted to world z in [-0.15,0.35]m; obstacle slab is [0.35,2.3]m. Revision-bound bridge zone uses actual collision rays with 20 cm approach-step allowance; experimental locomotion. Not for arbitrary multi-level scenes. Geometry candidates do not certify locomotion.",
    }
    (output / "map.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
