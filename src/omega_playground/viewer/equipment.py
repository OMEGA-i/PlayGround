"""Render-only baton and round shield in ASE ``right_hand`` / ``shield`` body coordinates.

Ported from ProtoMotions-ASE-newton ``utils/ase_manny_equipment.py``. The shield face matches the ASE shield
collider (0.30 m radius); the props never create collision shapes or policy degrees of freedom.
"""

import itertools

import numpy as np
import trimesh


def _cylinder(a, b, radius, sections=24):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mesh = trimesh.creation.cylinder(radius=radius, height=np.linalg.norm(b - a), sections=sections)
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], b - a))
    mesh.apply_translation((a + b) / 2)
    return mesh


def equipment(grips: dict, native: bool = False) -> list[dict]:
    """``grips[side] = (palm center, handle axis)`` in hand coordinates, from the skin palette.

    ``native``: the shield is mounted on the displayed hand (``NativeASEPalette.equipment_transform``), so its
    handle sits at UE5 Manny's palm in shield coordinates.
    """
    parts = []

    def add(name, body, mesh, color, roughness=0.6, metallic=0.0):
        parts.append({"name": name, "body": body, "mesh": mesh, "color": color,
                      "roughness": roughness, "metallic": metallic})

    black = (0.025, 0.031, 0.038)
    # 58 cm straight baton with a 13 cm ribbed rubber grip centered in the palm.
    g, baton_axis = grips["r"]
    add("baton_shaft", "right_hand", _cylinder(g + [0.065, 0, 0], g + [0.505, 0, 0], 0.016), (0.06, 0.075, 0.09),
        0.32, 0.55)
    add("baton_grip", "right_hand", _cylinder(g + [-0.075, 0, 0], g + [0.065, 0, 0], 0.020), black, 0.88)
    ribs = []
    for x in np.linspace(-0.061, 0.05, 12):
        ring = trimesh.creation.torus(major_radius=0.020, minor_radius=0.0018, major_sections=24, minor_sections=6)
        ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        ring.apply_translation(g + [x, 0, 0])
        ribs.append(ring)
    add("baton_ribs", "right_hand", trimesh.util.concatenate(ribs), (0.055, 0.06, 0.065), 0.9)
    for name, x, r in (("pommel", -0.077, 0.024), ("collar", 0.071, 0.024), ("cap", 0.507, 0.017)):
        add("baton_" + name, "right_hand", _cylinder(g + [x - 0.006, 0, 0], g + [x + 0.006, 0, 0], r), black, 0.45)

    radius = 0.30
    red, silver, blue = (0.55, 0.018, 0.025), (0.72, 0.76, 0.8), (0.015, 0.055, 0.32)

    def front(x, z=0.0):
        return 0.035 + 0.035 * max(0.0, 1.0 - (x * x + z * z) / radius ** 2)

    add("round_shield_back", "shield", _cylinder([0, 0, 0], [0, 0.027, 0], radius, 96), (0.18, 0.21, 0.24), 0.38, 0.75)

    def band(name, inner, outer, color):
        angles = np.linspace(0.0, 2 * np.pi, 129)
        verts = [[r * np.cos(a), front(r * np.cos(a), r * np.sin(a)), r * np.sin(a)]
                 for r in np.linspace(inner, outer, 5) for a in angles]
        faces = []
        for row in range(4):
            for j in range(128):
                k = row * 129 + j
                faces.extend([[k, k + 1, k + 129], [k + 1, k + 130, k + 129]])
        add(name, "shield", trimesh.Trimesh(vertices=verts, faces=faces, process=False), color, 0.3, 0.7)

    band("shield_outer_red", 0.245, 0.3, red)
    band("shield_silver_ring", 0.195, 0.245, silver)
    band("shield_inner_red", 0.145, 0.195, red)
    band("shield_blue_center", 0.0, 0.145, blue)
    points = []
    for i in range(10):
        a = np.pi / 2 + i * np.pi / 5
        r = 0.133 if i % 2 == 0 else 0.051
        x, z = r * np.cos(a), r * np.sin(a)
        points.append([x, front(x, z) + 0.0015, z])
    star = trimesh.Trimesh(vertices=[[0.0, front(0.0) + 0.0015, 0.0]] + points,
                           faces=[[0, 1 + (i + 1) % 10, 1 + i] for i in range(10)], process=False)
    star = star.subdivide().subdivide()
    star.vertices[:, 1] = [front(x, z) + 0.0015 for x, _, z in star.vertices]
    add("shield_silver_star", "shield", star, silver, 0.28, 0.8)
    rim = trimesh.creation.torus(major_radius=0.295, minor_radius=0.005, major_sections=96, minor_sections=12)
    rim.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    rim.apply_translation([0.0, 0.035, 0.0])
    add("shield_red_rim", "shield", rim, red, 0.3, 0.7)

    # Handle at the left palm, expressed relative to the shield body.
    if native:
        from omega_playground.viewer.native_skin import MANNY_GRIP_AXIS, MANNY_GRIP_CENTER

        grip = np.array([0.0, -0.07, -0.138947]) + MANNY_GRIP_CENTER["l"]
        handle_axis = MANNY_GRIP_AXIS["l"]
    else:
        grip = np.array([0.0, -0.07, -0.138947]) + grips["l"][0]
        handle_axis = grips["l"][1]
    add("shield_handle", "shield", _cylinder(grip - 0.064 * handle_axis, grip + 0.064 * handle_axis, 0.018), black, 0.85)
    mounts = []
    for offset in (-0.064, 0.064):
        end = grip + offset * handle_axis
        mounts.append(_cylinder([end[0], front(end[0], end[2]) - 0.01, end[2]], end, 0.009))
    add("shield_handle_mounts", "shield", trimesh.util.concatenate(mounts), (0.16, 0.18, 0.20), 0.35, 0.7)
    strap = []
    center = np.array([0.0, -0.07, 0.025])
    for a, b in itertools.pairwise(np.linspace(0, 2 * np.pi, 33)):
        strap.append(_cylinder(center + [0.052 * np.cos(a), 0.052 * np.sin(a), 0],
                               center + [0.052 * np.cos(b), 0.052 * np.sin(b), 0], 0.012, 10))
    add("shield_arm_strap", "shield", trimesh.util.concatenate(strap), black, 0.95)
    add("shield_strap_mounts", "shield",
        trimesh.util.concatenate([_cylinder([x, -0.07, 0.025], [x, front(0.075) - 0.01, 0.025], 0.010)
                                  for x in (-0.052, 0.052)]), (0.12, 0.14, 0.17), 0.4, 0.6)
    transform = trimesh.geometry.align_vectors([1, 0, 0], baton_axis)
    for part in parts:
        if part["body"] == "right_hand":
            part["mesh"].apply_translation(-g)
            part["mesh"].apply_transform(transform)
            part["mesh"].apply_translation(g)
    return parts


def asset_path(grips, native=False):
    """Content-addressed offline mesh variant; runtime never generates geometry."""
    import hashlib
    from pathlib import Path

    values = np.concatenate([np.asarray(value).reshape(-1) for side in ('l', 'r') for value in grips[side]])
    key = hashlib.sha256(np.rint(values*1e8).astype('<i8').tobytes() + bytes([native])).hexdigest()[:20]
    return Path(__file__).resolve().parents[3] / 'artifacts' / 'equipment' / f'{key}.npz'


def prepare_equipment(grips, native=False):
    path = asset_path(grips, native)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    parts = equipment(grips, native)
    arrays['count'] = len(parts)
    for i, part in enumerate(parts):
        for key in ('name', 'body', 'color', 'roughness', 'metallic'):
            arrays[f'{i}_{key}'] = part[key]
        arrays[f'{i}_vertices'] = np.asarray(part['mesh'].vertices)
        arrays[f'{i}_normals'] = np.asarray(part['mesh'].vertex_normals)
        arrays[f'{i}_faces'] = np.asarray(part['mesh'].faces)
    np.savez_compressed(path, **arrays)
    return path


def load_equipment(grips, native=False, prepared_path=None):
    from pathlib import Path
    path = asset_path(grips, native) if prepared_path is None else Path(prepared_path)
    if not path.is_file():
        raise ValueError('Armed assets missing; run tools/assets/prepare_armed_assets.py before starting the service')
    with np.load(path) as data:
        return [{key: data[f'{i}_{key}'].copy() if key in ('vertices', 'normals', 'faces', 'color')
                 else data[f'{i}_{key}'].item()
                 for key in ('name', 'body', 'color', 'roughness', 'metallic', 'vertices', 'normals', 'faces')}
                for i in range(int(data['count']))]
