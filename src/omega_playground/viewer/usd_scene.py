"""Offline USD visual-mesh import for the independent MJWarp renderer.

Only visible triangle meshes enter the visual model. Physics/collision layers
are never converted. Geometry is shared across USD instances and equal meshes.
Additional PBR channels are reported explicitly until renderer parity is added.
"""
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdShade


def split_transform(matrix):
    matrix = np.asarray(matrix, dtype=float)
    linear = matrix[:3, :3].T
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale < 1e-10):
        raise ValueError('Singular visual transform')
    rotation = linear/scale
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1
        rotation[:, 0] *= -1
    if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5):
        raise ValueError('Sheared visual transform requires offline baking')
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    return matrix[3, :3].copy(), quat, scale


def _attribute_faces(values, interpolation, faces):
    if interpolation in ('vertex', 'varying'):
        indices = faces
    elif interpolation == 'faceVarying':
        indices = np.arange(faces.size).reshape(faces.shape)
    elif interpolation == 'uniform':
        indices = np.repeat(np.arange(len(faces))[:, None], 3, axis=1)
    elif interpolation == 'constant':
        indices = np.zeros_like(faces)
    else:
        raise ValueError(f'Unsupported mesh interpolation {interpolation}')
    if indices.max(initial=0) >= len(values):
        raise ValueError('Visual attribute indices exceed its values')
    return indices


def texture_transform(source, depth=0):
    if source is None:
        return np.eye(3)
    if depth > 8:
        raise ValueError('Texture coordinate network is cyclic or too deep')
    node = UsdShade.Shader(source[0])
    kind = node.GetIdAttr().Get()
    if kind == 'UsdPrimvarReader_float2':
        if node.GetInput('varname').Get() != 'st':
            raise ValueError('Only the authored st primvar is supported')
        return np.eye(3)
    if kind != 'UsdTransform2d':
        raise ValueError(f'Unsupported texture coordinate node: {kind}')
    def value(name, default):
        result = node.GetInput(name).Get()
        return default if result is None else result
    scale = np.asarray(value('scale', (1., 1.)))
    angle = np.radians(value('rotation', 0.))
    c, sn = np.cos(angle), np.sin(angle)
    matrix = np.eye(3)
    matrix[:2, :2] = np.array([[c, -sn], [sn, c]])@np.diag(scale)
    matrix[:2, 2] = value('translation', (0., 0.))
    return matrix@texture_transform(node.GetInput('in').GetConnectedSource(), depth+1)


def add_usd_scene(spec, entry, texture_dir):
    """Append static world geoms; copy textures into an offline asset directory."""
    import shutil

    entry, texture_dir = Path(entry).resolve(), Path(texture_dir).resolve()
    texture_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(entry))
    if UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Visual scene must use Z-up metres')
    transforms = UsdGeom.XformCache()
    source_meshes, meshes, materials, textures = {}, {}, {}, {}
    warnings, instances, triangles = set(), 0, 0
    textured_materials = set()
    cutout_materials = {}
    face_varying_normals = {}
    texture_color_spaces = {}
    texture_wrap_modes = {}
    instance_transforms = []

    def material_for(prim):
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not material:
            return '', np.eye(3)
        shader = material.ComputeSurfaceSource()[0]
        if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            raise ValueError(f'Unsupported material surface: {material.GetPath()}')
        key = str(material.GetPath())
        if key in materials:
            return materials[key]
        color = shader.GetInput('diffuseColor').Get()
        opacity = shader.GetInput('opacity').Get()
        rgba = [*(color if color is not None else (.5, .5, .5)), opacity if opacity is not None else 1.]
        texture_name = ''
        uv_transform = np.eye(3)
        source = shader.GetInput('diffuseColor').GetConnectedSource()
        if source:
            node = UsdShade.Shader(source[0])
            if node.GetIdAttr().Get() != 'UsdUVTexture':
                raise ValueError(f'Unsupported color network: {material.GetPath()}')
            file = node.GetInput('file').Get()
            path = Path(file.resolvedPath)
            if not path.is_file():
                raise ValueError(f'Missing resolved texture: {file}')
            scale = node.GetInput('scale').Get()
            bias = node.GetInput('bias').Get()
            if bias is not None and not np.allclose(bias, 0):
                raise ValueError('Texture bias requires offline baking')
            # A connected RGB output replaces diffuseColor's fallback value.
            # Its alpha scale is unrelated to the separate surface opacity input.
            rgba[:3] = list(scale[:3]) if scale is not None else [1., 1., 1.]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            color_space = node.GetInput('sourceColorSpace').Get() or 'auto'
            if color_space not in ('auto', 'sRGB', 'raw'):
                raise ValueError(f'Unsupported texture sourceColorSpace: {color_space}')
            # Equal bytes may be connected with different encoding semantics.
            # Preserve auto as unresolved until file metadata is inspected.
            wraps = []
            for axis in ('wrapS', 'wrapT'):
                wrap = node.GetInput(axis).Get() or 'useMetadata'
                if wrap == 'useMetadata':
                    warnings.add('Texture useMetadata addressing unresolved; using repeat')
                    wrap = 'repeat'
                if wrap not in ('repeat', 'clamp', 'mirror', 'black'):
                    raise ValueError(f'Unsupported texture wrap mode: {wrap}')
                wraps.append(str(wrap))
            texture_key = (digest, str(color_space), *wraps)
            if texture_key not in textures:
                target = texture_dir/(digest+path.suffix)
                shutil.copyfile(path, target)
                texture_name = 'campus_tex_'+digest[:16]+'_'+str(color_space)+'_'+'_'.join(wraps)
                spec.add_texture(name=texture_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(target))
                textures[texture_key] = texture_name
                texture_color_spaces[texture_name] = str(color_space)
                texture_wrap_modes[texture_name] = wraps
            texture_name = textures[texture_key]
            st = node.GetInput('st').GetConnectedSource()
            uv_transform = texture_transform(st)
        name = 'campus_mat_'+str(len(materials))
        alpha_source = shader.GetInput('opacity').GetConnectedSource() if shader.GetInput('opacity') else None
        if alpha_source and texture_name and alpha_source[1] == 'a':
            alpha_node = UsdShade.Shader(alpha_source[0])
            alpha_file = alpha_node.GetInput('file')
            if (alpha_file and Path(alpha_file.Get().resolvedPath) == path
                    and np.allclose(texture_transform(alpha_node.GetInput('st').GetConnectedSource()), uv_transform)):
                cutoff = shader.GetInput('opacityThreshold').Get()
                cutout_materials[name] = dict(file=digest+path.suffix, wrap=wraps,
                                               threshold=float(cutoff or 0.5))
        for channel in ('normal', 'roughness', 'metallic', 'opacity', 'emissiveColor', 'occlusion'):
            inp = shader.GetInput(channel)
            if inp and inp.HasConnectedSource() and not (channel == 'opacity' and name in cutout_materials):
                warnings.add('Connected PBR channel not yet rendered: '+channel)
        target = spec.add_material(name=name, rgba=rgba)
        if texture_name:
            target.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
            textured_materials.add(name)
        for field in ('roughness', 'metallic'):
            value = shader.GetInput(field).Get()
            if value is not None:
                setattr(target, field, float(value))
        materials[key] = name, uv_transform
        return materials[key]

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == 'invisible' or imageable.ComputePurpose() in ('guide', 'proxy'):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f'Unsupported visible primitive {prim.GetPath()}: {prim.GetTypeName()}')
        mesh = UsdGeom.Mesh(prim)
        if UsdGeom.Subset.GetAllGeomSubsets(mesh):
            raise ValueError(f'Material subsets require splitting: {prim.GetPath()}')
        source = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        key = str(source.GetPath())
        if key not in source_meshes:
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if not np.all(counts == 3):
                raise ValueError(f'Non-triangle mesh requires triangulation: {prim.GetPath()}')
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
            if not len(points) or not len(faces):
                continue
            normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float32)
            normal_faces = (_attribute_faces(normals, mesh.GetNormalsInterpolation(), faces)
                            if normals.ndim == 2 and len(normals) else None)
            uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
            texcoords = np.asarray(uv.ComputeFlattened(), dtype=np.float32) if uv else None
            uv_faces = _attribute_faces(texcoords, uv.GetInterpolation(), faces) if uv else None
            source_meshes[key] = points, faces, normals, normal_faces, texcoords, uv_faces
        points, faces, normals, normal_faces, texcoords, uv_faces = source_meshes[key]
        pos, quat, scale = split_transform(transforms.GetLocalToWorldTransform(prim))
        material_name, uv_transform = material_for(prim)
        if material_name in textured_materials and uv_faces is None:
            raise ValueError(f'Textured visual mesh requires st coordinates: {prim.GetPath()}')
        vertices = points.copy()
        # MuJoCo requires four vertices even for a visual-only triangular shell.
        # An unreferenced centroid preserves every rendered triangle and UV index.
        if len(vertices) == 3:
            vertices = np.vstack((vertices, vertices.mean(axis=0)))
        # Keep source winding; instance scale is applied by the renderer.
        flip = mesh.GetOrientationAttr().Get() == 'leftHanded'
        order = [0, 2, 1] if flip else [0, 1, 2]
        fields = {'uservert': vertices.ravel(), 'userface': faces[:, order].ravel()}
        if normal_faces is not None:
            normalized = normals.copy()
            normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
            fields.update(usernormal=normalized.astype(np.float32).ravel(), userfacenormal=normal_faces[:, order].ravel())
        if uv_faces is not None:
            transformed_uv = (np.c_[texcoords, np.ones(len(texcoords))]@uv_transform.T)[:, :2]
            fields.update(usertexcoord=transformed_uv.astype(np.float32).ravel(), userfacetexcoord=uv_faces[:, order].ravel())
        digest = hashlib.sha256()
        for name, values in fields.items():
            digest.update(name.encode()); digest.update(values.tobytes())
        identity = digest.hexdigest()
        if identity not in meshes:
            name = 'campus_mesh_'+str(len(meshes))
            if normal_faces is not None and mesh.GetNormalsInterpolation() == 'faceVarying':
                face_varying_normals[name] = order
            spec.add_mesh(name=name, inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, **fields)
            meshes[identity] = name
            triangles += len(faces)
        spec.worldbody.add_geom(name='campus_geom_'+str(instances), type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname=meshes[identity], pos=pos, quat=quat, material=material_name,
                                contype=0, conaffinity=0, mass=0)
        instance_transforms.append({'geom': 'campus_geom_'+str(instances),
                                    'pos': pos.tolist(), 'quat': quat.tolist(), 'scale': scale.tolist()})
        instances += 1
    if not instances:
        raise ValueError('Scene contains no supported visible geometry')
    return {'instances': instances, 'unique_meshes': len(meshes), 'unique_triangles': triangles,
            'materials': len(materials), 'textures': len(textures), 'visual_gaps': sorted(warnings),
            'instance_transforms': instance_transforms, 'texture_color_spaces': texture_color_spaces,
            'texture_wrap_modes': texture_wrap_modes, 'cutout_materials': cutout_materials,
            'face_varying_normals': face_varying_normals}
