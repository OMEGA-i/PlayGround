import mujoco
import numpy as np
import pytest
from pxr import Usd, UsdGeom

from omega_playground.viewer.usd_scene import add_usd_scene, split_transform


def test_visual_import_reuses_geometry_and_ignores_hidden_collision(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path/'scene.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    for i in range(2):
        mesh = UsdGeom.Mesh.Define(stage, f'/mesh{i}')
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3]*4)
        mesh.CreateFaceVertexIndicesAttr(faces)
        UsdGeom.Xformable(mesh).AddTranslateOp().Set((i*3, 0, 0))
    hidden = UsdGeom.Cube.Define(stage, '/collider')
    hidden.CreateVisibilityAttr('invisible')
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    report = add_usd_scene(spec, tmp_path/'scene.usda', tmp_path/'textures')
    model = spec.compile()
    assert report['instances'] == 2 and report['unique_meshes'] == 1
    assert model.nmesh == 1 and model.ngeom == 2
    assert np.all(model.geom_contype == 0)
    np.testing.assert_allclose(model.geom_pos[1]-model.geom_pos[0], [3, 0, 0])


def test_transform_reflection_and_shear():
    matrix = np.diag([-2., 3., 4., 1.])
    matrix[3, :3] = [5, 6, 7]
    pos, quat, scale = split_transform(matrix)
    np.testing.assert_allclose(pos, [5, 6, 7])
    np.testing.assert_allclose(scale, [-2, 3, 4])
    np.testing.assert_allclose(quat, [1, 0, 0, 0])
    matrix[0, 1] = .5
    with pytest.raises(ValueError, match='Sheared'):
        split_transform(matrix)


def test_texture_transform_bakes_scale_rotation_translation():
    from pxr import Sdf, UsdShade

    from omega_playground.viewer.usd_scene import texture_transform
    stage = Usd.Stage.CreateInMemory()
    reader = UsdShade.Shader.Define(stage, '/reader')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(stage, '/transform')
    transform.CreateIdAttr('UsdTransform2d')
    transform.CreateInput('in', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    transform.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set((2, 3))
    transform.CreateInput('rotation', Sdf.ValueTypeNames.Float).Set(90)
    transform.CreateInput('translation', Sdf.ValueTypeNames.Float2).Set((4, 5))
    matrix = texture_transform((transform.ConnectableAPI(), 'result', None))
    np.testing.assert_allclose(matrix@np.array([1, 1, 1]), [1, 7, 1], atol=1e-7)


def test_single_triangle_remains_one_triangle(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path/'triangle.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, '/triangle')
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    add_usd_scene(spec, tmp_path/'triangle.usda', tmp_path/'textures')
    model = spec.compile()
    assert model.mesh_facenum[0] == 1
    assert model.mesh_vertnum[0] == 4


@pytest.mark.parametrize('texture_scale, expected_rgb', [
    (None, [1., 1., 1.]),
    ((.2, .4, .6, .1), [.2, .4, .6]),
])
def test_connected_diffuse_replaces_fallback_and_preserves_opacity(tmp_path, texture_scale, expected_rgb):
    from PIL import Image
    from pxr import Sdf, UsdShade

    Image.new('RGB', (2, 2), (160, 120, 80)).save(tmp_path/'color.png')
    stage = Usd.Stage.CreateNew(str(tmp_path/'material.usda'))
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, '/mesh')
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(
        [(0., 0.), (1., 0.), (0., 1.)])
    material = UsdShade.Material.Define(stage, '/material')
    shader = UsdShade.Shader.Define(stage, '/material/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    diffuse = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
    diffuse.Set((.1, .2, .3))
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.7)
    texture = UsdShade.Shader.Define(stage, '/material/texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./color.png'))
    if texture_scale is not None:
        texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(texture_scale)
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).Set((0., 0.))
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    diffuse.ConnectToSource(texture.ConnectableAPI(), 'rgb')
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    report = add_usd_scene(spec, tmp_path/'material.usda', tmp_path/'textures')
    model = spec.compile()
    np.testing.assert_allclose(model.mat_rgba[0], [*expected_rgb, .7], atol=1e-7)
    assert report['textures'] == 1
    assert model.mat_texid[0, mujoco.mjtTextureRole.mjTEXROLE_RGB] >= 0


def test_textured_mesh_without_uv_is_rejected_before_gpu_upload(tmp_path):
    test_connected_diffuse_replaces_fallback_and_preserves_opacity(tmp_path, None, [1., 1., 1.])
    stage = Usd.Stage.Open(str(tmp_path/'material.usda'))
    stage.GetPrimAtPath('/mesh').RemoveProperty('primvars:st')
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match='requires st coordinates'):
        add_usd_scene(mujoco.MjSpec(), tmp_path/'material.usda', tmp_path/'textures')


@pytest.mark.parametrize('encoding', ['sRGB', 'raw', 'auto'])
def test_texture_encoding_is_preserved_without_guessing(tmp_path, encoding):
    from pxr import Sdf, UsdShade
    test_connected_diffuse_replaces_fallback_and_preserves_opacity(tmp_path, None, [1., 1., 1.])
    stage = Usd.Stage.Open(str(tmp_path/'material.usda'))
    texture = UsdShade.Shader(stage.GetPrimAtPath('/material/texture'))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(encoding)
    stage.GetRootLayer().Save()
    spec = mujoco.MjSpec()
    report = add_usd_scene(spec, tmp_path/'material.usda', tmp_path/'textures')
    model = spec.compile()
    texture_id = model.mat_texid[0, mujoco.mjtTextureRole.mjTEXROLE_RGB]
    assert report['texture_color_spaces'][model.tex(int(texture_id)).name] == encoding


def test_same_texture_bytes_keep_distinct_encoding_bindings(tmp_path):
    from pxr import Sdf, UsdShade
    test_connected_diffuse_replaces_fallback_and_preserves_opacity(tmp_path, None, [1., 1., 1.])
    stage = Usd.Stage.Open(str(tmp_path/'material.usda'))
    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, '/material', layer, '/raw_material')
    Sdf.CopySpec(layer, '/mesh', layer, '/raw_mesh')
    for path, encoding in [('/material', 'sRGB'), ('/raw_material', 'raw')]:
        texture = UsdShade.Shader(stage.GetPrimAtPath(path+'/texture'))
        texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(encoding)
        shader = UsdShade.Shader(stage.GetPrimAtPath(path+'/surface'))
        shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(), 'rgb')
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        material.GetSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/raw_mesh')).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/raw_material')))
    layer.Save()
    spec = mujoco.MjSpec()
    report = add_usd_scene(spec, tmp_path/'material.usda', tmp_path/'textures')
    model = spec.compile()
    assert report['textures'] == 2
    assert set(report['texture_color_spaces'].values()) == {'sRGB', 'raw'}
    assert len(set(model.mat_texid[:, mujoco.mjtTextureRole.mjTEXROLE_RGB])) == 2


def test_same_image_preserves_distinct_sampler_bindings(tmp_path):
    from pxr import Sdf, UsdShade
    test_connected_diffuse_replaces_fallback_and_preserves_opacity(tmp_path, None, [1., 1., 1.])
    stage = Usd.Stage.Open(str(tmp_path/'material.usda'))
    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, '/material', layer, '/clamp_material')
    Sdf.CopySpec(layer, '/mesh', layer, '/clamp_mesh')
    for path, mode in [('/material', 'repeat'), ('/clamp_material', 'clamp')]:
        texture = UsdShade.Shader(stage.GetPrimAtPath(path+'/texture'))
        for axis in ('wrapS', 'wrapT'):
            texture.CreateInput(axis, Sdf.ValueTypeNames.Token).Set(mode)
        shader = UsdShade.Shader(stage.GetPrimAtPath(path+'/surface'))
        shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(), 'rgb')
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        material.GetSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/clamp_mesh')).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/clamp_material')))
    layer.Save()
    spec = mujoco.MjSpec()
    report = add_usd_scene(spec, tmp_path/'material.usda', tmp_path/'textures')
    model = spec.compile()
    assert report['textures'] == 2
    assert set(map(tuple, report['texture_wrap_modes'].values())) == {('repeat', 'repeat'), ('clamp', 'clamp')}
    assert len(set(model.mat_texid[:, mujoco.mjtTextureRole.mjTEXROLE_RGB])) == 2
    assert not any('wrap' in warning or 'addressing' in warning for warning in report['visual_gaps'])
