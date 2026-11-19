from types import SimpleNamespace

import numpy as np

from omega_playground.viewer import mjwarp_live


def test_live_skin_receives_authoritative_pose_before_render(monkeypatch):
    events = []
    visual = np.array([[4., 5., 6., 0., 0., 0., 1.]])
    physical = np.array([[1., 2., 3., 0., 0., 0., 1.]])
    original = physical.copy()

    class Renderer:
        def __init__(self, *args, **kwargs):
            assert kwargs['backface_culling'] is False
            assert kwargs['skin_meshes'] is True
        def look_at(self, *args):
            pass
        def render(self, poses):
            events.append('render')
            np.testing.assert_equal(poses, visual)
            return SimpleNamespace(numpy=lambda: np.zeros((1, 64, 64, 3)))

    class Painter:
        def log(self, bridge, poses):
            events.append('skin')
            np.testing.assert_equal(poses, original)

    class Bridge:
        def attach(self, renderer):
            events.append('attach')
        def refit(self):
            events.append('refit')
        def poses(self):
            return visual

    # This tests call ordering, not GPU readback (covered by test_gpu_pixels).
    monkeypatch.setattr(mjwarp_live, 'Rgb8Readback', lambda *args, **kwargs: SimpleNamespace(
        read=lambda frame: (frame.numpy()*255).astype(np.uint8)))
    monkeypatch.setattr(mjwarp_live, 'MjWarpPoseRenderer', Renderer)
    viewer = mjwarp_live.MjWarpViewer(SimpleNamespace(cam_fovy=[45.]),
        {'policy_body_names': ['skin'], 'body_names': ['skin'], 'armed': True}, width=64, height=64, skin=(Painter(), Bridge()))
    viewer.frame_sink = lambda *args: None
    viewer.show(physical, .5)
    assert events == ['attach', 'skin', 'refit', 'render']
    np.testing.assert_equal(physical, original)
    assert viewer.frames == 1
