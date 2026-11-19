"""GPU world contract: named joints, link-origin state, graph/eager parity, recovery."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU physics only")


@pytest.fixture
def world(tmp_path):
    import mujoco

    from omega_playground.ops.serve import configure_simulation_stream
    from omega_playground.sim.mjwarp_world import MJWarpWorld

    configure_simulation_stream(torch.device("cuda:0"))
    model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".005"/>
      <asset><mesh name="floor" vertex="-5 -5 -.2 5 -5 -.2 -5 5 -.2 5 5 -.2 -5 -5 0 5 -5 0 -5 5 0 5 5 0"/></asset>
      <worldbody><geom type="mesh" mesh="floor"/>
      <body name="root" pos="0 0 1.3"><freejoint/><geom type="sphere" size=".1"/>
      <body name="arm" pos="0 0 -.2"><joint name="hip"/><geom type="capsule" size=".05 .1" pos="0 0 -.1"/>
      <body name="tip" pos="0 0 -.3"><joint name="knee"/><geom type="sphere" size=".1" pos="0 0 -.1"/>
      </body></body></body></worldbody></mujoco>""")
    mujoco.mj_saveModel(model, str(tmp_path / "physics.mjb"))
    pd = {"kp": [20.0, 30.0], "kd": [2.0, 3.0], "effort_limit": [5.0, 8.0]}
    package = SimpleNamespace(
        sha256sum=lambda: "policy",
        physics_hz=200,
        body_names=["tip", "root", "arm"],
        dof_names=["knee", "hip"],
        robot={"anchor_body": "root", "default_root_height": 1.3, "default_dof_pos": [0.2, -0.1]},
        manifest={"pd": pd},
    )
    meta = {
        "schema": "omega.mjwarp-physics/1",
        "policy_sha256": "policy",
        "scene_sha256": "scene",
        "collision_sha256": "collision",
        "entry": "physics.mjb",
        "mujoco_version": mujoco.__version__,
    }
    (tmp_path / "asset.json").write_text(json.dumps(meta))
    (tmp_path / "SHA256SUMS.json").write_text(
        json.dumps(
            {
                name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                for name in ("asset.json", "physics.mjb")
            }
        )
    )
    return MJWarpWorld(
        package,
        tmp_path,
        SimpleNamespace(scene_sha256="scene", collision_sha256="collision"),
        [[0.0, 0.0]],
        "cuda:0",
    )


def test_named_state_and_recovery(world):
    view = world.robot_view()
    np.testing.assert_allclose(view.dof_pos.cpu(), [[0.2, -0.1]], atol=1e-6)
    np.testing.assert_allclose(view.root_pos.cpu(), [[0.0, 0.0, 1.3]], atol=1e-5)
    world.step()
    assert world.physics_steps == 4 and world.sim_time == 0.02
    world.reset_agents([0], [[1.0, 2.0]], [np.pi / 2])
    view = world.robot_view()
    np.testing.assert_allclose(view.root_pos.cpu(), [[1.0, 2.0, 1.3]], atol=1e-5)
    np.testing.assert_allclose(view.root_rot.cpu(), [[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-6)
    assert torch.count_nonzero(view.dof_vel) == 0
    assert world.physics_steps == 4  # Recovery does not rewind the global clock.


def test_graph_matches_four_eager_gpu_steps(world):
    import mujoco_warp as mw
    import warp as wp

    from omega_playground.sim.mjwarp_world import torque_pd

    world.target.add_(0.1)
    world.step()
    actual = world.view("qpos").clone()
    world.reset_agents([0])
    world.target.add_(0.1)
    for _ in range(4):
        wp.launch(
            torque_pd,
            2,
            inputs=[world.data.qpos, world.data.qvel, *world.pd, wp.from_torch(world.target)],
            outputs=[world.data.qfrc_applied],
            device="cuda:0",
        )
        mw.step(world.model, world.data)
    torch.testing.assert_close(world.view("qpos"), actual, atol=2e-6, rtol=1e-6)


def test_gpu_ground_missing_and_below(world):
    assert np.isnan(world.ground.heights([[50.0, 50.0]], allow_missing=True)[0])
    with pytest.raises(ValueError, match="No authored ground"):
        world.ground.heights([[50.0, 50.0]])
    np.testing.assert_allclose(world.ground.heights([[0.0, 0.0]], below=[0.5]), [0.0], atol=1e-6)


def test_nonfoot_contact_is_not_alone_a_fall(world):
    import warp as wp

    from omega_playground.eval.mjwarp_safety import MJWarpSafety

    safety = MJWarpSafety(world)
    bodies = world.cpu_model.geom_bodyid
    static = int(np.flatnonzero(bodies == 0)[0])
    robot = int(np.flatnonzero(bodies != 0)[0])
    wp.to_torch(world.data.nacon).fill_(1)
    wp.to_torch(world.data.contact.geom)[0] = torch.tensor([static, robot], device="cuda:0")
    wp.to_torch(world.data.contact.dist)[0] = -0.001
    assert safety.sample() is None
    assert safety.sample() is None
    assert safety.sample()["code"] == "nonfoot_contact"
