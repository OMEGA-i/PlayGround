"""The G1 obstacle route must respect the Player acceleration budget after rounding."""
from omega_playground.control.tensor_web_player import TensorWebPlayer


def test_obstacle_route_has_executable_rounded_corner():
    spec=dict(route=[[-6.,-12.5],[6.625,-10.625],[9.,-7.]],
        reference=dict(nominal_speed=.55,corner_radius=1.,slot_offsets=[[0,0]],preserve_path=False))
    player=TensorWebPlayer(spec,'cpu')
    assert 16 < player.reference.describe()['length_m'] < 18
    assert len(player.reference.xy)>300
