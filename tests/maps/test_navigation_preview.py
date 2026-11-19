import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image

from omega_playground.map.navigation import CampusNavigation
from omega_playground.map.preview import inspect_path, occupancy_png


def navigation(tmp_path):
    blocked = np.zeros((8,8), dtype=bool)
    blocked[3,3] = True
    support = np.zeros((8,8), dtype=np.float32)
    clearance = np.full((8,8), 3., dtype=np.float32)
    clearance[3,3] = -.1
    clearance[4,3] = .4
    support[7,0] = -np.inf
    clearance[7,0] = -.1
    np.savez(tmp_path/'layers.npz', support_height=support, clearance=clearance,
             collision_blocked=blocked, road_mask=np.zeros_like(blocked))
    (tmp_path/'map.json').write_text(json.dumps({'schema': 'omega.map/3','scene_sha256': 'scene',
        'collision_sha256': 'collision','bounds': [0,0,8,8],'resolution': 1}))
    return CampusNavigation(tmp_path)


def test_preview_checks_segment_interior_and_reports_world_coordinates(tmp_path):
    nav=navigation(tmp_path)
    result=inspect_path(nav,[[1,3.5],[6,3.5]])
    assert not result['valid']
    assert 3 <= result['invalid_points'][0][0] < 4
    assert '0.6 m' in result['message']
    assert inspect_path(nav,[[1,1],[6,1]])['valid']
    assert not inspect_path(nav,[[3.5,4.1],[3.5,4.9]])['valid']


def test_overlay_north_up_and_distinguishes_obstacle_from_clearance(tmp_path):
    nav=navigation(tmp_path)
    pixels=np.array(Image.open(BytesIO(occupancy_png(nav))))
    assert pixels[7-3,3].tolist()==[244,78,78,135]
    assert pixels[7-4,3].tolist()==[245,170,55,100]
    assert pixels[0,0,3] > 0
    assert pixels[6,1,3] == 0


def test_server_preview_and_overlay_use_same_map(tmp_path):
    import urllib.request

    from omega_playground.ops.server import OperationsServer
    nav=navigation(tmp_path)
    server=OperationsServer(SimpleNamespace(directory=tmp_path,meta=nav.meta),'scene',port=0)
    server.start()
    base=f'http://127.0.0.1:{server.http.server_port}'
    try:
        req=urllib.request.Request(base+'/api/drawn-path',json.dumps({'keypoints':[[1,3.5],[6,3.5]]}).encode(),
                                   headers={'X-Omega-Token':server.token})
        with urllib.request.urlopen(req) as r:plan=json.load(r)
        assert not plan['validation']['valid']
        assert plan['route']
        with urllib.request.urlopen(base+'/occupancy.png') as r:assert r.read()==occupancy_png(nav)
    finally:
        server.close()
