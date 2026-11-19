from omega_playground.map.navigation import campus_roads

SCENE = '752186cd58cc14179944a4a7c4e2f3b3c0b31887fcd0373b3928801237a48770'
LAYOUT = {'site_bounds': [-100, 190, -200, 144],
          'routes': [{'id': 'sports', 'width': 7, 'points': [[23, -193], [23, -4]]}]}


def test_perimeter_requires_exact_frozen_scene():
    roads, bounds, limit = campus_roads(LAYOUT, 'different-scene')
    assert roads == [{'id': 'sports', 'width': 7, 'points': [[23, 193], [23, 4]]}]
    assert bounds == [-100, -144, 190, 200]
    assert limit == 4


def test_perimeter_covers_both_carriageways_and_gate_extensions():
    roads, bounds, limit = campus_roads(LAYOUT, SCENE)
    indexed = {r['id']: r for r in roads}
    assert len(indexed) == len(roads)
    assert len([name for name in indexed if not name.startswith('walk-')]) == 19
    assert bounds == [-130, -172, 222, 229]
    assert limit == 4
    for side in ['north', 'south', 'west', 'east']:
        for lane in ['inner', 'outer']:
            assert f'boulevard-{side}-{lane}' in indexed
    for name in ['gate-north', 'gate-south']:
        assert name in indexed
    for road in roads:
        for x, y in road['points']:
            assert bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]


def test_marked_paths_join_existing_sports_and_north_roads():
    roads, _, _ = campus_roads(LAYOUT, SCENE)
    indexed = {r['id']: r for r in roads}
    for name, y in [('north', 111), ('middle', 84), ('south', 56)]:
        assert indexed['walk-court-'+name]['points'] == [[23, y], [80, y]]
        join = indexed[f'walk-court-park-{y}']['points']
        assert join[0] == [80, y]
        assert join[-1] in indexed['walk-park-meander']['points']
    assert indexed['walk-park-meander']['points'][-1] == [84, 4]
    assert indexed['walk-north-garden-bridge']['points'] == [[23, 157], [65, 157]]


def test_repaired_package_identity_enables_same_canonical_network():
    repaired = '5e33ec9ae98d6765ba0c07ecb26987ba29b97b283c0b9417c89bdabea7618415'
    assert campus_roads(LAYOUT, repaired) == campus_roads(LAYOUT, SCENE)


def test_october_package_retains_the_canonical_network():
    october = "0f26d54e0df6b65e590e39fd734e0542aa964f6368d5637526d432ae9baaecc8"
    assert campus_roads(LAYOUT, october) == campus_roads(LAYOUT, SCENE)


def test_entrance_candidate_retains_surveyed_walkway_network():
    candidate = '12d5918e2bc6b27571d64acb0eafeb562c94d7b1d6b376e1b368bc7ab9c767bd'
    assert campus_roads(LAYOUT, candidate) == campus_roads(LAYOUT, SCENE)
