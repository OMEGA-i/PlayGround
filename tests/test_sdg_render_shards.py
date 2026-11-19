"""A completed episode belongs to exactly one render worker for a fixed shard count."""
import importlib
from pathlib import Path
import uuid

import pytest


@pytest.mark.parametrize('count', [1,2,3,4])
def test_disjoint_complete_render_partition(monkeypatch, count):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/sdg'))
    module = importlib.import_module('watch_render')
    ids = [str(uuid.UUID(int=i)) for i in range(100)]
    partitions = [{uid for uid in ids if module.render_shard(uid,count)==index} for index in range(count)]
    assert set.union(*partitions) == set(ids)
    assert sum(map(len,partitions)) == len(ids)
    assert all(partitions)
    with pytest.raises(ValueError):
        module.render_shard(ids[0],0)
