"""Exclude robot collision proxies when authored visual geometry is available."""
import copy

import numpy as np


def render_model(model):
    """Return an independent model; retain collision-only bodies and world ground.

    MJWarp enables groups 0, 1 and 2 by default. Native robot MJCFs put
    collision duplicates in group 0, so group selection alone is insufficient.
    Group 5 is reserved here for hidden proxies in this render-only copy.
    """
    result = copy.copy(model)
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    visible = (model.geom_rgba[:, 3] > 0) & np.isin(model.geom_group, [0, 1, 2])
    visual_bodies = np.unique(model.geom_bodyid[~collision & visible])
    hide = collision & (model.geom_bodyid != 0) & np.isin(model.geom_bodyid, visual_bodies)
    result.geom_group[hide] = 5
    return result
