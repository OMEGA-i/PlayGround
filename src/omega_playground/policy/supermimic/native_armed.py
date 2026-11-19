"""Prepare named palm sockets from the native collision and appearance geometry offline."""
import numpy as np

from omega_playground.viewer.mjwarp_armed import add_prop_meshes


def bake(spec,model):
    import mujoco
    names={model.body(i).name.lower():model.body(i).name for i in range(1,model.nbody)}
    humanoid='lefthand' in names
    sockets={'forearm':names['leftforearm' if humanoid else 'left_elbow_link']}
    centers={}
    data=mujoco.MjData(model);mujoco.mj_kinematics(model,data)
    for side,word in [('l','left'),('r','right')]:
        key=word+'hand' if humanoid else word+'_wrist_yaw_link'
        name=names[key];w=model.body(name).id
        hand=model.body(names[word+'_rubber_hand_link']).id if word+'_rubber_hand_link' in names else w
        candidates=np.flatnonzero(model.geom_bodyid==hand)
        if not len(candidates):raise ValueError(f'No calibrated hand geometry for {name}')
        points=(data.geom_xpos[candidates]-data.xpos[w])@data.xmat[w].reshape(3,3)
        sign=-1 if humanoid and side=='r' else 1
        center=points[np.argmax(sign*points[:,0])].copy()
        if not humanoid:center[1]+=-.022 if side=='l' else .022
        centers[side]=center.tolist()
        sockets[side]=dict(name=name,rotation=np.diag([sign,sign,1]).tolist())
    return {**add_prop_meshes(spec,1),'kind':'supermimic_native','centers':centers,'sockets':sockets}
