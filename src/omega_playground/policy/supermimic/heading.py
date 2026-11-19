"""Place chunk-local references in world space; legacy slew is initialization-only."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


class HeadingFrame:
    def __init__(self, xy, heading=0.):
        self.world=np.array(xy,dtype=float)
        self.local=None
        self.heading=heading

    def place(self, reference, target):
        q=reference['qpos'].copy()
        local=q[:,:2].copy()
        delta=np.diff(local,axis=0,prepend=(local[:1] if self.local is None else self.local[None]))
        err=math.atan2(math.sin(target-self.heading),math.cos(target-self.heading))
        yaw=self.heading+np.clip(err,-np.arange(1,len(q)+1)*.012,np.arange(1,len(q)+1)*.012)
        spin=np.diff(yaw,prepend=self.heading)*50
        rot=Rotation.from_euler('z',yaw[:,None])
        delta3=rot.apply(np.column_stack((delta,np.zeros(len(q)))))
        q[:,:2]=self.world+np.cumsum(delta3[:,:2],axis=0)
        q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
        velocity=rot.apply(reference['root_linear_velocity'])
        angular=rot.apply(reference['root_angular_velocity_world'])
        angular[:,2]+=spin
        self.heading,self.local,self.world=yaw[-1],local[-1].copy(),q[-1,:2].copy()
        return dict(reference,qpos=q,root_linear_velocity=velocity,root_angular_velocity_world=angular)



def place_from_proprio(reference, state):
    """Rigid chunk-heading -> tracker-world transform anchored at measured state."""
    def yaw(q):
        w,x,y,z=q
        return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    q=reference['qpos'].copy()
    anchor=q[0].copy()
    rotation=Rotation.from_euler('z',yaw(state[3:7])-yaw(anchor[3:7]))
    translation=state[:3]-rotation.apply(anchor[:3])
    q[:,:3]=rotation.apply(q[:,:3])+translation
    q[:,3:7]=(rotation*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    transform=np.eye(4)
    transform[:3,:3]=rotation.as_matrix()
    transform[:3,3]=translation
    return dict(reference,qpos=q,
                root_linear_velocity=rotation.apply(reference['root_linear_velocity']),
                root_angular_velocity_world=rotation.apply(reference['root_angular_velocity_world']),
                world_from_chunk=transform)
