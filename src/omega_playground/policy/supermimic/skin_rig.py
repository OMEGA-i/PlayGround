"""Map physically distinct SIH2300 link frames to the authored UE skin calibration."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation


class SkinRig:
    def __init__(self,xml,calibration):
        import mujoco
        self.xml,self.asset_path=Path(xml),Path(calibration)
        tables=[]
        for path in (self.xml,self.asset_path):
            model=mujoco.MjModel.from_xml_path(str(path))
            data=mujoco.MjData(model)
            mujoco.mj_kinematics(model,data)
            tables.append({model.body(i).name:(data.xpos[i].copy(),data.xmat[i].reshape(3,3).copy()) for i in range(1,model.nbody)})
        physical,authored=tables
        if set(physical)!=set(authored):raise ValueError('Skin bones and physical skeleton differ')
        self.body_names=list(physical)
        def lateral_yaw(table):
            left=table['LeftArm'][0]-table['RightArm'][0]
            return np.arctan2(left[1],left[0])
        basis=Rotation.from_euler('z',lateral_yaw(physical)-lateral_yaw(authored)).as_matrix()
        self.adjust=Rotation.from_matrix(np.stack([physical[n][1].T@basis@authored[n][1] for n in self.body_names])).as_quat()
        # The same anatomical skeleton is expressed in two bind frames.
        for n in self.body_names:
            a=authored[n][0]-authored['Hips'][0]
            b=physical[n][0]-physical['Hips'][0]
            if np.linalg.norm(b-basis@a)>.03:raise ValueError(f'{n}: skin rest joint differs from SIH2300')
    def sha256sum(self):
        return hashlib.sha256(self.xml.read_bytes()).hexdigest()


def make_skin_viewer(visual,xml,calibration,scene_digest,video,width,height):
    from omega_playground.viewer.mjwarp_skin_asset import load_prepared_skin
    from omega_playground.viewer.mjwarp_live import MjWarpViewer
    import os
    os.environ['OMEGA_SOMA_GPU_SKIN']='1'
    rig=SkinRig(xml,calibration)
    meta=json.loads((visual/'asset.json').read_text())
    model,painter,bridge,meta=load_prepared_skin(visual,rig,scene_sha256=scene_digest,armed=meta['armed'])
    return MjWarpViewer(model,meta,visual_directory=visual,skin=(painter,bridge),width=width,height=height,
                        video=video,camera='chase',video_bitrate_kbps=4000)
