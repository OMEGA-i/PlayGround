"""GPU pose bridge for one live SuperMimic Tracker and its own independent cameras."""
from functools import partial
from pathlib import Path
import os
import hashlib
import json
from xml.etree import ElementTree as ET


def make_viewer(visual, xml_digest, scene_digest, video, width, height):
    import mujoco
    from omega_playground.viewer.mjwarp_live import MjWarpViewer
    meta = json.loads((visual/'asset.json').read_text())
    if (meta['schema'],meta['xml_sha256'],meta['scene_sha256']) != (
            'omega.supermimic-visual/1',xml_digest,scene_digest):
        raise ValueError('Visual bundle does not match physical robot and scene')
    model = mujoco.MjModel.from_binary_path(str(visual/meta['entry']))
    meta.setdefault('policy_body_names',meta['body_names'])
    return MjWarpViewer(model,meta,visual_directory=visual,width=width,height=height,video=video,camera='chase',
                        video_bitrate_kbps=4000)


class LiveView:
    def __init__(self, session, xml, robot, scene, scene_digest, spawn, video, callback,
                 width=960,height=540,avatar=None,armed=False):
        import torch
        from omega_playground.viewer.process import RenderProcess
        root=Path(__file__).resolve().parents[4]
        visual=root/'artifacts/mjwarp'/f'supermimic-{robot}-{scene}{"-armed" if armed else ""}'
        self.adjust=None
        if avatar in ('ue4','ue5'):
            from .skin_rig import SkinRig,make_skin_viewer
            if robot!='sih2300':raise ValueError('SIH2300 Tracker required for this skin')
            calib=root/'artifacts/skins/sih2300/sih_soma23_var00.xml'
            rig=SkinRig(xml,calib)
            names=rig.body_names
            self.adjust=torch.tensor(rig.adjust,dtype=torch.float32,device=session.device)
            visual=root/'artifacts/mjwarp'/f'supermimic-{avatar}-{scene}{"-armed" if armed else ""}'
            factory=partial(make_skin_viewer,visual,xml,calib,scene_digest,video,width,height)
        else:
            meta=json.loads((visual/'asset.json').read_text())
            names=meta['policy_body_names']
            factory=partial(make_viewer,visual,hashlib.sha256(Path(xml).read_bytes()).hexdigest(),
                            scene_digest,video,width,height)
        model=session.env.cpu_model
        source=[model.body(i).name for i in range(1,model.nbody)]
        self.order=torch.tensor([source.index(n) for n in names],device=session.device)
        contact=ET.parse(session.scene_xml).find('worldbody/geom')
        z=-float(contact.get('pos','0 0 0').split()[2])
        self.offset=torch.tensor([*spawn,z],device=session.device)
        self.renderer=RenderProcess(factory,callback,
                                    python_executable=os.environ.get('OMEGA_RENDER_PYTHON', root.parent/'venvs/pg-mjwarp-clean-001/bin/python'))
        self.renderer.camera_mode='chase'
        self.renderer.follow_body=names.index(model.body(session.env.root_bid).name)
        self.renderer.root_bodies=[self.renderer.follow_body]

    def show(self, session, tick, record=True):
        poses=session.physical_pose().index_select(0,self.order)
        if self.adjust is not None:
            import torch
            a,b=poses[:,3:],self.adjust
            xyz=a[:,3:]*b[:,:3]+b[:,3:]*a[:,:3]+torch.linalg.cross(a[:,:3],b[:,:3])
            w=a[:,3:]*b[:,3:]-(a[:,:3]*b[:,:3]).sum(-1,keepdim=True)
            poses[:,3:]=torch.cat((xyz,w),-1)
        poses[:,:3]+=self.offset
        self.renderer.show(poses,tick/50,record=record)

    def snapshot(self, session, tick):
        """Render the decision state and wait without advancing physics."""
        self.show(session,tick,record=False)
        if self.renderer.latest_sim_time != tick/50:
            raise RuntimeError('Decision camera and proprio physics timestamps differ')
        if not self.renderer.ego_jpeg:
            raise RuntimeError('Decision requires an ego camera frame')
        return dict(physics_step=tick*4, timestamp=tick/50,
                    ego_jpeg=self.renderer.ego_jpeg, main_jpeg=self.renderer.latest_jpeg)
