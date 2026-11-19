"""Render saved Joystick references as FK animation, explicitly not physics."""
import argparse
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sdg'))
from render_episode import Encoder
from omega_playground.viewer.mjwarp import MjWarpPoseRenderer
from omega_playground.viewer.gpu_pixels import Rgb8Readback
from render_plane import metric_frame


def render(root, robot):
    folder=root/'layer2'/robot
    tree=ET.parse(root/'layer1'/robot/'native/physics.xml')
    world=tree.getroot().find('worldbody')
    ET.SubElement(world,'camera',name='reference_camera',pos='0 -4 2',fovy='45')
    ET.SubElement(world,'light',name='reference_light',pos='0 0 6',dir='0 0 -1',directional='true',diffuse='.8 .8 .8')
    tree.write(folder/'render.xml')
    model=mujoco.MjModel.from_xml_path(str(folder/'render.xml'))
    cases=sorted(folder.glob('frame1000-seed7-*'))
    with np.load(cases[0]/'kinematics.npz') as data:names=data['body_names'].tolist()
    renderer=MjWarpPoseRenderer(model,names,width=960,height=540)
    pixels=Rgb8Readback((1,540,960),'cuda:0');encoder=Encoder(folder/'references.mp4',960,540)
    timeline=[];frame=0
    try:
        for case in cases:
            command=json.loads((case/'command.json').read_text())
            with np.load(case/'kinematics.npz') as data:poses=data['body_q'].copy()
            timeline.append(dict(first_video_frame=frame,case=case.name,command=command))
            for i in range(0,len(poses)-1,2):
                body=poses[i];target=body[0,:3]
                renderer.look_at(target+[-3.,-3.,1.5],target)
                im=Image.fromarray(pixels.read(renderer.render(body))[0].copy())
                label=f"{robot} REFERENCE ONLY (FK, no physics) | {command['name']} | {i/50:.2f}s"
                ImageDraw.Draw(im).text((12,12),label,fill='white',stroke_width=1,stroke_fill='black')
                encoder.write(np.asarray(im));frame+=1
                if i==60:
                    im.save(case/'reference-midpoint.jpg')
                    metric_frame(renderer,pixels,model,body,target,960,540).save(case/'qc-metric.jpg')
    finally:encoder.close()
    (folder/'video-timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('robot')
    args=p.parse_args();render(args.root,args.robot)
