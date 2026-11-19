"""Synchronized native/Playground native-plane replay (render-only camera/light)."""
import argparse
from pathlib import Path
import sys
import math
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sdg'))
from render_episode import Encoder
from omega_playground.viewer.mjwarp import MjWarpPoseRenderer
from omega_playground.viewer.gpu_pixels import Rgb8Readback


def metric_frame(renderer,pixels,model,body,target,width,height):
    target=target.copy();target[2]=1.25
    distance=1.25/math.tan(math.radians(float(model.cam_fovy[0]))/2)
    renderer.look_at(target+np.array([-.8660254037844386,-.5,0.])*distance,target)
    im=Image.fromarray(pixels.read(renderer.render(body))[0].copy());draw=ImageDraw.Draw(im)
    for z in np.arange(0,2.51,.5):
        row=max(0,min(height-1,round((height-1)/2-height/2*(z-1.25)/1.25)))
        draw.line([(0,row),(width-1,row)],fill=(60,220,220))
        draw.text((5,max(0,row-12)),f'{z:.1f} m',fill='white',stroke_width=1,stroke_fill='black')
    return im


def render(folder):
    tree=ET.parse(folder/'native/physics.xml')
    world=tree.getroot().find('worldbody')
    ET.SubElement(world,'camera',name='acceptance_camera',pos='0 -4 2',fovy='45')
    ET.SubElement(world,'light',name='acceptance_light',pos='0 0 6',dir='0 0 -1',directional='true',diffuse='.8 .8 .8')
    tree.write(folder/'render.xml')
    model=mujoco.MjModel.from_xml_path(str(folder/'render.xml'))
    data=[]
    for name in ('native','playground'):
        with np.load(folder/name/'body_states.npz') as archive:
            data.append(archive['body_q'].copy());names=archive['body_names'].tolist()
    renderer=MjWarpPoseRenderer(model,names,width=640,height=480)
    pixels=Rgb8Readback((1,480,640),'cuda:0')
    encoder=Encoder(folder/'comparison.mp4',1280,480)
    try:
        for i in range(0,min(map(len,data)),2):
            panels=[]
            target=data[0][i,0,:3]
            for name,poses in zip(('SMK native evaluator','Playground adapter'),data):
                renderer.look_at(target+[-3.,-3.,1.5],target)
                im=Image.fromarray(pixels.read(renderer.render(poses[i]))[0].copy())
                ImageDraw.Draw(im).text((12,12),f'{name} | {i/50:.2f}s',fill='white',stroke_width=1,stroke_fill='black')
                panels.append(np.asarray(im))
            frame=np.concatenate(panels,axis=1);encoder.write(frame)
            if i in (0,250,500):Image.fromarray(frame).save(folder/f'comparison-{i:04d}.jpg')
            if i==250:
                for name,poses in zip(('native','playground'),data):
                    metric_frame(renderer,pixels,model,poses[i],poses[i,0,:3],640,480).save(folder/f'qc-metric-{name}.jpg')
    finally:
        encoder.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    render(p.parse_args().folder)
