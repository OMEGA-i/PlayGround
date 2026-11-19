# Adapted from ProtoMotions ADD authored SOMA23 exporter.
"""Preserve ASE visual rigs for SOMA23 motion retargeting; export no physics or policy data."""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rest_skeleton(xml):
    rest = {}
    def visit(node, origin):
        if node.get('quat') or node.get('euler'):
            raise ValueError('This SOMA23 exporter requires identity rest body rotations')
        point = origin + np.fromstring(node.get('pos', '0 0 0'), sep=' ')
        rest[node.attrib['name']] = point
        for child in node.findall('body'): visit(child, point)
    for body in ET.parse(xml).findall('./worldbody/body'): visit(body, np.zeros(3))
    assert len(rest) == 23 and 'Hips' in rest
    return rest


def segment_transform(source, target):
    a, b = np.linalg.norm(source), np.linalg.norm(target)
    if min(a,b) < 1e-8: return np.eye(3)
    u,v=source/a,target/b
    cross=np.cross(u,v); angle=np.arccos(np.clip(u@v,-1,1))
    if np.linalg.norm(cross)<1e-8:
        cross=np.cross(u,np.eye(3)[np.argmin(abs(u))])
    rotation=Rotation.from_rotvec(cross/np.linalg.norm(cross)*angle).as_matrix()
    return rotation @ (np.eye(3)+(b/a-1)*np.outer(u,u))


def native_basis(bind, names, rest):
    """Map either native or pre-calibrated source yaw into the SOMA frame."""
    lateral = bind[names.index('left_upper_arm'), :3, 3]-bind[names.index('right_upper_arm'), :3, 3]
    desired = rest['LeftArm']-rest['RightArm']
    if np.linalg.norm(lateral[:2]) < 1e-6 or abs(lateral[2]) > .05:
        raise ValueError('Native source requires a horizontal left/right shoulder axis')
    yaw = np.arctan2(desired[1], desired[0])-np.arctan2(lateral[1], lateral[0])
    return Rotation.from_euler('z', yaw).as_matrix()


def export(source, output, xml, replace=False):
    robot=(source/'robot_skin.npz').exists()
    origin=source/('robot_skin.npz' if robot else 'manny.npz')
    data=np.load(origin,allow_pickle=False)
    rest=rest_skeleton(xml); names=list(rest); target=np.array(list(rest.values()))
    src_names=data['names'].tolist()
    basis = native_basis(data['bind'], src_names, rest) if robot else np.eye(3)
    bind=data['bind'][:,:3,3] @ basis.T
    anchors={'Hips':'pelvis','Head':'head'}
    if robot:
        anchors['Chest']='torso'
        if 'neck' in src_names:anchors['Neck1']='neck'
    else:
        spines=[n for n in src_names if n.startswith('spine_')]
        # SOMA23 Chest starts the long ribcage segment, not the uppermost
        # sternum joint. Manny's extra spine_04/05 follow that Chest segment.
        anchors.update(Spine1=spines[0],Spine2=spines[1],Chest=spines[2],Neck1='neck_01')
        if 'neck_02' in src_names: anchors['Neck2']='neck_02'
    for side,word,suffix in [('Left','left','l'),('Right','right','r')]:
        roles={'Arm':'upper_arm','ForeArm':'lower_arm','Hand':'hand','Leg':'thigh','Shin':'shin','Foot':'foot'} if robot else {
            'Shoulder':'clavicle','Arm':'upperarm','ForeArm':'lowerarm','Hand':'hand','Leg':'thigh','Shin':'calf','Foot':'foot','ToeBase':'ball'}
        for body,bone in roles.items(): anchors[side+body]=f'{word}_{bone}' if robot else f'{bone}_{suffix}'
    # Preserve authored geometry, weights and bind rotations (ASE's contract).
    # Only change the coordinate basis; motion retargeting happens at runtime.
    full_bind=data['bind'].copy()
    full_bind[:,:3,:3]=basis@full_bind[:,:3,:3]
    full_bind[:,:3,3]=bind
    if robot:
        parent_names={'pelvis':None,'torso':'pelvis','head':'neck' if 'neck' in src_names else 'torso'}
        if 'neck' in src_names:parent_names['neck']='torso'
        for side in ['left','right']:
            for bone,parent in [('upper_arm','torso'),('lower_arm',side+'_upper_arm'),
                                ('hand',side+'_lower_arm'),('thigh','pelvis'),
                                ('shin',side+'_thigh'),('foot',side+'_shin')]:
                parent_names[side+'_'+bone]=parent
        parents=np.array([src_names.index(parent_names[n]) if parent_names[n] else -1 for n in src_names])
    else:
        parents=data['parents']
    result={'names': data['names'],'bind': full_bind,'parents': parents,
                'soma_names': np.array(names),'soma_rest': target,
                'mapping': np.array([anchors.get(n,'') for n in names]),
                'slots': data['slots'],'format_version': np.array(2)}
    source_metadata=json.loads((source/'skin.json').read_text()) if (source/'skin.json').exists() else {}
    result['native_leg_frames']=np.array(source_metadata.get('kind')=='native_robot_visual')
    if 'semantic_frames' in data:
        result['semantic_frames']=basis@data['semantic_frames']
        result['semantic_names']=data['semantic_names']
    if output.exists():
        if not replace:raise FileExistsError(output)
        previous=json.loads((output/'skin.json').read_text())
        if not previous.get('render_only') or previous['source_sha256']!=hashlib.sha256(origin.read_bytes()).hexdigest():
            raise ValueError('Refusing to replace unrelated skin assets')
    output.mkdir(parents=True,exist_ok=replace)
    stats=[]
    for slot in range(int(data['slots'])):
        points=data[f's{slot}_points']@basis.T; normals=data[f's{slot}_normals']@basis.T
        for key,value in {'points': points,'normals': normals,
                              'joints': data[f's{slot}_joints'],'weights': data[f's{slot}_weights'],
                              'indices': data[f's{slot}_indices'],'uv': data[f's{slot}_uv']}.items():
            result[f's{slot}_{key}']=value
        shutil.copyfile(source/f'basecolor_{slot}.png',output/f'basecolor_{slot}.png')
        for channel in ['normal','roughness','metallic']:
            texture=source/f'{channel}_{slot}.png'
            if texture.exists():shutil.copyfile(texture,output/texture.name)
        stats.append({'vertices': len(points),'influences': data[f's{slot}_weights'].shape[1]})
    np.savez_compressed(output/'skin.npz',**result)
    metadata={'source': str(source),'source_sha256': hashlib.sha256(origin.read_bytes()).hexdigest(),
                  'skeleton_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),'render_only': True,'slots': stats,
                  'format_version': 2,'retarget': "authored_rig_fk",'source_type': 'rigid_robot_shell' if robot else 'weighted_character','anchors': anchors,
                  'source_to_soma_basis': basis.tolist()}
    (output/'skin.json').write_text(json.dumps(metadata,indent=2))
    metadata.update(name='soma23-'+output.name, appearance=output.name, kind='soma23',
                    rigid_robot=robot)
    (output/'skin.json').write_text(json.dumps(metadata,indent=2))
    sums={str(f.relative_to(output)):hashlib.sha256(f.read_bytes()).hexdigest()
          for f in output.rglob('*') if f.is_file() and f.name != 'SHA256SUMS.json'}
    (output/'SHA256SUMS.json').write_text(json.dumps(sums,indent=2))
    print(json.dumps({'name':metadata['name'],'slots':len(stats)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--xml',type=Path,required=True)
    p.add_argument('--replace-generated',action='store_true')
    a=p.parse_args();export(a.source,a.output,a.xml,a.replace_generated)
