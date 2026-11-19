"""Full-horizon Joystick inference, optionally conditioned on executed state history."""
import io
import json
from pathlib import Path

import numpy as np
import torch

from .reference import native_reference


class RollingJoystick:
    def __init__(self, workspace, assets, robot, xml, pins):
        from joystick.evaluate import verify_contract
        from joystick.representation import make_representation
        from joystick.selection import sha256
        from joystick.source import activate
        from joystick.stable_geometry import install
        from joystick.train import build_model
        from joystick.rolling import DEFAULT_HORIZON
        install()
        from motionbricks.motion_backbone.inference.motion_inference import motion_inference
        self.device = 'cuda:0'
        self.robot, self.xml = robot, Path(xml)
        self.xml_sha = sha256(self.xml)
        self.horizon = DEFAULT_HORIZON
        self.features = Path(assets)/'datasets/bones-seed-25fps'/robot
        run = Path(assets)/'runs'/robot/'fsq100k_25fps'
        manifest = json.loads((self.features/'manifest.json').read_text())
        rep = self.rep = make_representation(xml, self.features/'stats', embodiment=robot)
        tokenizer = assets / pins['tokenizer']['path']
        if sha256(tokenizer) != pins['tokenizer']['sha256']:
            raise ValueError('Joystick tokenizer digest mismatch')
        models = {}
        for stage in ('root','pose'):
            path = assets / pins[stage]['path']
            if sha256(path) != pins[stage]['sha256']:
                raise ValueError(f'Joystick {stage} digest mismatch')
            ck = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            audit = ck['joystick_audit']
            verify_contract(audit,manifest,self.features,self.xml,tokenizer)
            model,_ = build_model(activate(),stage,rep,audit['steps'],
                                  tokenizer=tokenizer if stage=='pose' else None,
                                  preset='official',schedule_steps=audit['schedule_steps'])
            model.load_state_dict(ck['state_dict'],strict=True)
            model.eval().to(self.device)
            model.move_supporting_nets_to_device(self.device)
            models[stage] = model
        class Engine(motion_inference):
            def _decode_motions_from_predicted_root_and_pose_tokens(self,batch,config,info):
                local,glob=super()._decode_motions_from_predicted_root_and_pose_tokens(
                    batch,dict(config,final_root_pred_mode='from_pose_module'),info)
                glob[...,self.global_motion_rep.indices['root']]=self.global_motion_rep.unnormalize(batch['pred_global_root_values'])
                local[...,self.local_motion_rep.indices['root']]=self.local_motion_rep.unnormalize(batch['pred_local_root_values'])
                return local,glob

            def _reapply_initial_root_info(self,batch):
                batch['pred_global_poses'][...,self.global_motion_rep.indices['global_root_pos_2d']]+=batch['reference_start_root_global_offsets']
                return batch['pred_global_poses']
        self.engine=Engine(models,models['pose'].args,self.device)
        rows={r['motion_id']:r for r in (json.loads(line) for line in (self.features/'catalog.jsonl').read_text().splitlines()) if r['split']=='train'}
        bank=json.loads((Path(workspace)/'configs/soma_primitives_25fps.json').read_text())['primitives']
        self.prompts={}
        g=self.engine.global_motion_rep
        for name,entry in bank.items():
            row=rows[entry['source_id']]
            data=np.load(self.features/row['feature_file'],mmap_mode='r')
            def read(start,count):
                offset=row['offset']+start
                x=torch.tensor(data[offset:offset+count].copy(),device=self.device)
                x=x[:,rep.dual_rep.indices['global_rep']][None]
                return g.change_first_heading(x,0.,False,False)
            phases=[p for p in entry['phases'] if p['name'] in ('left_support','right_support')] or entry['phases']
            self.prompts[name]=(read(entry['window_start25'],4),[read(p['frame25'],5) for p in phases])
        self.reset()

    def reset(self):
        self.context=None
        self.generation=0

    @torch.inference_mode()
    def prepare(self,vx,vy,observation=None):
        from joystick.primitive_conditions import align_target
        from joystick.web_generate import category
        from motionbricks.helper.data_training_util import extract_feature_from_motion_rep as extract
        seed,phases=self.prompts[category(vx,vy)]
        context=seed if self.context is None else self.context
        if observation is not None:
            from joystick.representation import full_body_motion, features
            states=np.asarray(observation['states'])
            if states.shape[0] != 4 or not np.isfinite(states).all():
                raise ValueError('Expected four finite executed states at 25 FPS')
            positions,rotations=full_body_motion(self.xml,states,observation['joint_names'])
            data=features(self.rep.dual_rep,positions.to(self.device),rotations.to(self.device))
            context=data[:,self.rep.dual_rep.indices['global_rep']][None]
            # Canonicalize the LAST observed heading, not the prediction tail.
            g=self.engine.global_motion_rep
            _,angles=g.compute_root_pos_and_rot(context,return_angle=True,return_quat=False)
            context=g.change_first_heading(context,angles[:,0]-angles[:,-1],False,False)
            self.context=context.clone()
        engine=self.engine
        g,l=engine.global_motion_rep,engine.local_motion_rep
        dual=engine.motion_rep.dual_rep
        start=context[:,:1,g.indices['global_root_pos']]
        command=torch.tensor([vx,vy],device=self.device)
        prompt=phases[self.generation%len(phases)]
        goal,local_goal=align_target(g,dual,prompt,start,command,self.horizon,25)
        if observation is not None:
            # Command velocities arrive in desired-facing coordinates. Rotate them
            # into this decision's measured heading frame; condition pose heading
            # separately instead of warping the generated trajectory afterwards.
            angle=float(observation['command_heading_delta'])
            c,s=np.cos(angle),np.sin(angle)
            vx_local,vy_local=c*vx-s*vy,s*vx+c*vy
            b,t,d=prompt.shape
            aligned=g.change_first_heading(prompt.reshape(b*t,1,d),angle,False,False).reshape(b,t,d)
            frames=torch.arange(self.horizon-4,self.horizon-4+t,device=self.device)
            aligned[...,g.indices['global_root_pos_2d']]=(
                start[...,[0,2]]+frames[None,:,None]/25*command.new_tensor([vy_local,vx_local]))
            local_aligned=dual.global_to_local(aligned,False,False,
                                              lengths=torch.full((b,),t,device=self.device,dtype=torch.long))
            goal,local_goal=aligned[:,:4],local_aligned[:,:4]
        boundary=torch.cat((context,goal),1)
        local_context=dual.global_to_local(context,False,False,lengths=torch.tensor([4],device=self.device))
        local=torch.cat((local_context,local_goal),1)
        mask=torch.ones((1,8),dtype=torch.bool,device=self.device)
        local_mask=mask.clone()
        local_mask[:,3]=False
        return dict(inputs=(extract(boundary,g,'root'),mask,extract(local,l,'root'),local_mask,
                            extract(local,l,'joint_positions_and_rotations'),mask),
                    context=context, observation=observation)

    @torch.inference_mode()
    def finish(self, prepared, out, n):
        context, observation = prepared['context'], prepared['observation']
        g = self.engine.global_motion_rep
        if int(n.item())*4 != self.horizon or out.shape[1] != self.horizon:
            raise ValueError('Joystick must return its complete configured horizon')
        future=out[:,4:]
        # In closed loop the current observed pose anchors interpolation at t=0.
        # No prediction is written back as the next observed context.
        clip=torch.cat((context[:,-1:],future),1)
        if observation is None:
            self.context=future[:,-4:].clone()
        inv=g.inverse(clip,is_normalized=False)
        delta=inv['global_joint_rots'][0]
        basis=delta.new_tensor([[0,1,0],[0,0,1],[1,0,0]])
        rotations=basis.T @ delta @ basis @ self.rep.skeleton.rest_global_rotations.to(delta)
        buf=io.BytesIO()
        np.savez(buf,embodiment=self.robot,fps=25,xml_sha256=self.xml_sha,
                 names=self.rep.skeleton.bone_order_names,
                 timestamps=np.arange(clip.shape[1])/25,joints=inv['posed_joints'][0,...,[2,0,1]].cpu().numpy(),
                 world_body_rotations=rotations.cpu().numpy())
        buf.seek(0)
        ref=native_reference(buf,self.xml,self.robot)
        ref['includes_current_anchor']=True
        ref['planner_future_frames']=int(future.shape[1])
        if observation is not None:
            ref['observation_tick']=int(observation['ticks'][-1])
        self.generation+=1
        return ref


    @torch.inference_mode()
    def release(self, vx, vy, observation=None):
        return self.release_many([self], [(vx, vy, observation)])[0]

    @staticmethod
    @torch.inference_mode()
    def release_many(planners, commands):
        if not planners or len(planners) != len(commands):
            raise ValueError('One command is required for every planner context')
        engine = planners[0].engine
        if any(p.engine is not engine or p.horizon != planners[0].horizon for p in planners):
            raise ValueError('Batched contexts must share the same resident model and horizon')
        prepared = [planner.prepare(*command) for planner, command in zip(planners, commands)]
        inputs = [torch.cat([p['inputs'][i] for p in prepared]) for i in range(6)]
        out, lengths = engine.predict(*inputs, planners[0].horizon//4,
            config={'pose_token_sampling_use_argmax':True, 'num_inference_step':4,
                    'final_root_pred_mode':'from_root_module'}, info={})
        return [planner.finish(p, out[i:i+1], lengths[i])
                for i, (planner, p) in enumerate(zip(planners, prepared))]
