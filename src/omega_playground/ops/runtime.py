"""Simulation-thread command handling; pause freezes all simulation and reference clocks."""

import json
import queue
from pathlib import Path


import numpy as np
import torch

from omega_playground.control.player_factory import device_player_enabled, mission_player
from omega_playground.control.teams import host_view, subset
from omega_playground.ops.robot import policy_choice


def validate_reference(player, world, navigation):
    for role in range(player.num_agents):
        xy = np.array([player.reference.sample(t, role)[0] for t in player.reference.times])
        from omega_playground.map.preview import inspect_path
        validation = inspect_path(navigation, xy, player.clearance)
        if not validation['valid']:
            raise ValueError(validation['message'])
        heights = world.ground.heights(xy)
        from omega_playground.map.terrain import support_continuous
        if not support_continuous(xy, heights, getattr(navigation, 'meta', {}).get('terrain_zones', [])):
            raise ValueError(f'Role {role} reference lacks continuous physical support')




class OperationsRuntime:
    def __init__(self, world, runner, teams, server, output, safety=None):
        self.world, self.runner, self.teams, self.server = world, runner, teams, server
        self.output, self.safety = Path(output), safety
        self.paused, self.hold = True, None
        self.reconfigure_assets = None
        self.reconfigure_request = None
        self.reconfigure_package = None
        self.armed = False
        self.reconfigure_armed = False
        self.avatar = None
        self.reconfigure_avatar = None
        self.acks = {}
        self.viewer = None
        self.task_states = {}
        self.warning_states = {}
        self.task_origins = {}
        # Interactive navigation reports tracking error without forcing a reset.
        # Offline acceptance players retain their strict divergence gate.
        for player in self.teams.players:
            player.tracking_divergence_hold = False
            player.safety_holds = False
        self.event_file = (self.output / 'events.jsonl').open('a')

    def event(self, event):
        event = {**event, 'sim_time': self.world.sim_time, 'physics_step': self.world.physics_steps}
        self.event_file.write(json.dumps(event) + '\n')
        self.event_file.flush()
        self.server.event(event)

    def selected(self, command):
        team = command.get('team', 0)
        if type(team) is not int or not 0 <= team < len(self.teams.players):
            raise ValueError('invalid team')
        ids = self.teams.ids[team]
        if 'agents' in command:
            raw = command['agents']
            if not isinstance(raw, list) or not raw or any(type(x) is not int for x in raw):
                raise ValueError('agents must be a nonempty integer list')
            if len(set(raw)) != len(raw) or not set(raw) <= set(ids):
                raise ValueError('agents must be unique members of the selected team')
            ids = np.asarray(raw, dtype=int)
        return team, ids

    def reset(self, team, ids, mode="in_place"):
        if mode not in ("in_place", "task_start"):
            raise ValueError("recovery mode must be in_place or task_start")
        player = self.teams.players[team]
        local = [int(np.flatnonzero(self.teams.ids[team] == i)[0]) for i in ids]
        view = self.world.robot_view()
        xy = view.root_pos[ids, :2].detach().cpu().numpy().copy()
        x, y, z, w = getattr(view, "heading_rot", view.root_rot)[ids].detach().cpu().numpy().T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        if mode == 'task_start':
            if team in self.task_origins:
                origins, headings = self.task_origins[team]
                xy, yaw = origins[local].copy(), headings[local].copy()
            else:
                targets = [player.reference.sample(0., role) for role in local]
                xy = np.array([p for p, _ in targets])
                yaw = np.array([heading for _, heading in targets])
        audit = self.world.reset_agents(ids, xy, yaw-getattr(view, "heading_offset", 0.))
        view = self.world.robot_view()
        tensor_ids = torch.as_tensor(ids, device=self.world.torch_device)
        if mode == 'task_start':
            player.reset(torch.as_tensor(local, device=self.world.torch_device),
                         subset(view, self.teams.ids[team]))
        player.recover_task(local, subset(view, self.teams.ids[team]), self.world.sim_time)
        audit.update({'recovery_mode': mode, 'position_xy': xy.tolist(), 'heading_rad': yaw.tolist(), 'state': 'task_retained',
                      'max_xy_shift_m': float(np.linalg.norm(view.root_pos[ids, :2].cpu().numpy()-xy, axis=-1).max())})
        self.runner.wbc.reset(tensor_ids)
        if self.runner.joystick is not None:
            self.runner.joystick.reset(tensor_ids, view)
        self.runner.request_reset(tensor_ids)
        if self.safety is not None:
            self.safety.nonfoot_count[ids] = 0
        # A hold only clears when every involved robot has explicitly been reset.
        if self.hold is not None:
            affected = set(self.hold.get('agents', []))
            affected.update(i for pair in self.hold.get('pairs', []) for i in pair)
            if not affected:
                affected = set(range(self.world.num_agents))
            affected -= set(ids)
            if affected:
                self.hold = {**self.hold, 'agents': sorted(affected), 'pairs': []}
            else:
                self.hold = self.teams.hold = None
                for p in self.teams.players:
                    p.hold = None
                self.runner.stopped = False
                self.runner.on_wbc.failure = None
        if self.viewer is not None:
            self.viewer.show(self.world.body_q_snapshot(), self.world.sim_time, record=False)
        self.event({'kind': 'reset_audit', **audit})
        return audit

    def execute(self, command):
        kind = command['kind']
        if self.reconfigure_request is not None:
            raise ValueError('队伍配置正在重建，请等待完成')
        if kind == 'select_scene':
            return self.select_scene(command.get('scene'))
        if kind == 'control_mode':
            from omega_playground.control.keyboard import KeyboardPlayer
            mode = command.get('mode')
            if mode not in ('auto', 'manual') or sum(map(len, self.teams.ids)) != 1:
                raise ValueError('Select Auto or Manual for a single agent')
            self.runner.keyboard = KeyboardPlayer(**self.teams.players[0].locomotion()) if mode == 'manual' else None
            self.paused = mode == 'auto'
            return {'mode': mode}
        if kind == 'keyboard':
            keyboard = getattr(self.runner, 'keyboard', None)
            if keyboard is None:
                raise ValueError('Manual control is inactive')
            keyboard.update(command.get('forward'), command.get('turn'))
            return {}
        if getattr(self.runner, 'keyboard', None) is not None and kind == 'drawn_path':
            raise ValueError('Switch to Auto to execute a path')
        if kind == 'configure_robot':
            from omega_playground.ops.scenes import SCENES
            from omega_playground.ops.robot import resolve_avatar
            armed = command.get('armed', self.armed)
            if type(armed) is not bool:
                raise ValueError('Armed 必须为布尔值')
            avatar = command.get('avatar', self.avatar)
            package, _ = resolve_avatar(self.world.package, 'sonic', avatar)
            if avatar == self.avatar and armed == self.armed:
                raise ValueError('机器人配置没有变化')
            spec = json.loads((Path('configs/scenes') / SCENES[self.server.active_scene][2]).read_text())
            assets = self.validate_configuration(package, avatar, armed, 1)
            config = {'teams': [spec], 'start_idle': True}
            self.reconfigure_armed = armed
            self.reconfigure_avatar = avatar
            path = self.output / 'next_player.json'
            path.write_text(json.dumps(config, indent=2) + '\n')
            self.reconfigure_assets = assets
            self.reconfigure_package = package.root.resolve()
            self.reconfigure_request = path
            self.paused = True
            return {'avatar': avatar, 'armed': armed}
        elif kind == 'pause':
            self.paused = True
        elif kind == 'resume':
            if self.hold is not None:
                raise ValueError('reset all affected robots before resuming a hold')
            self.paused = False
        elif kind == 'reset':
            team, ids = self.selected(command)
            return self.reset(team, ids, command.get('recovery_mode', 'in_place'))
        elif kind == 'locomotion':
            team, ids = self.selected(command)
            if set(ids) != set(self.teams.ids[team]):
                raise ValueError('Route locomotion settings apply to the whole team')
            if self.hold is not None:
                raise ValueError('Resolve the safety hold before changing locomotion')
            player = self.teams.players[team]
            player.configure_locomotion(command['mode'], command['speed'], self.world.sim_time)
            if getattr(self.runner, 'keyboard', None) is not None:
                self.runner.keyboard.configure(**player.locomotion())
            self.event({'kind': 'locomotion_changed', 'team': team, **player.locomotion()})
            return player.locomotion()
        elif kind == 'drawn_path':
            from dataclasses import asdict
            team, ids = self.selected(command)
            if set(ids) != set(self.teams.ids[team]):
                raise ValueError('Navigation applies to the whole team')
            ids = self.teams.ids[team]  # Keep role order even if an API caller lists all IDs in reverse.
            if self.hold is not None:
                raise ValueError('Resolve the safety hold before navigation')
            old = self.teams.players[team]
            view = self.world.robot_view()
            start = view.root_pos[int(ids[0]), :2].detach().cpu().numpy().tolist()
            from omega_playground.control.drawn_path import drawn_path
            if len(ids) != 1:
                raise ValueError('Hand-drawn paths require one active agent')
            plan = drawn_path(command.get('keypoints'))
            from omega_playground.map.preview import inspect_path
            validation = inspect_path(self.server.preview_navigation(), plan['route'], plan['clearance_m'])
            if not validation['valid']:
                raise ValueError(validation['message'])
            if np.linalg.norm(np.subtract(start, plan['route'][0])) > .35:
                raise ValueError('Robot moved from the stroke origin; redraw from its current position')
            spec = {'lateral_control': old.lateral_control, 'sonic_mode': old.sonic_mode, 'route': plan['route'], 'legs': plan.get('legs', [plan['route']]),
                    'task_mode': kind, 'ffpd': asdict(old.config),
                    'schedule': {'trigger': 'periodic', 'period_s': old.config.dt},
                    'reference': {'closed': False, 'nominal_speed': old.reference.nominal_speed,
                                  'corner_radius': old.reference.corner_radius, 'preserve_path': True,
                                  'slot_offsets': [[0., 0.]]},
                    'clearance_m': plan['clearance_m']}
            player_device = (self.teams.device if device_player_enabled(self.teams.device) and len(ids) == 1
                             else old.device)
            player = mission_player(policy_choice(self.world.package), spec, player_device, len(ids))
            player.tracking_divergence_hold = old.tracking_divergence_hold
            player.safety_holds = old.safety_holds
            player.navigation = self.server.navigation
            validate_reference(player, self.world, player.navigation)
            player.start_times[:] = self.world.sim_time
            # Preserve acceleration-limiter history; never reset physics or the WBC.
            player.controller.previous[:] = old.controller.previous
            player.initialized = True
            self.teams.players[team] = player
            qx, qy, qz, qw = getattr(view, "heading_rot", view.root_rot)[ids].detach().cpu().numpy().T
            self.task_origins[team] = (
                view.root_pos[ids, :2].detach().cpu().numpy().copy(),
                np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
            (self.output / f'navigation_team{team}_step{self.world.physics_steps}.json').write_text(
                json.dumps(plan, indent=2) + '\n')
            self.event({'kind': 'route_replanned', 'team': team, 'plan': plan})
            self.paused = False
            return plan
        elif kind == 'viewer':
            if self.viewer is None:
                raise ValueError('Viewer is not enabled; start serve with --video')
            mode, agent = command.get('camera'), command.get('agent', 0)
            if mode not in ('qc', 'chase', 'squad', 'top', 'free'):
                raise ValueError('invalid camera mode')
            if type(agent) is not int or not 0 <= agent < self.world.num_agents:
                raise ValueError('invalid follow agent')
            yaw, pitch, distance = [float(command.get(k, d)) for k, d in [('yaw', 30), ('pitch', -12), ('distance', 4)]]
            if not np.isfinite([yaw, pitch, distance]).all() or not -89 <= pitch <= 89 or not 1 <= distance <= 100:
                raise ValueError('invalid orbit camera values')
            self.viewer.camera_mode = mode
            self.viewer.follow_body = agent * len(self.world.package.body_names) + self.world.anchor
            self.viewer.orbit_yaw, self.viewer.orbit_pitch, self.viewer.orbit_distance = yaw, pitch, distance
            self.viewer.show(self.world.body_q_snapshot(), self.world.sim_time, record=False)
            if self.viewer.latest_jpeg is not None:
                self.server.publish_frame(self.viewer.latest_jpeg)
            else:
                frame = self.output / 'preview.png'
                self.viewer.viewer.save_screenshot(str(frame))
                self.server.viewer_frame = frame
        else:
            raise ValueError('unknown command')
        return {}

    def drain(self):
        while True:
            try:
                command = self.server.commands.get_nowait()
            except queue.Empty:
                return
            cid = command['id']
            if cid in self.acks:
                self.server.event(self.acks[cid])
                continue
            try:
                detail = self.execute(command)
                ack = {'id': cid, 'accepted': True, 'detail': detail}
            except (ValueError, KeyError, TypeError) as exc:
                ack = {'id': cid, 'accepted': False, 'error': str(exc)}
            if command['kind'] == 'keyboard' and ack['accepted']:
                continue  # Heartbeats must not fill the event log or receipt cache.
            self.acks[cid] = ack
            self.event({'kind': 'ack', 'command': command['kind'], **ack})

    def freeze(self, reason):
        affected = set(reason.get('agents', []))
        affected.update(i for pair in reason.get('pairs', []) for i in pair)
        reason = {**reason, 'teams': [i for i, ids in enumerate(self.teams.ids) if affected.intersection(ids)]}
        self.paused, self.hold = True, reason
        self.event({'kind': 'hold', **reason})

    def warn(self, source, warnings):
        current = {(w['code'], tuple(w.get('agents', [])),
                    tuple(tuple(p) for p in w.get('pairs', []))): w for w in warnings}
        previous = self.warning_states.get(source, {})
        for key in current.keys() - previous.keys():
            self.event({**current[key], 'kind': 'warning'})
        for key in previous.keys() - current.keys():
            self.event({**previous[key], 'kind': 'warning_cleared'})
        self.warning_states[source] = current

    def report_safety(self, reason):
        if reason and reason['code'] == 'nonfinite':
            if hasattr(self.safety, 'save_history'):
                self.safety.save_history()
            self.freeze(reason)
        else:
            self.warn('physics', [reason] if reason else [])

    def publish_warnings(self):
        for team, (player, ids) in enumerate(zip(self.teams.players, self.teams.ids, strict=True)):
            self.warn(team, [{**w, 'team': team, 'agents': ids[w['agents']].tolist()}
                             for w in player.warnings])

    def publish(self):
        self.publish_warnings()
        view = host_view(self.world.robot_view())
        report = np.concatenate([p.measure(subset(view, ids), self.world.sim_time)['route_error']
                                 for p, ids in zip(self.teams.players, self.teams.ids, strict=True)])
        positions = view.root_pos.cpu().numpy()
        x, y, z, w = getattr(view, "heading_rot", view.root_rot).cpu().numpy().T
        headings = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        speeds = torch.linalg.vector_norm(view.root_vel[:, :2], dim=-1).cpu().numpy()
        for i, player in enumerate(self.teams.players):
            if hasattr(player, 'status'):
                status = player.status()
                if self.task_states.get(i) != status:
                    self.event({'kind': 'task_phase', 'team': i, **status})
                    self.task_states[i] = status
        self.server.publish({'control_mode': 'manual' if getattr(self.runner,'keyboard',None) else 'auto', 'ready': self.reconfigure_request is None,
                             'reconfiguring': self.reconfigure_request is not None,
                             'physics_backend': getattr(self.world, 'backend', 'mujoco_warp'),
                             'wbc_policy': policy_choice(self.world.package), 'avatar': self.avatar, 'armed': self.armed,
                             'wbc_choices': getattr(self.server, 'policy_choices', ['sonic']),
                             'session_id': getattr(self.server, 'session_id', None), 'paused': self.paused, 'hold': self.hold,
                             'tracking_divergence_action': 'warn', 'safety_action': 'warn',
                             'error_metric': 'local_route_distance',
                             'sim_time': self.world.sim_time, 'physics_step': self.world.physics_steps,
                             'agents': [{'id': i, 'position': p.tolist(),
                                         'heading_deg': float(headings[i]),
                                         'speed_m_s': float(speeds[i])}
                                        for i, p in enumerate(positions)],
                             'teams': [{'id': i, 'agents': ids.tolist(),
                                        'player_implementation': type(player).__name__, 'player_device': str(player.device),
                                        'reference': [] if player.idle.all() else player.reference.xy[::10].tolist(),
                                        'locomotion': player.locomotion(),
                                        'idle_agents': ids[player.idle].tolist(),
                                        'task': {'mode': 'idle', 'phase': 'idle'} if player.idle.all() else player.status() if hasattr(player, 'status') else {
                                            'mode': 'initial', 'phase': 'complete' if self.world.sim_time >=
                                            float(np.max(player.start_times))+player.reference.duration else 'track'}}
                                       for i, (player, ids) in enumerate(zip(self.teams.players, self.teams.ids, strict=True))],
                             'error_m': np.linalg.norm(report, axis=-1).tolist(),
                             'viewer': None if self.viewer is None else {
                                 'camera': self.viewer.camera_mode,
                                 'agent': self.viewer.follow_body // len(self.world.package.body_names),
                                 'yaw': self.viewer.orbit_yaw, 'pitch': self.viewer.orbit_pitch,
                                 'distance': self.viewer.orbit_distance},
                             'rates': self.runner.rates(), 'performance': getattr(self, 'performance', None)})

    def close(self):
        try:
            self.event_file.close()
        finally:
            if self.runner is not None:
                self.runner.close()
