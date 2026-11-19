"""``scripted`` Player: a fixed, time-indexed command sequence (contracts §3.2)."""

import math
from dataclasses import replace

import numpy as np
import torch

from omega_playground.control.ffpd import FFPD, FFPDConfig
from omega_playground.control.locomotion import MODES, validate_locomotion
from omega_playground.control.reference import ReferenceTrajectory
from omega_playground.control.route_deviation import LocalRouteProjection
from omega_playground.policy.base import JoystickBatch, RobotBatchView
from omega_playground.policy.schedule import Schedule


class ScriptedPlayer:
    """Segments ``{"until_s", "heading_deg", "speed"[, "facing_deg", "mode", "height", "stand"]}``.

    Facing defaults to the heading, mode to 2, height to -1 (unspecified); ``stand`` sends a zero movement.
    """

    def __init__(self, spec: dict, num_agents: int, device: torch.device):
        self.name = spec.get("name", "scripted")
        self.schedule = Schedule(trigger="periodic", period_s=1.0)
        self.segments = sorted(spec["segments"], key=lambda s: s["until_s"])
        self.num_agents, self.device = num_agents, device

    def reset(self, agent_ids, view: RobotBatchView) -> None:
        pass

    def segment(self, sim_time: float) -> dict:
        for seg in self.segments:
            if sim_time < seg["until_s"] - 1e-9:
                return seg
        return self.segments[-1]

    def act(self, view: RobotBatchView, task, sim_time: float) -> JoystickBatch:
        seg = self.segment(sim_time)
        n, dev = self.num_agents, self.device

        def unit(deg):
            a = math.radians(deg)
            return torch.tensor([math.cos(a), math.sin(a), 0.0], device=dev).repeat(n, 1)

        movement = unit(seg["heading_deg"])
        return JoystickBatch(
            mode=torch.full((n,), int(seg.get("mode", 2)), dtype=torch.long, device=dev),
            speed=torch.full((n,), float(seg["speed"]), device=dev),
            movement=torch.zeros_like(movement) if seg.get("stand", False) else movement,
            facing=unit(seg.get("facing_deg", seg["heading_deg"])),
            vertical_speed=torch.zeros(n, device=dev),
            height=torch.full((n,), float(seg.get("height", -1.0)), device=dev),
        )


class RouteControlSupport:
    """Time-driven route reference with measured-state feedback; no pose or clock corrections."""

    def __init__(self, spec: dict, num_agents: int, device: torch.device):
        self.name = spec.get("name", "route_ffpd")
        self.lateral_control = spec.get("lateral_control", True)
        self.schedule = Schedule(trigger="periodic", period_s=1.0)
        self.config = FFPDConfig(**{**spec.get("ffpd", {}), "dt": 1.0})
        if abs(self.schedule.interval_s() - self.config.dt) > 1e-9:
            raise ValueError("FFPD dt must match the Player schedule")
        self.reference = ReferenceTrajectory(spec["route"], max_speed=self.config.max_speed,
                                             max_acceleration=self.config.max_acceleration,
                                             **spec.get("reference", {}))
        if len(self.reference.offsets) != num_agents:
            raise ValueError("Reference must declare one slot per agent")
        self.num_agents, self.device = num_agents, device
        self.start_time = float(spec.get("start_delay_s", 1.))
        self.start_times = np.full(num_agents, self.start_time)
        self.controller = FFPD(num_agents, self.config)
        self.initialized = False
        self.divergence = float(spec.get("divergence_m", 2.))
        self.divergence_seconds = float(spec.get("divergence_s", 1.))
        self.diverged_since = np.full(num_agents, np.nan)
        self.hold = self.warning = None
        self.tracking_divergence_hold = True
        self.safety_holds = True
        self.warnings = []
        self.idle = np.zeros(num_agents, dtype=bool)
        self.idle_xy = np.zeros((num_agents, 2))
        self.idle_yaw = np.zeros(num_agents)
        self.projections = {}
        self.navigation = None
        self.clearance = float(spec.get("clearance_m", .5))
        self.sonic_mode = spec.get("sonic_mode", "walk")

    def locomotion(self):
        return {'mode': self.sonic_mode, 'speed': self.reference.nominal_speed}

    def configure_locomotion(self, mode, speed, sim_time):
        speed = validate_locomotion(mode, speed)
        config = replace(self.config, max_speed=max(1., speed+.35))
        reference = self.reference.retimed(speed, config.max_speed, config.max_acceleration)
        starts = self.start_times.copy()
        for i, start in enumerate(starts):
            if sim_time >= start:
                distance = self.reference.distance_at(sim_time-start)
                starts[i] = sim_time-reference.time_at_distance(distance)
        self.reference, self.start_times = reference, starts
        self.config = self.controller.config = config
        self.sonic_mode = mode

    def reset(self, agent_ids, view: RobotBatchView) -> None:
        ids = np.asarray(agent_ids.cpu())
        self.controller.reset(ids, view.root_vel[agent_ids].detach().cpu().numpy())
        self.diverged_since[ids] = np.nan
        self.warning = None
        self.warnings = []
        self.projections = {k: v for k, v in self.projections.items() if k[1] not in ids}

    def recovery_time(self, role, position, reference):
        key = (id(reference.xy), role)
        projection = self.projections.get(key)
        if projection is None:
            projection = LocalRouteProjection(reference, role)
        _, distance = projection.project(position)
        self.projections[key] = projection
        return reference.time_at_distance(distance)

    def recover_task(self, agent_ids, view, sim_time):
        ids = np.asarray(agent_ids, dtype=int)
        times = [self.recovery_time(i, view.root_pos[i, :2].cpu().numpy(), self.reference) for i in ids]
        projections = self.projections.copy()
        RoutePlayer.reset(self, torch.as_tensor(ids, device=self.device), view)
        self.projections = projections
        self.start_times[ids] = sim_time-np.array(times)
        self.controller.previous[ids] = 0.
        self.initialized = True
        self.hold = None

    def recover_idle(self, agent_ids, view):
        ids = np.asarray(agent_ids, dtype=int)
        self.idle[ids] = True
        self.idle_xy[ids] = view.root_pos[ids, :2].detach().cpu().numpy()
        x, y, z, w = getattr(view, "heading_rot", view.root_rot)[ids].detach().cpu().numpy().T
        self.idle_yaw[ids] = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self.controller.previous[ids] = 0.
        self.hold = None
        if hasattr(self, 'motion_warning'):
            self.motion_warning = None

    def apply_idle(self, command):
        ids = np.flatnonzero(self.idle)
        command.mode[ids], command.speed[ids], command.movement[ids] = 0, -1., 0.
        yaw = self.idle_yaw[ids]
        command.facing[ids] = torch.as_tensor(np.c_[np.cos(yaw), np.sin(yaw), np.zeros(len(ids))],
                                             device=self.device, dtype=command.facing.dtype)
        self.controller.previous[ids] = 0.
        return command

    def route_error(self, position):
        reference = self.reference
        if hasattr(self, 'stage'):
            reference = self.reference.parts[self.stage]
        result, tangents = [], []
        for role, point in enumerate(position):
            if self.idle[role]:
                result.append(self.idle_xy[role]-point)
                tangents.append([np.cos(self.idle_yaw[role]), np.sin(self.idle_yaw[role])])
                continue
            key = (id(reference.xy), role)
            if key not in self.projections:
                self.projections = {k: v for k, v in self.projections.items() if k[1] != role}
                self.projections[key] = LocalRouteProjection(reference, role)
            result.append(self.projections[key].error(point))
            tangents.append(self.projections[key].tangent.copy())
        return np.array(result), np.array(tangents)

    def targets(self, sim_time):
        return [(self.idle_xy[i].copy(), self.idle_yaw[i]) if self.idle[i] else
                self.reference.sample(sim_time - self.start_times[i], i) for i in range(self.num_agents)]

    def measure(self, view, sim_time):
        target = np.array([p for p, _ in self.targets(sim_time)])
        ahead = np.array([p for p, _ in self.targets(sim_time + self.config.dt)])
        error = target - view.root_pos[:, :2].detach().cpu().numpy()
        speed = (ahead - target) / self.config.dt
        route_error, tangent = self.route_error(view.root_pos[:, :2].detach().cpu().numpy())
        reference = self.reference.parts[self.stage] if hasattr(self, 'stage') else self.reference
        progress_error = np.zeros(self.num_agents)
        for role in range(self.num_agents):
            if not self.idle[role]:
                elapsed = sim_time-(self.leg_started if hasattr(self, 'stage') else self.start_times[role])
                progress_error[role] = (reference.distance_at(elapsed)
                                       - self.projections[(id(reference.xy), role)].reference_distance)
        return {"reference_xy": target, "reference_velocity_xy": speed, "position_error": error,
                    "route_error": route_error, "route_tangent": tangent, "route_progress_error": progress_error,
                    "speed_saturated": self.controller.report.get("speed_saturated", np.zeros(self.num_agents, bool)),
                    "acceleration_saturated": self.controller.report.get("acceleration_saturated",
                                                                        np.zeros(self.num_agents, bool))}

    def lateral_feedback(self, measured):
        return ({"route_error": measured["route_error"], "route_tangent": measured["route_tangent"],
                 "route_progress_error": measured["route_progress_error"]}
                if self.lateral_control else {})

    def check(self, view, sim_time):
        measured = self.measure(view, sim_time)
        self.warnings = [self.motion_warning] if getattr(self, 'motion_warning', None) else []
        if not self.safety_holds and self.hold is not None:
            self.warnings.append(self.hold)
            self.hold = None
        error = np.linalg.norm(measured["position_error" if self.safety_holds else "route_error"], axis=-1)
        bad = (error > self.divergence) & ~self.idle
        self.diverged_since[~bad] = np.nan
        self.diverged_since[bad & np.isnan(self.diverged_since)] = sim_time
        ids = np.flatnonzero(bad & (sim_time - self.diverged_since >= self.divergence_seconds - 1e-9))
        self.warning = None
        if len(ids):
            reason = {"code": "tracking_divergence", "agents": ids.tolist(), "sim_time": sim_time,
                      "error_m": error[ids].tolist()}
            if self.safety_holds and self.tracking_divergence_hold:
                self.hold = reason
            else:
                self.warning = reason
                self.warnings.append(reason)
        if self.navigation is not None:
            ids = np.flatnonzero(~self.navigation.valid(view.root_pos[:, :2].detach().cpu().numpy(), self.clearance))
            if len(ids):
                reason = {"code": "off_navigable", "agents": ids.tolist(), "sim_time": sim_time}
                if self.safety_holds:
                    self.hold = reason
                else:
                    self.warnings.append(reason)
        return measured

    def act(self, view: RobotBatchView, task, sim_time: float) -> JoystickBatch:
        if not self.initialized:
            self.controller.reset(np.arange(self.num_agents), view.root_vel.detach().cpu().numpy())
            self.initialized = True
        measured = self.measure(view, sim_time)
        cmd = self.controller.update(measured["reference_xy"], measured["reference_velocity_xy"],
                                     view.root_pos.detach().cpu().numpy(), view.root_vel.detach().cpu().numpy(),
                                     **self.lateral_feedback(measured))
        speed = np.linalg.norm(cmd, axis=-1)
        movement = np.c_[cmd / np.maximum(speed[:, None], 1e-8), np.zeros(self.num_agents)]
        yaw = np.array([yaw for _, yaw in self.targets(sim_time)])
        facing = np.c_[np.cos(yaw), np.sin(yaw), np.zeros(self.num_agents)]
        if self.lateral_control:
            facing[:, :2] = measured["route_tangent"]
        waiting = (sim_time < self.start_times) | (sim_time >= self.start_times + self.reference.duration)
        movement[waiting] = 0
        speed[waiting] = -1

        def tensor(v):
            return torch.as_tensor(v, dtype=torch.float64, device=self.device)

        return self.apply_idle(JoystickBatch(mode=torch.as_tensor(np.where(waiting, 0, MODES[self.sonic_mode]["id"]), dtype=torch.long, device=self.device),
                             speed=tensor(speed), movement=tensor(movement), facing=tensor(facing),
                             vertical_speed=tensor(np.zeros(self.num_agents)),
                             height=tensor(np.full(self.num_agents, -1.))))


class RoutePlayer(RouteControlSupport):
    """Single-robot FFPD NPC executing a target trajectory."""
    def __init__(self, spec, num_agents, device):
        if num_agents != 1:
            raise ValueError('A Player controls one robot; use SquadMaster for multiple robots')
        super().__init__(spec, 1, device)
