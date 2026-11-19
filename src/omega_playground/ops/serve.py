"""Interactive shared-world runtime and reproducible headless multi-team runs."""

import json
import os
import signal
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from omega_playground.control.teams import TaskControllerBatch
from omega_playground.eval.run import Recorder, _git, _versions, summarize
from omega_playground.map.navigation import CampusNavigation
from omega_playground.ops.runtime import OperationsRuntime, validate_reference
from omega_playground.ops.server import OperationsServer
from omega_playground.policy.package import load
from omega_playground.policy.runner import Runner
from omega_playground.scene.package import load as load_scene


def configure_simulation_stream(device):
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)
    wp.set_stream(wp.stream_from_torch(stream))
    return stream


def serve(args):
    # Background shells may inherit SIGINT=ignore; preserve graceful trace/video finalization.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    # Bound the host-side Torch pool before CUDA graph/RTX initialization. On
    # large CPU hosts an unbounded pool can starve renderer startup and cadence.
    torch.set_num_threads(4)
    if any(v < 64 or v > 4096 or v % 2 for v in (args.render_width, args.render_height)):
        raise ValueError('render dimensions must be even integers between 64 and 4096')
    if args.auto_run_seconds is not None and (not np.isfinite(args.auto_run_seconds) or args.auto_run_seconds <= 0):
        raise ValueError('auto-run-seconds must be positive and finite')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    package, scene = load(args.package), load_scene(args.scene)
    if package.family != "sonic":
        raise ValueError("Expected a SONIC G1 policy")
    from omega_playground.ops.robot import default_avatar, policy_choice, resolve_avatar
    args.avatar = args.avatar or default_avatar(package)
    package, skin_path = resolve_avatar(package, policy_choice(package), args.avatar)
    args.package = str(package.root)
    from omega_playground.viewer.cadence import render_rate
    render_hz = render_rate(package, args.render_hz, getattr(args, "dlss_frame_generation", "off"))
    if torch.device(args.device).type != 'cuda':
        raise ValueError('Playground requires MJWarp on CUDA')
    cuda_render_ipc = os.environ.get('OMEGA_RENDER_CUDA_IPC') == '1'
    navigation = CampusNavigation(args.map, scene.scene_sha256, scene.collision_sha256)
    device = torch.device(args.device)
    configure_simulation_stream(device)
    config = json.loads(Path(args.player).read_text())
    specs = config.get('teams', [config])
    if len(specs) != 1 or len(specs[0].get('reference', {}).get('slot_offsets', [[0, 0]])) != 1:
        raise ValueError('A Playground scene requires exactly one robot')
    registry = None
    if getattr(args, 'prepared_assets', None):
        from omega_playground.ops.prepared_assets import PreparedAssetRegistry
        registry = PreparedAssetRegistry(args.prepared_assets)
        count = sum(len(spec.get('reference', {}).get('slot_offsets', [[0, 0]])) for spec in specs)
        selected = registry.select(package, scene.scene_sha256, args.avatar, args.armed, count)
        args.physics_asset, args.visual_model = selected['physics_asset'], selected['visual_model']
        args.render_environment = selected.get('render_environment')
    for spec in specs:
        spec['schedule'] = {'trigger': 'periodic', 'period_s': 1.0}
        spec.setdefault('ffpd', {})['dt'] = 1.0
    teams = TaskControllerBatch(specs, device, navigation, package.family)
    spawn = [player.reference.sample(0, i)[0].tolist() for player in teams.players for i in range(player.num_agents)]
    separation = np.linalg.norm(np.array(spawn)[:, None] - np.array(spawn)[None], axis=-1)
    np.fill_diagonal(separation, np.inf)
    if separation.min() < 1.4:
        raise ValueError('initial spawns must be separated by at least 1.4 m')
    from omega_playground.sim.mjwarp_world import MJWarpWorld
    from omega_playground.ops.initial_heading import initialize_route_heading
    world = MJWarpWorld(package, args.physics_asset, scene, spawn, str(device))
    ground = world.ground
    initial_facing = initialize_route_heading(world, teams, specs)
    for player in teams.players:
        validate_reference(player, world, navigation)
    if config.get('start_idle'):
        view = world.robot_view()
        for player, ids in zip(teams.players, teams.ids, strict=True):
            player.idle[:] = True
            player.idle_xy[:] = view.root_pos[ids, :2].cpu().numpy()
            player.idle_yaw[:] = initial_facing[ids]
    recorder = Recorder(world, output, teams)
    runner = Runner(world, package, teams, device, on_wbc=recorder)
    # Live timings measure host dispatch plus natural waits. Device-wide fences
    # on every stage would serialize the independent renderer with simulation.
    # End-to-end FPS/RTF below are measured from completed frames and wall time.
    runner.timer.cuda = False
    from omega_playground.eval.mjwarp_safety import MJWarpSafety
    world.contact_report_stride = runner.wbc_every
    safety = MJWarpSafety(world)
    server = OperationsServer(navigation, scene.scene_sha256, args.port,
                              video_path=output / 'video/rtx.mp4' if args.video else None,
                              video_mime=('video/mp4; codecs="hvc1.1.6.L123.B0"' if args.video_codec == 'h265'
                                          else 'video/mp4; codecs="avc1.42E01F"'))
    runtime = OperationsRuntime(world, runner, teams, server, output, safety)
    from omega_playground.ops.scenes import SCENES, resolve_scene, scene_id
    server.active_scene = scene_id(args.scene)
    server.scenes = list(SCENES)
    if registry is not None:
        server.prepared_options = [
            {'avatar': avatar, 'armed': armed}
            for (_, digest, avatar, armed, count), paths in registry.entries.items()
            if digest == scene.scene_sha256 and count == 1
            and all((path / 'SHA256SUMS.json').is_file() for path in paths.values())
        ]
        server.policy_choices = ['sonic']
    def select_scene(key):
        if world.num_agents != 1:
            raise ValueError('场景切换需要单机会话')
        try:
            scene_dir, map_dir, player, assets = resolve_scene(key, registry, package, runtime.avatar, runtime.armed)
        except OSError as exc:
            raise ValueError(f'场景资产未就绪：{exc}') from exc
        args.scene, args.map = str(scene_dir), str(map_dir)
        runtime.reconfigure_assets = assets
        runtime.reconfigure_package = package.root.resolve()
        runtime.reconfigure_avatar, runtime.reconfigure_armed = runtime.avatar, runtime.armed
        runtime.reconfigure_request = player
        return {'scene': key}

    runtime.select_scene = select_scene
    runtime.armed = args.armed
    runtime.avatar = args.avatar
    def validate_prepared_configuration(next_package, avatar, armed, count):
        if registry is not None:
            return registry.select(next_package, scene.scene_sha256, avatar, armed, count)
        if (next_package.sha256sum() != package.sha256sum() or avatar != args.avatar
                or armed != args.armed or (args.video and count != world.num_agents)):
            raise ValueError('该配置需要对应的离线 MJWarp 资产包，当前服务尚未加载该资产')
    runtime.validate_configuration = validate_prepared_configuration
    info = {'git': _git('rev-parse', 'HEAD'), 'avatar': args.avatar, 'armed': args.armed, 'versions': _versions(), 'config': config,
            'torch_cpu_threads': torch.get_num_threads(), 'policy_device': str(device),
            'joystick_runtime': getattr(runner.joystick, 'runtime_metadata', None),
            'initial_players': [{'implementation': type(p).__name__, 'device': str(p.device),
                                 'num_agents': p.num_agents} for p in teams.players],
            'stage_timing_mode': 'host_wall_without_device_fences',
            'realtime_pacing': args.realtime,
            'tracking_divergence_action': 'warn', 'safety_action': 'warn',
            'warning_sample_hz': world.physics_hz/getattr(world,'contact_report_stride',1),
            'error_metric': 'local_route_distance',
            'scene_sha256': scene.scene_sha256, 'world': world.describe(), 'rates': runner.rates(), 'render_hz': render_hz, 'video_codec': args.video_codec, 'frame_generation': 'off',
            'package_sha256sums': package.sha256sum()}
    (output / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
    viewer = None
    render_index = 1
    frame_wall_times = []
    if args.video:
        from functools import partial

        from omega_playground.viewer.process import RenderProcess
        def completed_frame(jpeg, sim_time, record, wall):
            server.ego_jpeg = getattr(viewer, "ego_jpeg", None)
            server.publish_frame(jpeg)
            if record:
                frame_wall_times.append(wall)
        def camera_height(root):
            return float(ground.ground_heights(torch.as_tensor(root[None], device=device))[0])
        common = {'video': output / 'video/rtx.mp4', 'width': args.render_width, 'height': args.render_height,
                      'fps': Fraction(str(render_hz)), 'camera': 'qc', 'video_codec': args.video_codec,
                      'video_bitrate_kbps': args.video_bitrate_kbps}
        import hashlib

        from omega_playground.viewer.process import mjwarp_scene_viewer
        environment = getattr(args, 'render_environment', None)
        factory = partial(mjwarp_scene_viewer, args.package, args.visual_model,
                          scene.scene_sha256, world.num_agents, str(device), armed=args.armed,
                          environment_path=environment, pbr=bool(environment), linear_output=bool(environment),
                          skin_manifest_sha256=(hashlib.sha256((skin_path/'SHA256SUMS.json').read_bytes()).hexdigest()
                                                if skin_path else None), **common)
        viewer = RenderProcess(factory, completed_frame, camera_height,
                               python_executable=args.render_python)
        viewer.root_bodies = [i * len(package.body_names) + world.anchor for i in range(world.num_agents)]
        viewer.camera_mode = args.camera
        runtime.viewer = viewer
        # Shader/scene warmup is initialization, not a measured simulation frame.
        viewer.show(world.body_q_snapshot(), world.sim_time, record=False)
        server.publish_frame(viewer.latest_jpeg)
    runtime.publish()
    server.start()
    if server.http is not None:
        print(f'Operations: http://127.0.0.1:{server.http.server_port}', flush=True)
    if args.auto_run_seconds and not getattr(args, 'start_paused', False):
        runtime.execute({'kind': 'resume'})
    wall0 = time.perf_counter()
    last_publish = 0.
    performance_window = deque()
    snapshots, times = [], []
    pose_archive = None
    if cuda_render_ipc:
        from omega_playground.eval.pose_archive import DevicePoseArchive
        pose_archive = DevicePoseArchive(device)
    pace_wall = None
    reference_recorder = None
    recording_finished = False
    lifecycle = {}
    try:
        if getattr(args, 'record_reference', False):
            from omega_playground.eval.reference_archive import ReferenceRecorder
            reference_recorder = ReferenceRecorder(output/'reference_windows.zip', package)
            def record_tick(tick):
                reference_recorder.append(tick)
                return recorder(tick)
            runner.on_wbc = record_tick
        while True:
            # Process commands only on a Player boundary, which also aligns Joystick and WBC.
            if world.physics_steps % runner.wbc_every == 0 or runtime.paused:
                runtime.drain()
            if runtime.reconfigure_request is not None:
                runtime.publish()
                time.sleep(1.)  # Give the browser time to receive its accepted receipt.
                break
            if args.auto_run_seconds and (world.sim_time >= args.auto_run_seconds or runtime.hold):
                break
            if runtime.paused:
                pace_wall = None
                performance_window.clear()
                runtime.performance = None
                runtime.publish()
                time.sleep(.02)
                continue
            if args.realtime:
                if pace_wall is None:
                    pace_wall, pace_sim = time.perf_counter(), world.sim_time
                delay = pace_wall + world.sim_time - pace_sim - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            runner.step()
            reason = recorder.failure if runner.stopped else None
            due = world.physics_steps % getattr(world, 'contact_report_stride', 1) == 0
            if not runner.stopped and safety is not None and due:
                with runner.timer('safety_snapshot'):
                    reason = safety.sample()
            if reason or due:
                runtime.report_safety(reason)
            if world.sim_time + 1e-9 >= render_index / render_hz:
                render_index += 1
                if pose_archive is not None:
                    body_q = world.body_q_tensor()
                    pose_archive.append(body_q, world.sim_time)
                else:
                    body_q = world.body_q_snapshot()
                    snapshots.append(body_q)
                    times.append(world.sim_time)
                if viewer is not None:
                    with runner.timer('render'):
                        viewer.show(body_q, world.sim_time)
            now = time.perf_counter()
            if now - last_publish > .1:
                frames = viewer.frames if viewer is not None else 0
                performance_window.append((now, world.sim_time, frames))
                while len(performance_window) > 2 and performance_window[1][0] < now - 2:
                    performance_window.popleft()
                begin, sim_begin, frame_begin = performance_window[0]
                if now > begin:
                    runtime.performance = {'rtf': (world.sim_time-sim_begin)/(now-begin),
                                           'wall_fps': (frames-frame_begin)/(now-begin)}
                runtime.publish()
                last_publish = now
        recording_finished = True
    except KeyboardInterrupt:
        recording_finished = True
    finally:
        loop_finished = time.perf_counter()
        lifecycle['loop_wall_seconds'] = loop_finished-wall0
        lifecycle['frames_delivered_at_loop_end'] = viewer.frames if viewer is not None else 0
        try:
            try:
                if reference_recorder is not None:
                    if recording_finished and reference_recorder.meta['ticks']:
                        reference_recorder.finish()
                    reference_recorder.close()
            finally:
                render_close_started = time.perf_counter()
                video = viewer.close() if viewer is not None else None
                lifecycle['render_drain_close_seconds'] = time.perf_counter()-render_close_started
        finally:
            shutdown_started = time.perf_counter()
            server.close()
            runtime.close()
            lifecycle['server_runtime_close_seconds'] = time.perf_counter()-shutdown_started
        lifecycle['cleanup_wall_seconds'] = time.perf_counter()-loop_finished
    serialization_started = time.perf_counter()
    if pose_archive is not None:
        snapshots, times = pose_archive.finish()
    trace = recorder.arrays()
    if trace:
        np.savez_compressed(output / 'trace.npz', **trace)
        metrics = summarize(trace, runner.wbc_every / package.physics_hz, .5)
        errors = np.linalg.norm(trace['position_error'], axis=-1)
        metrics['position_error_p95_per_agent_m'] = np.percentile(errors, 95, axis=0).tolist()
        metrics['local_route_error_p95_per_agent_m'] = np.percentile(
            np.linalg.norm(trace['route_error'], axis=-1), 95, axis=0).tolist()
    else:
        metrics = {}
    if snapshots:
        np.savez_compressed(output / 'body_states.npz', body_q=snapshots, sim_time=times)
    if len(frame_wall_times) > 1:
        intervals = np.diff(frame_wall_times)
        metrics['render_performance'] = {
            'wall_fps': (len(frame_wall_times)-1)/(frame_wall_times[-1]-frame_wall_times[0]),
            'frame_interval_p50_ms': 1000*float(np.median(intervals)),
            'frame_interval_p95_ms': 1000*float(np.percentile(intervals, 95)),
            'measured_frames': len(frame_wall_times),
        }
    lifecycle['trace_serialization_seconds'] = time.perf_counter()-serialization_started
    metrics['lifecycle'] = lifecycle
    # Record actual controllers, not just requested devices or environment flags.
    # Web navigation can replace the startup Player during this run.
    metrics['final_players'] = [
        {'team': i, 'implementation': type(player).__name__, 'device': str(player.device),
         'agents': teams.ids[i].tolist()}
        for i, player in enumerate(teams.players)
    ]
    metrics.update(sim_seconds=world.sim_time, wall_seconds=time.perf_counter() - wall0,
                   hold=runtime.hold, formation=safety.report() if safety else None,
                   timing=runner.timer.report(), video=video)

    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)
    if runtime.reconfigure_request is not None:
        from uuid import uuid4

        from omega_playground.ops.robot import restart_arguments
        next_output = output.parent / ('team_config_' + uuid4().hex[:12])
        if runtime.reconfigure_package != package.root.resolve():
            args.package = str(runtime.reconfigure_package)
            args.render_hz = None  # Follow the newly selected policy's control cadence.
        args.armed = runtime.reconfigure_armed
        args.avatar = runtime.reconfigure_avatar
        if runtime.reconfigure_assets:
            args.physics_asset = runtime.reconfigure_assets['physics_asset']
            args.visual_model = runtime.reconfigure_assets['visual_model']
            args.render_environment = runtime.reconfigure_assets.get('render_environment')
        argv = restart_arguments(args, runtime.reconfigure_request.resolve(), next_output.resolve())
        os.execv(argv[0], argv)
    return 1 if runtime.hold else 0
