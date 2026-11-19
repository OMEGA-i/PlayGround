import argparse
import json
import sys


def _verify_policy(args) -> int:
    from omega_playground.policy import package, verify

    report = verify.verify(package.load(args.package), device=args.device)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _run(args) -> int:
    import math
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('seconds must be positive and finite')
    args.auto_run_seconds = args.seconds
    args.start_paused = False
    args.port = None
    return _serve(args)


def _serve(args) -> int:
    import torch
    if torch.device(args.device).type != 'cuda':
        raise ValueError('Physics requires CUDA')
    if not args.prepared_assets and not args.physics_asset:
        raise ValueError('A prepared MJWarp physics asset or registry is required')
    profiler = None
    try:
        from omega_playground.ops.serve import serve
        if getattr(args, 'profile_python', False):
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        return serve(args)
    finally:
        if profiler is not None:
            from pathlib import Path
            profiler.disable()
            if Path(args.output).is_dir():
                profiler.dump_stats(str(Path(args.output)/'runtime_profile.pstats'))


def _bake_map(args) -> int:
    from pathlib import Path
    from omega_playground.map.navigation import build
    from omega_playground.map.terrain import BRIDGE_COLLISIONS
    from omega_playground.scene.package import load
    scene = load(args.scene)
    ground = None
    if scene.collision_sha256 in BRIDGE_COLLISIONS:
        import mujoco
        from omega_playground.sim.mjwarp_world import MeshGround
        from omega_playground.ops.prepared_assets import checked_metadata
        if not args.physics_asset:
            raise ValueError('Bridge height baking requires --physics-asset for this scene')
        asset = Path(args.physics_asset)
        meta = checked_metadata(asset)
        if (meta['schema'] != 'omega.mjwarp-physics/1' or meta['scene_sha256'] != scene.scene_sha256
                or meta['collision_sha256'] != scene.collision_sha256):
            raise ValueError('MJWarp collision asset does not match the scene')
        ground = MeshGround(mujoco.MjModel.from_binary_path(str(asset/meta['entry'])), 'cuda:0')
    print(json.dumps(build(scene.entry, args.output, args.resolution, terrain_ground=ground), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="omega-playground")
    sub = parser.add_subparsers(dest="command", required=True)

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--profile-python', action='store_true', help='Save host profile to the run directory')
    p.add_argument('--prepared-assets', help='Explicit offline MJWarp registry for configuration rebuilds')
    p.add_argument('--physics-asset', help='Offline prepared MJWarp physics asset')
    p.add_argument('--visual-model', help='Offline prepared MJWarp scene and robot model')
    p.add_argument('--render-environment', help='Offline HDR/MIS bundle; enables experimental linear MJWarp PBR')
    p.add_argument('--render-python', help='Isolated MJWarp interpreter')
    p.add_argument("--package", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--player", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--video", action="store_true")
    p.add_argument("--avatar", choices=["g1"], default=None)
    p.add_argument("--armed", action=argparse.BooleanOptionalAction, default=False, help="display shield and baton")
    p.add_argument("--video-codec", choices=["h265", "h264"], default="h265")
    p.add_argument("--video-bitrate-kbps", type=int, default=1500, help="HEVC CBR target; bounds network buffering")
    p.add_argument("--render-hz", type=float, default=None, help="default: align with WBC (50 Hz -> 25 FPS; 30 Hz -> 30 FPS)")
    p.add_argument("--dlss-frame-generation", choices=["off", "2x"], default="off",
                   help="reserved 2x output; rejected until offscreen generated-frame capture is supported")
    p.add_argument("--render-width", type=int, default=1280)
    p.add_argument("--render-height", type=int, default=720)
    p.add_argument("--realtime", action="store_true", help="pace simulation to wall time for live operation")
    p.add_argument("--camera", choices=["qc", "chase", "squad", "top"], default="qc")
    p.add_argument("--render-profile", choices=["quality", "realtime"], default="realtime")
    p.add_argument("--auto-run-seconds", type=float, default=None, help="headless QC: resume and exit after this time")
    p.add_argument('--start-paused', action='store_true',
                   help='Wait for an explicit task/resume even with --auto-run-seconds; keep its simulation-time exit bound')
    p.add_argument('--record-reference', action='store_true',
                   help='Diagnostic: archive exact WBC references/states for causal replay; adds IO overhead')
    sub.add_parser("serve", parents=[p], help="Campus operations workbench; starts paused").set_defaults(fn=_serve)
    batch = sub.add_parser("run", parents=[p], help="bounded MJWarp run without a Web listener")
    batch.add_argument("--seconds", required=True, type=float, help="simulation duration")
    batch.set_defaults(fn=_run)

    p = sub.add_parser("bake-map", help="bake collision support, clearance and the 2D map from a verified scene")
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resolution", type=float, default=.5)
    p.add_argument('--physics-asset', help='Prepared MJWarp physics bundle for bridge height queries')
    p.set_defaults(fn=_bake_map)

    p = sub.add_parser("verify-policy", help="hash checks + fixed-input parity of a policy package")
    p.add_argument("package")
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=_verify_policy)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
