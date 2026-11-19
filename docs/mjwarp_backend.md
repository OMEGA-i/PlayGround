# GPU MJWarp backend

The L20 workbench uses MJWarp for both active workflows. SONIC uses
`MJWarpWorld`; SuperMimic retains its release-owned MJWarp runtime. Physics runs
on CUDA only. The native world integrates four 5 ms substeps in one CUDA graph
per 50 Hz Tracker tick. Player and Joystick scheduling remain unchanged.

Named policy joints select the released PD gains, torque limits and targets;
body outputs use package order, XYZW quaternions and world-frame link-origin
velocities. Passive hand joints remain native model joints. Recovery resets
physics and policy history while retaining the task and global simulation clock.

After integration, GPU kinematics and COM velocity transforms refresh the state
without an extra constraint solve. Contact warnings sample the last physics
substep. Non-foot contact while upright is a contact warning, not by itself a
fall. Warnings never pause simulation.

Physics sends authoritative CUDA body poses to the separate MJWarp renderer.
Main and ego cameras remain independent. HEVC over WebSocket and the existing UI
are unchanged. GPU simulation does not imply a fully asynchronous application:
logging, warning summaries and browser telemetry still read small host snapshots.

## Prepared assets and startup

Run from the OmegaPlayground checkout. The validated environment is
`../venvs/pg-mjwarp-clean-001/bin/python`: Python 3.12, torch 2.14.1,
mujoco/mujoco-warp 3.12.0, warp-lang 1.17.0, onnx 1.23.0rc1,
onnxruntime-gpu 1.30.0 and wsproto (WebSocket transport). It also contains the existing USD/image/video dependencies.
Isaac Sim is not needed for live SONIC startup.

Offline compilation, once per policy/scene revision (output must not exist):

```bash
PYTHONPATH=src ../venvs/pg-mjwarp-clean-001/bin/python tools/assets/prepare_mjwarp_physics.py \
  --package artifacts/policies/sonic-v1.1-g1-cuda-batch \
  --scene artifacts/scenes/VibeCampus-v20261013 --center 19 4 \
  --output artifacts/mjwarp-physics/g1-campus
PYTHONPATH=src ../venvs/pg-mjwarp-clean-001/bin/python tools/assets/prepare_mjwarp_physics.py \
  --package artifacts/policies/sonic-v1.1-g1-cuda-batch \
  --scene artifacts/scenes/VibeObstacle-v20261103 --center -6 -12.5 \
  --output artifacts/mjwarp-physics/g1-obstacle
bash deploy/start.sh
```

The exporter combines the native MJCF with the scene's authored static convex
collision pieces in world coordinates. The bundles bind policy, scene and
collision hashes plus the MuJoCo compiler version. Runtime checks their
checksums and loads the MJB; it does not convert USD or mount equipment.
`configs/mjwarp-assets.json` selects scene/Armed visual variants. First-use
Warp kernel compilation still takes time; it is not asset conversion.

The local workbench listens on the configured nginx port (default 7777).
SONIC worker is `18782`, SuperMimic `18783`, selector `18784`.
`serve` and `run` execute GPU MJWarp. For Campus bridge map baking, pass
`--physics-asset artifacts/mjwarp-physics/g1-campus` to `bake-map` so the
height queries use the prepared collision meshes. The SONIC launcher enables
real-time pacing so lighter scenes do not advance faster than wall time.

## L20 validation, 2026-11-04

- GPU contract tests cover named ordering, reset position/yaw, graph versus four
  eager GPU steps, missing/below-root support, and upright non-foot warnings.
- Campus: 15 s straight replay, 750 Tracker ticks, minimum root height over
  support 0.694 m, no fallen frames. Main/ego rendering retained.
- Campus web Auto: 90° and 180° turns followed by 2 m paths completed, endpoint
  errors 0.110 m / 0.292 m, final speeds 0.067 / 0.056 m/s. These are short trials,
  not a guarantee of long-run or arbitrary terrain stability.
- Obstacle web Auto: 2 m path completed after both recovery modes; endpoint error
  about 0.22 m. In-place recovery preserved XY exactly; task-start recovery
  restored (-6, -12.5). Manual forward input moved about 3.07 m, followed by
  zero input and settling to about 0.063 m/s.
- Campus main+ego 1280×720: first replay 17.1 FPS; after removing the redundant
  fifth solve, web trial median windows were 19.6 / 20.7 FPS. Obstacle unpaced
  trial median was 31.9 FPS, so production pacing caps it to the 25 FPS video
  cadence. A subsequent production smoke test through nginx measured 24.98 FPS
  and RTF 0.999 over a 10 s run. Both workflow state endpoints report
  `physics_backend: mujoco_warp`. Campus has not yet met the 25 FPS target. Timings include rendering
  and readback; asynchronous host dispatch time is not GPU physics execution time.

Saved videos, state samples and inspected images are in
`artifacts/qc/l20-mjwarp/` locally, and `artifacts/qc/mjwarp-migration/` on L20.
The initial straight replay is L20 run `mjwarp-g1-qc-1793755677`.

