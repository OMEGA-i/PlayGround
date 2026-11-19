# OmegaPlayground release candidate

This source snapshot starts from `b452422bde3d25f402df896aa0ccc8185eb8932c`. The separately developed G1 latent experiment checkout is `fe02c34652a18a22c1c83f142a2567bd7fc6a88e`. That checkout adds latent rollout and diagnostics and changes the live planner/client/worker. The release example uses the candidate's native reference adapter and pinned G1 tracker.

[Run the G1 diagonal replay](examples/g1-diagonal/README.md). The candidate retains tracker pins in [trackers.json](configs/supermimic/trackers.json), and Joystick pins in [joysticks.json](configs/supermimic/joysticks.json). The independent tracker runtime used for the paired example is `d539e2c6ba8aa0fabdc6640221313879fa201bd6`. Scene packages, baked physics assets, policy binaries, robot XML/meshes, recorded captures and original design artwork are external inputs. Standalone installations can inspect the command-line interface without them.

## Source and dependency notices

- The [MJWarp PBR renderer](src/omega_playground/viewer/mjwarp_pbr_render.py) retains the Newton Developers' attribution. Its [Apache License 2.0 text](licenses/Apache-2.0.txt) accompanies this candidate.
- MuJoCo, MuJoCo Warp, Warp and the upstream tracker are separately installed dependencies. The G1 assets and pretrained policies have their own provenance and distribution terms. The example checks the tracker, joint ordering and robot XML identities before execution.
- Team-authored source is licensed under Apache-2.0. Upstream components retain their own terms.
