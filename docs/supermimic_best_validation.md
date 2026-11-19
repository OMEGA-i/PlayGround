# SuperMimic a-course validation-best integration — 2026-11-04

The live and offline entrypoints use `policy/supermimic/release.py` and the same [pinned receipt](../configs/supermimic/trackers.json). This supersedes the 2026-11-03 selection; the old receipt is historical evidence. Private checkpoints remain in shared storage and are not published as workbench web assets.

| Robot | Iterations | Replay steps | Tracker failure | Root XYZ RMSE (m) | Root XYZ max error (m) |
| --- | ---: | ---: | --- | ---: | ---: |
| g1 | 45000 | 500 | False | 0.460 | 0.939 |
| sih2300 | 20000 | 500 | False | 0.529 | 0.980 |
| h2 | 20000 | 500 | False | 0.911 | 1.976 |
| a3 | 15000 | 500 | False | 0.922 | 1.916 |

Validation used the held-out `walk3_subject3` clip, source frames 1000:1511 at 50 Hz, on an NVIDIA L20. Each run starts from the reference state and advances 500 Tracker steps / 2,000 MJWarp GPU substeps. No Joystick, heading re-anchoring, or scene collision geometry is involved in this isolated test. Campus is a rendering background only. Root errors are absolute XYZ distances, including accumulated drift; no-failure does not imply navigation or long-duration stop/turn acceptance.

`tools/supermimic/verify_supermimic_release.py` compares the frozen web metadata with each loaded checkpoint: named joint/body schema, source XML hash, layout dimensions, history=10, goal frames=11, 50 Hz control / 200 Hz physics, strict Actor state load including normalizer mean/std, default pose, action scale and checkpoint-derived PD gains. Runtime verifies the full checkpoint SHA-256 before loading. The training/export host and L20 have byte-identical `core.py`, `mjwarp.py`, `kinematics.py`, and `physx.py`; hashes are in the receipt.

All four replay videos were generated from saved physical body poses, and sampled frames were visually inspected. No missing robot or detached body binding was observed in those samples. This is not a full motion-dataset visual acceptance.

Private QC artifacts on L20 and the local workstation: `artifacts/qc/supermimic-best-20261104/<robot>/` contains `contract.json`, `reference50.npz`, `rollout.npz`, `body_states.npz`, `result.json`, `replay.mp4`, `ego.mp4`, and `qc.jpg`. `summary.json` records the measurements above. Source metadata snapshots are preserved separately in the QC root.

Online reload confirmed `/api/state`: `ready=true`, `avatar=sih2300`, `tracker_release=a-lafan-best-20261104`, `tracker_iterations=20000`, SHA-256 `7de00d07ef9784d19a3d8f4df43881f51050bcc2dd43f8356399ceb3d7d7cdc9`. The worker is paused and ready. Other embodiments resolve their new checkpoint when selected/rebuilt. Existing Joystick pins and SONIC remain unchanged.
