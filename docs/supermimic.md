# SuperMimic

The Playground workbench runs on Pod port 7777. Start or check the services with `bash deploy/start.sh`. Jupyter keeps port 8888. Every browser follows the same active world. Pause the simulation before switching between SONIC and SuperMimic in **机器人设置**.

The workbench controls one embodiment in Campus or Obstacle. In Auto, click or drag points on the map, edit the B-spline, and execute. FFPD reads the measured robot state at 1 Hz. In Manual, W/S move and A/D turn; releasing the keys clears the command. Rebuilding applies the chosen avatar and equipment. In-place and task-start recovery retain the drawn task.

SuperMimic keeps OmegaJoystick and the SuperMimicKit Tracker in their owner environments in sibling worktrees. The local execute-then-plan path releases all 60 future frames at 25 Hz, anchored on four executed proprio samples. It waits for 120 control intervals to finish before replanning. The Tracker streams eleven-frame windows at 50 Hz, preserving its ten-frame observation history across chunks; missing terminal lookahead holds the final pose with zero velocity. Each chunk uses a fixed measured-heading local-to-world transform. Decision camera snapshots wait for the matching physics timestamp. This revised path still requires GPU closed-loop acceptance; see [the export schema](sdg_eawm_export_schema.md). Its MJWarp training backend runs four GPU substeps per Tracker step. The live tracker anchors the reference window to measured yaw; the observed heading error remains available until the physical robot completes its turn. The authoritative device body poses feed a separate MJWarp renderer and head-mounted camera.

The four a-course validation-best Trackers (SIH2300 20k, G1 45k, H2 20k, A3 15k) are pinned in [the receipt](../configs/supermimic/trackers.json). H2 uses a frozen candidate Joystick (root step 35,000, pose step 20,000); its stopping behavior has not passed acceptance. UE4 and UE5 use the SIH2300 Tracker and their own precompiled visual bindings. An appearance/Armed combination is offered only when its prepared bundle exists in that scene.

For an offline diagnostic replay of the pinned release, use `python3 tools/supermimic/run_supermimic.py run g1 /path/to/new/output`. The public workbench uses the persistent GPU worker rather than this offline job.

The [locomotion isolation report](supermimic_locomotion_diagnosis.md) separates raw source motions, generated references, heading feedback and Auto control. In the tested 30-second, 8 m flat-ground task, A3 completed; H2 and SIH2300 remained in arrival correction. These releases are not uniformly accepted for Auto navigation. Use the same runner's `diagnose` stage to reproduce the ablations without changing the live world.

## H2 Joystick candidate received 2026-11-05

The completed 100k training delivery is received as a **candidate, not activated**. Keep `configs/supermimic/joysticks.json` unchanged until isolated comparison against the current H2 candidate passes. The earlier locomotion results used root 35k / pose 20k and do not measure this new pair.

Paths below are relative to `$OMEGA_SHARED_ROOT/joystick/runs/h2/fsq100k_25fps/`:

| Component | File | Selected step | SHA-256 |
| --- | --- | ---: | --- |
| Tokenizer | `tokenizer/last.ckpt` | 100000 | `4df5a9fe28b76b82e746e3bd95f1f101e64c42c485796b82896bc4394f1051d2` |
| Root | `root/best.ckpt` | 85000 | `a6c2d8697b1972f0e8faba1279db7093102d930056a7b788a9ea38dd2ed912e6` |
| Pose | `pose/best.ckpt` | 90000 | `c806fc195219e9821cf83ab5759c7824679fef1ec0ce0dfa3e5824edc0cac5ca` |

Receiver verification used read-only access to the shared originals: all three file digests match; root/pose `state_dict` tensors are finite; embedded checkpoint audits identify H2, 25 FPS planner output and 50 Hz Tracker input; their actual global steps are 85k/90k. The pose checkpoint binds the tokenizer **last** digest above, not tokenizer best.

Both embedded audits match feature manifest SHA `be688db47998f7fe7df539b8d758ea023627848fec36549a390f70234e618698`, independently checked against `datasets/bones-seed-25fps/h2/manifest.json`. Native XML `$OMEGA_SHARED_ROOT/supermimic/assets/robots/h2/robot.xml` hashes to `4f8e5fc93923157a3f5ba86c40f34b2def5239a79455677f68ae965402cdfd21`, matching the existing native H2 Tracker contract.

The producer reports all stages completed 100k training, finite last/best model and optimizer tensors, matching completion receipts and complete SwanLab synchronization receipts. Those broader optimizer/cloud-sync checks were not independently repeated by the receiver. Training losses and tokenizer gates are not physical closed-loop acceptance.

L20 SSH returned connection refused during receipt. No online worker was restarted, no active checkpoint pins changed, and no GPU inference or new-model physical replay is claimed. Pending acceptance is isolated load/contract verification followed by the same idle, forward, initial-turn and Auto arrival diagnostics before promotion.
