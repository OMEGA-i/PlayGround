# SuperMimic locomotion isolation — 2026-11-05

The investigation separates model identity, generated motion, native Tracker execution, live heading feedback, and Auto Player control. It does not treat an episode without a fall as successful navigation.

## Reproduction

On the active L20 host, from the Playground repository, with the SuperMimicKit environment and `PYTHONPATH=src`:

```bash
python tools/supermimic/run_supermimic.py diagnose h2 artifacts/qc/new-h2
python tools/supermimic/run_supermimic.py diagnose h2 artifacts/qc/new-h2-source --scenarios source_forward source_left source_idle
python tools/supermimic/run_supermimic.py diagnose h2 artifacts/qc/new-h2-auto --scenarios auto --steps 1500
python tools/supermimic/run_supermimic.py diagnose h2 artifacts/qc/new-h2-turn --scenarios auto_turn
```

Repeat for `a3` and `sih2300`. Use a new output directory. Joystick runs in its owner environment; Tracker and physics use the SuperMimicKit environment. GPU MJWarp runs at 200 Hz, Tracker at 50 Hz and Auto Player at 1 Hz. The diagnostic uses the actual live `World.select_command` adapter and Player, without web or rendering overhead. Output includes references, physical states, body poses, per-second Auto commands/phases, and replay metrics.

Saved evidence: `artifacts/qc/locomotion-isolation-20261105/` on L20 and the local workstation. Physics uses the native flat ground, not Campus/Obstacle collisions. The three `*-auto/auto/replay.mp4` videos render the saved physical states against the Obstacle background; their sampled frames were inspected. This isolates locomotion, not scene traversal.

## Identity and conversion

The pinned Tracker verifies checkpoint digest, iteration, robot identity, named joints/bodies, native XML digest and MJWarp training backend. References are mapped by joint name, sampled at 50 Hz, and retain SI-unit velocities. There is no observed G1/H2/A3/SIH2300 checkpoint or skeleton mix-up.

For generated forward, idle-to-forward and idle clips, no joint-limit clipping occurred. The largest per-release projection p95 is 0.035 rad for H2 and 0.041 rad for A3; SIH2300 is below 5e-16 rad. These measurements rule against a gross rotation-conversion fault in these tests, not every possible motion.

Joystick uses Bones-seed 25 FPS features. The currently pinned Trackers are the `a_lafan_50k` validation-best checkpoints: H2 20k, A3 15k and SIH2300 20k. Matching model/asset schemas do not establish that the Tracker can execute all Bones motion primitives.

## Matched-reference tests

Each saved reference is replayed twice from the same reference initial state, with identical checkpoints: native Tracker goals, and live goals with frame-zero yaw anchored to measured heading. Generation is bypassed during replay. Recorded LAFAN, generated forward, idle-to-forward and idle are distinct input cases; comparisons across cases are not matched-initial-pose experiments.

With live heading feedback, the 10-second, 0.65 m/s generated-forward cases ended at:

| Robot | Reference X (m) | Actual X (m) | Reference Y (m) | Actual Y (m) |
| --- | ---: | ---: | ---: | ---: |
| H2 | 6.304 | 3.063 | 0.145 | -0.875 |
| A3 | 6.151 | 5.913 | 0.052 | 0.098 |
| SIH2300 | 6.385 | 6.267 | -0.083 | -1.218 |

Generated idle, 10-second root XYZ RMSE with live heading feedback: H2 **0.480 m**, A3 **0.035 m**, SIH2300 **0.049 m**. No tilt/height termination was triggered in these short cases. That termination does not check trajectory error.

Removing live yaw feedback is not a general fix. For H2 generated forward, native goals accumulated roughly 90 degrees of heading drift; live feedback reduced that drift but did not repair forward speed and lateral tracking. It also does not improve every reference: H2's original Bones idle is worse with live anchoring than without it.

Raw Bones primitives bypass both Joystick generation and body-rotation conversion. With live heading feedback, root XYZ RMSE was:

| Robot | Original forward, 3.68 s | Original forward-left, 8.80 s | Original idle, 9.28 s |
| --- | ---: | ---: | ---: |
| H2 | 0.685 m | 2.131 m | 1.816 m |
| A3 | 0.139 m | 0.521 m | 0.044 m |
| SIH2300 | 0.357 m | 1.618 m | 0.034 m |

These source clips have their original speeds and poses; their metrics are not direct matched comparisons against the 0.65 m/s generated clips. The native Tracker/physics execution of supplied source motion is already poor in some cases, without either generator or conversion. For SIH2300 forward-left, native goals give 0.869 m RMSE versus 1.618 m with live yaw anchoring. The added heading feedback changes the learned goal distribution and is itself an integration tradeoff, not an independently accepted controller.

## Auto control findings

An 8 m straight task starts from the physical initial position. At 10 seconds:

- H2 reaches approximately `(1.22, -0.31)` m. Idle drift delays the turn/accelerate transition. At second 5 the Player issues a backwards correction to return toward its time-indexed path, followed by forward saturation.
- A3 reaches approximately `(4.81, 0.03)` m. It enters acceleration at second 2 and can track forward; it is not generally unable to walk.
- SIH2300 reaches approximately `(4.95, -0.92)` m. Its executed reference ends at `(5.21, +1.52)` m. The Player is asking for positive lateral correction, and the reference moves that way, but the physical agent continues on the opposite side.

For a 90-degree initial heading change, A3 enters acceleration around second 8 and SIH2300 around second 9. H2 is still in the turn phase at second 9, despite reaching approximately 85 degrees; the transition also requires low physical speed. Initial turning is slow, rather than categorically disconnected.

The Joystick predicts 64 frames at 25 FPS (2.56 s including context) and releases 20 new frames (0.8 s). Live code prefetches when 23 Tracker frames remain. **25 FPS is not a 25 Hz command/replanning rate.** The measured Auto traces show the reference remaining nearly stationary after the Player begins accelerating, while the Player raises its corrective velocity. This delay and the ability to request backward/lateral corrections interact poorly with incomplete low-level stop/strafe tracking.

A separate 30-second replay of the same 8 m straight task covers arrival:

| Robot | Final XY (m), target (8, 0) | Last phase | Observation |
| --- | --- | --- | --- |
| H2 | (8.245, -0.223) | settle | Repeated forward/backward and lateral correction; no completed stop |
| A3 | (8.040, 0.073) | complete | Completes after arrival corrections; complete by the 27 s sample |
| SIH2300 | (7.382, 0.042) | settle | Repeated corrective commands; no completed stop |

The 30-second runs generate their references reactively and are independent trials, not exact continuations of the saved 10-second runs. The existing live H2 world was also observed stuck in `decelerate` after more than 1,100 simulation seconds; it was not reset or interrupted for this experiment.

## Interpretation and next acceptance gate

The evidence supports a behavior-compatibility problem, compounded by Auto control timing, rather than one universal wrong-joint-order bug:

1. Establish per-embodiment execution of raw Bones idle/forward/diagonal primitives before tuning FFPD. Failure without Joystick generation cannot be fixed solely by regenerating plans. A reference may also be physically unsuitable; this test alone does not distinguish that from insufficient Tracker generalization.
2. Validate generated velocity changes and lateral commands against those same primitives. Record command, generated root velocity and measured velocity separately.
3. Align Auto command smoothing, reference release latency, and arrival/stop behavior with the measured execution capability. Do not widen safety/arrival thresholds merely to label tasks successful.
4. Preserve the current 1 Hz Player contract while measuring the actual command-to-motion delay. Do not silently treat the 25 FPS reference clock as the planner update frequency.

No production policy weights, Player gains, heading behavior or active user task were changed by this investigation. Changes add reproducible diagnostics and repair the offline renderer's prepared-texture directory argument.
