# First layered acceptance run — 2026-11-08

Correction: the runs below used MJWarp 3.8.0.3 without deterministic mode.
Their exact-equality failures are **not applicable as an integration rejection**.
They remain numerical observations, not evidence that the adapter is incorrect.
See [the isolated MJWarp 4 determinism audit](determinism_20261108.md).

Status: **no embodiment has passed the complete acceptance chain**. This report
does not change checkpoint selection, physics parameters or the online service.
Implementation and commands: [layered acceptance](layered_acceptance.md).

Evidence root on L20 persistent storage:
`$OMEGA_SHARED_ROOT/playground/results/acceptance-20261108-v2` (about 1.6 GB).
Local reports and videos: `artifacts/qc/acceptance-20261108/`.

## Matrix

| Embodiment | Tracker exact equality | Max difference between two root trajectories | Joystick independent | Plane closed loop | Scene / navigation |
|---|---|---:|---|---|---|
| G1 | not applicable (nondeterministic) | 0.081604 m | blocked, 48 cases recorded | blocked by gates 1/2 | blocked |
| A3 | not applicable (nondeterministic) | 0.585978 m | blocked, 48 cases recorded | blocked by gates 1/2 | blocked |
| H2 | not applicable (nondeterministic) | 1.970833 m | blocked, 48 cases recorded | blocked by gates 1/2 | blocked |
| SIH2300 | not applicable (nondeterministic) | 0.348390 m | blocked, 48 cases recorded | blocked by gates 1/2 | blocked |

The distance column is **native-evaluator execution versus adapter execution**,
not actual-versus-reference tracking error. All runs use 500 control steps on
the analytic native plane. Both branches recorded zero `done` flags, despite
the differences above. `done=false` is therefore not an equality gate.

The plane closed-loop runner was invoked for all four embodiments and correctly
refused execution on the failed/unqualified prerequisites. Scene/navigation
acceptance is neither run nor inferred from previous production quotas. Their
full unified executors remain follow-up work after the upstream gates.

## First reproducible differences

1. **GPU execution divergence exists even between two native loops.** G1 starts
   with identical state, history, reference and first action; first-step qvel
   differs by `2.384185791015625e-7`. Differences subsequently accumulate.
   Native/native controls gave maximum root differences of 0.162483 m for G1
   and 1.970833 m for H2. The latter nearly reproduces the adapter comparison.
   This does not establish a Playground-specific defect, nor does it justify
   accepting metre-scale drift. More native repeat/control-action trials are
   needed before defining a defensible numerical tolerance.
2. **Derived FK is not the same as the actor-consumed reference.** Raw reference
   body-state tails have small differences in A3/H2/SIH2300. SMK
   `Layout.goals` uses root and joint q/qdot and discards these appended FK
   states. `input-analysis.json` separately checks those consumed fields and
   the initial history/action: **all four match exactly** in this run. Do not attribute actor divergence to an unused
   FK tail merely because the raw arrays differ.
3. **Joystick pose anchoring does not ensure velocity continuity.** Input final
   joint velocity versus generated first-frame velocity has these maximum
   differences across the 48 cases:

   | G1 | A3 | H2 | SIH2300 |
   |---:|---:|---:|---:|
   | 7.839 rad/s | 9.583 rad/s | 6.059 rad/s | 14.376 rad/s |

   These are offline-reference boundary measurements, not observed physical
   impacts. They identify a concrete resampling/continuity issue for investigation.

## Joystick scope

240 cases total: three seeds × two initial motion histories × eight sequential
commands per embodiment. All 30 same-input/same-seed repeated initial requests
were bitwise identical. Finite values, timestamps/horizon, unit quaternions and
post-projection authored joint limits passed these cases. Projection clamping is
reported separately, so joint-limit checks do not hide clamped generator outputs.

Command XY velocity error, heading error, relative half-response time, root/joint
acceleration, anchor errors, FK foot heights and near-floor speed proxies are
saved per case. No versioned command-error/acceleration/slip acceptance limits
have been supplied; they remain **measurements**, not a pass. Histories after
the initial motion are generated references, not measured physical rollouts.
Actual-rollout history coverage remains outstanding.

## Saved evidence and visual inspection

Each Tracker branch includes `physics.mjb`, `physics.xml`, `trace.npz` (actual
state, qpos/qvel, PD target, action, reference window and observation history),
body poses and control-boundary GPU contact arrays. These contacts are sampled
at 50 Hz, not all 200 Hz physics substeps. `provenance.json` records checkpoint,
asset/motion hashes, actually loaded SMK checkout, evaluator source hash, runtime
versions and robot-specific physics settings. `resource-hashes.json` fingerprints
the compiled model and saved reference.

Five synchronized native/adapter videos and five reference-only command videos
were rendered. Representative saved comparison/reference frames were inspected;
this is **not** full-video manual review. Metric QC stills use 2.5 m vertical
framing and 0.5 m grid lines. Reference videos explicitly say FK/no physics.

| Embodiment | Native / adapter comparison | Joystick references only |
|---|---|---|
| G1 | `artifacts/qc/acceptance-20261108/layer1/g1/comparison.mp4` | `artifacts/qc/acceptance-20261108/layer2/g1/references.mp4` |
| A3 | `artifacts/qc/acceptance-20261108/layer1/a3/comparison.mp4` | `artifacts/qc/acceptance-20261108/layer2/a3/references.mp4` |
| H2 | `artifacts/qc/acceptance-20261108/layer1/h2/comparison.mp4` | `artifacts/qc/acceptance-20261108/layer2/h2/references.mp4` |
| SIH2300 | `artifacts/qc/acceptance-20261108/layer1/sih2300/comparison.mp4` | `artifacts/qc/acceptance-20261108/layer2/sih2300/references.mp4` |

The initial `acceptance-20261108` run (without `-v2`) had incorrect stream
ownership in the dual-instance diagnostic and is invalidated. It is retained
for traceability, excluded from this matrix. No baseline thresholds were widened.

Validation of the tooling: focused gate tests passed (2 tests); F-series lint and
Python compilation passed. Existing unrelated working-tree changes were preserved.
