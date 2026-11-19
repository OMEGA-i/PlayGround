# MJWarp 4 deterministic audit — 2026-11-08

## Implementation follow-up

The measurements below describe **unpatched** upstream 4.0.0. Two independent
implementation branches now fill its contact and constraint sorting hooks:

| Branch | Commit | Scope |
|---|---|---|
| `fix/deterministic-contact-order` | `6dc7134` | Canonical full-record contact ordering with stable GPU radix passes |
| `fix/deterministic-constraint-order` | `8bb30fd` | Canonical constraint rows, sparse Jacobian storage, Newton blocks, contact and DOF-friction addresses |

Both branches start at upstream `4d1c64bc077d725247e9bf5e2ac3f357749ff804`.
Combined validation uses `integration/deterministic-ordering`, commit `d9d36cb`.
Sorting stays on GPU and supports CUDA graph capture; neither change enables
deterministic mode by default. Contact keys use physical bit patterns, not
quantized positions. Constraint sorting preserves each object's internal axes.

The initial constraint prototype missed `dof_frictionadr`. A3's multiple friction
rows exposed this as intermittent NaNs after row reordering. The strengthened
regression failed all eight parameter combinations before the address-remapping
fix and passed all eight afterward. Replaying the saved failing A3 state now
produces finite state through all four physics substeps and forward evaluation.

At `d9d36cb`, A3 completed two independent 500-control-step native/Playground
comparisons: actor-consumed references, actions, PD targets, history, joint and
body state were bitwise equal, with zero root difference and zero done flags.
The two processes' saved execution traces also match exactly within each adapter.
Unused derived FK reference tails differ between adapters by up to `2.86e-6`;
this is not a claim that the complete raw reference archives match.

G1, H2 and SIH2300 were all rerun on final commit `d9d36cb`: each passed
500 steps with bitwise-equal physical execution, zero root difference and zero
done flags. G1 also passed an independent-process repeat on preceding commit
`f435a56`. These results establish
repeatability for the tested rigid-body configurations on the same L20 GPU,
not cross-device determinism, all flex models, navigation stability or visual QC.
No policy, gains, scene, production service or runtime was changed.

Latest full upstream suite on GPU: **2585 passed, 41 skipped** (81.78 s).
Ruff, formatting, YAML, lockfile and kernel-analyzer pre-commit checks all passed.
Contact-only full-suite validation previously passed 2577 tests, with 41 skipped.

A captured box-on-plane microbenchmark measured the following median milliseconds
per batched physics step (three samples of 500 steps, L20 GPU 1, Warp 1.19,
contact capacity 16 per world, constraint capacity 64, no overflow):

| Worlds | Mode | Upstream 4.0 | Combined patch |
|---:|---|---:|---:|
| 1 | NONE | 0.1534 | 0.1536 |
| 1 | CONTACTS | 0.1536 | 0.6225 |
| 1 | ALL | 0.3279 | 0.8418 |
| 32 | NONE | 0.1579 | 0.1579 |
| 32 | CONTACTS | 0.1578 | 0.6301 |
| 32 | ALL | 0.3925 | 0.9103 |

This intentionally conservative full-record sorting has a measurable cost.
The benchmark is not a humanoid throughput or rendering FPS result. Unpatched
CONTACTS invokes an empty hook, so its timing does not represent equivalent work.

Both implementation PRs are now published independently:
[contact ordering #1923](https://github.com/google-deepmind/mujoco_warp/pull/1923)
and [constraint ordering #1924](https://github.com/google-deepmind/mujoco_warp/pull/1924).
Existing documentation-only draft
[PR #1922](https://github.com/google-deepmind/mujoco_warp/pull/1922) remains separate.
The implementation branches, patches, bundle, logs and benchmark are preserved
locally under `artifacts/qc/deterministic-20261108/implementation/` (not tracked).
Remote per-run evidence remains under the root listed below, including
`friction-remap-adapter/{g1,h2,a3,sih2300}`, `friction-remap-repeat/a3` and
`a3-warm-failure/probe-fixed.json`.

The previous MJWarp 3.8.0.3 exact-equality rejection was not a valid integration
gate: deterministic execution was not enabled. Those traces remain useful as
observations, but cannot establish an adapter defect.

## Isolated runtime

Executed on L20 GPU 1, separately from the online service. No production runtime,
checkpoint, scene, gains or solver settings were changed. MJWarp/MuJoCo 4.0.0
were installed in an overlay; Warp 1.18.0 and 1.19.0 were tested separately.
The native SMK checkout is pinned to `1dc4f49` plus compatibility commit
`e360df3` (skip the removed `Model.stat` field on MJWarp 4). Playground tests
used `313d67d` / `3a1f2fd`; each run records its loaded code identities.

Official model control: `model.opt.deterministic = DeterminismType.ALL` (7),
set before warmup and CUDA graph capture, with readback from both models.
ALL selects the contact, constraint and atomic feature bits. Follow-up source
inspection found that contact and constraint sorting are still empty hooks in
4.0.0: `_sort_contacts` and `_sort_constraints` contain only TODO/pass.
The flag readback therefore does not establish full determinism support.
See [MJWarp 4 release](https://github.com/google-deepmind/mujoco_warp/releases/tag/v4.0.0)
and [Warp deterministic execution](https://nvidia.github.io/warp/v1.19/user_guide/execution_and_performance/deterministic_execution.html).

## Measurements

G1, identical checkpoint, recorded motion and initial observable state; analytic
native plane. Tracker 50 Hz, physics 200 Hz. Root difference below is between
the two runs, **not reference tracking error**.

| Warp | MJWarp mode | Comparison | Control steps | First differing physical state | Max root difference |
|---|---|---|---:|---:|---:|
| 1.18 | NONE | native / Playground | 500 | 1 | 0.059091 m |
| 1.18 | ALL | native / Playground | 500 | 3 | 0.119664 m |
| 1.19 | ALL | native / Playground | 500 | 1 | 0.269526 m |
| 1.19 | ALL | native / native | 100 | 2 | 0.000796 m |

All completed runs have identical actor-consumed references and initial actions,
and zero `done` flags. A full comparison of the saved native/Playground compiled
MuJoCo model arrays in the Warp 1.18 ALL run found **487 arrays, no differences**.
This does not prove that all internal GPU solver/cache state is identical.

The first qvel difference is small: `1.19e-7` with Warp 1.18 ALL,
`2.38e-7` with Warp 1.19 ALL, and `2.29e-6` in the native/native control.
The larger end-to-end differences develop after feedback through the actor.

The model-level ALL flag alone has not established repeatability in this harness.
Upstream PR [#1687](https://github.com/google-deepmind/mujoco_warp/pull/1687)
explicitly lists canonical contact/constraint sorting as future work. Thus ALL
is not a complete trajectory determinism guarantee in this release; the earlier
description of it as completed stable ordering was incorrect.
Native/native also diverges, so this is **not sufficient evidence of a Playground
adapter defect or an upstream engine bug**. Initial internal state, reset/capture
behavior and remaining nondeterministic operations require isolation. Larger
trajectory differences in these individual runs do not establish that the newer
runtime is less accurate or less stable.

## Global Warp RUN_TO_RUN

Setting `wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN` before
importing the simulator, together with model-level ALL, **fails during constructor
warmup**, before a trajectory can be recorded. Warp 1.19 raises
`WarpDeterminismError` for `ccd_kernel_builder__locals__ccd_kernel`: an
`atomic_add()` call, index or increment changes when the kernel is repeated to
assign stable slots. Warp explicitly says that outputs may already have changed
and must be discarded/reset. The process exited; the check was not bypassed.

Thus global deterministic mode is not a working replacement for this current
native CCD path. This is not a completed replay, and no equality or locomotion
result is assigned to it. The model-level ALL result remains a separate issue.
The error is not proof that the model-level divergence has the same cause.

## Reproduction and evidence

Runtime pins: `configs/acceptance/mjwarp4-runtime.txt` (isolated use only).
Remote evidence root:
`$OMEGA_SHARED_ROOT/playground/results/deterministic-20261108/`.
Local result/provenance copies: `artifacts/qc/deterministic-20261108/`.
Completed subdirectories: `mjw4-none`, `mjw4-all`, `mjw4-wp119-all`,
`mjw4-wp119-native-control`. The failed global run has only a startup log:
`global-native-startup.log`; its intended output directory was never created.
Saved traces contain state, actions, history, PD targets, contacts and models.

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$OMEGA_SHARED_ROOT/venvs/pg-mjw4-deterministic-overlay:$OMEGA_SHARED_ROOT/SuperMimicKit-pg-mjw4-audit:src \
../SuperMimicKit/.venv/bin/python tools/acceptance/tracker_plane.py \
  --robot g1 --deterministic --native-control --steps 100 --output NEW_DIRECTORY
```

Add `--warp-deterministic` to set Warp RUN_TO_RUN before importing simulator
kernels. The first global experiment used an equivalent Python launcher setting
the global flag before `runpy.run_path`:
`python -u -c 'import warp as wp; wp.config.deterministic=wp.DeterministicMode.RUN_TO_RUN; import runpy; runpy.run_path("tools/acceptance/tracker_plane.py",run_name="__main__")'`
followed by the same robot, deterministic, native-control, steps and output arguments.
Ordinary nondeterministic runs now return `not_applicable_nondeterministic`
instead of claiming an integration failure. Original evidence files are preserved.
