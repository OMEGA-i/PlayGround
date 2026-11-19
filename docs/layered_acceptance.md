# SuperMimic layered integration acceptance

This acceptance suite is separate from production qualification. A successful
navigation episode, `done=false`, or an SDG quota cannot replace an upstream gate.
Configuration: `configs/acceptance/supermimic.json`.

First measured results and videos: [2026-11-08 report](acceptance_20261108.md).

Exact physical equality requires a validated deterministic runtime. Nondeterministic
MJWarp differences alone cannot reject an adapter. See the
[MJWarp 4 deterministic audit](determinism_20261108.md); model-level ALL has not
yet established native/native repeatability in this harness.

| Gate | Scope | Required evidence |
|---|---|---|
| 1 Tracker integration | Native SMK analytic `type="plane"`, fixed motion | Reference, initial state, history, actions, PD targets, physical states, exact loaded code/asset/checkpoint identity |
| 2 Joystick independently | Reference generation and interface, no Tracker | Commands, input histories, complete chunks, reproducibility, kinematics, anchoring and command-response measurements |
| 3 Plane closed loop | Joystick + Tracker | Requires gates 1 and 2; starts/stops, speed changes, turns, replanning, seeds/initial headings |
| 4 Scene contact controls | Replace support only, then add geometry | Fixed motion first, then Joystick; identical initial conditions; hash-pinned collision representation |
| 5 Full navigation | Campus/Obstacle and paths | Goal, clearance, contact and valid episode ratio; requires gates 1–4 |

Gates 1 and 2 run independently. The result matrix blocks downstream acceptance
when a prerequisite is failed, missing, or lacks a justified threshold. The current
executable suite implements gates 1 and 2, a prerequisite-gated plane closed-loop
command suite, and synchronized gate-1 rendering. Gate 3 refuses execution on
failed prerequisites; `plane_loop.py --diagnostic` explicitly produces non-acceptance
diagnostics if needed. Gates 4–5 have versioned scope/configuration but are **not yet
implemented in this runner**; production tools are not credited to those gates.

## Running

From the Playground checkout on a GPU host, with the released shared assets:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  ../SuperMimicKit/.venv/bin/python tools/acceptance/run.py \
  --output $OMEGA_SHARED_ROOT/playground/results/acceptance-RUN \
  --stages tracker joystick

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  ../venvs/pg-mjwarp-clean-001/bin/python tools/acceptance/run.py \
  --output $OMEGA_SHARED_ROOT/playground/results/acceptance-RUN \
  --stages render
```

Outputs are never overwritten. `--robots` selects a subset. Stage errors are
recorded per embodiment; they do not become passes. `matrix.json` always includes
all five embodiments.

## Gate 1 methodology and limits

`tracker_plane.py` uses the installed SMK `Motions`, evaluation-schema preparation,
`Actor`, `Layout`, and `MJWarp` with no `scene_xml`. It reproduces the evaluator
operations beside the actual Playground `TrackingSession.append/step`. It verifies
that the compiled ground is an analytic plane. Robot-specific checkpoint settings,
PD, CCD, action scaling and friction are retained. Heading rewriting is disabled.

The native branch consumes `Motions.get` including its source body states; the
Playground branch rebuilds FK through its actual adapter. Both branches start at
LAFAN `walk3_subject3` frame 1000 and run 500 controls at 50 Hz, physics at 200 Hz.
This is a bounded integration trace, not a reproduction of the complete SMK
test37 benchmark. It intentionally continues after a failure flag to retain an
equal-length diagnostic trace; failure flags remain recorded.

Every simulator uses its own Torch/Warp stream. A first experimental run without
explicit stream switching is invalidated and must not be used to diagnose policy
quality. Valid initial run: `acceptance-20261108-v2`.

Initial acceptance is strict equality. Absolute/rms/first-index differences and
root displacement are reported. No empirical GPU tolerance has yet been accepted.
`--native-control` substitutes evaluator operations in the second world to measure
native/native numerical variation; these are diagnostics, not adapter passes.

Each branch saves compiled model/XML, reference/history/action arrays, PD targets
(`ctrl`), qpos/qvel, actual body states and GPU contact arrays. Contacts are sampled
at **50 Hz control boundaries**, not every 200 Hz substep; `nacon`/world IDs delimit
active contact entries. No CPU physics is advanced. Rendering adds only camera and
light to a separate model and replays the saved physical body transforms.

## Gate 2 methodology and limits

`joystick_reference.py` loads pinned Joystick and codec releases via the production
client. It covers three seeds, two initial motion windows, and successive stand,
walk, speed change, stop, left/right/reverse and stop commands. Every initial
history is explicitly labelled offline motion; later histories are synthesized
from the first second of generated references. They are **not physical rollouts**.

The suite saves original and world-placed references, inputs, commands, repeated
same-seed outputs and per-case metrics. It checks finite data, named joints,
authored joint limits, unit quaternions, anchor/horizon/timestamp conventions, and
deterministic repetition. Quaternion tolerance is 32 float32 eps; all command,
acceleration and slip metrics remain measurements until a versioned acceptance
contract supplies defensible thresholds.

Foot geometry heights and near-floor tangential speed are FK-only estimates, not
physical contacts. The 2 cm near-floor band is a labelled diagnostic proxy, not an
acceptance margin. Reference projection clamping and reconstruction error are
reported so clipped joint limits cannot conceal generator problems.

Until response/slip/acceleration limits, real-rollout history tests and reference
visual QC are completed, a clean interface test is still not a Joystick pass.
