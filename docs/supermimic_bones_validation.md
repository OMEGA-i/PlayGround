# BONES-selected Tracker integration — 2026-11-07

Current release: `bones-selected-20261107`, pinned in [trackers.json](../configs/supermimic/trackers.json). The previous [a-course receipt](supermimic_best_validation.md) is historical. Live and offline callers use the same exact paths, SHA-256 and completed-iteration pins. Joystick, XML assets, physics parameters and skins are unchanged. SIH2300 drives its native/UE4/UE5 avatars.

| Robot | Course | Completed iterations | Owner BONES test return | Failed motions / 13023 | Local replay steps | Tracker failure | Root XYZ RMSE (m) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| G1 | c_lafan25k_bones25k | 45000 | 0.8721432892 | 27 | 500 | false | 1.084 |
| H2 | c_lafan25k_bones25k | 35000 | 0.7763815857 | 77 | 500 | false | 3.545 |
| A3 | c_lafan25k_bones25k | 30000 | 0.8250864530 | 65 | 500 | false | 2.802 |
| SIH2300 | c_lafan25k_bones25k | 35000 | 0.8505380129 | 46 | 500 | false | 0.942 |

A3 c30k was explicitly chosen by the user for fewer failures instead of b25k's higher return. Checkpoints were selected on validation within each course; the course choice used BONES test comparisons. The owner BONES numbers are selection evidence, not a new independent post-selection evaluation performed by Playground. Evidence paths are pinned per robot in the receipt.

## Contract and execution checks

`verify_supermimic_release.py ROBOT - OUTPUT` audits the checkpoint-native deployment without exporting or publishing web weights. It checks the complete file SHA, internal iteration + 1, frozen XML hash, named joint/body ordering, matching training/runtime configuration, strict Actor load and finite normalizer/parameter buffers, actuator joint order, default pose, action scale and checkpoint-derived PD gains. Normalizers remain those in the exact checkpoint. The native inference architecture uses ten history states and eleven goal samples; each 50 Hz Tracker step advances four 200 Hz MJWarp GPU substeps.

All four models ran on L20 GPU 0 with the same held-out LAFAN `walk3_subject3` source frames 1000:1511, 500 control steps each. No Joystick, measured-heading re-anchoring or Campus collision geometry is used in this isolated replay. Campus is only the rendering background. This follows the previous a-course replay protocol and confirms actual loading and stepping of every selected checkpoint.

Global XYZ error includes accumulated drift and is larger on this clip than in the historical a-course replay. In particular, H2/A3 drift remains substantial. `failed=false` is the Tracker's own termination criterion, not acceptance of trajectory accuracy, foot/knee contacts, stop/turn control, or full BONES coverage. Those remain separate task/contact gates under OmegaSDG.

Private artifacts: `artifacts/qc/supermimic-bones-20261107/<robot>/` contains the audited contract, named reference, physical rollout/body poses, result and rendered replay. The root `summary.json` records numerical errors. Checkpoints remain in private shared storage.

## Online load

After reloading the paused G1 workbench, `/api/state` reported `ready=true`, `paused=true`, `tracker_release=bones-selected-20261107`, 45000 completed iterations and SHA `0cc0add882481c9e96f22f2b7227303185c256d311da16321fca95f9aa1d6cbe`. Other embodiments use their pinned checkpoints when selected/rebuilt; all four were separately exercised above. The release label is captured at session construction so later manifest edits cannot relabel an already loaded model.
