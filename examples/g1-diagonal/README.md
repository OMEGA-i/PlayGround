# Replay the Joystick G1 diagonal reference

Keep the two candidate checkouts side by side. On macOS or Linux, check the installed CLI:

```sh
uv sync --frozen --python 3.11
uv run --frozen omega-playground --help
```

On a CUDA host, first generate `reference25/custom_single.npz` with the paired Joystick example. Use the Joystick interpreter for MuJoCo, NumPy and the 25→50 Hz conversion. The separate tracker environment must provide the pinned SuperMimic runtime, PyTorch, CUDA, MuJoCo and MJWarp. Provide the local G1 XML/meshes and released 45,000-step tracker. The runtime verifies checkpoint hash, iteration, XML and joint names.

```sh
PYTHONPATH="$PWD/src" "$JOYSTICK_PYTHON" examples/g1-diagonal/replay.py \
  --reference "$RUN_DIR/reference25/custom_single.npz" --xml "$G1_XML" \
  --checkpoint "$G1_TRACKER" --tracker-python "$TRACKER_PYTHON" \
  --tracker-repo "$TRACKER_REPO" --output "$RUN_DIR/physics"
```

`reference50.npz` stores named joint references at 20 ms intervals. `body_states.npz`, `rollout.npz`, `trajectory.json` and `result.json` report simulated body poses, recorded states, root tracks and physical termination status. This example uses a ground plane. The website's scene demonstration is a separate experiment. See [RELEASE.md](../../RELEASE.md) for dependency and distribution status.
