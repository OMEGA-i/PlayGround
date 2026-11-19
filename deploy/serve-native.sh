#!/usr/bin/env bash
# Native SONIC policy, offline authored collisions, GPU MJWarp physics/render.
set -euo pipefail
pg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$pg_root"
export OMEGA_PORT=${OMEGA_PORT:-18782}
export OMEGA_PREPARED_ASSETS=${OMEGA_PREPARED_ASSETS:-configs/mjwarp-assets.json}
export PYTHONPATH="$pg_root/src${PYTHONPATH:+:$PYTHONPATH}"
scene_row=$(python3 -c 'from omega_playground.ops.scenes import SCENES; import sys; print(*SCENES[sys.argv[1]])' "${OMEGA_SCENE:-campus}")
read -r scene map player <<< "$scene_row"
physics_python=${OMEGA_MJWARP_PYTHON:-$pg_root/../venvs/pg-mjwarp-clean-001/bin/python}
render_python=${OMEGA_RENDER_PYTHON:-$physics_python}
export HF_HOME=${HF_HOME:-$pg_root/../hf_cache}
export OMEGA_RENDER_CUDA_IPC=${OMEGA_RENDER_CUDA_IPC:-1}
export OMEGA_RENDER_SUPERSAMPLE=${OMEGA_RENDER_SUPERSAMPLE:-1}
# Fail at startup if the live video or policy dependencies are missing.
"$physics_python" -c 'import mujoco_warp, torch, onnx, onnxruntime, wsproto'
exec "$physics_python" -m omega_playground.cli serve \
  --prepared-assets "$OMEGA_PREPARED_ASSETS" \
  --package artifacts/policies/sonic-v1.1-g1-cuda-batch --avatar g1 --no-armed \
  --scene "artifacts/scenes/$scene" --map "artifacts/maps/$map" \
  --player "configs/scenes/$player" \
  --output "artifacts/runs/mjwarp_g1_$(date -u +%Y%m%dT%H%M%SZ)_$$" \
  --port "${OMEGA_PORT:-18782}" --start-paused --realtime --video \
  --render-python "$render_python" --render-width 1280 --render-height 720 \
  --video-bitrate-kbps 6000 --camera chase "$@"
