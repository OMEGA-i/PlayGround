#!/usr/bin/env bash
# Start the existing workbench on the Pod's dedicated NodePort target.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src"
logs=artifacts/runs/services
port=${OMEGA_PUBLIC_PORT:-7777}
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo "Invalid OMEGA_PUBLIC_PORT" >&2; exit 2; }
mkdir -p "$logs"
start() {
  local name=$1 url=$2; shift 2
  if curl --fail --silent --max-time 2 "$url" >/dev/null; then return; fi
  if [[ -s "$logs/$name.pid" ]] && kill -0 "$(cat "$logs/$name.pid")" 2>/dev/null; then return; fi
  nohup "$@" </dev/null >"$logs/$name.log" 2>&1 &
  echo "$!" >"$logs/$name.pid"
}
start selector http://127.0.0.1:18784/active python3 tools/workbench/workbench_selector.py
start native http://127.0.0.1:18782/api/state bash deploy/serve-native.sh
start supermimic http://127.0.0.1:18783/api/state "${OMEGA_SUPERMIMIC_PYTHON:-../SuperMimicKit/.venv/bin/python}" -u tools/supermimic/serve_supermimic_live.py --scene "${OMEGA_SCENE:-campus}" --robot "${OMEGA_SUPERMIMIC_ROBOT:-g1}"
conf="$PWD/$logs/playground.conf"
sed "s/@PUBLIC_PORT@/$port/g" deploy/playground.conf >"$conf.tmp"
mv "$conf.tmp" "$conf"
sudo -n nginx -t -c "$conf"
if [[ -s /tmp/omega-playground-nginx.pid ]] && sudo -n kill -0 "$(cat /tmp/omega-playground-nginx.pid)" 2>/dev/null; then
  sudo -n nginx -s reload -c "$conf"
else
  sudo -n nginx -c "$conf"
fi
