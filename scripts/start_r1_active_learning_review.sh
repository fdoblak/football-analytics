#!/usr/bin/env bash
# Active-learning + blind holdout_v2 dual review (foreground WSL helper).
set -euo pipefail
REPO_ROOT="/home/fdoblak/projects/football-analytics"
PYTHON="/home/fdoblak/miniconda3/envs/ai-dev/bin/python"
SERVER_PY="${REPO_ROOT}/scripts/r1_independent_gt_review_server.py"
RUNTIME="/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4_active_learning"
VIDEO="/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
HOST="127.0.0.1"
PORT="8768"
PID_FILE="${RUNTIME}/server.pid"
LOG_FILE="${RUNTIME}/server_wrapper.log"
EXPECTED_SHA="97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"

mkdir -p "${RUNTIME}"
log() { echo "$1"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >>"${LOG_FILE}"; }
log "=== start_r1_active_learning_review.sh ==="
if [[ ! -f "${RUNTIME}/draft_annotations.json" ]]; then
  log "ERROR: active learning draft missing"
  exit 5
fi
DIGEST="$("${PYTHON}" -c "import hashlib; from pathlib import Path; p=Path(r'${VIDEO}'); h=hashlib.sha256();
f=p.open('rb');
[h.update(c) for c in iter(lambda: f.read(1<<20), b'')];
f.close();
print(h.hexdigest())")"
if [[ "${DIGEST}" != "${EXPECTED_SHA}" ]]; then
  log "ERROR: SOURCE_SHA_MISMATCH"
  exit 6
fi
HEALTH="$("${PYTHON}" - <<'PY'
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8768/health", timeout=2) as r:
        body = json.loads(r.read().decode())
except Exception:
    print("DOWN"); raise SystemExit(0)
ok = (
    body.get("status")=="ok"
    and body.get("service")=="r1_independent_gt_review"
    and body.get("active_learning") is True
)
print("OK" if ok else "MISMATCH")
PY
)"
if [[ "${HEALTH}" == "OK" ]]; then
  log "INFO: active learning server already healthy"
  exit 10
fi
if [[ "${HEALTH}" == "MISMATCH" ]]; then
  log "ERROR: port 8768 occupied by non-AL service"
  exit 8
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo $$ >"${PID_FILE}"
log "Starting AL+holdout_v2 review on http://${HOST}:${PORT}/"
exec "${PYTHON}" "${SERVER_PY}" --host "${HOST}" --port "${PORT}" --runtime "${RUNTIME}" --video "${VIDEO}" --active-learning
