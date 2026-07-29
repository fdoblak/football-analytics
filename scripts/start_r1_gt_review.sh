#!/usr/bin/env bash
# Start R1 independent GT review server in the foreground (Windows launcher helper).
# Optional: REPAIR_MODE=train-empty-complete or --repair train-empty-complete
set -euo pipefail

REPO_ROOT="/home/fdoblak/projects/football-analytics"
PYTHON="/home/fdoblak/miniconda3/envs/ai-dev/bin/python"
SERVER_PY="${REPO_ROOT}/scripts/r1_independent_gt_review_server.py"
RUNTIME="/home/fdoblak/workspace/independent_gt_review/own_video_97b298e4"
VIDEO="/home/fdoblak/football_data/videos/raw_matches/own_video_97b298e4/original.mp4"
HOST="127.0.0.1"
PORT="8766"
PID_FILE="${RUNTIME}/server.pid"
LOG_FILE="${RUNTIME}/server_wrapper.log"
EXPECTED_SHA="97b298e41a82b567a7d68bd2322993bea34492b1cbb58362b0d72ca4a5471160"
REPAIR_MODE="${REPAIR_MODE:-}"

# Parse optional --repair <mode>
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repair)
      REPAIR_MODE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${RUNTIME}"

log() {
  local msg="$1"
  echo "${msg}"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ${msg}" >>"${LOG_FILE}"
}

log "=== start_r1_gt_review.sh repair_mode=${REPAIR_MODE:-none} ==="

if [[ ! -x "${PYTHON}" ]]; then
  log "ERROR: ai-dev python missing: ${PYTHON}"
  exit 2
fi
if [[ ! -f "${SERVER_PY}" ]]; then
  log "ERROR: server script missing: ${SERVER_PY}"
  exit 3
fi
if [[ ! -f "${VIDEO}" ]]; then
  log "ERROR: canonical video missing: ${VIDEO}"
  exit 4
fi
if [[ ! -f "${RUNTIME}/draft_annotations.json" ]]; then
  log "ERROR: draft missing - run prepare_r1_f2a_independent_gt.py first"
  exit 5
fi

DIGEST="$("${PYTHON}" -c "import hashlib; from pathlib import Path; p=Path(r'${VIDEO}'); h=hashlib.sha256();
f=p.open('rb');
import sys
[h.update(c) for c in iter(lambda: f.read(1<<20), b'')];
f.close();
print(h.hexdigest())")"
if [[ "${DIGEST}" != "${EXPECTED_SHA}" ]]; then
  log "ERROR: SOURCE_SHA_MISMATCH got=${DIGEST}"
  exit 6
fi

HEALTH_CODE="$(
  REPAIR_MODE="${REPAIR_MODE}" "${PYTHON}" - <<'PY'
import json
import os
import urllib.request

want = os.environ.get("REPAIR_MODE") or None
try:
    with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2) as r:
        body = json.loads(r.read().decode("utf-8"))
except Exception:
    print("DOWN")
    raise SystemExit(0)
ok = (
    body.get("status") == "ok"
    and body.get("service") == "r1_independent_gt_review"
    and body.get("source_id") == "own_video_97b298e4"
    and body.get("repair_mode") == want
)
print("OK" if ok else "MISMATCH")
PY
)"

if [[ "${HEALTH_CODE}" == "OK" ]]; then
  log "INFO: matching healthy R1 GT review already on ${HOST}:${PORT}; not starting duplicate"
  log "Close this window. Use the existing server / browser."
  exit 10
fi

if [[ "${HEALTH_CODE}" == "MISMATCH" ]]; then
  log "ERROR: port ${PORT} has a different R1 GT mode (repair mismatch). Close that server window first."
  exit 8
fi

PORT_STATE="$("${PYTHON}" - <<'PY'
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", 8766))
except OSError:
    print("FREE")
else:
    print("BUSY")
finally:
    s.close()
PY
)"
if [[ "${PORT_STATE}" == "BUSY" ]]; then
  log "ERROR: port ${PORT} occupied by non-matching process; refusing to kill it"
  exit 7
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo $$ >"${PID_FILE}"
chmod 600 "${PID_FILE}" 2>/dev/null || true

EXTRA=()
if [[ -n "${REPAIR_MODE}" ]]; then
  EXTRA+=(--repair "${REPAIR_MODE}")
fi

log "Starting review server on http://${HOST}:${PORT}/ repair=${REPAIR_MODE:-none}"
log "Do not close this window during review."
exec "${PYTHON}" "${SERVER_PY}" --host "${HOST}" --port "${PORT}" --runtime "${RUNTIME}" --video "${VIDEO}" "${EXTRA[@]+"${EXTRA[@]}"}"
