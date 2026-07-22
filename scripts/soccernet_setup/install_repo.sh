#!/usr/bin/env bash
# Install a single SoccerNet repo into an isolated environment.
# Usage: bash install_repo.sh <repo-name>
set -Eeuo pipefail

REPO="${1:?repo name required}"
TIMESTAMP="${SOCCERNET_INSTALL_TS:-20260712_170330}"
LOGDIR="/home/fdoblak/logs/soccernet_full_install_${TIMESTAMP}"
SOCCERNET_ROOT="${HOME}/projects/soccernet"
VENV_ROOT="${HOME}/.venvs/soccernet"
MODEL_ROOT="${HOME}/models/soccernet"
REPO_ENVS="${HOME}/projects/football-analytics/requirements/repo_envs"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
export PATH="${HOME}/.local/bin:${PATH}"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "${LOGDIR}/${REPO}.install.log"; }

run_cmd() {
  local desc="$1"; shift
  log "CMD: $*"
  if "$@" >> "${LOGDIR}/${REPO}.install.log" 2>&1; then
    log "OK: $desc"
    return 0
  else
    log "FAIL: $desc (exit $?)"
    return 1
  fi
}

mkdir -p "$LOGDIR" "$VENV_ROOT" "$MODEL_ROOT" "$REPO_ENVS"
log "=== Installing $REPO ==="

case "$REPO" in
  sn-trackeval)
    ENV=sn-trackeval
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
      run_cmd "create conda env" conda create -n "$ENV" python=3.10 -y
    fi
    conda activate "$ENV"
    run_cmd "pip install editable" pip install -e "${SOCCERNET_ROOT}/sn-trackeval"
    pip freeze > "${REPO_ENVS}/${ENV}.freeze.txt"
    python -m pip check
    ;;
  sn-echoes)
    ENV=sn-echoes
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
      run_cmd "create conda env" conda create -n "$ENV" python=3.10 -y
    fi
    conda activate "$ENV"
    python "${SOCCERNET_ROOT}/sn-echoes/stats.py" --help 2>/dev/null || python -c "import json,glob; print('json files', len(glob.glob('${SOCCERNET_ROOT}/sn-echoes/Dataset/**/*.json', recursive=True)))"
    ;;
  *)
    log "No install handler for $REPO in shell script; use Python orchestrator"
    exit 2
    ;;
esac

log "=== Done $REPO ==="
