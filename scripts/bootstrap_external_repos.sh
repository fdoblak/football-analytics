#!/usr/bin/env bash
# Bootstrap script for SoccerNet external reference repositories.
# Idempotent: verifies existing clones, clones only when missing.
# Does NOT install requirements, download datasets, or delete anything.

set -Eeuo pipefail

SOCCERNET_ROOT="${HOME}/projects/soccernet"

declare -A REPOS=(
  ["ActiveSpotting"]="https://github.com/SoccerNet/ActiveSpotting.git"
  ["PTS-baseline"]="https://github.com/SoccerNet/PTS-baseline.git"
  ["SoccerNet"]="https://github.com/SoccerNet/SoccerNet.git"
  ["SoccerNet-v3"]="https://github.com/SoccerNet/SoccerNet-v3.git"
  ["sn-banner"]="https://github.com/SoccerNet/sn-banner.git"
  ["sn-calibration"]="https://github.com/SoccerNet/sn-calibration.git"
  ["sn-caption"]="https://github.com/SoccerNet/sn-caption.git"
  ["sn-depth"]="https://github.com/SoccerNet/sn-depth.git"
  ["sn-echoes"]="https://github.com/SoccerNet/sn-echoes.git"
  ["sn-gamestate"]="https://github.com/SoccerNet/sn-gamestate.git"
  ["sn-grounding"]="https://github.com/SoccerNet/sn-grounding.git"
  ["sn-jersey"]="https://github.com/SoccerNet/sn-jersey.git"
  ["sn-mvfoul"]="https://github.com/SoccerNet/sn-mvfoul.git"
  ["sn-nvs"]="https://github.com/SoccerNet/sn-nvs.git"
  ["sn-reid"]="https://github.com/SoccerNet/sn-reid.git"
  ["sn-spotting"]="https://github.com/SoccerNet/sn-spotting.git"
  ["sn-teamspotting"]="https://github.com/SoccerNet/sn-teamspotting.git"
  ["sn-trackeval"]="https://github.com/SoccerNet/sn-trackeval.git"
  ["sn-tracking"]="https://github.com/SoccerNet/sn-tracking.git"
)

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

verify_repo() {
  local name="$1"
  local url="$2"
  local path="${SOCCERNET_ROOT}/${name}"

  if [[ ! -d "${SOCCERNET_ROOT}" ]]; then
    log "ERROR: SoccerNet root missing: ${SOCCERNET_ROOT}"
    return 1
  fi

  if [[ -d "${path}/.git" ]]; then
    log "VERIFY: ${name} already cloned at ${path}"
    local origin
    origin="$(git -C "${path}" remote get-url origin 2>/dev/null || true)"
    if [[ "${origin}" != "${url}" ]]; then
      log "WARNING: ${name} origin mismatch. Expected: ${url}, Found: ${origin}"
    fi
    if [[ -n "$(git -C "${path}" status --porcelain 2>/dev/null)" ]]; then
      log "WARNING: ${name} has uncommitted local changes (dirty working tree)"
    else
      log "OK: ${name} working tree is clean"
    fi
    git -C "${path}" fetch --prune --quiet 2>/dev/null || log "WARNING: fetch failed for ${name}"
    log "HEAD: $(git -C "${path}" rev-parse --short HEAD) on branch $(git -C "${path}" branch --show-current)"
    return 0
  fi

  if [[ -d "${path}" ]]; then
    log "ERROR: ${path} exists but is not a git repository. Manual intervention required."
    return 1
  fi

  log "CLONE: ${name} from ${url}"
  git clone "${url}" "${path}"
  log "OK: ${name} cloned. HEAD=$(git -C "${path}" rev-parse --short HEAD)"
}

main() {
  log "Starting external repo bootstrap"
  log "SoccerNet root: ${SOCCERNET_ROOT}"

  mkdir -p "${SOCCERNET_ROOT}"

  local failed=0
  for name in $(printf '%s\n' "${!REPOS[@]}" | sort); do
    if ! verify_repo "${name}" "${REPOS[$name]}"; then
      failed=$((failed + 1))
    fi
  done

  if [[ "${failed}" -gt 0 ]]; then
    log "Bootstrap finished with ${failed} error(s)"
    exit 1
  fi

  log "Bootstrap finished successfully (${#REPOS[@]} repositories)"
}

main "$@"
