#!/usr/bin/env bash
set -euo pipefail

# GLM-5.3-Flash EXL3 on AMD Strix Halo (gfx1151) — one-command deploy.
#
# Order of operations: preflight -> build image (cached) -> download checkpoint
# (resumable, host-side) -> run container -> wait for /health -> print endpoints.
# Idempotent: re-running skips build/download and reports "already running" if
# the container is up.
#
# Precedence (highest first):
#   1. shell env vars already exported when you run ./start.sh
#   2. values in ./.env
#   3. inline defaults below

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

IMAGE="${IMAGE:-glm53-flash-strix:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-strix}"
PORT="${PORT:-8080}"                          # host publish port (container listens on 8000)
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/.cache}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/.glm53.log}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"        # seconds; cold-cache compile is minutes
MAX_JOBS="${MAX_JOBS:-$(nproc)}"              # kernel build parallelism (hipcc/ninja)

# Model download settings (overridable via .env / exported env, same as above).
HF_MODEL="${HF_MODEL:-yamz-labs/GLM-5.3-Flash-EXL3-Yamz}"
HF_REVISION="${HF_REVISION:-}"                  # empty = latest; pin a commit for reproducibility

# ---- preflight ---------------------------------------------------------------
command -v podman >/dev/null 2>&1 || { echo "podman is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is not on PATH"; exit 1; }

[[ -e /dev/kfd && -d /dev/dri ]] || { echo "ERROR: /dev/kfd and/or /dev/dri not found — AMD GPU driver not loaded"; exit 1; }
if ! id -nG | grep -qw render; then
  echo "warning: current user is not in the 'render' group; rootless /dev/kfd access may fail"
fi

# memlock: informational only. Rootless podman clamps --ulimit memlock to the
# user session's hard limit; the flag below takes effect if it's already raised.
# If the engine later dies with HSA "Cannot allocate memory" pin errors, raise it
# once (optional, needs sudo — see README troubleshooting).
MEMLOCK_HARD="$(systemctl show "user@$(id -u).service" -p LimitMEMLOCK --value 2>/dev/null || echo unknown)"
if [[ "${MEMLOCK_HARD}" != "infinity" ]]; then
  echo "note: user session memlock limit is '${MEMLOCK_HARD}' (not infinity)"
  echo "      if you see HSA pin errors at runtime, raise it once (see README troubleshooting)"
fi

# ---- image -----------------------------------------------------------------
if podman image exists "${IMAGE}"; then
  echo "Image ${IMAGE} exists; skipping build."
else
  echo "Building image ${IMAGE} (MAX_JOBS=${MAX_JOBS}) ..."
  # --format docker: the Containerfile's HEALTHCHECK is ignored under the
  # default OCI format (podman warns and drops it).
  podman build \
    --format docker \
    --jobs="${MAX_JOBS}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    -f "${SCRIPT_DIR}/Containerfile" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"
  echo "[$(date -Is)] image ${IMAGE} built"
fi

# ---- model checkpoint (host-side, resumable) ---------------------------------
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model not found at ${MODEL_DIR} — downloading via huggingface-cli ..."
  mkdir -p "${MODEL_DIR}"

  # --revision must be omitted (not passed empty): huggingface-cli rejects ''.
  REV_ARGS=()
  if [[ -n "${HF_REVISION}" ]]; then REV_ARGS=(--revision "${HF_REVISION}"); fi

  # Try huggingface-cli first (from huggingface_hub Python package).
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download \
      "${HF_MODEL}" \
      "${REV_ARGS[@]}" \
      --local-dir "${MODEL_DIR}" \
      --resume-download
  elif python3 -c "import huggingface_hub" 2>/dev/null; then
    HF_REVISION_ENV="${HF_REVISION}" MODEL_DIR_ENV="${MODEL_DIR}" HF_MODEL_ENV="${HF_MODEL}" python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ['HF_MODEL_ENV'],
    revision=os.environ.get('HF_REVISION_ENV') or None,
    local_dir=os.environ['MODEL_DIR_ENV'],
    resume_download=True,
)
"
  else
    echo "ERROR: huggingface-cli or huggingface_hub not available on host."
    echo "       Install with:  pip install huggingface_hub[cli]"
    echo "       Or manually download the model to ${MODEL_DIR}"
    exit 1
  fi

  # Verify the download produced at least the expected marker files.
  if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: download completed but no config.json found in ${MODEL_DIR}"
    exit 1
  fi
  if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    echo "ERROR: download completed but no model.safetensors.index.json found in ${MODEL_DIR}"
    exit 1
  fi
  echo "[$(date -Is)] model checkpoint ready at ${MODEL_DIR}"
else
  echo "Model checkpoint found at ${MODEL_DIR}; skipping download."
fi

# ---- container lifecycle (idempotent) ----------------------------------------
if podman ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if podman ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    exit 0
  fi
  # Stopped container exists — remove it so we can recreate with fresh config.
  echo "Removing stopped container ${CONTAINER_NAME} ..."
  podman rm "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${CACHE_DIR}"

echo "Starting ${CONTAINER_NAME} (port: ${PORT}, model: ${MODEL_DIR})"
echo "Model: ${MODEL_DIR} -> /models (ro)   Cache: ${CACHE_DIR} -> /cache"

# --ipc host is load-bearing (GPU runtime dies at startup without it; no
# shm_size substitutes). --group-add keep-groups carries the host render group
# into the rootless container for /dev/kfd. :z relabels binds for SELinux.
podman run -d \
  --name "${CONTAINER_NAME}" \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups \
  --ipc host \
  --ulimit memlock=-1:-1 \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/models:ro,z" \
  -v "${CACHE_DIR}:/cache:z" \
  -e MODEL_PATH=/models -e HOST=0.0.0.0 -e PORT=8000 \
  -e "MAX_JOBS=${MAX_JOBS}" \
  "${IMAGE}" \
  >/dev/null

echo "Spawned container ${CONTAINER_NAME}"

# ---- readiness ---------------------------------------------------------------
log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]]; then
    # The follower runs in its own process group (setsid below): signal the whole
    # group so `podman logs -f` dies together with `tee`. Killing only the
    # pipeline's last pid leaves podman alive, and the `wait` below then blocks
    # forever right after a successful start.
    kill -- "-${log_follow_pid}" 2>/dev/null \
      || kill "${log_follow_pid}" 2>/dev/null || true
    wait "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Stream the engine log to the terminal AND record it in .glm53.log.
# setsid isolates the `podman logs -f | tee` pipeline in its own process group so
# cleanup can reap both ends (see above); without it, start.sh never returns.
setsid bash -c 'podman logs -f "$1" 2>&1 | tee -a "$2"' _ "${CONTAINER_NAME}" "${LOG_FILE}" &
log_follow_pid=$!

READY_URL="http://127.0.0.1:${PORT}/health"
echo "Waiting for HTTP readiness at ${READY_URL} (timeout ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
heartbeat=0
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! podman ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out after ${READY_TIMEOUT}s waiting for readiness"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  # The log itself is streaming above; only a light heartbeat every ~30s.
  if (( heartbeat % 6 == 0 )); then
    echo "  still starting..."
  fi
  heartbeat=$((heartbeat + 1))
  sleep 5
done

echo "GLM-5.3-Flash is ready"
echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
echo "Served model id: ${HF_MODEL}"
echo "Recommended sampling: temperature 1.0, top_p 0.95, no repetition penalty"
echo "Try: curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"${HF_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
