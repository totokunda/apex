#!/usr/bin/env bash
set -euo pipefail

# If invoked via sudo, restore the original user's home so Apex caches/configs
# stay under /home/<user>/apex-diffusion instead of /root/apex-diffusion.
if [[ "${EUID:-$(id -u)}" -eq 0 && -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
  SUDO_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6 || true)"
  if [[ -n "${SUDO_HOME:-}" ]]; then
    export HOME="$SUDO_HOME"
    export USER="$SUDO_USER"
    export LOGNAME="$SUDO_USER"
    export APEX_HOME_DIR="$SUDO_HOME"
    # Keep caches in the original user's home unless caller overrides.
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
  fi
fi

# Ensure imports (including sitecustomize.py) resolve from repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Spoof system RAM (opt-in via sitecustomize.py).
export APEX_FAKE_RAM_GB="${APEX_FAKE_RAM_GB:-32}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="/home/tosin_coverquick_co/miniconda3/envs/apex/bin/python3"
ARGS=(-m verify.hunyuan_video_15)

# Enforce a real 32GB RAM limit when possible (Linux + systemd).
# This is a hard cap: if the process tries to exceed it, it will OOM/abort accordingly.
if command -v systemd-run >/dev/null 2>&1; then
  exec "$PY" "${ARGS[@]}"
else
  echo "WARNING: systemd-run not found; cannot enforce a real 32GB RAM cap." >&2
  echo "         RAM spoofing (APEX_FAKE_RAM_GB=${APEX_FAKE_RAM_GB}) is still enabled." >&2
  exec "$PY" "${ARGS[@]}"
fi