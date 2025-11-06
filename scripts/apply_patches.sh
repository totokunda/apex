#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/apply_patches.sh [options]

Apply all .patch files from a patches directory to the repository working tree.

Options:
  --dir <dir>      Path to patches directory (default: patches)
  --dry-run | -n   Validate patches without applying changes
  --reverse | -R   Reverse-apply patches (unapply)
  --3way           Use 3-way merge when applying with git (default)
  --no-3way        Disable 3-way merge for git apply
  -h | --help      Show this help and exit

Environment:
  PATCHES_DIR      Same as --dir

Notes:
  - Only files ending with .patch are processed; other files are ignored.
  - If inside a git repo, uses 'git apply' (with optional 3-way). If that fails
    or not in a git repo, falls back to the 'patch' command with -p1.
  - In --dry-run mode, the script exits non-zero if any patch would fail.
EOF
}

PATCHES_DIR=${PATCHES_DIR:-patches}
DRY_RUN=false
REVERSE=false
THREE_WAY=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --dir" >&2; exit 1; }
      PATCHES_DIR="$2"; shift 2 ;;
    --dry-run|-n)
      DRY_RUN=true; shift ;;
    --reverse|-R)
      REVERSE=true; shift ;;
    --3way)
      THREE_WAY=true; shift ;;
    --no-3way)
      THREE_WAY=false; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

# Move to repo root if in a git repository
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT_DIR"

if [[ ! -d "$PATCHES_DIR" ]]; then
  echo "Patches directory not found: $PATCHES_DIR" >&2
  exit 1
fi

# Collect .patch files only, sorted for deterministic order
mapfile -t PATCH_FILES < <(find "$PATCHES_DIR" -maxdepth 1 -type f -name "*.patch" | sort)

if [[ ${#PATCH_FILES[@]} -eq 0 ]]; then
  echo "No .patch files found in $PATCHES_DIR. Nothing to do."
  exit 0
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT=true
else
  IN_GIT=false
fi

apply_with_git() {
  local patch_file="$1"
  local args=()
  $THREE_WAY && args+=(--3way)
  $REVERSE && args+=(-R)
  if $DRY_RUN; then
    git apply --check "${args[@]}" "$patch_file"
  else
    git apply "${args[@]}" "$patch_file"
  fi
}

apply_with_patch_cmd() {
  local patch_file="$1"
  local args=(-p1)
  $REVERSE && args+=(-R)
  if $DRY_RUN; then
    patch --dry-run "${args[@]}" < "$patch_file"
  else
    patch "${args[@]}" < "$patch_file"
  fi
}

success_count=0
failure_count=0

echo "Using patches directory: $PATCHES_DIR"
echo "Mode: ${DRY_RUN:+dry-run }${REVERSE:+reverse }${THREE_WAY:+3-way }apply"

for patch_path in "${PATCH_FILES[@]}"; do
  patch_name=$(basename "$patch_path")
  echo "Applying $patch_name ..."

  if $IN_GIT; then
    if apply_with_git "$patch_path"; then
      ((success_count++))
      echo "✔ Applied $patch_name via git"
    else
      echo "git apply failed for $patch_name, attempting 'patch' fallback..."
      if apply_with_patch_cmd "$patch_path"; then
        ((success_count++))
        echo "✔ Applied $patch_name via patch"
      else
        ((failure_count++))
        echo "✖ Failed to apply $patch_name" >&2
      fi
    fi
  else
    if apply_with_patch_cmd "$patch_path"; then
      ((success_count++))
      echo "✔ Applied $patch_name via patch"
    else
      ((failure_count++))
      echo "✖ Failed to apply $patch_name" >&2
    fi
  fi
done

echo "Summary: ${success_count} applied, ${failure_count} failed${DRY_RUN:+ (dry-run)}"

if [[ $failure_count -gt 0 ]]; then
  exit 1
fi

exit 0


