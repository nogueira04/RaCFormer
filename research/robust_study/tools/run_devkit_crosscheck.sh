#!/bin/bash
# Launch devkit_crosscheck.py under the isolated cross-check venv.
#
# The script itself refuses to run unless it is started this way (isolated mode, no user site, no
# inherited PYTHONPATH, nuscenes resolving inside the venv), so this wrapper exists to make the
# correct invocation the easy one. All arguments are forwarded verbatim.
set -euo pipefail

VENV="${DEVKIT_VENV:-$HOME/venvs/nusc-devkit-check}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$VENV/bin/python" ]; then
  echo "FATAL: cross-check venv not found at $VENV" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONNOUSERSITE=1

exec "$VENV/bin/python" -I -u "$HERE/devkit_crosscheck.py" --venv "$VENV" "$@"
