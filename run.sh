#!/usr/bin/env bash
set -euo pipefail
replay_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
replay_python="${REPLAY_PYTHON:-$replay_dir/.venv/bin/python}"
if [[ ! -x "$replay_python" ]]; then
    echo "Set REPLAY_PYTHON to a Python with requirements.txt installed, or create $replay_dir/.venv" >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "$replay_python" "$replay_dir/replay.py" "$@"
