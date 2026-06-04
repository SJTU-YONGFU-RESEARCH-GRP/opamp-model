#!/usr/bin/env bash
# Fail if opamp-model contains forbidden legacy import/path references.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Primary guard: Python scanner (skips guard implementation files).
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    "${ROOT_DIR}/.venv/bin/python" -m pytest tests/test_independence.py -q
elif command -v python3 >/dev/null 2>&1; then
    PYTHONPATH="${ROOT_DIR}/src" python3 -m pytest tests/test_independence.py -q
else
    echo "No python available for independence test." >&2
    exit 1
fi

echo "Independence check passed."
