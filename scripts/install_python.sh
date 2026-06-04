#!/usr/bin/env bash
# Create a virtual environment and install opamp-model from pyproject.toml.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
MIN_PYTHON_VERSION="3.10"
INSTALL_DEV=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create a Python virtual environment and install opamp-model in editable mode.

Options:
  --dev       Install dev dependencies (pytest, ruff). Default.
  --no-dev    Runtime dependencies only.
  --recreate  Remove an existing virtual environment before installing.
  -h, --help  Show this message.

Environment:
  VENV_DIR    Virtual environment path (default: ${ROOT_DIR}/.venv)
EOF
}

version_ge() {
    local left="${1#python}"
    local right="${2#python}"
    IFS='.' read -r left_major left_minor _ <<<"${left}"
    IFS='.' read -r right_major right_minor _ <<<"${right}"
    if (( left_major > right_major )); then return 0; fi
    if (( left_major < right_major )); then return 1; fi
    (( left_minor >= right_minor ))
}

find_python() {
    local candidate version
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if ! command -v "${candidate}" >/dev/null 2>&1; then
            continue
        fi
        version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if version_ge "${version}" "${MIN_PYTHON_VERSION}"; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

RECREATE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) INSTALL_DEV=1 ;;
        --no-dev) INSTALL_DEV=0 ;;
        --recreate) RECREATE=1 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

PYTHON="$(find_python)" || {
    echo "Error: Python ${MIN_PYTHON_VERSION}+ required." >&2
    exit 1
}

if [[ "${RECREATE}" -eq 1 ]] && [[ -d "${VENV_DIR}" ]]; then
    rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

if [[ "${INSTALL_DEV}" -eq 1 ]]; then
    python -m pip install -e "${ROOT_DIR}[dev]"
else
    python -m pip install -e "${ROOT_DIR}"
fi

echo "Done. Activate: source ${VENV_DIR}/bin/activate"
