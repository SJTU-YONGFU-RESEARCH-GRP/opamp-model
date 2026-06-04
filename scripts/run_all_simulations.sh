#!/usr/bin/env bash
# Batch: AC, STB, noise, PSRR, slew, THD for python / ngspice / spectre (when available).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
EXTRA_ARGS=()
SKIP_MISSING=0

BENCH_SCRIPTS=(
    run_ac.py
    run_stb.py
    run_noise.py
    run_psrr.py
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- EXTRA_ARGS...]

Run all op-amp benches for each available engine under:
  \${OUTPUT_ROOT}/{python,ngspice,spectre}/

Transient benches (slew, THD) run for the python engine only.

Options:
  --output-root DIR   Base output directory (default: ${ROOT_DIR}/outputs).
  --skip-missing      Skip ngspice/spectre when binaries are absent.
  --ideal             Pass --ideal to each bench script.
  -h, --help          Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --skip-missing) SKIP_MISSING=1; shift ;;
        --ideal) EXTRA_ARGS+=(--ideal); shift ;;
        --) shift; EXTRA_ARGS+=("$@"); break ;;
        -h | --help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="python3"
fi

run_bench() {
    local engine="$1"
    local script="$2"
    local out="${OUTPUT_ROOT}/${engine}"
    if [[ "${engine}" != "python" ]]; then
        if [[ "${SKIP_MISSING}" -eq 1 ]]; then
            if [[ "${engine}" == "ngspice" ]] && ! command -v ngspice >/dev/null 2>&1; then
                echo "Skipping ngspice (${script}) — not on PATH."
                return 0
            fi
            if [[ "${engine}" == "spectre" ]] && ! command -v spectre >/dev/null 2>&1; then
                echo "Skipping spectre (${script}) — not on PATH."
                return 0
            fi
        fi
    fi
    echo "=== ${engine}: ${script} ==="
    "${PYTHON}" "${ROOT_DIR}/scripts/${script}" \
        --simulator "${engine}" \
        --output-dir "${out}" \
        "${EXTRA_ARGS[@]}"
}

write_report() {
    local engine="$1"
    echo "=== ${engine}: REPORT.md ==="
    "${PYTHON}" "${ROOT_DIR}/scripts/write_engine_report.py" \
        --simulator "${engine}" \
        --output-dir "${OUTPUT_ROOT}/${engine}" || true
}

for engine in python ngspice spectre; do
    for script in "${BENCH_SCRIPTS[@]}"; do
        run_bench "${engine}" "${script}"
    done
    write_report "${engine}"
done

echo "=== python: run_slew.py ==="
"${PYTHON}" "${ROOT_DIR}/scripts/run_slew.py" \
    --simulator python \
    --output-dir "${OUTPUT_ROOT}/python" \
    "${EXTRA_ARGS[@]}"

echo "=== python: run_thd.py ==="
"${PYTHON}" "${ROOT_DIR}/scripts/run_thd.py" \
    --simulator python \
    --output-dir "${OUTPUT_ROOT}/python" \
    "${EXTRA_ARGS[@]}"

write_report python

echo "Batch complete under ${OUTPUT_ROOT}/"
echo "Open outputs/<engine>/REPORT.md for figures and bench report links."
