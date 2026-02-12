#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

STATE="${1:-anionic}"
WORKERS="${WORKERS:-24}"
EXHAUST="${EXHAUST:-8}"

PY_MAIN="${PY_MAIN:-python}"
PY_RLS="${PY_RLS:-${ROOT_DIR}/.venv_rlscore/bin/python}"
if [[ ! -x "${PY_RLS}" ]]; then
  PY_RLS="${PY_MAIN}"
fi

run_step() {
  local step="$1"
  shift
  echo "[$(date '+%F %T')] START ${step}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step}.log"
  echo "[$(date '+%F %T')] DONE  ${step}"
}

run_step "01_prepare_ligands" "${PY_MAIN}" "${ROOT_DIR}/src/01_prepare_ligands.py" --workers "${WORKERS}"
run_step "02_prepare_receptors" "${PY_MAIN}" "${ROOT_DIR}/src/02_prepare_receptors.py"
run_step "03_run_docking_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/03_run_docking.py" --state "${STATE}" --workers "${WORKERS}" --exhaustiveness "${EXHAUST}"
run_step "04_score_to_kd_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/04_score_to_kd.py" --state "${STATE}" --hsa-calib strong
run_step "05_process_exposure" "${PY_MAIN}" "${ROOT_DIR}/src/05_process_exposure.py"
run_step "06_compute_obi_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/06_compute_obi.py" --state "${STATE}" --cref 1e-7 --hsa-calib strong --thyroid-weights auto --calibrate-t4-kd --t4-free-target 1.6e-11 --mc-samples 200
run_step "06_compute_obi_${STATE}_dsf" "${PY_MAIN}" "${ROOT_DIR}/src/06_compute_obi.py" --state "${STATE}" --cref 1e-7 --hsa-calib dsf --thyroid-weights auto --calibrate-t4-kd --t4-free-target 1.6e-11 --mc-samples 200 --out-tag dsf
run_step "07_qsar_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/07_qsar.py" --state "${STATE}" --endpoint OBI_ref --split scaffold --repeats 10 --seed 2026
run_step "08_kronrls_${STATE}" "${PY_RLS}" "${ROOT_DIR}/src/08_kronrls.py" --state "${STATE}" --reg-grid 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1
run_step "09_validation_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/09_validation.py" --state "${STATE}"
run_step "11_redocking" "${PY_MAIN}" "${ROOT_DIR}/src/11_redocking.py" --exhaustiveness 16 --num-modes 9 --seed 2026 --rmsd-pass 2.0
run_step "12_audit_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/12_audit_checks.py" --state "${STATE}"
run_step "10_make_report_${STATE}" "${PY_MAIN}" "${ROOT_DIR}/src/10_make_report.py" --state "${STATE}"

echo "Pipeline finished. Summary report:"
echo "  ${ROOT_DIR}/results/reports/pipeline_summary_${STATE}.md"
