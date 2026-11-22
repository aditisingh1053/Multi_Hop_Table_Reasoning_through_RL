#!/usr/bin/env bash
# Reader-in-the-loop PPO (RiL-PPO) experiments on a GPU machine (Kaggle, Colab or a lab server).
# Resumable: re-running skips every finished step. Usage:
#   SMOKE=1 bash scripts/run_loop_gpu.sh     # ~5 min end-to-end check of every stage (after data preparation)
#   bash scripts/run_loop_gpu.sh             # everything, 3 seeds (roughly 4-5 h on a T4)
#   SEEDS=2 NO_LONG=1 bash scripts/run_loop_gpu.sh   # quick version (roughly 1.5-2 h)
# Switches: SEEDS=n, NO_LONG=1 (no 100-iteration runs), SKIP_PLUS=1 (no RiL-PPO++ extensions),
#           SKIP_OPT=1 (no critic-free optimisers), NO_SC=1 (no test-time self-consistency)
# Stages run in priority order and every finished run is kept, so a session that hits the
# time limit can simply be started again: it continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

# --- data (identical to scripts/run_all_gpu.sh) ---
mkdir -p data/raw data/processed results/logs
[ -d data/raw/HybridQA ] || git clone --depth 1 https://github.com/wenhuchen/HybridQA data/raw/HybridQA
[ -d data/raw/WikiTables-WithLinks ] || git clone --depth 1 --no-checkout \
    https://github.com/wenhuchen/WikiTables-WithLinks data/raw/WikiTables-WithLinks
[ -f data/processed/wikitables.jsonl ] || $PY scripts/extract_wikitables.py
[ -f data/processed/test.pkl ] || $PY scripts/build_instances.py --n_train 10000 --n_val 1000
$PY tests/test_core.py

if [ "${SMOKE:-0}" = "1" ]; then
    $PY scripts/run_loop_suite.py --smoke
    echo "LOOP SMOKE TEST PASSED"
    exit 0
fi

EXTRA=""
if [ "${NO_LONG:-0}" = "1" ]; then EXTRA="$EXTRA --no-long"; fi
if [ "${SKIP_PLUS:-0}" = "1" ]; then EXTRA="$EXTRA --skip-plus"; fi
if [ "${SKIP_OPT:-0}" = "1" ]; then EXTRA="$EXTRA --skip-opt"; fi
if [ "${NO_SC:-0}" = "1" ]; then EXTRA="$EXTRA --no-sc"; fi
$PY scripts/run_loop_suite.py --seeds "${SEEDS:-3}" $EXTRA
$PY scripts/loop_report.py || true

$PY - <<'PYEOF'
import shutil, pathlib
tmp = pathlib.Path("bundle_tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
shutil.copytree("results/loop", tmp / "results" / "loop", ignore=shutil.ignore_patterns("actor_final.pt"))
shutil.make_archive("loop_results_bundle", "zip", tmp)
shutil.rmtree(tmp)
print("wrote loop_results_bundle.zip")
PYEOF
echo "LOOP ALL DONE"
