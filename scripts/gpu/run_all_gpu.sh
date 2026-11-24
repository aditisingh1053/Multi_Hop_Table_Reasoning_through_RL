#!/usr/bin/env bash
# End-to-end pipeline for a GPU machine (Kaggle, Colab or a lab server). Resumable: re-running
# skips every step whose output already exists. Usage:
#   bash scripts/gpu/run_all_gpu.sh            # full suite (5 seeds main methods, 3 seeds ablations)
#   SMOKE=1 bash scripts/gpu/run_all_gpu.sh    # 2-3 minute end-to-end check, writes results_smoke/
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONUNBUFFERED=1

mkdir -p data/raw data/processed results/logs
[ -d data/raw/HybridQA ] || git clone --depth 1 https://github.com/wenhuchen/HybridQA data/raw/HybridQA
[ -d data/raw/WikiTables-WithLinks ] || git clone --depth 1 --no-checkout \
    https://github.com/wenhuchen/WikiTables-WithLinks data/raw/WikiTables-WithLinks
[ -f data/processed/wikitables.jsonl ] || $PY scripts/data/extract_wikitables.py
[ -f data/processed/test.pkl ] || $PY scripts/data/build_instances.py --n_train 10000 --n_val 1000
$PY tests/test_core.py

if [ "${SMOKE:-0}" = "1" ]; then
    $PY scripts/train/run_suite.py --smoke
    echo "SMOKE TEST PASSED"
    exit 0
fi

$PY scripts/data/data_stats.py
$PY scripts/train/run_suite.py --seeds-main "${SEEDS_MAIN:-5}" --seeds-ablation "${SEEDS_ABL:-3}"
$PY scripts/analysis/make_assets.py || echo "make_assets failed (not needed on the GPU machine)"

# everything the laptop needs to build the report
$PY - <<'PYEOF'
import shutil, pathlib
tmp = pathlib.Path("bundle_tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
shutil.copytree("results", tmp / "results", ignore=shutil.ignore_patterns("actor_final.pt"))
shutil.copytree("report/generated", tmp / "report_generated", dirs_exist_ok=True)
shutil.make_archive("results_bundle", "zip", tmp)
shutil.rmtree(tmp)
print("wrote results_bundle.zip")
PYEOF
echo "ALL DONE"
