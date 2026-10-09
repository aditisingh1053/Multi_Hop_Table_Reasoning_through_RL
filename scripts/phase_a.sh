#!/usr/bin/env bash
# Phase A: validation baselines, then one seed of BC and PPO to check timing and learning.
set -e
cd "$(dirname "$0")/.."
PY=/c/Users/aksgu/AppData/Local/Programs/Python/Python312/python.exe
export PYTHONUNBUFFERED=1
$PY scripts/run_baselines.py --split val
$PY scripts/train.py --method bc --seed 0
$PY scripts/train.py --config configs/ppo_main.yaml --seed 0
