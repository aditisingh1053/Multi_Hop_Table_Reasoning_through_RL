"""Shared helpers for the command-line scripts."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

if sys.platform == "win32":
    # keep Windows from idle-sleeping while an experiment runs (released when the process exits)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

import numpy as np
import torch

torch.set_num_threads(min(8, os.cpu_count() or 8))  # fixed count: cached reader outputs stay reproducible

from mhtrl.env import Bank

DATA = Path(os.environ.get("MHTRL_DATA", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("MHTRL_RESULTS", ROOT / "results"))
CONFIGS = ROOT / "configs"


def config_path(name: str) -> Path:
    """configs/<group>/<name>.yaml for a config name such as 'loop_full' (names are unique)."""
    hits = sorted(CONFIGS.rglob(f"{name}.yaml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one configs/**/{name}.yaml, found {len(hits)}")
    return hits[0]


# Training uses the first 5,000 questions of the table-ordered training split (whole tables,
# except the last one), which keeps reader-in-the-loop training affordable on a laptop CPU.
TRAIN_LIMIT = int(os.environ.get("MHTRL_TRAIN_LIMIT", 5000))


def load_bank(split: str, k_passages: int = 5, limit=None) -> Bank:
    if split == "train" and limit is None:
        limit = TRAIN_LIMIT
    return Bank(str(DATA / f"{split}.pkl"), k_passages=k_passages, limit=limit)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)


class Tee:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()
