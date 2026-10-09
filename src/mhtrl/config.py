"""YAML experiment configs with dotted command-line overrides (e.g. rl.iters=30)."""
from dataclasses import fields
from pathlib import Path

import yaml

from .env import EnvConfig, RewardConfig
from .rl.algos import RLConfig

SECTIONS = {"env": EnvConfig, "reward": RewardConfig, "rl": RLConfig}


def _cast(cls, key, value):
    ftype = {f.name: f.type for f in fields(cls)}[key]
    t = ftype if isinstance(ftype, type) else {"int": int, "float": float, "bool": bool, "str": str}[ftype]
    if value is None:            # an empty override such as rl.init_actor=
        return "" if t is str else value
    if t is bool and isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return t(value)


def load_config(path: str = "", overrides=()):
    raw = {}
    if path:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    for o in overrides:
        key, value = o.split("=", 1)
        sec, name = key.split(".", 1)
        raw.setdefault(sec, {})[name] = yaml.safe_load(value)
    out = {}
    for sec, cls in SECTIONS.items():
        vals = raw.get(sec, {}) or {}
        out[sec] = cls(**{k: _cast(cls, k, v) for k, v in vals.items()})
    out["name"] = raw.get("name", Path(path).stem if path else "run")
    return out
