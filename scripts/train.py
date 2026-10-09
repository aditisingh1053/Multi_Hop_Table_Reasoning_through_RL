"""Train one selector (bc | ppo | grpo | reinforce) for one seed.

Usage:
  python scripts/train.py --method bc  --seed 0
  python scripts/train.py --config configs/ppo.yaml --seed 0 [--set rl.iters=30 ...]
Outputs go to results/runs/<name>_s<seed>/.
"""
import argparse
from dataclasses import asdict

from common import RESULTS, Tee, load_bank, save_json

from mhtrl.config import load_config
from mhtrl.reader import DEFAULT_READER, ReaderCache
from mhtrl.rl.algos import train_rl
from mhtrl.rl.bc import train_bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="", help="bc, or leave empty to use the config's rl.algo")
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--bc_epochs", type=int, default=8)
    ap.add_argument("--train_limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    env_cfg, rc, rl = cfg["env"], cfg["reward"], cfg["rl"]
    rl.seed = args.seed
    name = "bc" if args.method == "bc" else cfg["name"]
    out = RESULTS / "runs" / f"{name}_s{args.seed}"
    log = Tee(out / "train.log")
    log(f"config: env={asdict(env_cfg)} reward={asdict(rc)} rl={asdict(rl)}")

    train_bank = load_bank("train", env_cfg.k_passages, args.train_limit)
    val_bank = load_bank("val", env_cfg.k_passages)
    cache = ReaderCache(DEFAULT_READER)
    if args.method == "bc":
        best = train_bc(train_bank, val_bank, cache, env_cfg, str(out), seed=args.seed,
                        epochs=args.bc_epochs, log=log)
        save_json({"best_epoch": best["epoch"], "best_val_F1": best["val_F1"]}, out / "best.json")
    else:
        if rl.init_actor:
            rl.init_actor = rl.init_actor.replace("{seed}", str(args.seed))
        best = train_rl(train_bank, val_bank, cache, env_cfg, rc, rl, str(out), log=log)
        save_json({"best_iter": best["iter"], "best_val_F1": best["val_F1"]}, out / "best.json")
    log(f"done: {out}")


if __name__ == "__main__":
    main()
