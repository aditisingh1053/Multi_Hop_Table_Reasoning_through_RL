"""Dataset statistics for the report (Table 'datastats') and reachability checks.

Writes results/data_stats.json, report/generated/table_data.tex and report/generated/stats.tex.
Usage: python scripts/data_stats.py
"""
import numpy as np
from common import RESULTS, ROOT, load_bank, save_json

from mhtrl.data import PASSAGE
from mhtrl.env import EnvConfig, Episode


def reachable(p, cfg):
    """True if some gold path can be completed by the environment's candidate sets."""
    for path in p.gold_paths:
        ep = Episode(p, cfg)
        ok = True
        for n in path:
            if n not in ep.candidates():
                ok = False
                break
            ep.step(n)
            if ep.done and n != path[-1]:
                ok = False
                break
        if ok:
            return True
    return False


def stats(bank, cfg):
    items = bank.items
    n_paths = np.array([len(p.gold_paths) for p in items])
    has = n_paths > 0
    psg_paths = [sum(len(path) == 2 for path in p.gold_paths) / len(p.gold_paths) for p in items if p.gold_paths]
    cand0 = [len(Episode(p, cfg).candidates()) for p in items]
    reach = [reachable(p, cfg) for p in items if p.gold_paths]
    types = [p.answer_type for p in items]
    return {
        "questions": len(items),
        "tables": len({p.inst["table_id"] for p in items}),
        "rows": float(np.mean([p.n_rows for p in items])),
        "passages": float(np.mean([(p.kind == PASSAGE).sum() for p in items])),
        "nodes": float(np.mean([p.n for p in items])),
        "edges": float(np.mean([sum(len(a) for a in p.adj) / 2 for p in items])),
        "cand0": float(np.mean(cand0)),
        "with_gold": float(100 * has.mean()),
        "one_gold": float(100 * (n_paths == 1).mean()),
        "gold_paths": float(n_paths[has].mean()),
        "passage_paths": float(100 * np.mean(psg_paths)),
        "reachable": float(100 * np.mean(reach)),
        "answer_in_some_entity": float(100 * np.mean([p.contains.any() for p in items])),
        "type_table": sum(t == "table" for t in types),
        "type_passage": sum(t == "passage" for t in types),
    }


def main():
    cfg = EnvConfig()
    banks = {s: load_bank(s, cfg.k_passages) for s in ("train", "val", "test")}
    out = {s: stats(b, cfg) for s, b in banks.items()}
    train_tables = {p.inst["table_id"] for p in banks["train"].items}
    test_items = banks["test"].items
    out["test_tables_in_train"] = len({p.inst["table_id"] for p in test_items} & train_tables)
    out["test_questions_with_train_table"] = sum(p.inst["table_id"] in train_tables for p in test_items)
    out["val_tables_in_train"] = len({p.inst["table_id"] for p in banks["val"].items} & train_tables)
    save_json(out, RESULTS / "data_stats.json")
    save_json(sorted(p.qid for p in test_items if p.inst["table_id"] in train_tables),
              RESULTS / "test_qids_seen_table.json")
    for k, v in out.items():
        print(k, v)

    gen = ROOT / "report" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Questions", "questions", "{:,}"), ("Distinct tables", "tables", "{:,}"),
        ("Rows per table", "rows", "{:.1f}"), ("Linked passages per table", "passages", "{:.1f}"),
        ("Entities (nodes) per question", "nodes", "{:.1f}"), ("Row--passage edges", "edges", "{:.1f}"),
        ("Candidates at the first step", "cand0", "{:.1f}"),
        ("Questions with a gold path (\\%)", "with_gold", "{:.1f}"),
        ("Questions with exactly one gold path (\\%)", "one_gold", "{:.1f}"),
        ("Gold paths per question (if any)", "gold_paths", "{:.2f}"),
        ("Gold paths ending in a passage (\\%)", "passage_paths", "{:.1f}"),
        ("Gold path reachable under the cap $K$ (\\%)", "reachable", "{:.1f}"),
        ("Answer string in some entity (\\%)", "answer_in_some_entity", "{:.1f}"),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Statistics of the per-question entity graphs. Train and validation are drawn from "
             r"disjoint sets of HybridQA training tables; test is the full official dev set.}",
             r"\label{tab:datastats}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r" & Train & Validation & Test (dev) \\", r"\midrule"]
    for label, key, fmt in rows:
        vals = " & ".join(fmt.format(out[s][key]) for s in ("train", "val", "test"))
        lines.append(f"{label} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (gen / "table_data.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    macros = {
        "nTrain": f"{out['train']['questions']:,}", "nVal": f"{out['val']['questions']:,}",
        "nTest": f"{out['test']['questions']:,}", "nTrainTables": f"{out['train']['tables']:,}",
        "nValTables": f"{out['val']['tables']:,}", "nTestTables": f"{out['test']['tables']:,}",
        "testTablesInTrain": f"{out['test_tables_in_train']:,}",
        "testQuestionsWithTrainTable": f"{out['test_questions_with_train_table']:,}",
        "nTestTypeTable": f"{out['test']['type_table']:,}", "nTestTypePassage": f"{out['test']['type_passage']:,}",
        "testReachable": f"{out['test']['reachable']:.1f}", "testWithGold": f"{out['test']['with_gold']:.1f}",
    }
    (gen / "stats.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()),
                                   encoding="utf-8")
    print("wrote", gen)


if __name__ == "__main__":
    main()
