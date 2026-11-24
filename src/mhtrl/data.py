"""Loads HybridQA and turns each question into a graph of table rows and linked passages.

Rows are written as "header is value ; ...", passages are the Wikipedia pages linked from the
table's cells, and an edge joins a row to the passages it links to. Gold evidence comes from
HybridQA's traced answer nodes (found by string matching, so it is noisy).
"""
import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROW, PASSAGE = 0, 1
MAX_PASSAGE_WORDS = 300  # passages are Wikipedia lead sections (~200 words on average)


@dataclass
class Instance:
    qid: str
    question: str
    answer: str
    table_id: str
    title: str
    n_rows: int
    node_kind: List[int]                 # ROW or PASSAGE
    node_text: List[str]                 # text shown to the reader and the encoder
    node_ref: List[object]               # row index (int) or passage link (str)
    adj: List[List[int]]                 # undirected row<->passage links
    gold_nodes: List[Tuple] = field(default_factory=list)   # raw traced answer nodes
    gold_paths: List[Tuple[int, ...]] = field(default_factory=list)
    answer_type: Optional[str] = None    # 'table' / 'passage' from dev_reference.json

    @property
    def n_nodes(self) -> int:
        return len(self.node_kind)


def link_to_name(link: str) -> str:
    name = link[len("/wiki/"):] if link.startswith("/wiki/") else link
    return urllib.parse.unquote(name).replace("_", " ")


def linearise_row(title: str, header: List[str], cells: List[str]) -> str:
    parts = [f"{h} is {v}" for h, v in zip(header, cells) if v.strip()]
    return f"{title} : " + " ; ".join(parts)


def build_instance(q: dict, table_rec: dict, answer_type: Optional[str] = None) -> Instance:
    t = table_rec["table"]
    passages: Dict[str, str] = table_rec["passages"]
    title = t.get("title", "") or ""
    header = [h[0] for h in t["header"]]
    data = t["data"]

    node_kind, node_text, node_ref = [], [], []
    for r, row in enumerate(data):
        cells = [c[0] for c in row]
        node_kind.append(ROW)
        node_text.append(linearise_row(title, header, cells))
        node_ref.append(r)
    n_rows = len(data)

    link_to_node: Dict[str, int] = {}
    edges = set()
    for r, row in enumerate(data):
        for cell in row:
            for link in cell[1]:
                if link not in passages or not passages[link].strip():
                    continue
                if link not in link_to_node:
                    link_to_node[link] = len(node_kind)
                    words = passages[link].split()[:MAX_PASSAGE_WORDS]
                    node_kind.append(PASSAGE)
                    node_text.append(f"{link_to_name(link)} : " + " ".join(words))
                    node_ref.append(link)
                edges.add((r, link_to_node[link]))

    adj: List[List[int]] = [[] for _ in node_kind]
    for r, p in sorted(edges):
        adj[r].append(p)
        adj[p].append(r)

    gold_nodes = [tuple([n[0], tuple(n[1]), n[2], n[3]]) for n in q.get("answer-node", [])]
    gold_paths = set()
    for _, (r, _c), link, kind in gold_nodes:
        if r >= n_rows:
            continue
        if kind == "table":
            gold_paths.add((r,))
        elif link in link_to_node:
            gold_paths.add((r, link_to_node[link]))

    return Instance(
        qid=q["question_id"], question=q["question"], answer=q.get("answer-text", ""),
        table_id=q["table_id"], title=title, n_rows=n_rows, node_kind=node_kind,
        node_text=node_text, node_ref=node_ref, adj=adj, gold_nodes=gold_nodes,
        gold_paths=sorted(gold_paths), answer_type=answer_type,
    )


def load_questions(hybridqa_dir: Path, split: str) -> List[dict]:
    name = f"{split}.traced.json" if split in ("train", "dev") else f"{split}.json"
    with open(Path(hybridqa_dir) / "released_data" / name, encoding="utf-8") as f:
        return json.load(f)


def load_dev_answer_types(hybridqa_dir: Path) -> Dict[str, str]:
    with open(Path(hybridqa_dir) / "released_data" / "dev_reference.json", encoding="utf-8") as f:
        ref = json.load(f)
    types = {qid: "table" for qid in ref["table"]}
    types.update({qid: "passage" for qid in ref["passage"]})
    return types


def load_tables(wikitables_jsonl: Path, table_ids) -> Dict[str, dict]:
    wanted = set(table_ids)
    out = {}
    with open(wikitables_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["table_id"] in wanted:
                out[rec["table_id"]] = rec
    return out
