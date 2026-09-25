# Learning Multi-Hop Table Reasoning through RL
### PPO-Based Entity Selection for Decision Making

Course project — Optimization Methods for Machine Learning, 7th semester.

---

## One-paragraph summary

Multi-hop table QA requires chaining evidence across rows, columns and linked tables. At every step, a reasoning system must decide **which entity to expand next** — and today that decision is made by embedding similarity (TableRAG), a traversal heuristic (hybrid graphs), or a greedy LLM prompt (Chain-of-Table, Dater). All three are myopic: they score the immediate relevance of the next hop rather than the probability that the *final* answer is correct, so errors compound across hops. We formulate entity selection as a finite-horizon MDP and learn the selection policy with **PPO + GAE** in an actor-critic setup, using a fully verifiable reward (answer exact-match + gold-evidence recall + hop-efficiency and validity penalties — no reward model). The policy is a schema-agnostic pointer network over frozen sentence embeddings, ~1.1M trainable parameters, and drops into existing table-reasoning frameworks as a replacement for the heuristic selector.

---

## Documents

| Document | Contents |
|---|---|
| [docs/IMPLEMENTATION_OVERVIEW.md](docs/IMPLEMENTATION_OVERVIEW.md) | Full technical plan: MDP formulation, state/action/reward, network architecture, PPO objective and hyperparameters, training curriculum, 9 baselines, ~20 ablations, metrics, repo layout, 8-week schedule |
| [docs/COMPUTE_REQUIREMENTS.md](docs/COMPUTE_REQUIREMENTS.md) | **Deliverable 1.** Hardware, disk, ~125 GPU-hour budget, API cost estimate, risks |
| [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) | **Deliverable 2.** Phase status, completed work, identified gap, Week-1 actions |
| [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) | **Deliverable 3.** 24-paper survey in 5 clusters, synthesis, remaining gaps |
| [papers/README.md](papers/README.md) | Annotated bibliography — every paper with its role and reading priority |

## Paper collection (24 PDFs)

```
papers/
├── 01_rl_foundations/                        (3)  PPO, GAE, GRPO/DeepSeekMath
├── 02_rl_multihop_structured_reasoning/      (4)  DeepPath, MINERVA, M-Walk, RL for text+table QA
├── 03_rl_llm_reasoning_agents/               (4)  Search-R1, R1-Searcher, Table-R1 ×2
├── 04_table_reasoning_frameworks/            (5)  Chain-of-Table, TableRAG, Binder, Dater, Hybrid Graphs
└── 05_datasets_benchmarks/                   (8)  MMQA, HybridQA, OTT-QA, WikiTQ, MultiTabQA, TAT-QA, MultiHiertt, TabFact
```

## Where this sits in the literature

| | Curated KG | Single table | **Multi-table / hybrid** |
|---|---|---|---|
| **Heuristic / prompted selection** | — | Chain-of-Table, Dater, Binder | TableRAG, Hybrid Graphs |
| **Learned (RL) selection** | DeepPath, MINERVA, M-Walk | Table-R1, Table-r1 | ⬅ **this project** |

## Status

**Phase 0 (design + literature) complete. No code written yet** — implementation starts Week 1 with data loading, entity-graph construction, and the gold-evidence reachability check that gates the whole formulation. See [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md).

## Datasets

- **Primary:** MMQA (ICLR 2025) — 5,000 QA over 3,312 tables, annotated primary/foreign keys and gold SQL
- **Secondary:** HybridQA — 72K multi-hop questions over hyperlinked table cells
- **Regression:** WikiTableQuestions · **Optional OOD:** OTT-QA subset
