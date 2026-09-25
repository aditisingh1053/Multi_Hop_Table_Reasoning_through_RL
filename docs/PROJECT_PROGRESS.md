# 2. Current Project Progress

**Project:** Learning Multi-Hop Table Reasoning through RL — PPO-Based Entity Selection
**Status as of:** 2026-09-25
**Overall phase:** Phase 0 complete — *design and literature*. Implementation (Phase 1) not yet started.

---

## 2.1 Honest status summary

| Phase | Status |
|---|---|
| **Phase 0 — Scoping, formulation, literature** | ✅ **Complete** |
| Phase 1 — Data + environment | ⬜ Not started (Week 1–2) |
| Phase 2 — Baselines | ⬜ Not started (Week 3) |
| Phase 3 — Policy + BC | ⬜ Not started (Week 4) |
| Phase 4 — PPO training | ⬜ Not started (Week 5) |
| Phase 5 — Ablations + comparison | ⬜ Not started (Week 6–7) |
| Phase 6 — Report + seminar | ⬜ Not started (Week 8) |

**No code has been written and no experiments have been run yet.** The repository currently contains the literature collection and the design documents. This is the correct state at the start of Week 1, and Phase 0 was the right thing to finish first: the compute plan in `COMPUTE_REQUIREMENTS.md` is only credible because the formulation was pinned down before any code.

---

## 2.2 Completed work

### ✅ Problem formulation (design frozen)
- Multi-hop table reasoning cast as a finite-horizon MDP with `H = 4`.
- **State:** question embedding ⊕ evidence embedding ⊕ focused-entity embedding ⊕ 16 structural scalars (≈1168-d).
- **Action:** variable-size candidate entity set `C_t` (top-`K = 20`) plus `STOP`, scored by a pointer/attention head so the policy is schema-agnostic and transfers across tables.
- **Reward:** composite and fully verifiable — answer EM (λ=1.0) + gold-evidence recall (λ=0.3) + per-hop penalty (−0.02) + invalid-action penalty (−0.1). **No reward model**, which is the decision that keeps the project inside budget.
- **Optimiser:** PPO with clipped surrogate + GAE(γ=0.99, λ=0.95), shared trunk actor-critic, ~1.1M trainable parameters.

### ✅ Environment design
- Entity graph induced from tables with four edge types: `same_row`, `same_col`, `fk_join` (from MMQA PK/FK annotations), `hyperlink` (HybridQA), `value_match`.
- Candidate generation reuses TableRAG-style schema + cell retrieval.

### ✅ Dataset selection and justification
| Dataset | Role | Why chosen |
|---|---|---|
| **MMQA** (ICLR 2025) | primary | 2–3 interlinked tables per instance with **annotated primary/foreign keys and gold SQL** → free oracle-path supervision and a verifiable evidence reward. This annotation is the reason the project is tractable. |
| **HybridQA** | secondary | 72K multi-hop questions over hyperlinked table cells → a ready-made entity graph with gold answer locations. |
| WikiTQ | regression | single-hop sanity check |
| OTT-QA | optional OOD | full open retrieval is out of budget; subset only |

### ✅ Experimental protocol fixed in advance
- **9 methods** under an identical reader and identical evidence budget: random, embedding-greedy, BM25, LLM-prompted selection, behaviour cloning, oracle-upper-bound, **PPO**, GRPO, REINFORCE.
- **~20 ablations** across reward components, GAE λ, γ, clip ε, epochs, horizon `H`, candidate width `K`, state features, reader size.
- **3 seeds, mean ± std on everything learned.** Single-seed RL numbers do not go in the report.
- Metrics fixed: answer EM/F1, path accuracy, evidence recall/precision, avg. hops, reader calls + tokens, sample-efficiency curves, critic explained variance, per-update KL.

### ✅ Literature survey — 24 papers collected, read-prioritised and annotated
See `LITERATURE_SURVEY.md` and `papers/README.md`. Grouped into RL foundations (3), RL for multi-hop structured reasoning (4), RL-trained reasoning/search agents (4), table-reasoning frameworks (5), datasets/benchmarks (8). Every paper has a stated role: method / prior art / base framework / data.

### ✅ Compute plan
Tier A (~125 GPU-hours total, single 8–16 GB GPU, ~$25–30 optional API) with a documented risk table and a scope-down fallback. See `COMPUTE_REQUIREMENTS.md`.

### ✅ Repository scaffold and 8-week plan
Directory structure defined down to module level, with a weekly exit criterion for each of the 8 weeks.

---

## 2.3 The gap we identified (project justification)

From the survey, three findings define the contribution:

1. **RL for *structured* multi-hop reasoning is well-established but pre-LLM and KG-only** — DeepPath (2017), MINERVA (2018), M-Walk (2018). None of them operate on tables, where the graph must be *induced* from schema and cell values rather than read off a curated KG.
2. **RL for table reasoning exists but is single-table.** Table-R1 (2505.12415) learns to select a minimal *region* of one table; Table-r1 (2506.06137) learns program-based reasoning on small LMs. Neither addresses cross-table hop selection.
3. **Modern table-reasoning frameworks all make the selection decision heuristically or by greedy LLM prompt** — TableRAG by retrieval similarity, Chain-of-Table and Dater by prompting, hybrid graphs by a traversal heuristic. All are myopic with respect to final answer correctness.

The one paper genuinely close to us is **Ribeiro et al. 2024, *QA with Texts and Tables through Deep RL* (2407.04858)** — RL for the text+table retrieval path on OTT-QA. Our differentiation: entity-level (not module-level) actions, PPO with a learned critic (not their policy-gradient setup), a verifiable composite reward, and a controlled PPO/GRPO/REINFORCE/BC comparison under a fixed reader. **This paper is the first reading assignment**, and if a close read shows the overlap is larger than it appears, the fallback is to pivot toward the Table-R1 cross-table generalisation framing, which is unoccupied either way.

---

## 2.4 Immediate next actions (Week 1)

| # | Action | Owner | Exit criterion |
|---|---|---|---|
| 1 | Close-read 2407.04858, 2505.12415, 2501.17767; write a 1-page delta note | all | Contribution statement confirmed or pivoted |
| 2 | Download MMQA + HybridQA; write `src/data/loaders.py` to a common schema | | Both load; counts match published stats |
| 3 | Implement `src/data/graph_builder.py` | | Graph stats: nodes/edges per instance, hop distance to gold |
| 4 | **Reachability check** | | Gold evidence reachable within `H=4` for ≥95% of MMQA train — *go/no-go for the whole formulation* |
| 5 | Extract oracle paths from MMQA gold SQL | | ≥90% of train instances yield a valid oracle entity sequence |
| 6 | `requirements.txt`, env setup, GPU access confirmed | | `train.py --smoke` runs a random agent |

Action #4 is the gate. If gold evidence is not reliably reachable within the horizon, the action space or edge set is wrong and must be fixed in Week 1 — before any RL code is written.

---

## 2.5 Open questions to resolve during implementation

1. **Is there enough headroom?** If the oracle upper bound (B6) sits close to the embedding-greedy baseline (B2), selection is not the bottleneck on MMQA and we need a harder split (filter to 3-table / 3-hop instances). Measured in Week 3, before writing the PPO trainer.
2. **Reader cache hit rate** — drives the whole time budget. Measured in Week 2.
3. **Does the pointer policy generalise across schemas**, or does it overfit to MMQA's 138 domains? Tested by HybridQA transfer in Week 7.
4. **PPO vs GRPO at this scale** — GRPO's critic-free group normalisation may simply be better when episodes are this short and rewards this sparse. Genuinely open; either result is reportable.
