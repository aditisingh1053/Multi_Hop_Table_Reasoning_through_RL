# 3. Literature Survey Completed So Far

**Project:** Learning Multi-Hop Table Reasoning through RL — PPO-Based Entity Selection
**Date:** 2026-09-25
**Papers collected:** 24 (all PDFs in `papers/`, organised in 5 thematic folders)
**Papers close-read:** 0 of 24 — *scheduled Week 1.* Survey below is at abstract/method level from the papers themselves and the surrounding literature. Annotations get revised after close reading.

---

## 3.1 Survey structure

| Cluster | # | Question it answers |
|---|---|---|
| A. RL foundations | 3 | What exactly are we optimising, and how? |
| B. RL for multi-hop reasoning over structured data | 4 | Has "RL picks the next entity" been done before? |
| C. RL-trained reasoning & search agents | 4 | What is the current recipe for multi-turn RL over a retrieval tool? |
| D. Table reasoning frameworks | 5 | What do we plug into, and what do we replace? |
| E. Datasets & benchmarks | 8 | Where do we train and evaluate? |

---

## Cluster A — RL foundations (3 papers)

**Schulman et al. 2017, PPO** (1707.06347). Clipped surrogate objective `min(rÂ, clip(r,1±ε)Â)` as a first-order stand-in for trust-region methods, permitting multiple epochs of minibatch updates per rollout batch. This is our optimiser; the ratio clipping is what keeps updates stable when the policy's action set changes size from step to step.

**Schulman et al. 2016, GAE** (1506.02438). Exponentially-weighted advantage estimator `Â_t = Σ(γλ)^l δ_{t+l}` trading bias against variance via λ. Especially relevant here: our episodes are 2–5 steps with reward concentrated at the terminal step, so λ has a large, measurable effect — a natural ablation for an optimization course.

**Shao et al. 2024, DeepSeekMath / GRPO** (2402.03300). Replaces the learned critic with group-relative reward normalisation across sampled rollouts of the same prompt. Cheaper (no value network) and now the default in LLM reasoning work. We implement it as our second optimiser to get a PPO-vs-GRPO comparison under identical rewards.

**Takeaway for us.** The optimizer choice is a real experimental question, not boilerplate: PPO's learned critic may or may not pay for itself at 2–5 step horizons. That comparison is a core result of the project.

---

## Cluster B — RL for multi-hop reasoning over structured data (4 papers)

**Xiong et al. 2017, DeepPath** (1707.06690). First to frame KG path finding as an MDP with policy gradients; reward = global accuracy + path efficiency + path diversity. Our reward shaping is descended from this. Key limitation: it requires both source *and* target entity, so it is link prediction, not QA.

**Das et al. 2018, MINERVA** (1711.05851, ICLR). Query-conditioned graph walker with an unknown target and a learned `STOP` action; LSTM history encoding; policy gradient with entropy regularisation. Beats DeepPath with a single model rather than per-relation classifiers. **This is the closest architectural ancestor of our policy** — our variable-size action scoring and STOP action come from here.

**Shen et al. 2018, M-Walk** (1802.04394). Adds MCTS-based policy improvement to graph walking, addressing the sparse-reward problem directly. Motivates our optional inference-time beam/MCTS decoding ablation.

**Ribeiro et al. 2024, QA with Texts and Tables through Deep RL** (2407.04858). ⭐ **The nearest neighbour to this project.** Applies deep RL to learn the information-retrieval path for hybrid text+table QA on OTT-QA, explicitly motivated by the fact that gold paths are not annotated in such datasets. Actions are retrieval/reasoning module choices rather than individual entities.

**Takeaway for us.** The "reasoning as sequential decision-making with RL" framing is well-validated — but on curated knowledge graphs (2017–18) or at module granularity (2024). Nobody has done **entity-level selection over a graph induced from table schemas and cell values, optimised with PPO against a verifiable reward.** That is the gap.

---

## Cluster C — RL-trained reasoning & search agents (4 papers)

**Jin et al. 2025, Search-R1** (2503.09516). Multi-turn RL where the LLM interleaves reasoning with retrieval calls. Three techniques we adopt: interleaved rollouts, **retrieved-token loss masking** (never compute policy loss on environment-inserted tokens), and an outcome-only reward. Reports +41% (Qwen2.5-7B) over RAG baselines across 7 QA datasets.

**Song et al. 2025, R1-Searcher** (2503.05592). Two-stage outcome-based RL — a format/validity reward first, then an answer-correctness reward — with no process reward and no distillation cold start. Direct evidence that our sparse-reward design is trainable, and the source of our Stage 1 → Stage 2 curriculum.

**Table-R1: Region-based RL for Table Understanding** (2505.12415). ⭐ RL where the model learns to extract the **minimum relevant table region** before answering, folding region extraction into the reasoning chain; +14.36 points average across three benchmarks and several base models. This is the strongest existing evidence that *learned selection* beats *heuristic selection* in table reasoning — but it is single-table.

**Table-r1: Self-supervised + RL for Program-based Table Reasoning in Small LMs** (2506.06137). Two-stage method: a layout-transformation self-supervised task, then a mix-paradigm GRPO variant preferring program-based reasoning with text-based reasoning as fallback. Demonstrates RL for table reasoning works on *small* models — our compute-feasibility precedent.

**Takeaway for us.** The 2025 literature has converged on a clear recipe: outcome-verifiable rewards, multi-turn rollouts with tool calls, staged reward curricula, GRPO or PPO. Our job is to instantiate that recipe for cross-table entity selection — and Table-R1 tells us the payoff for learned selection is large (double digits) in the single-table case.

---

## Cluster D — Table reasoning frameworks (5 papers)

**Wang et al. 2024, Chain-of-Table** (2401.04398). Evolves the table itself through a chain of atomic operations (select, group, sort, add-column), SOTA on WikiTQ/FeTaQA/TabFact. The operation-selection step is an LLM prompt — a heuristic decision we can replace with a learned policy (**injection point #2**).

**Chen et al. 2024, TableRAG** (2410.04739, NeurIPS). Query expansion + separate schema and cell retrieval + frequency-aware truncation, for million-token tables. Provides our **candidate-generation machinery** and our primary heuristic baseline (B2).

**Cheng et al. 2023, Binder** (2210.02875). Binds LM API calls into SQL/Python programs, executed symbolically. Relevant because MMQA ships gold SQL: Binder-style execution gives us a deterministic answer reader and a verifiable reward signal.

**Ye et al. 2023, Dater** (2301.13808). Decomposes both evidence (sub-table selection) and questions (sub-questions) before reasoning. Its sub-table decomposition is a strong non-RL competitor on precisely our selection problem.

**Hybrid Graphs for Table-and-Text QA using LLMs** (2501.17767). ⭐ Constructs an explicit entity graph linking table cells to text passages, then prunes/traverses it heuristically. **This is our environment** — we reuse the graph construction and replace the traversal heuristic with PPO (**injection point #1**).

**Takeaway for us.** Every one of these frameworks contains a selection decision made by similarity, heuristic, or greedy prompt. None optimises that decision against final answer correctness. That is a clean, well-defined slot for an RL policy, and it means our method is a *drop-in module* rather than a competing system — which makes the comparison fair and the contribution legible.

---

## Cluster E — Datasets & benchmarks (8 papers)

| Dataset | Scale | Multi-hop? | Why in the survey |
|---|---|---|---|
| **MMQA** (ICLR 2025) | 5,000 QA, 3,312 tables, 138 domains | ✅ 2–3 linked tables | **Primary.** Annotated PK/FK + gold SQL → free oracle paths and verifiable evidence reward. Evaluation covers multi-table retrieval, text-to-SQL, multi-table QA, and PK/FK selection — the last two are almost exactly our task. |
| **HybridQA** (2004.07347) | 72K QA, 13K tables | ✅ table+text | **Secondary.** Cell hyperlinks give a natural entity graph with gold answer locations. |
| **OTT-QA** (2010.10439) | 45,841 QA, 400K+ tables, 5M passages | ✅ open-domain | Held-out OOD test; full retrieval exceeds budget so subset only. Also the dataset used by our nearest-neighbour paper (2407.04858). |
| WikiTQ (1508.00305) | 22K QA | ✗ single-table | Regression check — must not degrade single-hop. |
| MultiTabQA (2305.12820) | multi-table, tabular answers | ✅ | Extra multi-table training signal; join reasoning. |
| TAT-QA (2105.07624) | 16K QA, finance | ✅ hybrid + arithmetic | Optional; numerical hops. |
| MultiHiertt (2206.01347) | hierarchical tables + text | ✅ | Optional hard transfer. |
| TabFact (1909.02164) | 118K statements | ✗ | Needed for apples-to-apples numbers against Chain-of-Table / Binder / Dater. |

**Takeaway for us.** MMQA is the right primary dataset, and the reason is specifically its **PK/FK and gold-SQL annotations**: they convert an otherwise unsupervised path-finding problem into one with an oracle, which gives us behaviour cloning for a warm start, a dense shaping reward, and an upper bound to measure headroom against. Without that, a one-semester RL project on this task would be considerably riskier.

---

## 3.2 Synthesis: the four-way gap

| | Curated KG | Single table | **Multi-table / hybrid** |
|---|---|---|---|
| **Heuristic / prompted selection** | — | Chain-of-Table, Dater, Binder | TableRAG, Hybrid Graphs |
| **Learned (RL) selection** | DeepPath, MINERVA, M-Walk | Table-R1, Table-r1 | ⬅ **our project** (adjacent: Ribeiro 2024, at module granularity) |

The bottom-right cell is where we sit. The literature supplies everything needed to fill it: the environment (2501.17767), candidate generation (2410.04739), the training recipe (2503.09516, 2503.05592), the optimiser (1707.06347, 1506.02438, 2402.03300), the evidence that learned selection pays off (2505.12415), the evidence small models suffice (2506.06137), and the annotated data (MMQA).

## 3.3 Survey gaps still to close

1. **Close-read the 7 starred papers** — abstract-level annotation is not enough to defend novelty at a seminar. Week 1.
2. **Verify the 2407.04858 delta** in detail; it is the only paper that could subsume our contribution.
3. **Check for 2026 work** on RL for multi-table reasoning — the field moves in months. Repeat the search at Week 4 and again at Week 7 before submission. (Several recent-looking titles surfaced during collection — e.g. work on TableQA with verifiable reasoning-trace rewards, TableGPT-R1, and multi-agent TableQA frameworks — which should be checked and added if relevant.)
4. **Extract concrete baseline numbers** (WikiTQ/TabFact/HybridQA/MMQA) from Chain-of-Table, TableRAG, Dater, Binder into a comparison table, so our results are contextualised rather than free-floating.
5. **Add classical RL references** for the report's background section (Sutton & Barto; REINFORCE; A3C) — textbook citations, not downloads.
