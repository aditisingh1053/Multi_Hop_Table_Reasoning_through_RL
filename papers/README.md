# Paper Collection — Annotated Bibliography

24 papers, grouped by the role each plays in the project.

Project: **Learning Multi-Hop Table Reasoning through RL: PPO-Based Entity Selection for Decision Making**

Role legend: **[METHOD]** we build directly on it · **[PRIOR]** closest prior art, we must beat it or differentiate from it · **[BASE]** framework we plug into · **[DATA]** dataset/benchmark. ⭐ = read first.

---

## 01 — RL Foundations (the optimiser we actually implement)

| Paper | arXiv | Role | Why it is in the stack |
|---|---|---|---|
| Schulman et al., *Proximal Policy Optimization Algorithms* | 1707.06347 | **[METHOD]** | The clipped surrogate objective at the core of our trainer: ratio clipping, multiple epochs per batch, composite loss (policy + value + entropy bonus). Implemented in `src/rl/ppo.py`. |
| Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | 1506.02438 | **[METHOD]** | GAE(γ, λ). Directly relevant: our episodes are short (2–5 hops) with a sparse terminal reward, so λ is a genuinely meaningful knob and a clean ablation axis. |
| Shao et al., *DeepSeekMath* (introduces **GRPO**) | 2402.03300 | **[METHOD]** | Group Relative Policy Optimization — critic-free, group-normalised rewards. Our secondary optimiser, enabling a PPO-vs-GRPO comparison, which is the live question in RL-for-reasoning right now. |

## 02 — RL for Multi-Hop Reasoning over Structured Data (closest prior art)

| Paper | arXiv | Role | Why it is in the stack |
|---|---|---|---|
| Xiong et al., *DeepPath: A RL Method for Knowledge Graph Reasoning* | 1707.06690 | **[PRIOR]** | First formulation of path finding as an MDP with policy gradients. We inherit the "reasoning = sequential decision problem" framing and its reward shaping (global accuracy + path efficiency + diversity). Its limitation, which we avoid: it needs both source *and* target entity. |
| Das et al., *MINERVA: Go for a Walk and Arrive at the Answer* (ICLR 2018) | 1711.05851 | **[PRIOR]** | The canonical query-conditioned graph walker: target entity unknown, agent learns when to STOP. Our action space and STOP action are modelled on this. Our delta: the graph is *induced from tables* (PK/FK joins, cell hyperlinks), so the candidate set is noisy and schema-dependent rather than a curated KG. |
| Shen et al., *M-Walk: Learning to Walk over Graphs using MCTS* | 1802.04394 | **[PRIOR]** | Shows RNN state encoding + MCTS-augmented policy improvement beats plain policy gradient under sparse rewards. Motivates our optional beam/MCTS inference-time decoding ablation. |
| Ribeiro et al., *Question Answering with Texts and Tables through Deep RL* | 2407.04858 | **[PRIOR]** ⭐ | **The most directly comparable paper.** Uses deep RL to learn the retrieval/reasoning path for text+table QA (OTT-QA) precisely because gold paths are unannotated. Read first — our contribution must be positioned against it. Our delta: they select *modules/actions*, we select *entities*, with PPO and a learned critic. |

## 03 — RL-Trained Reasoning & Search Agents (methodological template)

| Paper | arXiv | Role | Why it is in the stack |
|---|---|---|---|
| Jin et al., *Search-R1* | 2503.09516 | **[METHOD]** ⭐ | The modern recipe for multi-turn RL over a retrieval tool: interleaved reason/retrieve rollouts, **retrieved-token loss masking**, outcome-only reward. Our rollout loop and the "no policy loss on environment-inserted tokens" rule come from here. |
| Song et al., *R1-Searcher* | 2503.05592 | **[METHOD]** | Two-stage outcome-based RL with no process reward and no distillation cold start. Evidence our sparse-reward design is trainable; source of the format-reward → answer-reward curriculum in Stage 1. |
| *Table-R1: Region-based RL for Table Understanding* | 2505.12415 | **[PRIOR]** ⭐ | RL where the agent learns to extract the **minimal relevant table region** before answering (+14.4 points avg). This is single-table sub-table selection — the closest existing "RL picks what part of the table to look at" work. Our project is its multi-hop / cross-table generalisation. |
| *Table-r1: Self-supervised + RL for Program-based Table Reasoning in Small LMs* | 2506.06137 | **[METHOD]** | GRPO-style RL works on *small* models for table reasoning. Our compute-feasibility precedent: justifies a 0.5B–1.5B backbone instead of 7B+. |

## 04 — Table Reasoning Frameworks (what we plug the policy into)

| Paper | arXiv | Role | Why it is in the stack |
|---|---|---|---|
| Wang et al., *Chain-of-Table* | 2401.04398 | **[BASE]** | Iteratively applies table operations to evolve the table along a reasoning chain. Operation selection is currently done by LLM prompting — **injection point #2 for our learned policy**. |
| Chen et al., *TableRAG* (NeurIPS 2024) | 2410.04739 | **[BASE]** ⭐ | Schema + cell retrieval with query expansion and frequency-aware truncation. Gives us (a) the scalable cell-retrieval machinery that produces our **candidate entity set** each hop, and (b) our primary heuristic-retrieval baseline. |
| Cheng et al., *Binder* | 2210.02875 | **[BASE]** | Neural-symbolic: binds LM calls into SQL/Python. MMQA ships gold SQL, so Binder-style execution serves as both answer reader and a source of verifiable reward. |
| Ye et al., *Dater* | 2301.13808 | **[BASE]** | Decomposes evidence (sub-table selection) and question (sub-questions). Its sub-table decomposition is a strong non-RL baseline for exactly the selection problem we learn. |
| *Hybrid Graphs for Table-and-Text QA using LLMs* | 2501.17767 | **[BASE]** ⭐ | Builds an explicit entity graph linking table cells to text passages. **This is the environment we walk on** — we reuse the graph construction and replace their traversal/pruning heuristic with a PPO policy. |

## 05 — Datasets & Benchmarks

| Paper | Source | Role | Notes for us |
|---|---|---|---|
| Wu et al., *MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions* (ICLR 2025) | ICLR proceedings | **[DATA]** ⭐ | **Primary dataset.** 5,000 samples, 3,312 tables, 138 domains; 2–3 interlinked tables per instance with **annotated primary/foreign keys and gold SQL**. Those annotations give us free supervision for oracle paths and reward shaping — the reason this is the right choice. |
| Chen et al., *HybridQA* | 2004.07347 | **[DATA]** ⭐ | **Secondary dataset.** 72K multi-hop questions, 13K tables, cells hyperlinked to passages → a ready-made entity graph with gold answer locations. Ideal fit for an entity-selection action space. |
| Chen et al., *Open Question Answering over Tables and Text* (OTT-QA) | 2010.10439 | **[DATA]** | Hard open-domain variant (400K+ tables, 5M passages, hyperlinks removed). Held-out generalisation test, time permitting — full-scale retrieval exceeds a course compute budget. |
| Pasupat & Liang, *Compositional Semantic Parsing on Semi-Structured Tables* (WikiTQ) | 1508.00305 | **[DATA]** | Single-table standard benchmark. Sanity/regression set: the policy must not hurt single-hop performance. |
| Pal et al., *MultiTabQA* | 2305.12820 | **[DATA]** | Multi-table QA with tabular answers; extra multi-table training signal and a comparison point on join reasoning. |
| Zhu et al., *TAT-QA* | 2105.07624 | **[DATA]** | Hybrid financial QA with arithmetic. Optional; stresses numerical hops. |
| Zhao et al., *MultiHiertt* | 2206.01347 | **[DATA]** | Multiple hierarchical tables + text, long documents. Optional hard transfer setting. |
| Chen et al., *TabFact* | 1909.02164 | **[DATA]** | Table fact verification, used by Chain-of-Table / Binder / Dater — needed for apples-to-apples baseline numbers. |

---

## Reading order for the team

1. **2407.04858** (RL for text+table QA) — our nearest neighbour; defines the gap.
2. **2505.12415** (Table-R1 region RL) — RL for selection inside tables.
3. **2501.17767** (Hybrid graphs) — the environment.
4. **2410.04739** (TableRAG) — candidate generation + baseline.
5. **2503.09516** (Search-R1) — the RL training loop to copy.
6. **1707.06347 + 1506.02438** (PPO + GAE) — the maths we implement.
7. **MMQA** + **2004.07347** (HybridQA) — the data.

Everything else is cited for positioning, ablations, or optional extensions.
