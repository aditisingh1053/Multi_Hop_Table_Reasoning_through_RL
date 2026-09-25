# Implementation Overview

**Project:** Learning Multi-Hop Table Reasoning through RL — PPO-Based Entity Selection for Decision Making
**Course:** Optimization Methods for Machine Learning (7th sem)
**Date:** 2026-09-25

---

## 1. The problem, stated precisely

Multi-hop table QA requires chaining evidence across rows, columns and linked tables. Every existing framework — Chain-of-Table, TableRAG, Dater, hybrid-graph methods — has to answer the same question at every step:

> Given the question and what I have gathered so far, **which entity do I expand next?**

Today that decision is made by one of three things: lexical/embedding similarity (TableRAG), a hand-written heuristic (hybrid graphs), or a greedy LLM prompt (Chain-of-Table, Dater). All three are *myopic*: they optimise the immediate relevance of the next hop, not the probability that the final answer is correct. Errors compound multiplicatively across hops, and this is the dominant failure mode reported in the multi-hop table QA literature.

**Our claim.** Entity selection is a sequential decision problem with delayed reward, so it should be *learned* with RL rather than hand-scored. We train a PPO actor-critic policy that selects the next entity to expand, rewarded by final answer correctness, and drop it into an existing table-reasoning framework as a replacement for the heuristic selector.

**What makes this a good course project for an optimization course:** the deliverable is not "an LLM does table QA." It is a concrete, small, fully-inspectable optimization problem — a clipped surrogate objective, GAE advantage estimation, an actor-critic with a learned baseline, reward shaping, and a variance/bias trade-off we can measure. The LLM is a fixed component of the environment, not the thing being optimised.

---

## 2. MDP formulation

We formalise multi-hop table reasoning as a finite-horizon MDP `M = (S, A, P, R, γ, H)` with horizon `H = 4` hops (tunable; 2–3 covers most of MMQA).

### 2.1 The environment: an entity graph induced from tables

For each question instance we build a graph `G = (V, E)` offline (following *Hybrid Graphs*, 2501.17767):

- **Nodes `V`** — one node per candidate *entity*, where an entity is any of:
  - a **cell value** (`table_id, row_id, col_id, value`)
  - a **row** (a record, for join-style hops)
  - a **column/schema element** (for aggregation hops)
  - a **linked passage** (HybridQA hyperlink target)
- **Edges `E`** — typed, and derived from data, not guessed:
  - `same_row` / `same_col` (intra-table structural)
  - `fk_join` (MMQA primary/foreign-key annotations — exact, free supervision)
  - `hyperlink` (HybridQA cell → Wikipedia passage)
  - `value_match` (equal or fuzzy-equal cell values across tables — our only noisy edge type)

### 2.2 State

```
s_t = [ q_emb ;  E_t_emb ;  c_emb ;  φ_t ]
```

| Component | Dim | What it is |
|---|---|---|
| `q_emb` | 384 | frozen sentence-encoder embedding of the question |
| `E_t_emb` | 384 | mean-pooled embedding of evidence gathered so far (the reasoning context) |
| `c_emb` | 384 | embedding of the currently-focused entity |
| `φ_t` | ~16 | hand-crafted scalars: hop index `t/H`, `|C_t|`, fraction of `q` tokens already covered by `E_t`, #distinct tables touched, #visited nodes, max/mean candidate similarity, whether an answer-typed cell is present |

Total state dim ≈ **1168**. Deliberately small: the three embeddings are produced by a **frozen** encoder, so the entire trainable policy is a small MLP. This is what makes the project run on one modest GPU (see `COMPUTE_REQUIREMENTS.md`).

### 2.3 Action space

At step `t` the environment exposes a **variable-size candidate set** `C_t ⊆ V` (the top-`K` unvisited neighbours of visited nodes, `K = 20`, ranked by TableRAG-style retrieval), plus a terminal action:

```
A_t = C_t ∪ {STOP}
```

Because `|A_t|` varies, we use a **pointer/attention-style scoring head** rather than a fixed softmax over a fixed action set:

```
score(a) = w_out · tanh( W_s s_t + W_a a_emb )          # a_emb = 384-d entity embedding + type one-hot
π(a | s_t) = softmax over a ∈ A_t of score(a)           # invalid/visited actions masked to -inf
```

This is the standard trick from MINERVA (1711.05851) and is what lets one policy generalise across tables of different schemas — the policy never learns "column 3", it learns *relational features*.

### 2.4 Transition

`P(s_{t+1} | s_t, a_t)` is deterministic given the environment:
1. Add `a_t` to the evidence set `E_{t+1} = E_t ∪ {a_t}`, mark visited.
2. Expand `a_t`: fetch its neighbours in `G`, re-rank, form `C_{t+1}`.
3. If `a_t = STOP` or `t = H`: terminate, hand `E_t` to the **reader** (a frozen LLM or Binder-style SQL executor) which produces the answer.

### 2.5 Reward

Terminal reward with light shaping. All shaping terms are ablatable — a required experiment, not a nice-to-have.

```
R = λ_ans · R_answer  +  λ_ev · R_evidence  +  λ_eff · R_efficiency  +  λ_inv · R_invalid
```

| Term | Definition | Default λ | Purpose |
|---|---|---|---|
| `R_answer` | exact match (1/0) on the final answer; token-F1 for free-form | **1.0** | the true objective; verifiable, no reward model needed |
| `R_evidence` | recall of gold evidence cells in `E_T` (from MMQA gold SQL / HybridQA answer-node annotation) | **0.3** | dense-ish signal to beat the cold-start problem |
| `R_efficiency` | `−0.02` per hop taken | **1.0** | discourages exhaustive expansion; makes STOP meaningful |
| `R_invalid` | `−0.1` per selection of an already-visited or unreachable node | **1.0** | format/validity shaping, à la R1-Searcher Stage 1 |

Discount `γ = 0.99`, GAE `λ = 0.95` (both swept).

**Why RLVR-style verifiable reward, not a reward model:** answer correctness is checkable by string match against gold, and evidence recall is checkable against gold SQL. No reward model to train, no reward hacking surface, no extra GPU. This is the single biggest compute saving in the design.

### 2.6 Policy and value networks

```
Encoder:  all-MiniLM-L6-v2 (22M params)  — FROZEN, embeddings precomputed and cached
Trunk:    Linear(1168 → 512) → ReLU → LayerNorm → Linear(512 → 256) → ReLU
Actor:    pointer head as in §2.3                       (~0.4M params)
Critic:   Linear(256 → 64) → ReLU → Linear(64 → 1)      (~0.02M params)
```

**Total trainable parameters ≈ 1.1M.** Shared trunk, separate heads (standard actor-critic). Tier-B stretch goal swaps the trunk for a LoRA-adapted Qwen2.5-1.5B-Instruct.

---

## 3. The optimizer (the part the course cares about)

### 3.1 PPO objective

```
L(θ) = E_t [ L_clip(θ)  −  c_v · L_value(θ)  +  c_e · H[π_θ](s_t) ]

L_clip  = min( r_t(θ) Â_t ,  clip(r_t(θ), 1−ε, 1+ε) Â_t ),   r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
L_value = ( V_θ(s_t) − V_t^targ )²                            (clipped value loss)
Â_t     = Σ_{l=0}^{T−t−1} (γλ)^l δ_{t+l},                     δ_t = r_t + γV(s_{t+1}) − V(s_t)
```

Hyperparameters (starting point, from PPO paper + Search-R1 practice):

| | |
|---|---|
| clip ε | 0.2 |
| epochs per batch | 4 |
| rollout batch | 256 episodes |
| minibatch | 64 transitions |
| lr | 3e-4 (actor+critic, Adam), cosine decay |
| `c_v` / `c_e` | 0.5 / 0.01 (entropy annealed 0.01 → 0.001) |
| advantage norm | per-minibatch |
| grad clip | 0.5 |
| KL penalty | optional, adaptive, target 0.01 (for Tier-B LLM policy) |

### 3.2 Training curriculum

| Stage | What | Why |
|---|---|---|
| **0. Behaviour cloning** | supervised cross-entropy on **oracle paths** derived from MMQA gold SQL (the join path *is* the gold entity sequence) and HybridQA answer-cell annotations | RL from scratch on a sparse terminal reward in a 20-way action space wastes most of the budget on random walks. BC gives a warm start in minutes. Also our imitation-learning baseline. |
| **1. PPO, shaped reward** | full reward with `λ_ev = 0.3` | escape the cold start |
| **2. PPO, outcome-only** | anneal `λ_ev → 0` | stop the policy from gaming evidence recall; optimise what we actually measure |

Reader-LLM outputs are **cached by (question, frozenset(evidence))**. Since the policy revisits similar evidence sets constantly, cache hit rates should be high after the first epoch — this is what makes the reward function cheap enough to call hundreds of thousands of times.

---

## 4. Baselines and ablations

Every row is a run we must produce. This is the experimental core of the report.

### Baselines (all share the same reader LLM, so only selection differs)
| # | Baseline | Point it makes |
|---|---|---|
| B1 | Random valid entity | lower bound / is the task nontrivial |
| B2 | Greedy embedding similarity (TableRAG-style) | the standard retrieval heuristic |
| B3 | BM25 lexical greedy | non-neural heuristic |
| B4 | LLM zero-shot selection (prompt the reader to pick) | the Chain-of-Table/Dater style selector |
| B5 | Behaviour cloning on oracle paths | does RL beat plain imitation? **the key comparison** |
| B6 | Oracle path (gold evidence handed to reader) | upper bound — tells us how much headroom selection has at all |
| **B7** | **PPO (ours)** | |
| B8 | GRPO (ours, critic-free) | PPO vs GRPO under identical rewards |
| B9 | REINFORCE + learned baseline | ablates the clipped surrogate itself |

### Ablations
- Reward components: drop `R_evidence`, drop `R_efficiency`, outcome-only.
- GAE λ ∈ {0, 0.5, 0.95, 1.0}; γ ∈ {0.9, 0.99, 1.0} — bias/variance, directly reportable.
- Clip ε ∈ {0.1, 0.2, 0.3}; epochs ∈ {1, 4, 10} — the PPO stability story.
- Horizon `H` ∈ {2, 3, 4, 6}; candidate width `K` ∈ {5, 10, 20, 40}.
- Critic on/off (→ REINFORCE), entropy bonus on/off.
- State features: with/without `φ_t` scalars; with/without evidence embedding.
- Reader model size (0.5B / 1.5B / 7B / API) — does a better policy compensate for a weaker reader? Nice result if yes.

---

## 5. Evaluation

| Metric | What it measures |
|---|---|
| **Answer EM / token-F1** | headline task performance |
| **Path / hop accuracy** | does the selected trajectory contain all gold evidence |
| **Evidence recall@T, precision@T** | selection quality, decoupled from the reader |
| **Avg. hops to STOP** | efficiency; is the agent learning to stop |
| **#reader calls & tokens per question** | practical cost — a real selling point if PPO is cheaper *and* better |
| **Sample efficiency curve** (reward vs env steps) | the RL story; compare PPO/GRPO/REINFORCE |
| **Explained variance of the critic** | is the value function actually learning |
| **KL(π_new ‖ π_old) per update** | PPO stability diagnostics |

Splits: MMQA train/dev/test as released; HybridQA dev as in-domain transfer; WikiTQ as single-hop regression check; OTT-QA subset as out-of-domain (optional).
Report **mean ± std over 3 seeds** for every learned configuration. Non-negotiable — single-seed RL numbers are not evidence.

---

## 6. Repository layout

```
oml/
├── papers/                         # 24 PDFs, grouped, + annotated bibliography
├── docs/
│   ├── IMPLEMENTATION_OVERVIEW.md  # this file
│   ├── COMPUTE_REQUIREMENTS.md
│   ├── PROJECT_PROGRESS.md
│   └── LITERATURE_SURVEY.md
├── src/
│   ├── data/
│   │   ├── loaders.py              # MMQA, HybridQA, WikiTQ → common schema
│   │   ├── graph_builder.py        # tables + PK/FK + hyperlinks → entity graph G
│   │   └── oracle_paths.py         # gold SQL / answer cells → oracle entity sequences
│   ├── env/
│   │   ├── table_env.py            # Gymnasium-style MDP: reset/step/candidates/mask
│   │   ├── candidates.py           # TableRAG-style top-K candidate generation
│   │   └── reward.py               # composite verifiable reward
│   ├── models/
│   │   ├── encoder.py              # frozen MiniLM + on-disk embedding cache
│   │   ├── policy.py               # trunk + pointer actor + critic
│   │   └── reader.py               # frozen answer LLM / SQL executor + response cache
│   ├── rl/
│   │   ├── ppo.py                  # clipped surrogate, GAE, update loop
│   │   ├── grpo.py                 # critic-free variant
│   │   ├── reinforce.py            # ablation
│   │   └── bc.py                   # behaviour cloning warm start
│   ├── baselines/                  # random, bm25, embedding-greedy, llm-select, oracle
│   └── eval/
│       ├── metrics.py
│       └── run_eval.py
├── configs/                        # YAML per experiment (one file per table row)
├── scripts/                        # download_data.sh, precompute_embeddings.py, train.py, sweep.py
├── results/                        # logs, checkpoints, csv, plots
└── requirements.txt
```

## 7. Software stack

`torch` · `transformers` · `sentence-transformers` · `faiss-cpu` · `datasets` · `pandas` · `duckdb` (SQL execution for MMQA gold queries) · `gymnasium` (env interface only) · `hydra`/`omegaconf` (configs) · `wandb` or tensorboard (logging) · `matplotlib`.

**We implement PPO ourselves (~300 lines)** rather than pulling in TRL/verl. For an optimization course this is the right call: the grader can read the surrogate objective, and we keep full control over the variable-size action space, which is awkward in off-the-shelf LLM-RL libraries. TRL is the fallback only for the Tier-B LoRA policy.

---

## 8. Eight-week plan

| Week | Deliverable | Exit criterion |
|---|---|---|
| **1** | Data + graph. MMQA and HybridQA loaded into a common schema; `graph_builder.py` producing `G`; stats on graph size, hop distance to gold | Gold evidence is reachable within `H=4` for ≥95% of MMQA train |
| **2** | Environment + reward. `table_env.py` passes a random-agent smoke test; embeddings precomputed and cached; reader wired with caching | Random agent runs 10k episodes end-to-end; B1 and B6 numbers in hand |
| **3** | Baselines B1–B4, B6. Oracle paths extracted | Baseline table populated; **headroom between B2 and B6 confirmed** — if that gap is small, pivot to a harder split before writing any RL code |
| **4** | Policy net + behaviour cloning (B5) | BC beats B2 on evidence recall |
| **5** | **PPO**. Trainer, GAE, logging, 3 seeds | Training reward curve rises above BC; critic explained-variance > 0 |
| **6** | Ablations: reward terms, λ/γ, clip, horizon, K | ~20 runs logged; the two or three plots for the report exist |
| **7** | GRPO + REINFORCE comparison; transfer eval (HybridQA, WikiTQ) | Full results table complete with mean ± std |
| **8** | Report, slides, seminar talk, code cleanup | Reproducible: one command per table row |

**Built-in fallback.** If PPO on the full graph proves unstable, the guaranteed-deliverable version is the *single-table sub-table selection* setting (the Table-R1 formulation): smaller action space, denser reward, same optimizer and same experimental structure. We keep the plots and the analysis either way, so the project cannot fail outright — only scope down.

## 9. What we claim as contribution

1. A clean MDP formulation of cross-table entity selection with a **verifiable** composite reward (no reward model).
2. A **schema-agnostic pointer policy** that transfers across tables, trained with PPO + GAE.
3. A controlled empirical comparison — heuristic vs. LLM-prompted vs. imitation vs. PPO vs. GRPO vs. REINFORCE — under an identical reader and identical evidence budget, which as far as our survey shows does not yet exist for multi-hop *table* reasoning.
4. An analysis of *where* RL helps: hop count, table count, and reader strength.

This is deliberately an empirical/optimization contribution rather than a claim of SOTA on a leaderboard, which is the right size for a one-semester course project.
