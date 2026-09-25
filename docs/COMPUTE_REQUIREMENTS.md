# 1. Compute Requirements

**Project:** Learning Multi-Hop Table Reasoning through RL — PPO-Based Entity Selection
**Date:** 2026-09-25

The design was chosen *so that* compute stays inside a course budget. Three decisions do the work:

1. **The policy is tiny (~1.1M trainable params).** The sentence encoder is frozen and its embeddings are precomputed once, so no backprop ever touches a transformer in the main (Tier-A) configuration.
2. **The reward is verifiable, not learned.** Answer EM against gold + evidence recall against gold SQL. No reward model → no second network to train, no preference data.
3. **The reader LLM is frozen and cached** by `(question, frozenset(evidence))`. It is part of the environment, not part of the optimization.

---

## 1.1 Configuration tiers

| | **Tier A — core (planned)** | **Tier B — stretch (optional)** |
|---|---|---|
| Policy | frozen MiniLM embeddings + 1.1M-param MLP/pointer net | LoRA on Qwen2.5-1.5B-Instruct (~18M trainable) |
| Optimiser | PPO, GRPO, REINFORCE, BC | PPO/GRPO with token-level rollouts |
| GPU needed | **1 × 8–16 GB** (T4, RTX 3060/4060, A4000) | 1 × 24–48 GB (RTX 4090, A100-40) |
| Runs feasible | all ~35 planned runs | 2–4 runs only |
| Status | **committed** | only if Weeks 1–6 land on schedule |

Tier A is the project. Tier B is an upgrade we attempt only with spare time, and the report stands without it.

---

## 1.2 Hardware

| Resource | Requirement | Notes |
|---|---|---|
| GPU | 1 × 8–16 GB CUDA | Free tier is viable: Colab T4 / Kaggle P100 (30 h/week). No multi-GPU, no distributed training anywhere in the plan. |
| CPU | 8 cores | Environment stepping (graph expansion, FAISS lookup, string matching) is CPU-bound; we run 8 parallel envs to keep the GPU fed. |
| RAM | 32 GB recommended, 16 GB workable | Entity graphs for a whole MMQA split held in memory; drop to on-disk shards at 16 GB. |
| Disk | **~60 GB** | breakdown below |
| Reader LLM | 4-bit Qwen2.5-7B-Instruct locally (≈6 GB VRAM), **or** an API | API is the fallback if local generation is too slow; cost estimate below. |

### Disk breakdown
| Item | Size |
|---|---|
| MMQA (3,312 tables, 5,000 QA) | ~1 GB |
| HybridQA + linked Wikipedia passages | ~8 GB |
| WikiTQ, TabFact, MultiTabQA | ~2 GB |
| Precomputed cell/entity embeddings (384-d fp16) | ~6 GB |
| FAISS indices | ~3 GB |
| Reader response cache (JSON/SQLite) | ~5 GB, grows with training |
| Checkpoints (~35 runs × 3 seeds, tiny nets) | ~2 GB |
| Local reader model weights (4-bit 7B) | ~5 GB |
| Headroom / logs / plots | ~25 GB |
| **Total** | **≈60 GB** |

---

## 1.3 Time budget (estimates)

These are engineering estimates from the parameter counts and per-step costs, not measured numbers — Week 2 produces the first real timings and this table gets revised then.

**One-off preprocessing**
| Job | Cost |
|---|---|
| Embed all cells/schema/passages (MiniLM, batch 512) | ~1.5 GPU-hours |
| Build entity graphs + FAISS indices | ~2 CPU-hours |
| Extract oracle paths from gold SQL | ~20 CPU-minutes |

**Per training run (Tier A)**
| Item | Estimate |
|---|---|
| Env steps per run | ~200k (≈60k episodes at 3.3 hops avg) |
| Policy forward/backward | negligible (1.1M params) — **not the bottleneck** |
| Reader calls, epoch 1 | ~60k, mostly cache misses |
| Reader calls, later epochs | expected 70–90% cache hit rate |
| **Wall-clock, one seed** | **~2–4 hours on a T4** |
| Behaviour cloning warm start | ~10 minutes |

**Whole project (Tier A)**
| Bucket | Runs | GPU-hours |
|---|---|---|
| Preprocessing | — | 2 |
| Baselines B1–B4, B6 (no training) | 5 | 4 |
| BC (B5), 3 seeds | 3 | 1 |
| PPO main, 3 seeds | 3 | 9 |
| Ablations (reward, λ/γ, clip, epochs, H, K, features) | ~20 | 50 |
| GRPO + REINFORCE, 3 seeds each | 6 | 18 |
| Transfer eval (HybridQA, WikiTQ, OTT-QA subset) | — | 6 |
| Re-runs, debugging, failed configs (×1.4 contingency) | — | 36 |
| **Total** | ~37 | **≈125 GPU-hours** |

**125 GPU-hours over 8 weeks ≈ 16 h/week** — inside Kaggle's free 30 h/week quota, or a few weeks of Colab Pro, or a couple of days on one lab machine.

**Tier B, if attempted:** ~8–12 h per LoRA run on a 4090/A100 → ~40 GPU-hours for 3–4 runs.

---

## 1.4 Reader-LLM cost, if we use an API instead of local

Worst case, assuming **zero** caching:

- 37 runs × 60k reader calls = 2.2M calls — **not affordable, so this path is only for final evaluation, never for training.**

Realistic plan: **local 4-bit model for all training**, API only for final test-set evaluation and for the "stronger reader" ablation:

| Use | Calls | ~Tokens | Est. cost (small/cheap model, ~$0.3/M in + $1.5/M out) |
|---|---|---|---|
| Final eval, 3 datasets × 9 methods × ~1k questions | ~27k | ~30M in / 1M out | **~$10–15** |
| Reader-strength ablation | ~10k | ~12M in / 0.4M out | **~$5** |
| Buffer | | | ~$10 |
| **Total** | | | **≈$25–30** |

Cost figures are order-of-magnitude — they depend on which model we pick and are worth re-checking at Week 7 before spending anything.

---

## 1.5 What we are explicitly *not* doing (and why the budget holds)

| Avoided | Why | Saved |
|---|---|---|
| Fine-tuning a 7B+ policy with PPO | needs 4× A100 for actor+critic+ref+old-policy copies | ~1000s of GPU-hours |
| Training a reward model | reward is verifiable by string/SQL match | a whole second training pipeline |
| Full OTT-QA open retrieval (400K tables, 5M passages) | index build alone is days | ~100 GPU-hours |
| Multi-GPU / distributed RL | single-GPU is sufficient at this policy size | setup complexity, debugging time |
| Re-embedding per training step | encoder is frozen, embeddings cached on disk | ~80% of nominal training time |

## 1.6 Risks to the compute estimate

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reader cache hit rate is much lower than hoped → reward becomes the bottleneck | medium | shrink candidate width `K`, batch reader calls across parallel envs, switch reader to a 0.5B model for training and only use 7B at eval |
| PPO needs far more than 200k env steps to converge | medium | BC warm start already mitigates; also reduce `H` and `K`, which shrinks the effective search space fast |
| HybridQA passage corpus inflates disk past 60 GB | low | keep only passages linked from tables we actually use |
| Free-tier GPU quota interruptions | medium | checkpoint every 5k steps; all runs resumable; runs are short (2–4 h) by design |
