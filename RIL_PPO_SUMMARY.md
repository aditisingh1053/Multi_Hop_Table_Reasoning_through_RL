# Reader-in-the-Loop PPO (RiL-PPO): what we built and what we found

CS769 project extension: RL entity selection for multi-hop table and passage QA on HybridQA.
Full run finished on 2026-11-22 on the local RTX 3050 (WSL). Raw numbers are in
[results/loop/REPORT.md](results/loop/REPORT.md); per-run logs and configs are in [results/loop/runs/](results/loop/runs/).

---

## 1. Starting point

The original project trains a PPO pointer policy (actor and critic) to pick up to H=3 entities (table rows or
linked passages) per question. A frozen extractive reader (MiniLM-SQuAD2) then reads each selected
entity, and the answer with the highest confidence wins.

The previous best was **PPO from scratch: 27.6 F1**. We diagnosed two problems with it:

1. **The policy is blind to the reader.** It chooses hop 2 without knowing what hop 1 actually produced, so
   it cannot follow a bridge entity (true multi-hop reasoning).
2. **Max-confidence aggregation loses answers.** The answer is in the selected evidence 46% of the time, but
   picking the best span among those entities would give 38.7 F1, while max-confidence gives only 27.6.
   About 11 F1 points are lost purely at the aggregation step.

## 2. What we built

### 2.1 RiL-PPO (main contribution)

| Component | What it does | Where |
|---|---|---|
| **Bridge feedback** | After each hop the reader's span from that entity is added to the observation. Eleven new candidate features show whether the bridge span appears in a candidate's text, title or row, plus the reader's confidence and whether its span overlaps the question. Nine new state scalars are also added. The policy can now "follow" the answer of hop 1. | `src/mhtrl/env.py` (`reader_loop`) |
| **Learned COMMIT action** | After STOP, the policy takes one more action that chooses *which selected entity's answer is final*. The logit carries a residual prior, `prior · reader_score` with `prior` initialised to 5. An untrained head therefore reproduces max-confidence, and RL only has to learn the corrections. | `env.py` (`commit`), `policy.py` |
| **Consistent reward shaping** | The potential-based shaping was generalised so that it still telescopes with the extra commit step, which keeps the optimal policy unchanged. | `env.episode_rewards` |
| **Diagnostics** | Every evaluation also reports F1 for max-confidence and for the best answer in the selected set. Together these split the error into selection, reading and aggregation. | `rollout.evaluate_sets` |

### 2.2 RiL-PPO++ extensions (from the literature, each one can be switched on or off)

| Extension | Idea | Source |
|---|---|---|
| Rao-Blackwellised COMMIT | The COMMIT choice is cheap to evaluate exactly, so it uses the exact expected gradient over all commit choices instead of a single sample. | variance reduction |
| Privileged (asymmetric) critic | The critic sees training-only information, such as gold-path hits and answer presence. | Pinto et al. 2018 |
| Self-imitation learning | Replays past high-return episodes. | Oh et al. 2018 |
| Action dropout | Randomly hides candidates from the sampler (never STOP or COMMIT) to force exploration. | Lin et al. 2018 (multi-hop KG RL) |
| EMA weights | Keeps a Polyak-averaged copy of the policy and uses it for evaluation if it scores better. | |
| Critic-free optimisers | RLOO, GRPO and Dr. GRPO. | Ahmadian 2024; Shao 2024; Liu 2025 |
| Test-time self-consistency | Votes over 8 sampled evidence paths. | Wang et al. 2023 |

### 2.3 Baselines added

- `bridge_hop`: a hand-written reader-in-the-loop rule (read the best row, then hop to the passage containing its span).
- `read_all`: read every entity, about 49 reads per question.

### 2.4 Engineering

- `scripts/run_loop_suite.py` is a resumable driver. Its stages are baselines, then core, then ++, then optimisers, then 100-iteration runs, then self-consistency, then RoBERTa transfer.
- `scripts/loop_report.py` writes the report, with paired bootstrap tests, a gain decomposition, results on unseen tables and training curves.
- `scripts/run_loop_gpu.sh` and `scripts/make_gpu_bundle.py` support Kaggle, Colab and local runs. The Kaggle problems are fixed: the recursive dataset glob, CRLF line endings, and CUDA device placement in the tests.
- There are 17 unit tests (`tests/test_core.py`), and all pass on CUDA.
- The local run on the RTX 3050 4 GB under WSL took about 4 hours end to end.

---

## 3. Results (HybridQA dev, 3,466 questions, 3 seeds, mean ± std)

### 3.1 Headline

| Selector | EM | F1 | Answer in evidence |
|---|---|---|---|
| Oracle gold path (upper bound) | 48.4 | 55.6 | 96.9 |
| Read every entity (49 reads) | 17.0 | 21.5 | 97.1 |
| BM25 top-3 | 15.9 | 20.8 | 37.7 |
| Bridge hop (hand rule) | 14.3 | 19.0 | 37.6 |
| PPO from scratch, 40 iterations (previous best) | 22.2 ± 1.0 | 27.6 ± 1.0 | 46.1 |
| PPO from scratch, 100 iterations | 22.8 ± 1.0 | 28.0 ± 1.2 | 45.3 |
| **RiL-PPO, 40 iterations** | 24.0 ± 1.5 | 29.9 ± 1.6 | 51.5 |
| **RiL-PPO, 100 iterations (best)** | **31.0 ± 1.4** | **37.5 ± 1.8** | **57.4** |

**At 100 iterations, RiL-PPO gains +9.5 F1 over PPO given the same budget**: 95% CI [8.6, 10.6], p < 0.001, and every
RiL seed beats every PPO seed. The gain is **+16.7 F1 over BM25**, and the system reaches about 67% of the oracle F1.
Plain PPO plateaus (27.6 → 28.0 from 40 to 100 iterations), while RiL-PPO keeps improving (35.3 → 42.8 validation F1).
Having the reader in the loop is what makes extra training pay off.

### 3.2 Ablation: both parts matter (40 iterations)

| Variant | F1 | Δ vs PPO |
|---|---|---|
| PPO from scratch | 27.6 | – |
| + bridge feedback only | 28.7 | +1.1 (p = 0.002) |
| + learned COMMIT only | 29.0 | +1.4 (p < 0.001) |
| + both (RiL-PPO) | 29.9 | +2.3 (p < 0.001) |

The two parts add up roughly linearly. The full model beats each single-part ablation significantly (p ≤ 0.008).

### 3.3 Where the gain comes from

| | Answer in evidence | Best in set | Max-conf | Final F1 | COMMIT gain | Answers changed by COMMIT |
|---|---|---|---|---|---|---|
| PPO, 100 iterations | 45.3 | 37.9 | 28.0 | 28.0 | 0 | 0% |
| RiL-PPO, 100 iterations | 57.4 | 47.4 | 29.5 | 37.5 | **+8.0** | 33.5% |

Both mechanisms contribute:

- **Better selection**: the answer is in the evidence +12 points more often, thanks to bridge feedback.
- **Learned arbitration**: COMMIT recovers 8 F1 of the aggregation gap and overrides max-confidence on a third of the questions.

### 3.4 The RiL-PPO++ extensions (40 iterations, compared with RiL-PPO at 29.9)

| Extension | F1 | Verdict |
|---|---|---|
| Action dropout | **31.1 ± 2.6** | Helps: +1.2, p < 0.001 |
| Rao-Blackwellised COMMIT | 30.1 ± 1.2 | Neutral on F1 (+0.2, not significant); lower variance, more answers in evidence (54.1) |
| EMA weights | 29.9 | No effect: the EMA copy was never chosen (0/3 seeds), so this is identical to RiL-PPO |
| Privileged critic | 26.6 | Hurts (−3.3). The critic fits better (explained variance 0.61 vs 0.48), but the policy gets worse. |
| Self-imitation | 26.5 | Hurts (−3.4): it replays early lucky episodes and causes premature commitment |
| **All combined (++)** | 28.5 | Worse than RiL-PPO; at 100 iterations it is 27.0 (−10.6 vs RiL-PPO 100 iterations) |

**Honest negative result:** stacking the extensions hurts, because the two harmful ones (SIL and the privileged critic)
dominate. The ++ 100-iteration run peaks at iteration 68 and stalls. The useful finding is that **action dropout** is
the one extension worth keeping. The recommended next configuration is RiL-PPO + dropout (+ RB COMMIT) at 100 iterations,
which has not been run yet.

### 3.5 Optimisers (RiL environment, 40 iterations)

| Optimiser | F1 |
|---|---|
| RLOO (critic-free) | **30.2 ± 1.6** (matches PPO+GAE at 29.9; difference not significant) |
| Dr. GRPO | 28.4 |
| GRPO | 17.7: collapses to selecting 1 entity; 46% of groups have zero advantage |

GRPO's per-group std normalisation breaks under sparse rewards. Dr. GRPO, which drops that normalisation, fixes the collapse
(+10.7 F1), and so does RLOO. RLOO matches PPO with no critic.

### 3.6 Robustness

- **Unseen tables** (questions on tables never seen in training): RiL-PPO at 100 iterations scores 37.4 vs PPO at 27.8. The gain carries over unchanged.
- **Swap in a different reader** (RoBERTa-base, first 1,000 questions, with no retraining): RiL-PPO scores 32.0 vs PPO at 28.5. The policy transfers, and it benefits from the stronger reader inside the loop.
- **Self-consistency (k=8)**: the change is within ±0.2 F1, so it is not worth roughly 2× the reader cost. Greedy decoding is already near the policy's mode.

---

## 4. Takeaways for the report and slides

1. **The main claim:** putting the reader inside the RL loop, through bridge observations plus a learned COMMIT,
   raises F1 from 28.0 to 37.5 (+34% relative) at the same training budget, with p < 0.001 across all seeds. The gain also holds on unseen tables and with an unseen reader.
2. The mechanism is visible in the diagnostics: better evidence (+12 points answer recall) and learned arbitration (+8 F1).
3. Plain PPO saturates by about 40 iterations; RiL-PPO keeps learning.
4. Literature tricks are not free: action dropout helps, while SIL and the privileged critic hurt. RLOO is a valid critic-free alternative, and vanilla GRPO collapses under sparse multi-hop rewards.

## 5. Caveats

- There are 3 seeds per method; standard deviations are around 1–2.6 F1.
- "Test" here is the HybridQA dev set. Models were selected by validation F1 on a held-out 1,000-question split of train.
- The ++ variants were run with fixed hyper-parameters (for example SIL coefficient 1.0); retuning could change their verdicts.

## 6. Suggested next run (about 30 minutes locally)

Run RiL-PPO + action dropout + RB COMMIT for 100 iterations with 3 seeds. Based on the 40-iteration results it is the most likely to beat 37.5.
