# Multi-Hop Table Reasoning through Reinforcement Learning

CS769 (Optimization in Machine Learning) course project, IIT Bombay.
Aakash Gupta (23B0953) and Aditi Singh (23B1053). TA: Jahanvi Rajput.

## Overview

Some questions need a table **and** the text linked from it. For example: *"The driver who
finished in position 4 in the 2004 United States Grand Prix was of what nationality?"* To answer it,
you find the row with Pos = 4 (Jenson Button), follow the link to his Wikipedia page, and read
"British". Each hop is a decision about **what to read next**.

We treat that decision as a reinforcement-learning problem on HybridQA:

- **Environment.** Each question becomes a graph of about 49 entities: table rows, and the passages
  linked from their cells. At every step the agent picks one entity or STOP, with at most 3 picks.
- **Reader.** A frozen extractive QA model (MiniLM fine-tuned on SQuAD 2.0) answers from the
  selected entities. The reward is the answer's F1, minus a small cost per entity read.
- **Policy.** A pointer network trained with PPO + GAE. Each candidate gets a score from the same
  network, so the number of candidates can change.
- **Reader-in-the-Loop PPO (RiL-PPO, our contribution):**
  - *Bridge feedback.* After each hop, the reader's answer goes back into the state, so the policy
    can follow it to the next entity (row → "Jenson Button" → his page).
  - *Learned COMMIT.* After STOP, the policy chooses which selected entity gives the final answer,
    instead of trusting the reader's most confident span. It starts out identical to that rule
    and learns when to overrule it.

![architecture](slides/architecture/architecture.png)

## Results

All selectors use the same frozen reader. The test set is the HybridQA dev set (3,466 questions),
and learned methods are averaged over 3 seeds.

| Selector | EM | F1 |
|---|---|---|
| Greedy rank-then-hop (HYBRIDER-style) | 12.9 | 17.4 |
| BM25 top-3 (best fixed rule) | 15.9 | 20.8 |
| PPO, 100 iterations | 22.8 | 28.0 |
| **RiL-PPO, 100 iterations (ours)** | **31.0** | **37.5** |
| Oracle gold evidence (upper bound) | 48.4 | 55.6 |

- RiL-PPO beats PPO by +9.5 F1 with the same training budget (95% CI [8.6, 10.6], p < 0.001).
  Every RiL-PPO seed beats every PPO seed.
- Both parts help. Bridge feedback puts the answer in the evidence more often (57% vs 45%), and
  COMMIT adds 8 F1 over the reader's max-confidence rule.
- The gain holds on tables never seen in training (37.4 vs 27.8 F1). It also holds with a different
  reader, RoBERTa, used without retraining (32.0 vs 28.5).
- Ideas from the literature: action dropout (+1.2) and RLOO (≈ PPO, no critic) help, while
  self-imitation and a privileged critic hurt. Vanilla GRPO collapses on these sparse rewards (17.7);
  Dr. GRPO fixes it (28.4).

## Repository layout

```
src/mhtrl/             library
  data.py              HybridQA → per-question entity graphs
  env.py               the MDP: candidates, features, bridge feedback, COMMIT, rewards
  policy.py            pointer-network actor and critic
  reader.py            frozen QA reader with an on-disk cache
  rollout.py           batched episodes and evaluation
  baselines.py         fixed selectors (BM25, greedy hop, bridge hop, oracle, ...)
  rl/algos.py          PPO, GRPO / Dr. GRPO, RLOO, REINFORCE and the extensions
  rl/bc.py             behaviour cloning
configs/baseline/      first experiment suite (PPO, BC, REINFORCE, GRPO, ablations)
configs/ril_ppo/       Reader-in-the-Loop PPO suite
scripts/data/          download, extract tables, build graphs and embeddings
scripts/train/         train one config, or run a whole suite
scripts/eval/          evaluate checkpoints; demo.py runs inference on one question
scripts/analysis/      result tables, figures and reports
scripts/gpu/           one-command runs for Kaggle, Colab or any Linux GPU machine
tests/test_core.py     17 unit tests
results/               metrics and logs of every run (checkpoints are not committed)
report/                LaTeX progress report (report/main.pdf)
slides/                final/ and progress/ presentations, architecture/ diagram
```

## Running

### 1. Setup

Use Linux or WSL with Python 3.10–3.12. A CUDA GPU is recommended; a 4 GB laptop GPU is enough.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # on a GPU machine, install a CUDA build of torch first
python tests/test_core.py           # should print 17 "ok" lines
```

### 2. Data (once, about 15–20 minutes)

```bash
git clone --depth 1 https://github.com/wenhuchen/HybridQA data/raw/HybridQA
git clone --depth 1 --no-checkout https://github.com/wenhuchen/WikiTables-WithLinks data/raw/WikiTables-WithLinks
python scripts/data/extract_wikitables.py
python scripts/data/build_instances.py --n_train 10000 --n_val 1000
```

This writes `data/processed/{train,val,test}.pkl`: the entity graphs plus frozen MiniLM embeddings.

### 3. Training

One configuration and one seed:

```bash
python scripts/train/train.py --config configs/ril_ppo/loop_full_long.yaml --seed 0
```

The output goes to `results/runs/<name>_s<seed>/`: `actor_best.pt`, `history.json` and `train.log`.

The whole RiL-PPO suite (48 runs; data preparation included; resumable):

```bash
SMOKE=1 bash scripts/gpu/run_loop_gpu.sh      # 5-minute end-to-end check
bash scripts/gpu/run_loop_gpu.sh              # about 4 h on an RTX 3050 laptop GPU or a Kaggle T4
SEEDS=2 NO_LONG=1 bash scripts/gpu/run_loop_gpu.sh   # shorter version, about 1.5–2 h
```

If it stops, run the same command again; finished runs are skipped. The first (baseline) suite runs
with `bash scripts/gpu/run_all_gpu.sh`.

### 4. Evaluation and inference

```bash
# EM / F1 on all 3,466 test questions
python scripts/eval/evaluate.py --runs results/loop/runs/loop_full_long_s0 --split test

# inference on one question: every hop, the reader's answers, the COMMIT choice, the final answer
python scripts/eval/demo.py --idx 2 --fresh          # --fresh runs the reader live (no cache)
python scripts/eval/demo.py --find "nationality"     # first question containing a phrase
python scripts/eval/demo.py                          # a random question
```

The demo loads in about 20 s, then answers a question in 0.2–3 s.

### 5. Reports and figures

```bash
python scripts/analysis/loop_report.py    # results/loop/REPORT.md: tables, significance tests, curves
python slides/final/make_figures.py       # figures for the final presentation
```

### Kaggle / Colab

```bash
python scripts/gpu/make_gpu_bundle.py     # writes gpu_bundle.zip (code only, about 100 KB)
```

Upload the zip as a Kaggle dataset, and turn on a GPU and Internet in the notebook. Then run:

```bash
unzip gpu_bundle.zip && cd mhtrl
pip install -q pyyaml tqdm
SMOKE=1 bash scripts/gpu/run_loop_gpu.sh && bash scripts/gpu/run_loop_gpu.sh
```

Download `loop_results_bundle.zip` when it finishes and unzip it in the repository root.

### Rebuilding the slides

The decks are HTML, printed to PDF with Chrome and converted to PPTX (one image per slide, with
speaker notes):

```bash
"C:/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --no-pdf-header-footer \
    --virtual-time-budget=20000 --print-to-pdf=slides/final/final_presentation.pdf \
    file:///<path-to-repo>/slides/final/deck.html
python slides/final/build_pptx.py              # needs pymupdf and python-pptx
python slides/architecture/make_architecture.py   # architecture diagram (PNG step in its header)
```
