# 5-minute evaluation script

About 750 spoken words, plus a 1-minute live demo.

**Show these slides:** 4 (approach) → 6 (our method) → 7 (results) → 8 (ablations) → live demo.

**Before you start:**
1. Open Ubuntu.
2. Run `cd /mnt/d/7th_sem/oml && source ~/mhtrl_venv/bin/activate`.
3. Type `python scripts/demo.py --idx 2 --fresh` and leave it ready to run. Loading takes about 45 s, so press Enter while you are on slide 7.

---

## 0:00 – 0:30 · Opening (Slide 1 or 4)

Our project is multi-hop question answering over Wikipedia tables and their linked passages, on the HybridQA dataset. To answer a question, the system must decide **which table rows and passages to read, and when to stop**. We train that decision with reinforcement learning, and a frozen reader model then extracts the answer from what was selected.

## 0:30 – 1:45 · Code and method (Slide 4)

The code has four main parts, all in `src/mhtrl/`.

1. **Data and environment**, in `data.py` and `env.py`.
   - For each question we build a graph with one node per table row and one per linked passage, about 50 entities.
   - All text is embedded once with a frozen MiniLM encoder.
   - The environment is an MDP. The state is the question plus what we have read so far. The actions are "pick an entity" or "STOP", with at most three picks.
2. **Reward.** When the episode ends, a frozen question-answering reader, MiniLM trained on SQuAD 2, reads the selected evidence. The reward is the F1 of its answer, minus a small cost per entity read.
3. **Policy**, in `policy.py`. It is a pointer network: every candidate is scored by the same small network, so the number of choices can change per question. A separate critic estimates the expected reward.
4. **Training**, in `rl/algos.py`. We use PPO with GAE advantages and a clipped update, which keeps training stable. We compare it with behaviour cloning, REINFORCE, GRPO and RLOO, all under the same reward.

The code has 17 unit tests, and the whole experiment suite runs in about 4 hours on my laptop GPU.

## 1:45 – 2:45 · Our extension: Reader-in-the-Loop PPO (Slide 6)

Plain PPO stopped improving at about 28 F1, and we found two reasons.
- First, it chooses the second hop blind: it never sees what the reader found in the first hop.
- Second, the final answer is just the reader's most confident one, and that loses about 11 F1.

**We designed two fixes. This is our own contribution.**
1. **Bridge feedback.** After every hop we run the reader and add its answer to the state, as features like "does this answer appear in this candidate?". This lets the policy follow a name from a row to its passage.
2. **A learned COMMIT action.** After STOP, the policy chooses which selected entity gives the final answer. It starts from the reader's confidence, so it can never be worse than the baseline at the start.

We also tested ideas from other papers: action dropout from Lin et al., self-imitation, a privileged critic, RLOO and Dr. GRPO.

## 2:45 – 3:45 · Results (Slides 7 and 8)

We evaluate on all 3,466 HybridQA dev questions, with 3 seeds, the same reader for every method, and EM and F1 as the metrics.

- The best fixed heuristic, BM25, gets **20.8 F1**.
- Plain PPO gets **28.0**.
- **Our Reader-in-the-Loop PPO gets 37.5 F1 and 31.0 EM**, which is **+9.5 F1 over PPO with the same training budget**.
  - It is significant: p < 0.001, and every one of our runs beats every PPO run.
  - It also holds on tables never seen in training (37.4 vs 27.8) and with a different reader, RoBERTa (32.0 vs 28.5).

On slide 8, the ablation shows both fixes matter. Of the extensions from papers, only action dropout helped. Self-imitation and the privileged critic hurt, and plain GRPO collapsed. Those are honest negative results.

## 3:45 – 5:00 · Live demo (terminal)

Now the inference.

*[Press Enter on `python scripts/demo.py --idx 2 --fresh`.]*

This loads both trained models and the dev set. The `--fresh` flag means the reader runs live on the GPU, with nothing precomputed.

The question is: *"The driver who finished in position 4 in the 2004 US Grand Prix was of what nationality?"*

**Our model:**
- Hop 1 picks the table row with position 4, Jenson Button.
- Hop 2 follows that name to his Wikipedia passage. This is the bridge feedback at work.
- The reader extracts "British" from that passage.
- COMMIT chooses that answer: **correct, F1 1.0**. Max-confidence alone would have answered "Jenson Button", which is wrong.

**Plain PPO** jumps straight to Barrichello's passage and answers "Brazilian", which is wrong.

So the demo shows both of our ideas on one question: following the bridge, and choosing the right final answer.

*[If time allows: `python scripts/demo.py` shows a random question. It may fail, since our F1 is 37.5. Say so honestly.]*

Thank you.

---

## If they ask

| Question | Answer |
|---|---|
| Where is inference code? | `scripts/demo.py` for one question, `scripts/evaluate.py` for the full dev set. |
| Run the full evaluation? | `python scripts/evaluate.py --runs results/loop/runs/loop_full_long_s0 --split test` (about 1 min). |
| Which checkpoint? | `results/loop/runs/loop_full_long_s0/actor_best.pt`, chosen by validation F1. |
| Is anything trained at test time? | No. Greedy actions from the trained policy; the reader is frozen. |
| Why is loading slow? | About 45 s to load the prepared dev set and embeddings; each question then takes 1–2 s. |
