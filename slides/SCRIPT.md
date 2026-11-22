# Presentation script (about 15 minutes)

Read it as written. Each slide has a rough time. **[OURS]** marks our own contribution and **[FROM PAPER]** marks borrowed ideas. These labels are reminders for you; the spoken text already says it.

Total: about 2,100 words, roughly 15 minutes at a calm pace.

---

## Slide 1 — Title (0:30)

Good morning everyone. We are Aakash Gupta and Aditi Singh, and our project is on multi-hop question answering over tables using reinforcement learning.

In one line: when a question needs evidence from a table *and* from linked text, the system has to decide **what to read next, and when to stop**. We treat that decision as a reinforcement learning problem and train it with PPO. Later in the talk I'll present our own extension, which we call **Reader-in-the-Loop PPO**. It is the main new contribution of this project.

---

## Slide 2 — Problem and motivation (1:30)

Let me start with an example. The question is: "What is the middle name of the player with the second most career rushing yards?"

To answer it you have to do three things:
- first, find the row in the table where the rank is 2;
- second, follow the link from that player's name to his Wikipedia passage;
- third, read the middle name from that passage.

Each step is a decision about *what to read next*. If the first step is wrong, even a perfect reader cannot recover.

Today, every existing system makes this decision with a fixed rule:
- HYBRIDER uses a greedy classifier;
- TableRAG uses embedding similarity;
- Chain-of-Table uses an LLM prompt;
- ODYSSEY uses a similarity threshold plus graph search, and in its own error analysis, 14 out of 41 errors come from this selection step.

None of these systems learns the choice from whether the final answer was right, and they all stop after a fixed number of steps.

So our goal is to **learn both what to read and when to stop, directly from answer quality**, using reinforcement learning.

---

## Slide 3 — Related work (1:00)

I'll keep this short. Three papers are closest to us.

1. **José et al., BRACIS 2024**: they used PPO and DQN to choose between "retrieve text", "retrieve table" or "answer". Their best agent did no better than a fixed rule. They themselves said that whole tables are too coarse an action, and suggested working at the row level. That is exactly what we do.
2. **Learning to Reduce** and **Table-R1** use RL to pick rows and columns, but only once, inside a single table. There is no multi-hop.
3. **s3** is the cleanest design: a PPO-trained searcher picks documents while a frozen generator answers. But it works only on plain text.

Our project combines these ideas: **row- and passage-level actions, a learned stop, a frozen reader, and PPO**, on tables with linked text. That combination is not in any of these papers.

---

## Slide 4 — Dataset and approach (2:30)

**Data.** We use HybridQA: Wikipedia tables whose cells link to Wikipedia passages. For each question we build a small graph: one node per table row and one node per linked passage. That's about 49 entities per question. We train on 5,000 questions, pick the best checkpoint on 1,000 questions from *different* tables, and test on the full official dev set of 3,466 questions.

Now the reinforcement learning setup, which is a Markov decision process.

- **State.** The question, the entities chosen so far, and the last one chosen. All text is embedded once with a frozen MiniLM sentence encoder, plus 22 simple hand-made features such as word overlap.
- **Actions.** Pick any row, one of the top passages, a passage linked from a row we already picked, or **STOP**. At most three picks.
- **Reward.** When the episode ends, a **frozen** question-answering reader reads the chosen evidence and gives an answer. The reward is the F1 score of that answer, minus a small cost of 0.02 per entity so the agent does not read everything.
- **Reward shaping.** We also give a small bonus for picking gold evidence. We add it as *potential-based shaping*, from **Ng et al., 1999**. The idea: the bonus is the *change* in a potential function between steps, so it all cancels out over a full episode. That's why it speeds up learning without changing which policy is best.

**The policy** is a **pointer network**. The number of candidates changes from question to question, so instead of a fixed output layer we score each candidate with the same small network and take a softmax over them. A separate **critic** network predicts the expected reward from the current state.

**The training algorithm is PPO**, from **Schulman et al., 2017**. In simple words:
- We play many episodes with the current policy.
- For each action we compute an **advantage**: how much better it turned out than the critic expected. We compute it with **GAE**, Generalised Advantage Estimation, which blends short-term and long-term estimates to reduce noise.
- Then we push up the probability of good actions and push down bad ones. PPO **clips** the update so the new policy cannot move too far from the old one in one step, which keeps training stable.

We compare against three other learners, all with the same reward:
- **Behaviour cloning**: plain supervised learning that copies the annotated gold evidence.
- **REINFORCE**: the simplest policy gradient, with no critic and no clipping.
- **GRPO**, from the DeepSeek-Math paper: no critic; instead it samples a group of answers per question and compares each one to the group average.

---

## Slide 5 — Progress so far (1:30)

These are our first results. Every method feeds the *same* frozen reader, so the only difference is what evidence gets selected.

- If you hand the reader the correct evidence, it gets 55.7 F1. That's our upper bound.
- The best fixed rule, BM25 top-3, gets 20.8.
- Our PPO policy trained from scratch gets 28.8, which is **8 points above the best heuristic**. That's statistically significant: the 95% confidence interval is 6.9 to 9.1.
- It also holds on tables never seen in training, and with a different reader.

Two negative results:
- Starting PPO from the behaviour-cloned policy got stuck exactly where cloning was.
- GRPO collapsed to picking just one entity.

But we were not satisfied. We looked closely at *why* PPO stops at around 28, and that led to our main contribution.

---

## Slide 6 — Our method: Reader-in-the-Loop PPO (2:30)

This slide is our own work. We found **two leaks** in plain PPO.

**Leak one: the policy chooses the second hop blind.** It picks hop 1, and the reader never tells it what hop 1 actually said. So if hop 1 is the player's row and the reader extracts "Walter Payton", the policy doesn't know that "Walter Payton" is the name it should follow next. Yet following that bridge entity *is* multi-hop reasoning.

**Leak two: the final answer is chosen badly.** The reader reads every selected entity and the most confident answer wins. We measured this: if we always picked the best answer among the selected entities, we'd get 38.7 F1. Max-confidence gives only 27.6. **About 11 points are lost just at the last step.**

So we made two fixes. **Both are our own design.**

**Fix one: bridge feedback.** After every hop, we run the reader on what was just picked and put its answer into the state. For every remaining candidate we compute 11 new features, such as:
- does the reader's answer appear in this candidate's text?
- is it this passage's title?
- does it appear in this row?
- how confident was the reader?
- does the answer just repeat words from the question?

Now the policy can literally *follow* the answer from one hop to the next.

**Fix two: a learned COMMIT action.** After the agent says STOP, it takes one more action: it chooses *which* of its selected entities gives the final answer. The trick is in how we set it up. The score for each choice is a learned part plus five times the reader's own confidence. So at the very start, before any training, COMMIT behaves exactly like max-confidence. It can never start worse than the old system, and RL only has to learn *when to override* the reader.

One more detail: adding this extra step would break the reward shaping, so we generalised the shaping formula so it still cancels out correctly. That keeps the guarantee from Ng et al. intact.

At the bottom of the slide are ideas we **took from other papers** and tested on top of our method. I'll explain each one on slide 8.

---

## Slide 7 — Results (2:00)

Here is the main result. Everything is the mean of 3 seeds on all 3,466 dev questions.

- Plain PPO trained for 100 iterations: **28.0** F1.
- Our Reader-in-the-Loop PPO, with the exact same training budget: **37.5** F1.

That's **+9.5 F1**. The 95% confidence interval is 8.6 to 10.6, p is below 0.001, and *every single* RiL-PPO run beats *every* PPO run. Exact match goes from 22.8 to 31.0, and we're now at about two-thirds of the oracle.

Look at the training dynamics. Plain PPO barely moves from 40 to 100 iterations: 27.6 to 28.0. It has saturated. Our method keeps learning, from 29.9 to 37.5, because the reader's feedback gives it new information to learn from.

We can also see *where* the gain comes from:
- The correct answer is inside the selected evidence 57% of the time instead of 45%. That's fix one, bridge feedback, finding better evidence.
- The learned COMMIT adds 8 F1 on top of max-confidence, and it changes the answer on about a third of the questions. That's fix two.

It also generalises:
- On tables never seen in training: 37.4 versus 27.8.
- When we swap in a different, bigger reader, RoBERTa, without retraining: 32.0 versus 28.5.

And note the hand-written "bridge hop" rule, which follows the reader's answer with a fixed rule, gets only 19. So the idea of hopping is not enough; the gain comes from *learning* it.

---

## Slide 8 — What helped and what did not (2:30)

**Left table: our ablation.** Bridge feedback alone gives +1.1, COMMIT alone +1.4, and both together +2.3. All are significant, so both parts of our method matter.

**Middle table: ideas from other papers**, each added on top of our method. Let me explain each one simply.

- **Action dropout**, from **Lin et al., 2018**, an RL paper on knowledge graphs. During training we randomly hide some candidates, so the agent is forced to explore other paths instead of always taking the same one. This was the **one clear win: 31.1**.
- **Rao-Blackwellised COMMIT.** Our COMMIT step has only two or three choices, so instead of sampling one and getting a noisy gradient, we compute the exact average over all choices. The idea of Rao-Blackwellisation is a classic variance-reduction trick; applying it to our COMMIT step is our own. It was neutral on F1, 30.1, but it found the answer more often.
- **EMA weights**, Polyak averaging: we keep a smoothed copy of the network. It was never better than the normal one, so there was no change.
- **Privileged critic**, from **Pinto et al., 2018**: the critic gets extra information during training, such as where the gold evidence is. The critic predicted better, but the policy got *worse*, 26.6.
- **Self-imitation learning**, from **Oh et al., 2018**: replay the agent's own best past episodes. It hurt, 26.5, because it keeps copying early lucky episodes and stops exploring.

Combining all of them gave 28.5, worse than our plain method. So the honest finding is that these tricks do not simply stack; only action dropout is worth keeping.

**Right table: different optimisers**, all inside our new environment.
- **RLOO**, "REINFORCE leave-one-out" from **Ahmadian et al., 2024**: sample several episodes per question, and for each one use the *average of the others* as the baseline. With no critic at all it matched PPO, 30.2.
- **GRPO** collapsed to 17.7. It divides by the standard deviation of the group's rewards, and when most rewards are zero, that division blows up the noise.
- **Dr. GRPO**, from **Liu et al., 2025**, removes that division, and it recovers to 28.4. So we confirmed the failure mechanism.

Finally, **self-consistency**, from **Wang et al., 2023**: sample 8 different evidence paths and take a majority vote. It changed F1 by at most 0.2 and costs twice the reader calls, so it's not worth it here.

---

## Slide 9 — Takeaways and next steps (1:00)

To summarise:
- Plain PPO entity selection stops at about 28 F1 because it selects blind and picks the final answer by max-confidence.
- **Our contribution, Reader-in-the-Loop PPO, with bridge feedback and a learned COMMIT action, reaches 37.5 F1, +9.5, and the gain holds on unseen tables and an unseen reader.**
- From the literature, action dropout and RLOO help; self-imitation and the privileged critic hurt; and vanilla GRPO fails on sparse rewards unless you remove its normalisation.

Next we want to:
- train our method with action dropout for longer, with five seeds;
- go to four hops and cell-level nodes;
- put an LLM reader inside the loop;
- and as a stretch goal, multi-table questions on MMQA.

Thank you. We are happy to take questions.

---

## Quick answers for likely questions

| Question | Short answer |
|---|---|
| What is new here? | Bridge feedback (reader output in the RL state) and the COMMIT action with a residual max-confidence prior. Also: the shaping generalised to the commit step, and the Rao-Blackwellised commit gradient. |
| Why is the reader frozen? | Cheaper, and it isolates selection: any gain comes only from choosing better evidence, and the policy transfers to a new reader. |
| Why PPO and not just supervised learning? | Gold evidence is noisy (found by string matching, unique for only ~56% of questions). RL learns from the answer itself, and cloning got only 19 F1. |
| Is +9.5 just more training? | No. PPO got the same 100 iterations and stayed at 28. |
| Why 3 seeds? | Compute. The standard deviation is 1–2 F1, much smaller than the 9.5 gap, and the bootstrap p < 0.001. |
| Why 28.8 on slide 5 but 27.6/28.0 later? | Slide 5 is the earlier experiment suite; the new slides use a re-run baseline under identical settings for a fair comparison. |
