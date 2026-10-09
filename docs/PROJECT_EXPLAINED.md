# Learning Multi-Hop Table Reasoning through Reinforcement Learning — the whole project, explained from scratch

This document explains the project from the beginning. It assumes no background in question answering or reinforcement learning, and builds up every idea before it is used. If you already know a topic, skip its section; the table of contents is ordered so that each part only depends on the parts before it.

**Contents**

1. [The project in one page](#1-the-project-in-one-page)
2. [Background I: question answering over tables and text](#2-background-i-question-answering-over-tables-and-text)
3. [Background II: reinforcement learning from zero](#3-background-ii-reinforcement-learning-from-zero)
4. [The problem statement, precisely](#4-the-problem-statement-precisely)
5. [The literature: what exists and what is missing](#5-the-literature-what-exists-and-what-is-missing)
6. [Our formulation: the selection problem as an MDP](#6-our-formulation-the-selection-problem-as-an-mdp)
7. [Our method: the policy and how it is trained](#7-our-method-the-policy-and-how-it-is-trained)
8. [Implementation: code, data pipeline, safeguards, compute](#8-implementation-code-data-pipeline-safeguards-compute)
9. [Experimental setup](#9-experimental-setup)
10. [Results](#10-results)
11. [What the results mean, and their limits](#11-what-the-results-mean-and-their-limits)
12. [Next steps and challenges](#12-next-steps-and-challenges)
13. [Where everything is, and how to reproduce it](#13-where-everything-is-and-how-to-reproduce-it)
14. [Glossary](#14-glossary)
15. [Key references](#15-key-references)

---

## 1. The project in one page

**The question.** Some questions about a table cannot be answered by looking at a single cell. Take this real question from the HybridQA dataset:

> *"What is the middle name of the player with the second most National Football League career rushing yards?"*

The table "List of National Football League career rushing yards leaders" has one row per player. To answer, you must

1. find the row whose **Rank** is 2 → *Walter Payton*;
2. follow the hyperlink in that row's *Player* cell to Walter Payton's Wikipedia passage;
3. read the passage: *"Walter **Jerry** Payton (July 25, 1954 …)"* → the answer is **Jerry**.

Each step is a decision about **which piece of evidence to look at next**. Questions like this are called **multi-hop** questions: the answer is reached by hopping from one piece of evidence to another.

**What existing systems do.** Every table question-answering system has to make these choices. Today they make them with **fixed rules**: pick the most similar row by an embedding score (TableRAG), keep the top-ranked cell from a classifier (HYBRIDER), ask a large language model to choose (Chain-of-Table, Dater), or prune a graph with a similarity threshold and a breadth-first search (ODYSSEY). None of these rules is trained on whether the *final answer* comes out right, and all of them stop after a fixed number of steps.

**What this project does.** We treat "which entity next, or stop?" as a **sequential decision problem** and learn the decisions with **reinforcement learning (RL)**:

- a small neural network (the **policy**) looks at the question and the evidence collected so far, and picks the next table row or linked passage, or decides to **STOP**;
- a fixed, pre-trained **reader** model then answers the question from the chosen evidence;
- the policy is rewarded when the reader's answer is right (and pays a small cost for every entity it selects);
- the policy is trained with **PPO** (Proximal Policy Optimisation), a standard RL algorithm, together with **GAE** (Generalised Advantage Estimation) and a learned **critic**.

**What we found (preliminary results, full HybridQA dev set, 3,466 questions).** With the same frozen reader for every method:

| Evidence selector | Answer F1 (%) |
|---|---|
| Oracle: the correct evidence handed to the reader (upper bound) | 55.7 |
| **PPO trained from scratch (ours)** | **28.8** |
| Best fixed heuristic (BM25 top-3) | 20.8 |
| Imitating the annotated evidence (behaviour cloning) | 19.0 |
| HYBRIDER-style greedy rule | 17.4 |

PPO from scratch improves on the best heuristic by **8.0 F1 points** and on behaviour cloning by **9.8 points**, with statistical confidence intervals far from zero. Surprisingly, *starting* PPO from the behaviour-cloned policy (the usual recipe) removed the gain completely, and a popular alternative optimiser, GRPO, made things worse. Sections 10 and 11 explain why.

---

## 2. Background I: question answering over tables and text

### 2.1 Question answering and "readers"

**Question answering (QA)** is the task of producing an answer string for a natural-language question. A common design splits it into two parts:

- a **retriever** or **selector** decides *which* evidence (paragraphs, table rows, …) to look at;
- a **reader** reads that evidence and extracts the answer.

An **extractive reader** answers by pointing to a span (a contiguous piece) of the text it was given. It is a neural network (here a small Transformer model) that, for every word position, predicts a score for "the answer starts here" and "the answer ends here"; the best start–end pair is the answer. The readers we use were trained on **SQuAD 2.0**, a large dataset of questions about Wikipedia paragraphs. SQuAD 2.0 also contains unanswerable questions, so these readers can say "no answer here" — they produce a **null score** for that option. We use the difference "best span score minus null score" as a confidence.

### 2.2 Measuring answer quality: EM and F1

- **Exact match (EM)**: 1 if the predicted answer equals the gold answer after normalisation (lower-casing, removing punctuation and the words *a/an/the*, collapsing spaces), else 0.
- **F1**: treats both answers as bags of words and computes the harmonic mean of precision (what fraction of predicted words are in the gold answer) and recall (what fraction of gold words were predicted). "Walter Jerry Payton" against gold "Jerry" gets partial credit; "Jerry" gets full credit.

Reported numbers are averages over questions, in percent. We use exactly the normalisation of HybridQA's official evaluation script.

### 2.3 Tables, passages and "multi-hop"

A **table** has a header (column names) and rows of cells. In **HybridQA**, many cells are **hyperlinked** to Wikipedia passages (for example, a player's name links to that player's article). So the evidence for a question is a mixture of structured data (table rows) and unstructured text (passages).

A question is **multi-hop** when no single piece of evidence answers it. In HybridQA, the typical pattern is *table row → linked passage* (as in the Walter Payton example) or *passage → table row* (the condition is in a passage, the answer in the table).

### 2.4 How evidence is usually found: lexical and dense retrieval

- **BM25** is a classic *lexical* scoring function: a piece of text scores high if it contains the question's words, especially rare words, weighted by how often they appear. It is excellent at matching names and numbers.
- **Dense retrieval** uses a neural **sentence encoder** that maps any text to a vector (an **embedding**, here 384 numbers). Texts with similar meaning get vectors pointing in similar directions, measured by **cosine similarity**. It is good at paraphrases, weaker at exact names or numbers.
- **Top-k selection** keeps the k highest-scoring pieces of evidence.

We also use a **hybrid score** (cosine similarity plus normalised BM25).

### 2.5 Why selection matters so much

A reader can only answer from what it is given. If the right row or passage is missing, the answer is lost; if too much irrelevant text is included, the reader may pick a confident but wrong span (a **distractor**). In our experiments, giving the reader the correct evidence yields 55.7 F1, while the best fixed heuristic gives 20.8 F1: **more than 30 F1 points are lost purely to selection.** That gap is what this project tries to close.

---

## 3. Background II: reinforcement learning from zero

### 3.1 Agent, environment, reward

Reinforcement learning is about learning to make **sequences of decisions**. An **agent** interacts with an **environment** in steps:

1. the agent observes the current **state** $s_t$;
2. it chooses an **action** $a_t$;
3. the environment moves to a new state $s_{t+1}$ and gives a **reward** $r_t$ (a number: higher is better).

A complete run from start to finish is an **episode**. The agent's goal is to maximise the total reward of an episode, its **return**. Crucially, a reward may arrive only at the end (here: "was the final answer right?"), so the agent must figure out which earlier decisions deserved the credit. This is the **credit assignment** problem.

In our project: the *state* is the question plus the evidence chosen so far; an *action* is "select this row/passage" or "STOP"; the *reward* is mostly the reader's answer F1 at the end.

### 3.2 Markov decision processes (MDPs)

The formal model is a **Markov decision process**: a set of states, a set of actions available in each state, transition rules (what happens after an action), rewards, a **discount factor** $\gamma \in [0,1]$ (how much future rewards count; we use $\gamma=1$, i.e. all steps count equally) and a **horizon** $H$ (maximum number of steps; we use 3).

### 3.3 Policies, values and advantages

- A **policy** $\pi_\theta(a \mid s)$ gives a probability for each action in each state. $\theta$ are the neural network's weights. A **stochastic** policy is used during training (so the agent explores); at test time we act **greedily** (always take the most probable action).
- The **value** $V^\pi(s)$ is the expected return from state $s$ if you follow $\pi$.
- The **advantage** $A^\pi(s,a)$ is how much better action $a$ is than the policy's average behaviour in $s$. Positive advantage → do this more often; negative → less often.

### 3.4 Policy gradients and REINFORCE

The **policy gradient** idea: increase the probability of actions that led to higher-than-expected return. The simplest algorithm, **REINFORCE** (Williams, 1992), updates

$$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(G_t - b(s_t)\big),$$

where $G_t$ is the return from step $t$ onwards and $b(s_t)$ is a **baseline** (any estimate that does not depend on the action). Subtracting a baseline does not bias the gradient but greatly reduces its noise. A **learned baseline** is a second network that predicts the expected return.

### 3.5 Actor–critic methods

An **actor–critic** method has two networks: the **actor** (the policy) and the **critic** (a value function $V_\phi(s)$). The critic's predictions are used to compute advantages. We use *separate* networks for actor and critic, which large implementation studies found to work better than sharing.

### 3.6 Generalised Advantage Estimation (GAE)

How do we estimate the advantage of each step? Two extremes:

- **Monte-Carlo**: $\hat A_t = G_t - V(s_t)$ — unbiased but noisy;
- **one-step TD** (temporal difference): $\hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — low noise, but wrong if the critic is wrong.

**GAE** (Schulman et al., 2016) blends them with a parameter $\lambda$:

$$\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\,\delta_{t+l}.$$

$\lambda=1$ gives Monte-Carlo, $\lambda=0$ gives one-step TD; we use $\lambda = 0.95$. (The PPO paper prints the last exponent as $T-t+1$, which is a typo; the form above is the consistent one and is what our code implements.)

### 3.7 PPO: Proximal Policy Optimisation

Plain policy gradient uses each batch of experience once. **PPO** (Schulman et al., 2017) reuses each batch for several passes (**epochs**) of small updates, while preventing the policy from changing too much. It looks at the **probability ratio** between the new and the old policy,

$$\rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)},$$

and maximises the **clipped surrogate objective**

$$L^{CLIP} = \mathbb{E}_t\left[\min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\right],$$

with $\epsilon=0.2$. Clipping removes the incentive to push the ratio beyond $1\pm\epsilon$, so each update stays "proximal" to the old policy. We add an **entropy bonus** (rewarding a policy that keeps some randomness, so it keeps exploring) and stop an iteration early if the **KL divergence** (a measure of how different the new policy is from the old one) grows too large.

### 3.8 Critic-free group methods: GRPO

**GRPO** (Group Relative Policy Optimisation, introduced with DeepSeekMath, 2024) is popular for training large language models. It removes the critic: for each question it runs a **group** of $G$ episodes (we use 4) and gives each episode the advantage

$$\hat A_i=\frac{R_i-\mathrm{mean}(R_1..R_G)}{\mathrm{std}(R_1..R_G)+\varepsilon}.$$

So an episode is "good" if it beats its siblings on the same question. A known weakness (Liu et al., 2025, "Dr. GRPO"): dividing by the group's standard deviation can blow up tiny differences between episodes. This matters for our results.

### 3.9 Imitation learning and warm starts

**Behaviour cloning (BC)** is supervised learning on demonstrations: if we know which actions an expert would take, train the policy to predict them. It is often used as a **warm start** before RL, because RL from a random policy can waste a lot of time. Known weaknesses: the policy copies the expert's mistakes, and it may behave badly in states the expert never visited (errors compound; Ross et al., 2011).

### 3.10 Reward shaping, and why its form matters

Sparse rewards (only at the end) make learning slow, so one adds **shaping** rewards along the way. Ng, Harada and Russell (1999) proved a precise result: a shaping reward of the form

$$F(s,a,s') = \gamma\,\Phi(s') - \Phi(s)$$

for some "potential" function $\Phi$ (with potential 0 at the end of the episode) **never changes which policy is optimal**. The reason is simple: over a whole episode these terms cancel out (they **telescope**), so every episode's total reward is unchanged; only *when* the reward arrives changes. Any other kind of extra reward — for example a bonus paid at the end for "having found the right evidence" — *does* change what the agent is optimising.

### 3.11 Variable action sets: masking and pointer networks

In our problem the set of possible actions changes from question to question (different tables have different rows) and from step to step (already-selected entities cannot be chosen again). Two standard tools handle this:

- **Pointer-style scoring** (Pointer Networks, Vinyals et al., 2015): instead of a fixed output layer with one neuron per action, the network computes a score for each candidate from the candidate's own features, then applies a softmax over the candidates that exist.
- **Action masking**: invalid actions get a score of minus infinity (in practice $-10^9$), so their probability is exactly zero. Huang and Ontañón (2022) proved this keeps policy gradients valid and showed that the mask must be stored and re-applied when PPO recomputes probabilities — otherwise training becomes unstable.

---

## 4. The problem statement, precisely

Given a question $q$, a table $T$ and the passages linked from its cells, a **selector** chooses an ordered list of evidence entities $E=(e_1,\dots,e_n)$ (table rows and passages). A **frozen reader** $\rho$ maps $(q,E)$ to an answer. The selector is the only learned component.

Goals:

1. learn a selector, with RL, that makes the frozen reader as accurate as possible, using as little evidence as reasonable;
2. learn **when to stop** instead of using a fixed number of steps;
3. compare PPO with simpler and alternative optimisers (behaviour cloning, REINFORCE, GRPO) under identical conditions;
4. design the selector as a **drop-in module** that could replace the selection step of existing pipelines without touching their reasoner.

---

## 5. The literature: what exists and what is missing

The literature search covered arXiv, the ACL Anthology, OpenReview, Semantic Scholar and conference proceedings up to October 2026. More than one hundred candidate papers were checked; every retained paper's title, authors, venue and identifier were verified. The full synthesis (≈15,000 words, with 73 verified BibTeX entries) is in `reports/RL entity selection for table QA.md`; the per-topic notes are in `research_notes/`. Here is what matters, grouped into four families.

### 5.1 RL that walks graphs (knowledge graphs)

The idea "an agent hops from entity to entity, choosing the next step with a policy" comes from **knowledge-graph reasoning**:

- **DeepPath** (Xiong et al., EMNLP 2017) — first to treat path-finding in a knowledge graph as an MDP with policy gradients. It needed supervised pre-training on breadth-first-search paths; without it, "very poor convergence". Also introduced an efficiency reward (shorter paths better).
- **MINERVA** (Das et al., ICLR 2018) — the template for our policy: at each node, every outgoing edge is scored against a summary of the walk so far, with a softmax over whatever edges exist (an order-invariant **pointer** over a variable set). Stopping was emulated by a self-loop.
- **Lin et al.** (EMNLP 2018) — added reward shaping and **action dropout** (randomly hiding some actions to fight "spurious paths" — paths that reach the right answer for the wrong reason). FB15k-237 MRR from 29.3 to 40.7.
- **M-Walk** (Shen et al., NeurIPS 2018) — the only classic walker with an explicit **STOP** action, scored by its own network.
- Recent LLM agents on graphs (**KG-R1**, **Search-on-Graph-R1**, **Graph-R1**, **GraphRAG-R1**, 2025–2026) — train language models with GRPO. Search-on-Graph-R1 shows a supervised warm start matters a lot there (88.7 vs 63.0 on CWQ).

**What does not transfer:** these methods assume a fixed set of relation types and an ID embedding per entity. In HybridQA every test table is new, so our policy must describe candidates by their *content* (text embeddings and features), and its state is a growing *set* of evidence, not a single position.

### 5.2 RL for table reasoning

- **MAPO** (Liang et al., NeurIPS 2018) and **MeRL** (Agarwal et al., ICML 2019) — learned *programs* over tables from answer-only rewards (WikiTableQuestions ~44–47%); they warn about "spurious" programs that get the right answer by luck.
- **Learning to Reduce** (Lee et al., 2024) — PPO-trained models pick relevant columns, then rows, for a frozen GPT-4 reader; 87% column recall on HybridQA without training on it. Selection happens **once**, inside **one** table, and the reward scores the selection, not the final answer.
- **Table-R1, region-based** (Wu et al., 2025) — the model outputs which rows and columns it will look at, rewarded by overlap with the gold region plus answer correctness; +14.36 points on average. Single table, one-shot.
- **Reasoning-Table** (Lei et al., 2025) — GRPO with an evidence reward that only counts if the answer is right.
- **OpenTable-R1** (Qiu et al., 2025) — the one paper that genuinely learns *which tables to fetch* with RL (search + SQL tools): EM 12.0 → 86.2. One gold table per question, no joins.

### 5.3 Table–text and multi-table systems (and how they select evidence)

- **HybridQA / HYBRIDER** (Chen et al., Findings of EMNLP 2020) — the dataset we use and its reference model: link → rank cells with a BERT classifier → hop → read, **greedy top-1** at each step. Stage accuracies 87.4 / 87.9 / 89.2 % compound before reading; 44.0 EM / 50.7 F1 on dev.
- Later supervised systems (**DocHopper**, **MuGER²**, **MITQA**, **TACR**, **S³HQA**) reach up to 68.4 EM / 75.3 F1 on dev with fine-tuned readers, but still select evidence with fixed rules (keep top-5 rows; rules by question type…). None is trained against the final answer.
- **OTT-QA** (Chen et al., ICLR 2021) — open-domain version; its authors abandoned iterative retrieval because "early mistakes can propagate".
- **ODYSSEY** ("Hybrid Graphs for Table-and-Text based QA using LLMs", NAACL 2025) — builds a graph of cells and linked entities and prunes it with a 0.8 similarity threshold and 3-hop breadth-first search; GPT-4 reader; **14 of its 41 analysed errors come from the selection step**.
- **Chain-of-Table** (ICLR 2024), **TableRAG** (NeurIPS 2024), **Dater** (SIGIR 2023), **Binder** (ICLR 2023) — LLM frameworks for single tables; selection by greedy LLM choice, embedding top-k, prompting, or implicitly inside generated programs.
- **MMQA** (ICLR 2025), **JAR** (ACL 2024), **MURRE** (COLING 2025) — multi-table retrieval; MURRE leaves "learning from downstream SQL feedback" to future work. MMQA's official data link has since expired.

### 5.4 RL-trained evidence selection and search agents (text)

- **José et al.** (BRACIS 2024, arXiv 2407.04858) — **the closest precedent**. A PPO/DQN agent on OTT-QA chooses between three *modules* — retrieve texts, retrieve tables, answer — with a reward of 2 / F1 / −0.5 and −0.02 per retrieval. Its best agent scored **19.03 F1, exactly the same as the fixed "always retrieve text" sequence**; the authors note it converged to "the trivial solution". They observed tables were too coarse and suggested splitting them into rows.
- **s3** (EMNLP 2025) — the cleanest "learned selector, frozen reader" design: a PPO-trained searcher selects documents and decides when to stop; a frozen generator answers. Text only.
- **StepSearch** (EMNLP 2025) — PPO + GAE with a per-step reward for newly covered gold documents; +5.7 to +15.2 points.
- **Search-R1** (COLM 2025) — PPO vs GRPO for search agents: GRPO converges faster but can collapse; PPO is more stable. A follow-up study found intermediate retrieval rewards did not help.
- **IGPO** (ICLR 2026) — shows outcome-only GRPO suffers "advantage collapse" when all rollouts in a group score the same.

### 5.5 The twelve most relevant papers

| # | Paper | Why it matters | What it does not do |
|---|---|---|---|
| 1 | José et al. 2024 | Only RL controller for table–text QA | Module-level actions; tied a fixed heuristic |
| 2 | s3 (2025) | Frozen reader + PPO selector with stop | Text only |
| 3 | StepSearch (2025) | PPO + GAE + evidence-gain reward | Text only; no explicit stop |
| 4 | Learning to Reduce (2024) | RL table selector for a frozen LLM | One shot, one table, no passages |
| 5 | Table-R1 region (2025) | RL rewards the looked-at region | One region, one table |
| 6 | Reasoning-Table (2025) | Answer-gated evidence reward | Cites evidence, does not select step by step |
| 7 | HybridQA / HYBRIDER (2020) | Our dataset and the greedy pipeline to replace | Greedy, no stop, not answer-trained |
| 8 | ODYSSEY (2025) | Hybrid graph with hand-set pruning | Nothing learned |
| 9 | MINERVA + Lin et al. (2018) | Pointer policy, shaping, action dropout | Knowledge graphs only |
| 10 | M-Walk (2018) | Explicit STOP | Knowledge graphs; tree search |
| 11 | MURRE + MMQA (2025) | Hop-wise multi-table retrieval | No learning from answers |
| 12 | Search-R1 (2025) + follow-up | PPO vs GRPO evidence | LLM token policies over text |

### 5.6 The seven research gaps

1. **G1 — No RL method selects table entities one at a time with a learned STOP.**
2. **G2 — No RL selector has been shown to beat a fixed heuristic in table–text QA** (the one attempt tied it).
3. **G3 — Stopping is hand-coded everywhere** (fixed steps, fixed k, BFS depth, an LLM saying "None").
4. **G4 — Whether an evidence reward helps is disputed, and its two forms (policy-invariant shaping vs a terminal bonus) have never been compared.**
5. **G5 — PPO with a learned critic and GAE has never been tested on a small pointer policy**; all PPO-vs-GRPO evidence is from large language models.
6. **G6 — No paper swaps a learned selector into an existing framework while keeping the reasoner fixed.**
7. **G7 — Choosing which tables to join has never been trained on downstream feedback.**

This project addresses G1–G6 in the preliminary study; G7 is planned.

---

## 6. Our formulation: the selection problem as an MDP

### 6.1 Building a graph for every question

For each question we build a small graph:

- **row nodes**: one per table row, written as text: *"Table title : Rank is 2 ; Player is Walter Payton ; … ; Yards is 16,726"*;
- **passage nodes**: one per Wikipedia passage linked from a cell of the table (first 300 words, prefixed with the entity name);
- **edges**: a row is connected to every passage linked from one of its cells.

On average a question has **49 entities** (15.9 rows + 33.2 passages).

**Gold evidence.** HybridQA ships "traced" answer locations: every cell or linked passage that contains the answer string. We turn each location into a **gold path**: `(row)` if the answer is in a row, `(row, passage)` if it is in a passage linked from that row. These labels are **noisy** (the dataset's authors estimate ~15% are wrong, because a string can appear in the wrong place — Emmitt Smith's article also contains "Jerry") and often **multi-valued** (only 55.9% of dev questions have exactly one gold path). We therefore treat them as *a set of acceptable answers*, never as one fixed target.

### 6.2 State, actions, termination

- **State**: the question, the ordered list of entities selected so far, and the step number.
- **Actions** (the **candidate set**): every row not yet selected; the 5 passages that best match the question (hybrid score); every passage linked from an already-selected row; and **STOP** (not allowed at the first step, so the reader always gets something). At most 48 candidates (more are cut, keeping linked ones first).
- **Termination**: when the policy chooses STOP, or after **H = 3** selections.
- We checked that every gold path is reachable under these rules (100% of questions).

### 6.3 The reward

- **Answer reward**: the reader's F1 on the selected evidence, paid at the end.
- **Entity cost**: −0.02 for every selected entity (so STOP has a reason to exist).
- **Evidence signal** (optional): $\psi(E)$ = how much of the best-covered gold path the evidence contains (0, 0.5 or 1). It can enter in two ways:
  - **potential-based shaping** with weight $\kappa$: per-step reward $\kappa(\psi_{t+1}-\psi_t)$, minus everything again at the end — the main configuration uses $\kappa=0.5$;
  - **terminal bonus** with weight $\lambda_{ev}$: $\lambda_{ev}\,\psi(E_{final})$ added at the end (an ablation).

**How the reader turns several entities into one answer.** The reader reads each selected entity separately with the question, gets its best span and a confidence (span score minus null score), and the answer is the span of the most confident entity (the way open-domain systems such as DrQA combine passages). Two consequences: (1) reader outputs can be **cached per (question, entity)** and reused by every method, seed and iteration — this is what makes training affordable; (2) adding an irrelevant entity can hurt, if the reader is confidently wrong about it.

### 6.4 Two mathematical facts about the evidence reward

1. **Shaping changes no episode's total reward.** With $\gamma = 1$ and potential 0 at the start and end, the shaping terms telescope to zero. So shaping cannot change which policy is best; it can only speed up learning for a method that looks at per-step rewards (PPO with GAE does; see 2).
2. **GRPO cannot see shaping at all.** GRPO only uses each episode's *total* reward, which shaping does not change. So a "PPO vs GRPO with the same reward" comparison also compares *credit assignment*, not just optimisers.

The objective the policy optimises (with shaping only) is therefore: **expected reader F1 minus 0.02 × number of selected entities.**

---

## 7. Our method: the policy and how it is trained

### 7.1 Representing states and candidates

All texts are embedded once with a frozen sentence encoder (**all-MiniLM-L6-v2**, 384 numbers per text). Nothing about the encoder is trained, so embeddings are computed per table and cached.

- **State vector (1,160 numbers)**: question embedding + average embedding of selected evidence + embedding of the last selected entity + 8 scalars (progress, rows/passages selected, how much of the question is covered, best candidate scores, type of the last entity).
- **Each candidate**: its embedding + 14 scalars (row or passage; cosine and BM25 scores and rank; whether it is linked to the evidence or to the last entity; row position — useful for "second most …" questions; question-word overlap and *new* overlap; number match; degree; redundancy with the evidence).

No feature refers to a column index or an entity ID, so the same policy works on tables it has never seen.

### 7.2 The networks

- **Actor (762,114 parameters)**: two small networks map the state and each candidate to 256-number vectors $h$ and $g_a$; a scorer computes $u(a)=w^\top\mathrm{ReLU}(W[h; g_a; h\odot g_a])$; STOP has its own head. Invalid candidates are masked; a softmax gives the policy.
- **Critic (728,065 parameters)**: its own copies of the encoders; predicts the expected return from the state and a summary (mean and max) of the candidates.

### 7.3 Behaviour cloning on set-valued targets

From the gold paths we build oracle trajectories: *a gold row → (if the answer is in a passage) a gold passage of that row → STOP*. Because several gold rows/passages may be acceptable, the loss is **minus the log of the total probability of all acceptable actions** at each step. Trained 8 epochs; the epoch with the best validation F1 is kept.

### 7.4 PPO training loop (per seed)

For 40 iterations:

1. sample 1,024 training questions; run one stochastic episode per question;
2. ask the reader (cache first) for answers; compute rewards;
3. compute critic values, GAE advantages ($\gamma=1$, $\lambda=0.95$) and value targets;
4. 4 epochs of minibatch (256) updates: clipped objective ($\epsilon=0.2$), entropy bonus 0.01, critic regression; Adam optimiser, learning rates decayed linearly, gradient clipping 0.5; stop early if KL > 0.045;
5. every 5 iterations, evaluate the greedy policy on the validation set; keep the best.

Two settings differ between variants: **warm-started PPO** starts from the cloned policy, uses actor learning rate $10^{-4}$ and trains only the critic for the first 5 iterations; **PPO from scratch** starts from random weights, uses $3\times10^{-4}$ and no critic warm-up.

### 7.5 The comparison optimisers

- **REINFORCE with learned baseline**: Monte-Carlo advantages, one full-batch gradient step per iteration (learning rate $10^{-3}$), same critic.
- **GRPO**: 256 questions × 4 episodes per iteration, group-normalised advantages, same clipped objective, no critic.

All are given exactly the same data, rewards, starting policy and 1,024 episodes per iteration.

### 7.6 Integration as a drop-in module

The selector is a function: *(question, table, linked passages) → ordered evidence list*. In a HYBRIDER-style pipeline it replaces linking/ranking/hopping; in TableRAG it replaces top-k retrieval; in Chain-of-Table or Dater it would be a row pre-filter; in ODYSSEY it would replace the threshold-and-BFS pruning. The reasoner stays unchanged.

---

## 8. Implementation: code, data pipeline, safeguards, compute

### 8.1 Code layout

| Path | What it does |
|---|---|
| `src/mhtrl/data.py` | builds the per-question graph and gold paths |
| `src/mhtrl/env.py` | candidates, features, termination, reward (shaping/bonus) |
| `src/mhtrl/policy.py` | pointer actor and critic |
| `src/mhtrl/reader.py` | frozen reader with a persistent (question, entity) cache |
| `src/mhtrl/rollout.py` | batched episodes, masking, evaluation metrics |
| `src/mhtrl/rl/algos.py` | PPO, GRPO and REINFORCE in one trainer |
| `src/mhtrl/rl/bc.py` | behaviour cloning |
| `src/mhtrl/baselines.py` | random, BM25/dense/hybrid top-k, greedy hop, oracle |
| `scripts/run_suite.py` | runs everything in one resumable process |
| `scripts/make_assets.py` | writes every table, figure and number macro of the report |
| `tests/test_core.py` | unit tests |

PPO, GRPO, REINFORCE and BC are written directly (about 400 lines), not taken from a library, so every term of the objective is visible.

### 8.2 Data pipeline and obstacles met

1. Clone the HybridQA questions and the WikiTables-WithLinks tables/passages from GitHub.
2. **Obstacle:** some table file names contain characters Windows forbids (`:`, `"`), so a normal checkout fails. **Fix:** read every file straight out of git's object store (`git cat-file --batch`).
3. **Obstacle:** HybridQA's preprocessed archive link is dead (HTTP 404), and the Hugging Face copy is an old loading script without the gold answer locations. **Fix:** use the raw JSON files.
4. Build graphs and embeddings (230,512 entity texts) — `scripts/build_instances.py`.
5. Splits: tables shuffled with a fixed seed; a 10,000-question training pool and 1,000 validation questions on *different tables*; experiments use the first 5,000 training questions (1,000 tables); the test set is the full official dev set (3,466 questions; test labels are hidden by the authors). 262 dev tables also appear in our training subset (inherited from HybridQA's own splits), so we also report results on unseen tables.

### 8.3 Safeguards against error

- **No leakage**: BM25 statistics from training data only; checkpoints chosen on validation; the test set used once per model.
- **Unit tests**: official metric normalisation; candidate generation follows the graph; shaping leaves every return unchanged (and the terminal bonus does not); GAE reduces to Monte-Carlo at $\lambda=1$ and one-step TD at $\lambda=0$; masked actions get exactly zero probability.
- **Runtime checks**: no entity selected twice; STOP only when allowed; masks stored and re-applied in every update.
- **Reproducibility**: fixed seeds; reader cache keyed by reader and (question, entity); fixed thread count (answers identical across thread counts, scores differ by ≤ 1.1×10⁻⁵).
- **Generated results**: tables, figures and the main numbers in the report are written by scripts, not typed.
- **Cross-machine check**: training ran on a GPU machine that rebuilt the data; for every dev question, the gold-path coverage it recorded matches the laptop's copy of the data exactly.

### 8.4 The compute story

- The laptop (Intel i7-1360P, no GPU) ran on battery for long stretches; Windows throttled the CPU 4–7×, and it slept for ~8 hours overnight. Data preparation took 5.2 hours.
- On the laptop one reader call costs ~120 ms (after a batching fix that removed wasteful padding — 3× faster). A full PPO run would take hours, so training moved to a GPU.
- A **pilot run** on the laptop (validation data only) showed warm-started PPO not improving and a weak critic (explained variance < 0.2). Before the real runs, three settings were changed for *all* optimisers: episodes per iteration 512 → 1,024, critic learning rate 5×10⁻⁴ → 10⁻³, critic warm-up 3 → 5 iterations.
- The whole suite — baselines, 5 behaviour-cloning runs, **42 RL runs**, all evaluations — ran in **1.17 hours on one NVIDIA T4 GPU** (Python 3.13, PyTorch 2.11, transformers 5.16). One PPO run takes 1–3 minutes there.

---

## 9. Experimental setup

- **Data**: train 5,000 questions; validation 1,000 (unseen tables); test = HybridQA dev, 3,466 questions (1,349 with the answer in a cell, 2,025 in a passage, 92 computed).
- **Reader**: `deepset/minilm-uncased-squad2` (33M parameters; 76.2 EM / 79.6 F1 on SQuAD 2.0 dev), never trained on tables. Second reader for a transfer test: `deepset/roberta-base-squad2` (124M; 79.9 / 83.0), never used in training. (Dynamic int8 quantisation was tried and rejected: at most ~7% faster and it changed 48% of answers.)
- **Selectors compared**: random walk (k = 2, 3); BM25, dense (TableRAG-style) and hybrid top-k (k = 1, 2, 3); greedy rank-then-hop (HYBRIDER-style); oracle (upper bound); behaviour cloning; PPO warm-started; REINFORCE; GRPO; PPO from scratch.
- **Metrics**: EM, F1 (overall, in-table, in-passage); **AnsRecall** (answer string present in the evidence); **Loc** (the entity holding a traced answer location was selected); **Path** (a whole gold path was covered); right-answer-without-location rate; number of entities and words passed to the reader.
- **Seeds**: 5 for BC, warm-started PPO, REINFORCE and GRPO; 3 for PPO from scratch and each ablation.
- **Statistics**: mean ± standard deviation over seeds; differences tested with a **paired bootstrap** — resample the 3,466 questions with replacement 10,000 times and look at the distribution of the difference between two methods (per-question scores averaged over seeds); the 95% interval and a one-sided p-value come from that distribution. **Probability of improvement** = fraction of (run A, run B) pairs where A scores higher.

---

## 10. Results

### 10.1 Headroom: selection really is the bottleneck

Given the correct evidence (1.58 entities on average), the reader reaches **55.7 F1**. The best fixed heuristic reaches **20.8**, HYBRIDER-style greedy hopping **17.4**, a random walk below 10. Dense retrieval (the TableRAG-style step) is the weakest non-random heuristic (13.7 F1 at k = 2) — rows of one table look almost identical to a sentence encoder, while BM25 matches the exact names and numbers that identify the right row.

### 10.2 Main results (dev set, same frozen reader for all)

| Selector | EM | F1 | F1 (answer in table) | F1 (answer in passage) | Answer in evidence | Entities | Words |
|---|---|---|---|---|---|---|---|
| Random walk, k=2 (3 seeds) | 5.5 | 8.6 | 14.6 | 4.9 | 12.7 | 2.00 | 136 |
| BM25 top-3 | 15.9 | 20.8 | 21.7 | 21.0 | 37.7 | 3.00 | 333 |
| Dense top-2 (TableRAG-style) | 10.0 | 13.7 | 19.2 | 10.7 | 21.1 | 2.00 | 151 |
| Hybrid top-3 | 14.9 | 19.8 | 22.1 | 18.9 | 37.6 | 3.00 | 297 |
| Greedy rank-then-hop | 12.9 | 17.4 | 21.2 | 15.7 | 30.5 | 2.00 | 184 |
| Behaviour cloning (5) | 14.1 | 19.0 ± 0.3 | 21.0 | 18.4 | 36.4 | 1.78 | 163 |
| GRPO, warm-started (5) | 11.5 | 16.1 ± 0.5 | 21.9 | 12.9 | 25.8 | 1.33 | 92 |
| PPO + GAE, warm-started (5) | 14.1 | 18.9 ± 0.4 | 21.4 | 18.1 | 35.4 | 1.70 | 152 |
| REINFORCE, warm-started (5) | 15.0 | 20.0 ± 0.5 | 21.5 | 19.9 | 37.2 | 1.82 | 169 |
| **PPO + GAE from scratch (3)** | **23.3** | **28.8 ± 0.3** | **35.7** | **25.2** | **50.3** | 2.63 | 331 |
| Oracle gold path | 48.4 | 55.7 | 78.6 | 42.9 | 96.9 | 1.58 | 135 |

### 10.3 Are the differences real? (paired bootstrap)

| Comparison | ΔF1 | 95% interval | p | P(A > B) |
|---|---|---|---|---|
| PPO from scratch − behaviour cloning | +9.80 | [8.51, 11.09] | < 0.001 | 1.00 |
| PPO from scratch − BM25 top-3 | +7.97 | [6.86, 9.07] | < 0.001 | 1.00 |
| PPO from scratch − greedy hop | +11.36 | [10.00, 12.72] | < 0.001 | 1.00 |
| PPO from scratch − warm-started PPO | +9.84 | [8.54, 11.13] | < 0.001 | 1.00 |
| PPO from scratch − REINFORCE | +8.74 | [7.47, 10.02] | < 0.001 | 1.00 |
| Warm-started PPO − behaviour cloning | −0.04 | [−0.31, 0.25] | 0.61 | 0.50 |
| REINFORCE − behaviour cloning | +1.06 | [0.55, 1.58] | < 0.001 | 1.00 |
| GRPO − behaviour cloning | −2.90 | [−3.44, −2.37] | — | 0.00 |

Every PPO-from-scratch run beats every run of every other method.

### 10.4 What the winning policy does differently

- Behaviour cloning copies the gold-path pattern: **always exactly one row**, plus a passage for 78% of questions.
- PPO from scratch learned something else: three entities for 73% of questions, but only **0.69 rows** on average; for **35.4%** of questions it selects **no row at all** and takes passages directly from the best-matching passage list.
- Result: the answer string is in its evidence for **50.3%** of questions (36.4% for BC), and it selects the entity holding a traced answer location for **49.9%** (37.4% for BC).
- It uses about as much text as BM25 top-3 (331 vs 333 words) but scores 8 F1 points more — it is not winning by reading more, but by reading better.
- Not luck: correct answers obtained without selecting any traced answer location occur for at most 0.53% of questions, for any method.

### 10.5 Learning dynamics

- **PPO from scratch** starts at 15.8 validation F1 (random network), passes every other method within 5 iterations, and was **still improving at iteration 40** (best 33.9, reached at the last iteration in all 3 seeds). Its training and validation scores stay close (0.36 vs 0.34), so it generalises. Over training it uses slightly fewer entities (2.86 → 2.69) and stops explicitly more often (9% → 22% of episodes).
- **Warm-started PPO and REINFORCE** stay near the cloned policy (PPO within 1 point, REINFORCE +1–2 on validation) although their critics learn well (explained variance ~0.77) and their *training* reward rises (0.22 → ~0.29). PPO's updates are tiny (KL 0.003 per update, 3% of ratios clipped, vs 0.006 and 8% from scratch), and the policy only becomes *more certain* of its cloned choices (entropy 0.80 → 0.40). Behaviour cloning itself overfits: training accuracy 0.51 → 0.82 while validation loss is lowest after epoch 3.
- **GRPO** collapses: validation F1 23.2 → ~13, entities per episode 1.7 → 1.08. On average 65% of its groups have identical returns (no learning signal). Our explanation (an inference, not a measurement): when all four answers in a group are equally wrong, the returns differ only by the 0.02 entity cost; dividing by the tiny standard deviation turns that into a large advantage for shorter episodes, so the policy learns to stop early — exactly the weakness pointed out by Dr. GRPO.
- **A selection quirk**: because the warm-started critic warm-up lasts 5 iterations, the validation check at iteration 5 still measures the cloned policy; for PPO seed 0 and a few ablation seeds that check was the best, so those "PPO" models equal the cloned policy. No later iterate was better, so this reflects the flat curves rather than changing any conclusion.

### 10.6 Ablations (warm-started PPO, 3 seeds each)

| Factor | Settings | F1 range |
|---|---|---|
| Evidence term | none / shaping κ=0.5 / terminal bonus 0.3 / EM reward | 19.0 – 19.2 |
| GAE λ | 0 / 0.95 / 1 | 18.8 – 19.5 |
| Entity cost c | 0 / 0.02 / 0.05 / 0.1 | 18.9 – 19.6 |

All within 0.8 F1 — about twice the seed noise — because warm-started PPO barely moves. These ablations must be repeated from scratch to be informative.

### 10.7 Generalisation

- **Unseen tables** (3,163 dev questions whose table never appears in training): PPO from scratch 28.6 F1, behaviour cloning 18.8, BM25 top-3 20.6 — the same picture.
- **A second reader** (RoBERTa-base, never used in training; first 1,000 dev questions; each selector's evidence unchanged, only the reader swapped). The ranking does not change:

  | Selector | F1, training reader (MiniLM) | F1, unseen reader (RoBERTa) |
  |---|---|---|
  | PPO from scratch (3 seeds) | 28.7 | **30.8** |
  | BM25 top-3 | 20.5 | 21.1 |
  | REINFORCE, warm-started (5) | 20.5 | 21.1 |
  | Behaviour cloning (5) | 19.1 | 20.5 |
  | PPO, warm-started (5) | 18.9 | 20.0 |
  | Greedy rank-then-hop | 18.4 | 18.0 |
  | GRPO, warm-started (5) | 15.7 | 16.2 |
  | Oracle gold path | 56.7 | 61.4 |

  So the learned selection is not tied to the quirks of the reader it was trained with. All RoBERTa numbers were computed in one environment (laptop CPU) from the stored selections; as a consistency check, re-reading BC seed 0 there gave 16.0 EM / 20.4 F1 versus 15.8 / 20.2 on the GPU.

---

## 11. What the results mean, and their limits

**Answers to the research questions.**

1. *Does a learned entity-level selector beat the heuristics?* Yes — when trained by PPO from scratch: +8.0 F1 over the best heuristic, +11.4 over HYBRIDER-style greedy hopping, +9.8 over imitation. Warm-started PPO does not.
2. *What does the learned STOP do?* The policy adapts the amount of evidence per question (1, 2 or 3 entities) and stops more often as training goes on; but most episodes still reach the 3-step limit, so the limit may be too tight.
3. *Does the evidence reward help, and does its form matter?* Not measurable yet: nothing moved in the warm-started regime.
4. *How do the optimisers compare?* From the cloned start: REINFORCE slightly better (+1.1), PPO unchanged, GRPO worse (−2.9).
5. *Is behaviour cloning a good warm start?* **Here, no** — the main surprise. The literature (DeepPath, Search-on-Graph-R1) reports that RL needs a supervised start. We found the opposite, likely because (a) the traced paths teach a strategy (row first, then passage) that the reader does not prefer, (b) the cloned policy overfits, and (c) its low entropy and PPO's small, clipped steps keep it where it is.

**Relation to the closest precedent.** José et al.'s module-level PPO collapsed to a fixed strategy. Our entity-level PPO learned question-dependent selection that beats every fixed heuristic we tried — to our knowledge the first such result for an RL selector in table–text QA (within our literature search).

**Limitations (stated plainly).**

- The reader is small and reads each entity separately; absolute scores are far below fine-tuned HybridQA systems (~75 F1), so this is a comparison of *selection strategies*, not of complete QA systems.
- The policy is optimised for one reader's way of combining evidence (the second-reader test checks how much that matters).
- The gold evidence is noisy (~15% wrong), which affects BC and the evidence metrics (hence the extra answer-location metrics).
- Only 5,000 training questions and 40 iterations; PPO from scratch was still improving; it has 3 seeds (others 5).
- **Confound**: PPO from scratch used a larger learning rate (3×10⁻⁴) and no critic warm-up; warm-started PPO used 10⁻⁴. Warm start and step size are not yet separated (REINFORCE from the cloned start with 10⁻³ also stayed near BC, which hints that the start matters more).
- Test numbers are on HybridQA's dev set (test labels hidden).

---

## 12. Next steps and challenges

**Plan to the final submission, in order of priority:**

1. **Untangle warm start vs step size**: cloned or random start × actor learning rate 10⁻⁴ or 3×10⁻⁴; if cloning still hurts, try early-stopped cloning, a larger entropy bonus, or a decaying KL penalty to the cloned policy.
2. **Repeat the reward, GAE and entity-cost ablations from scratch**, 5 seeds each (gaps G3, G4).
3. **Fix GRPO** (no standard-deviation normalisation, as in Dr. GRPO) and run REINFORCE from scratch, to finish the optimiser comparison (G5).
4. **Scale up**: full training set, more iterations, horizon 4, cell- and column-level nodes.
5. **Integration with LLM reasoners** (G6): use the trained selector as the pruning step of an ODYSSEY-style hybrid graph and as a row pre-filter for Chain-of-Table, with the reasoner fixed; report accuracy, prompt tokens and latency.
6. **Stretch — multi-table joins** (G7) on MMQA with a reward from executing gold SQL.

**Challenges and blockers:** compute (the laptop throttles; the full suite needs ~1.2 h on a free T4); LLM readers cost money or a larger GPU and must be cached; noisy gold evidence; MMQA's official data link has expired and the only mirror has no licence.

---

## 13. Where everything is, and how to reproduce it

| Item | Location |
|---|---|
| Project report (PDF + LaTeX) | `report/main.pdf`, `report/` |
| Literature synthesis + BibTeX | `reports/RL entity selection for table QA.md` |
| Search notes | `research_notes/` |
| Code | `src/mhtrl/`, `scripts/`, `configs/`, `tests/` |
| All results | `results/` (`baselines/`, `runs/<name>_s<seed>/`, `logs/`, `pilot/`) |
| Running on a GPU | `GPU_INSTRUCTIONS.md` |
| Slides | the "Multi-Hop Table Reasoning via RL" deck (link given in chat) |

```bash
bash scripts/run_all_gpu.sh             # GPU: data, baselines, 47 runs, evaluation (~1.2 h on a T4)
python scripts/reread_second_reader.py  # CPU: second-reader transfer
python scripts/make_assets.py           # tables, figures, number macros
cd report && tectonic -X compile main.tex
python tests/test_core.py               # unit tests
```

---

## 14. Glossary

| Term | Meaning |
|---|---|
| Action masking | Giving impossible actions zero probability by setting their score to −∞ |
| Actor / critic | The policy network / the value-predicting network |
| Advantage | How much better an action was than the policy's average in that state |
| BM25 | Classic word-matching relevance score |
| Behaviour cloning (BC) | Supervised learning that imitates demonstrated actions |
| Bootstrap (paired) | Resampling questions to measure how uncertain a difference between two methods is |
| Clipping (PPO) | Limiting the policy-probability ratio to [1−ε, 1+ε] in the objective |
| Dense retrieval | Ranking texts by embedding (vector) similarity |
| EM / F1 | Exact-match accuracy / word-overlap score of an answer |
| Embedding | A vector of numbers representing a text's meaning |
| Entropy | How random a policy is; a bonus keeps it exploring |
| Episode | One complete run from the first decision to STOP |
| Explained variance | How much of the actual returns the critic's predictions account for (1 = perfect, 0 = useless) |
| Frozen | Used as is, never trained |
| GAE (λ) | Advantage estimator mixing one-step and full-episode estimates |
| GRPO | Critic-free method comparing episodes within a group of the same question |
| Gold path | A traced (row) or (row, passage) location of the answer |
| Greedy policy | Always taking the most probable action |
| HybridQA | Dataset of questions over Wikipedia tables with hyperlinked passages |
| KL divergence | A measure of how much the policy changed |
| MDP | Markov decision process: states, actions, transitions, rewards |
| Multi-hop | Needing several pieces of evidence chained together |
| Oracle | A cheating upper bound that is given the correct evidence |
| Pointer network | Scores candidates from their own features, so the action set can vary |
| Policy | The decision rule: probabilities over actions in each state |
| Potential-based shaping | Extra per-step reward γΦ(s′)−Φ(s) that never changes the best policy |
| PPO | Proximal Policy Optimisation: clipped, multi-epoch policy-gradient method |
| Reader | The model that extracts the answer from the selected evidence |
| REINFORCE | The basic policy-gradient algorithm |
| Return | Total reward of an episode (from a step onwards) |
| Seed | The random number that fixes a run; different seeds measure run-to-run variation |
| STOP | The action that ends selection and hands the evidence to the reader |
| Warm start | Initialising RL from a pre-trained (here: cloned) policy |

---

## 15. Key references

Full, verified BibTeX for all of these is in `report/references.bib`.

- Chen et al. 2020. *HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data.* Findings of EMNLP. arXiv:2004.07347.
- Chen et al. 2021. *Open Question Answering over Tables and Text* (OTT-QA). ICLR. arXiv:2010.10439.
- José et al. 2024. *Question Answering with Texts and Tables through Deep Reinforcement Learning.* BRACIS. arXiv:2407.04858.
- Lee et al. 2024. *Learning to Reduce: Optimal Representations of Structured Data in Prompting LLMs.* arXiv:2402.14195.
- Wu et al. 2025. *Table-R1: Region-based Reinforcement Learning for Table Understanding.* arXiv:2505.12415.
- Lei et al. 2025. *Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning.* arXiv:2506.01710.
- Qiu et al. 2025. *OpenTable-R1.* arXiv:2507.03018.
- Agarwal, Devaguptapu, S. 2025. *Hybrid Graphs for Table-and-Text based Question Answering using LLMs* (ODYSSEY). NAACL. arXiv:2501.17767.
- Wang et al. 2024. *Chain-of-Table.* ICLR. arXiv:2401.04398 · Chen et al. 2024. *TableRAG.* NeurIPS. arXiv:2410.04739 · Ye et al. 2023. *Dater.* SIGIR. arXiv:2301.13808 · Cheng et al. 2023. *Binder.* ICLR. arXiv:2210.02875.
- Wu et al. 2025. *MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions.* ICLR · Zhang et al. 2025. *MURRE.* COLING · Chen et al. 2024. *JAR* (join-aware multi-table retrieval). ACL. arXiv:2404.09889.
- Xiong et al. 2017. *DeepPath.* EMNLP. arXiv:1707.06690 · Das et al. 2018. *MINERVA (Go for a Walk and Arrive at the Answer).* ICLR. arXiv:1711.05851 · Lin et al. 2018. *Multi-Hop Knowledge Graph Reasoning with Reward Shaping.* EMNLP. arXiv:1808.10568 · Shen et al. 2018. *M-Walk.* NeurIPS. arXiv:1802.04394.
- Jiang et al. 2025. *s3.* EMNLP. arXiv:2505.14146 · Wang et al. 2025. *StepSearch.* EMNLP. arXiv:2505.15107 · Jin et al. 2025. *Search-R1.* COLM. arXiv:2503.09516 · Wang et al. 2026. *IGPO.* ICLR. arXiv:2510.14967.
- Williams 1992. *REINFORCE.* Machine Learning 8 · Schulman et al. 2016. *GAE.* ICLR. arXiv:1506.02438 · Schulman et al. 2017. *PPO.* arXiv:1707.06347 · Shao et al. 2024. *DeepSeekMath (GRPO).* arXiv:2402.03300 · Liu et al. 2025. *Understanding R1-Zero-Like Training (Dr. GRPO).* COLM. arXiv:2503.20783.
- Ng, Harada, Russell 1999. *Policy Invariance under Reward Transformations.* ICML · Huang & Ontañón 2022. *A Closer Look at Invalid Action Masking.* FLAIRS. arXiv:2006.14171 · Vinyals et al. 2015. *Pointer Networks.* NeurIPS. arXiv:1506.03134.
- Andrychowicz et al. 2021. *What Matters for On-Policy Deep Actor-Critic Methods?* ICLR · Engstrom et al. 2020. *Implementation Matters in Deep RL.* ICLR · Henderson et al. 2018. *Deep RL that Matters.* AAAI · Agarwal et al. 2021. *Deep RL at the Edge of the Statistical Precipice.* NeurIPS.
- Rajpurkar et al. 2018. *SQuAD 2.0.* ACL · Reimers & Gurevych 2019. *Sentence-BERT.* EMNLP · Wang et al. 2020. *MiniLM.* NeurIPS · Chen et al. 2017. *Reading Wikipedia to Answer Open-Domain Questions (DrQA).* ACL.
