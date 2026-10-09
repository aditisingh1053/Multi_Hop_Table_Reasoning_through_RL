# Prior RL never picks table entities hop by hop

No published paper trains the policy this project proposes. That policy walks a table-text or multi-table evidence graph one entity at a time (cell, row, column, linked passage or joinable table), learns when to STOP, and is optimised with PPO and GAE on answer correctness, evidence recall and a per-hop cost. The closest precedent is José et al. (BRACIS 2024). They trained PPO and DQN controllers that choose only among "retrieve texts", "retrieve tables" and "answer" on OTT-QA, and the best agent converged to the trivial always-retrieve-text policy at 19.03 F1 ([José et al.](https://arxiv.org/abs/2407.04858)). Everything else that matters overlaps the project only in part, and falls into four groups:

1. **Table RL that selects one region inside one table:** Learning to Reduce, region-based Table-R1 and Reasoning-Table.
2. **RL search agents over text:** s3, StepSearch and Search-R1. They select evidence and stop, trained with PPO and step rewards.
3. **RL graph walkers on knowledge graphs:** MINERVA, Lin et al. and M-Walk. They supply the pointer head and the STOP action.
4. **Table-text and multi-table pipelines:** HYBRIDER, ODYSSEY, Chain-of-Table, TableRAG and MURRE. Their selection steps are greedy, rule-based or zero-shot, and those steps are exactly where a learned selector would plug in.

The sharpest of the seven gaps below is empirical, not architectural: nobody has shown that any RL selector beats a fixed heuristic in table-text QA, because the one attempt could not.

HybridQA is the right primary benchmark, because its traced answer nodes supply gold evidence for behaviour cloning and a recall reward. Those labels are about 15% wrong, and only 54.5% of dev questions have a single node, so the reward must accept any of several targets.

A correct implementation must:

- **replay stored action masks** when PPO recomputes probabilities;
- **write recall shaping as γΦ(s′)−Φ(s) with a zero terminal potential** if it is to leave the optimal policy unchanged;
- **recognise that critic-free group baselines cannot see such shaping at all**;
- **report at least five seeds with bootstrap confidence intervals**.

## Four groups of prior work overlap the project, none completely

A paper counts as directly relevant here if it does one of four things:

- it learns evidence selection with RL in or near tables;
- it defines the table-text or multi-table environment and pipeline that a selector would plug into;
- it supplies the sequential-decision architecture (a pointer over a variable candidate set plus STOP);
- it fixes optimisation rules that the implementation must obey.

The comparison below shows the shape of the gap: every precedent covers some columns, and none covers all of them.

| Work | Evidence | Action unit | Sequential with STOP? | RL algorithm | Evidence-level reward | Cost term | Frozen reader |
|---|---|---|---|---|---|---|---|
| [José et al. 2024](https://arxiv.org/abs/2407.04858) | Tables + text (OTT-QA) | Module: retrieve texts / retrieve tables / answer | Yes; "answer" ends the episode | PPO, DQN | None | −0.02 per retrieval | Yes (FiE) |
| [Learning to Reduce](https://arxiv.org/abs/2402.14195) | Single table | Relevant columns, then rows, generated once | No | PPO + adaptive KL | Asymmetric relevance (missing ≫ extra) | Small penalty per irrelevant item | Yes (GPT-4) |
| [Table-R1, region](https://arxiv.org/abs/2505.12415) | Single table | Row and column index sets inside the chain of thought | No | TARPO (GRPO variant) | Region IoU, decaying weight | No | No |
| [Reasoning-Table](https://arxiv.org/abs/2506.01710) | Single table (oracle table on OTT-QA) | Cited cells | No | GRPO | Cell Jaccard, gated by answer | No | No |
| [OpenTable-R1](https://arxiv.org/abs/2507.03018) | Table corpus | BM25 table-search and SQL calls | Multi-turn; answer ends | Async GRPO + LoRA | Not described | No | No |
| [s3](https://arxiv.org/html/2505.14146) | Text | Queries, document selection, stop | Yes, explicit stop | PPO | Gain over naive RAG (outcome) | No | Yes |
| [StepSearch](https://arxiv.org/html/2505.15107) | Text | Search round | Implicit (answer) | PPO + GAE | Gold-document information gain | Redundancy penalty | No |
| [MINERVA](https://arxiv.org/abs/1711.05851) | Knowledge graph | Next edge (pointer) | NO_OP self-loop, fixed horizon | REINFORCE | None | No | n/a |
| [M-Walk](https://arxiv.org/abs/1802.04394) | Knowledge graph | Next neighbour or STOP | Yes, explicit STOP score | MCTS + Q-learning | None | No | n/a |
| [KG-R1](https://arxiv.org/abs/2509.26383) | Knowledge graph | One-hop KG query | Answer ends; at most 5 turns | GRPO | Binary: any gold entity retrieved | No | No |
| [HYBRIDER](https://aclanthology.org/2020.findings-emnlp.91/) | Table + linked passages | Cell, then neighbouring cell or passage | Greedy top-1, two fixed steps | None (weak supervision) | n/a | n/a | No (trained reader) |
| [ODYSSEY](https://arxiv.org/html/2501.17767) | Table-text graph | Graph nodes | Breadth-first search (BFS) to depth 3 | None (zero-shot) | n/a | n/a | Yes (LLM) |
| [MURRE](https://aclanthology.org/2025.coling-main.386.pdf) | Multiple tables | One table per hop | Beam search; LLM early stop; at most 3 hops | None | n/a | n/a | Yes (LLM) |
| **This project** | Tables + text + joins | Cell, row, column, passage, joinable table | Yes, explicit STOP | PPO + GAE after behaviour cloning | Gold-evidence recall | Per-hop cost | Yes |

Three precedents come closest, each from a different direction:

- **José et al.** share the hybrid evidence, the sequential decision, the per-step cost and the frozen reader. But they act on modules rather than entities.
- **s3** shares the explicit selection and stop actions, PPO and the frozen generator. But it works on text passages, and rewards only the gain in answer accuracy over naive retrieval ([s3](https://arxiv.org/html/2505.14146)).
- **StepSearch** shares PPO with GAE and a reward built from gold-evidence gain minus redundancy. But a single LLM both searches and answers, over text ([StepSearch](https://arxiv.org/html/2505.15107)).

The project's novelty therefore rests on the combination, not on any single ingredient: entity-level actions over structured evidence, an explicit STOP, a critic trained with GAE, and a recall-plus-cost reward.

Most of the 2025-2026 wave of RL for tables sidesteps selection altogether.

- **Table-R1 (inference-time scaling, EMNLP 2025)** runs GRPO over fully serialised tables capped at 4,096 prompt tokens, with no region choice. Its Llama-3.1-8B variant still reaches 81.2 on WikiTQ, against 68.0 for GPT-4.1 ([Yang et al.](https://aclanthology.org/2025.emnlp-main.1040.pdf)).
- **TableMind** trains a Python-tool agent with a GRPO variant, and lists "extending it to multi-table reasoning" as future work ([TableMind](https://arxiv.org/html/2509.06278v4)).
- **Formula-R1** is the rare PPO user. Its reward for spreadsheet formulas is 1 if correct, 0.2 if executable but wrong, and 0 otherwise ([Formula-R1](https://arxiv.org/html/2505.23667v3)).
- **OpenTable-R1** is the one paper that genuinely learns which tables to fetch. It trains a Qwen3-4B agent with asynchronous GRPO to issue BM25 table searches and SQL calls. Open WikiTable EM rises from 12.0 zero-shot, to 72.1 after cold start, to **86.2** after RL, while average turns fall from 8.8 to 6.8. Each question still has one gold table, and there are no joins ([OpenTable-R1](https://arxiv.org/abs/2507.03018)).

OpenTable-R1 rules out any claim that "RL table selection is unexplored". The defensible claim is narrower: entity-level, multi-hop chaining across tables and linked text. A community index of 42 text-table RL papers lists José et al. as its only multi-hop text-plus-table entry ([Awesome-RL-Table-Understanding](https://github.com/lyc61c/Awesome-RL-Table-Understanding)).

Several adjacent papers were judged tangential. They are named here once, and are not cited further or included in the appendix:

- **Table-Critic:** a prompting-based multi-agent refiner, not RL.
- **SQL-R1 and Arctic-Text2SQL-R1:** RL text-to-SQL that receives the full schema, with no learned table choice.
- **V-tableR1:** multimodal.
- **TableGPT-R1, Mixture-of-Minds and the program-based Table-r1:** code- or tool-calling agents whose row and column filtering happens implicitly inside generated code over a single table.
- **Classic text-QA RL (R³, Coarse-to-Fine QA) and further KG walkers (DIVA, CPL, DacKGR, RLH):** they add nothing that the selected papers do not already supply.

## Twelve papers set the bar, led by a PPO controller that failed

The ranking weighs how many of the project's design decisions each paper constrains, either as a precedent to extend or as a result to beat.

| Rank | Paper | Why it is on the list | What it does not do |
|---|---|---|---|
| 1 | José et al., BRACIS 2024 | Only RL hop-wise controller for table+text QA. Its reward (2 / F1 / −0.5, plus −0.02 per retrieval) is the project's reward without the recall term | Module-level actions over averaged block embeddings; single run; tied the always-retrieve-text heuristic at F1 19.03 |
| 2 | s3, EMNLP 2025 | Frozen generator; explicit document selection and stop; PPO; gain-over-RAG reward; about 2.4k training examples | Text only; no recall or per-step cost term |
| 3 | StepSearch, EMNLP 2025 | PPO + GAE with a per-round gold-document information gain minus redundancy, plus answer F1. This is the project's reward decomposition | Text only; the LLM both searches and answers; no explicit STOP |
| 4 | Learning to Reduce, arXiv 2024 | PPO-trained table evidence selector for a frozen LLM; recall-weighted asymmetric reward; 87% zero-shot column recall on HybridQA | One shot, single table, no passages; reward on selection, not on the answer |
| 5 | Table-R1 (region), arXiv 2025 | RL rewards which rows and columns the model looks at; decaying region weight; consistency penalty | One-shot region inside the chain of thought; single table; no multi-table validation |
| 6 | Reasoning-Table, arXiv 2025 | Answer-gated cell-evidence reward; trains on HybridQA and OTT-QA | Oracle table on OTT-QA; evidence is cited, not selected sequentially |
| 7 | HybridQA / HYBRIDER, Findings EMNLP 2020 | The environment and its gold answer nodes. The greedy rank-then-hop pipeline is the canonical slot to replace | Stages trained independently on noisy weak labels; greedy top-1; no STOP |
| 8 | ODYSSEY, NAACL 2025 | Hybrid table-text graph pruned by hand-set rules (0.8 threshold, 3-hop BFS); 14 of 41 sampled errors come from selection | Zero-shot; no learning; evaluated on 500-question samples |
| 9 | MINERVA (ICLR 2018) with Lin et al. (EMNLP 2018) | MDP template; order-invariant pointer over variable candidates; spurious-path analysis; reward shaping; action dropout | Knowledge graphs only; ID embeddings; STOP emulated by a self-loop; REINFORCE |
| 10 | M-Walk, NeurIPS 2018 | Explicit STOP score from a neighbourhood summary; policy and value share logits; set-like state | Knowledge graphs only; MCTS + Q-learning; sparse binary reward |
| 11 | MURRE (COLING 2025) with MMQA (ICLR 2025) | Hop-wise multi-table retrieval with an LLM early stop; leaves downstream SQL feedback as future work; multi-table benchmark | No learning from answers or SQL; MMQA has no training split and its official data link has expired |
| 12 | Search-R1 (COLM 2025) with Jin et al. (2025) | Direct PPO-versus-GRPO stability comparison; retrieved-token masking; controlled evidence that naive retrieval rewards do not help | LLM token policies over text; outcome reward only |

Several more papers narrowly miss the list:

- **KG-R1 and Search-on-Graph-R1 (SoG-R1)** are the modern knowledge-graph analogues. KG-R1 rewards retrieving any gold entity, and loses 4.03 and 2.68 Hits@1 on WebQSP and CWQ without that reward. SoG-R1 shows that a supervised warm start before GRPO matters enormously: **CWQ 88.7 with SFT plus RL, against 63.0 with RL alone** ([KG-R1](https://arxiv.org/abs/2509.26383); [SoG-R1](https://arxiv.org/abs/2607.18481)).
- **MMOA-RAG** restricts a document Selector's actions to document IDs, which it says "can significantly reduce the exploration space". It warm-starts with SFT and then optimises with multi-agent PPO, mirroring this project's behaviour-cloning-then-PPO pipeline ([MMOA-RAG](https://arxiv.org/html/2501.15228)).
- **Asai et al.** learn STOP with supervision: an end-of-evidence vector is a negative candidate at every step before the gold path ends. That is a template for STOP labels during behaviour cloning ([Asai et al.](https://arxiv.org/pdf/1911.10470)).
- **MAPO and MeRL** are the classic table-RL answer to spurious success. MAPO's memory buffer reached 46.3 on WikiTableQuestions as a 10-model ensemble. MeRL's meta-learned auxiliary reward discounts accidentally correct programs and lifted the single model from 42.9 to 44.1 ([MAPO](https://arxiv.org/abs/1807.02322); [MeRL](https://arxiv.org/abs/1902.07198)).
- **Chain-of-Table and TableRAG** are integration targets rather than methods to extend. They are covered with the other frameworks below.

### José et al. is the closest MDP and the result to beat

José et al. cast open-domain table-text QA as an episode of at most four steps ([José et al.](https://arxiv.org/abs/2407.04858)):

- **Episode:** the first retrieval fetches 10 passages (text or table segments) that seed 10 "blocks". Each later retrieval adds 4 new passages per block.
- **Actions:** Retrieve Texts, Retrieve Tables and Generate Answer.
- **Observation:** 11 vectors, one for the question and one averaged embedding per block.
- **Reward:** terminal. It is 2 for an exact match, token F1 for partial overlap and −0.5 otherwise, plus −0.02 per retrieval action.
- **Frozen modules:** the retriever (BM25 or a Tri-encoder) and the Fusion-in-Encoder reader from Chain-of-Skills.
- **Learners:** DQN and PPO from Stable-Baselines3, with MLP, LSTM, GRU and Transformer feature extractors.

**The best agent, PPO with a Transformer extractor over BM25, scored EM 14.81 and F1 19.03. That is identical to the fixed "always retrieve text" baseline (A1→A1→A1→A3), and the authors say the agent was "converging to the trivial solution of only picking the best action on average"** ([José et al.](https://arxiv.org/abs/2407.04858)). It sits below the non-RL iterative retriever with a cross-block reader (20.7 F1), and far below Chain-of-Skills (63.2 F1 on dev). Each configuration was trained once, because a single run took over a week. The BM25 agents saw only 100k steps, against 1M for the Tri-encoder.

The paper does not do what this project proposes ([José et al.](https://arxiv.org/abs/2407.04858)):

- It has no entity-level action, so it cannot pick a cell, row, passage or join.
- Its state averages away which evidence was selected.
- Its reward has no evidence term.
- Its critic is the library's default value head, with no analysis of GAE or reward shaping.
- It never runs HybridQA's closed setting or any multi-table data.

The authors' own diagnosis points the same way. They write that "table retrieval proved noisy ... as tables often contain excessive data with relevant information typically in a single row". BM25 action sequences that start with tables scored only 4-8 F1, against 16-19 for sequences that start with text, and the authors suggest splitting tables into rows.

**With three coarse actions and a terminal reward, the best response is simply the action that is best on average, so the collapse was predictable.** The paper left three levers unused: finer actions, a dense evidence signal and a warm start.

### s3 and StepSearch supply the PPO recipe, but only over text

**s3 is the cleanest "learned selector, frozen reader" design in any modality.** Its searcher "generates queries, retrieves documents, selects a subset of useful evidence, and decides whether to continue searching". A frozen generator then answers from the accumulated evidence. Gradients flow only through the searcher, which is trained with PPO "due to its strong empirical stability" ([s3](https://arxiv.org/html/2505.14146)).

- **Reward (Gain Beyond RAG):** the generator's accuracy with s3's evidence minus its accuracy with naive top-k retrieval.
- **Data filter:** training keeps only questions the generator gets wrong under naive retrieval, cutting 169,615 questions to 70,286.
- **Efficiency:** **s3 needed 20 PPO steps on about 2.4k examples, against 2,100 steps and 170k examples for Search-R1, and cut wall-clock time 33-fold.**
- **Results:** 58.9% average generation accuracy on six QA datasets. Removing document selection hurt four of the six datasets ([s3](https://arxiv.org/html/2505.14146)).

s3 has no evidence-recall term, no per-step cost and no structured evidence. Its baseline-relative reward and data filter transfer directly to this project. For example, the reward could score the frozen reader's gain over a HYBRIDER-style greedy evidence set.

**StepSearch supplies the reward decomposition.** Its "StePPO" runs PPO with GAE, and places a per-round reward on the last token of each search round ([StepSearch](https://arxiv.org/html/2505.15107)). That per-round reward has two parts:

- **Information gain:** the marginal improvement in TF-IDF cosine coverage of the gold documents, measured against a running maximum.
- **Redundancy penalty:** the fraction of the round's documents that were already retrieved earlier.

The global reward adds answer F1 and a keyword-match "search-key" reward, and retrieved tokens are masked out of the loss. The method gains **5.7, 9.1, 10.0 and 15.2 absolute points** over standard RL baselines on four multi-hop benchmarks.

StepSearch is the only paper found that pairs GAE with a gold-evidence-coverage step reward, which is almost exactly the project's correctness-plus-recall-plus-cost design. It lacks an explicit STOP, a frozen reader and structured evidence.

One caution follows from the telescoping argument in the optimisation section below. **Summing per-round marginal coverage gains yields final coverage. So a step reward like StepSearch's works as a terminal recall bonus, not as policy-invariant shaping.**

### Table-specific RL scores one region inside one table

Three papers reward a policy for where it looks in a table. All three do it once, inside one table.

**Learning to Reduce** trains two FLAN-T5-Large models with PPO to output first the relevant columns and then the relevant rows for a frozen GPT-4 reader ([Learning to Reduce](https://arxiv.org/abs/2402.14195)).

- **Training:** PPO with an adaptive KL penalty and NLPO-style top-p masking at p=0.9.
- **Gold evidence:** rows and columns are deleted from WikiTableQuestions tables for as long as the SQUALL SQL still returns the answer; what survives is gold.
- **Reward:** asymmetric. Relevant items earn a reward, irrelevant ones a small penalty, and missing necessary ones a large penalty.
- **Result:** **applied zero-shot to HybridQA tables, the column selector reaches 87% recall.** A RoBERTa classifier reaches 70% and zero-shot GPT-4 about 71.5%.

It is a 5-page preprint. It ignores linked passages, rewards the selection rather than the downstream answer, and makes one selection with no STOP.

**The region-based Table-R1** makes the region an explicit output, T_reg = {column indices, row indices}, emitted inside the chain of thought ([Table-R1 region](https://arxiv.org/html/2505.12415v3)).

- **Training:** Region-Enhanced SFT, then TARPO, a GRPO variant.
- **Reward:** r = α_t·IoU(region) + (1−α_t)·answer. The weight α_t = γe^(−ρτ) decays over training, and a consistency penalty applies when the region and answer advantages disagree in sign.
- **Results:** a 14.36-point average gain across base models and benchmarks, and 67.5% fewer response tokens than GRPO. With Qwen3-8B it scores 73.87 on WikiTQ and 72.50 on WikiSQL, against GPT-4o's 68.40 and 47.60 ([Table-R1 region](https://arxiv.org/abs/2505.12415)).
- **Stated limits:** 3B-8B models only; the region is inserted once and never refined; no validation on "multi-table QA scenarios".

The decaying schedule is a published answer to a question this project faces: how much to pay for evidence once the policy has learned to find it.

**Reasoning-Table** pays for cited cells, with R = R_ans·(1 + λ1·R_pos) + λ2·R_fmt ([Reasoning-Table](https://arxiv.org/html/2506.01710v1)).

- R_pos is the Jaccard similarity between the cells the model tags and ground-truth cells. The ground truth comes from intersecting five correct LLM samples.
- Because the evidence term is multiplied by answer correctness, evidence pays only when the answer is right.
- It trains on HybridQA and OTT-QA among a dozen datasets, but **supplies the oracle table for OTT-QA**, so table choice is never learned.
- Its abstract claims a 4.0% margin over Claude-3.7-Sonnet ([Reasoning-Table](https://arxiv.org/abs/2506.01710)).

The classic table-RL era offers one more lesson. MAPO and MeRL learned programs, not evidence, over single tables. MAPO's authors warn that "high-reward but spurious programs will bias the gradient" ([MAPO](https://arxiv.org/abs/1807.02322)). An evidence-recall term is the supervised modern counterpart of MeRL's auxiliary reward against accidental success ([MeRL](https://arxiv.org/abs/1902.07198)).

### HYBRIDER and ODYSSEY expose the heuristic slots a policy can fill

**HYBRIDER**, released with HybridQA, is the canonical pipeline a selector would replace. It links question spans to cells and passages, ranks cells with a BERT classifier, hops to a neighbouring cell in the same row, and then reads ([HybridQA](https://aclanthology.org/2020.findings-emnlp.91/)).

- **Training:** the three models are trained independently, on "somewhat noisy" reasoning paths synthesised from where the answer string appears.
- **Inference:** greedy. The authors write: "we use greedy search at first two steps to remain only the highest probably cell" ([HybridQA](https://arxiv.org/abs/2004.07347)).
- **Per-stage accuracy on dev:** linking 87.4%, ranking 87.9%, hopping 89.2%, reading 62%.
- **End-to-end:** HYBRIDER-Large scores 44.0/50.7 on dev and 43.8/50.6 on test (EM/F1).

Every later supervised system keeps the same pattern of weakly supervised, myopic selection, and none is trained against the final answer:

- MITQA keeps a fixed top-5 rows ([MITQA](https://arxiv.org/abs/2112.07337)).
- S³HQA applies rules keyed on question type ([S³HQA](https://arxiv.org/abs/2305.11725)).
- MuGER² compares a table score against a passage score ([MuGER²](https://arxiv.org/abs/2210.10350)).
- TACR returns the top-k cells, and admits it is "relatively worse in row selection" than in column selection ([TACR](https://arxiv.org/abs/2305.14682)).

**ODYSSEY (NAACL 2025)** is the LLM-era equivalent, and the most natural plug-in point for a learned selector over a hybrid graph ([ODYSSEY](https://arxiv.org/html/2501.17767)).

- **Graph:** it links question-relevant sub-table cells to named entities and their documents.
- **Matching:** question entities are matched to nodes with instructor-xl embeddings, keeping "the highest-ranked entity surpassing a 0.8 threshold".
- **Pruning:** "a 3-hop traversal" by BFS. When nothing matches, it falls back to the whole table and passages, in about 10% of HybridQA cases.
- **Results (GPT-4):** 58.4/71.8 EM/F1 on 500 sampled HybridQA dev questions, against 55.4 EM with the full context; 62.02/73.02 on 500 OTT-QA questions.
- **Errors:** **14 of 41 sampled errors come from the selection module** (8 entity matching, 2 entity extraction, 4 entity-header mapping). This bounds the headroom a better selector can claim.
- **Latency:** 3.6-5 s per question, against 1-1.5 s for the zero-shot baseline.

ODYSSEY reports its token savings two ways: reductions of 45% (HybridQA) and 53% (OTT-QA) in one place, and a 32.65% reduction in average HybridQA reader input in another ([ODYSSEY](https://arxiv.org/abs/2501.17767)). Any token-cost comparison against it should name the measure.

The single-table frameworks the project names are integration targets with greedy or similarity-based selection:

- **Chain-of-Table** prompts the LLM to pick the next operation, including `f_select_row` and `f_select_column`, using "a greedy search strategy" rather than sampling. With PaLM 2 it reaches 86.61 on TabFact and 67.31 on WikiTQ, using 50% fewer queries than Binder and 75% fewer than Dater ([Chain-of-Table](https://arxiv.org/abs/2401.04398)).
- **TableRAG** selects columns and cell values by embedding similarity under a token budget, then runs a program-aided agent. It scores 49.2 on ArcadeQA and 45.5 on BirdQA, against 37.7 and 39.6 for a row-and-column retrieval baseline. That gap comes from what gets selected ([TableRAG](https://arxiv.org/abs/2410.04739)).
- **Dater** prompts the LLM to reduce "huge evidence (a huge table) into sub-evidence (a small table)" ([Dater](https://arxiv.org/abs/2301.13808)).
- **Binder** leaves selection implicit in a single generated SQL or Python program ([Binder](https://arxiv.org/abs/2210.02875)).

RE-Tab's deterministic, LCS-based reward over intermediate sub-tables is the only RL-style signal proposed so far for Chain-of-Table-like state sequences. Only its abstract was verified ([RE-Tab](https://arxiv.org/abs/2601.22530)).

The table below maps each framework's selection step to the role a learned policy would take.

| Framework | Selection step today | Role for a learned policy | Evidence of headroom |
|---|---|---|---|
| HYBRIDER | BERT ranker + hop model, greedy top-1 at two steps | Choose a cell, then a cell or passage, then STOP | Stage accuracies of 87.4 / 87.9 / 89.2 compound before a 62% reader ([HybridQA](https://arxiv.org/abs/2004.07347)) |
| MITQA | Fixed top-5 rows, then a joint row-span reranker | Replace the fixed k with a learned STOP | Keeping 5 rows instead of 1 raises gold-row recall by 8-9% ([MITQA](https://arxiv.org/abs/2112.07337)) |
| S³HQA | Rule-based hybrid selector keyed on question type | Replace the rules | Removing the selector costs 2.9 EM ([S³HQA](https://arxiv.org/abs/2305.11725)) |
| ODYSSEY | Entity match above 0.8, BFS to depth 3 | Add a node or STOP, over the hybrid graph | 14 of 41 sampled errors come from selection ([ODYSSEY](https://arxiv.org/html/2501.17767)) |
| N2N-GQA | GraphRank (α=0.85); 5-10 nodes per intermediate hop, 12-25 at the last | Replace the node budgets | 48.80 EM on 500 OTT-QA questions, 8.1 below fine-tuned COS ([N2N-GQA](https://arxiv.org/html/2601.06603)) |
| CORE / COS | Fixed budgets (100 tables, 200 rows, top-100 passages); zero-shot T0 chainer | Replace the chainer and its budget | Oracle links lift a baseline's dev EM from 28.1 to 35.2 ([CORE](https://arxiv.org/abs/2210.12338); [COS](https://arxiv.org/abs/2305.03130)) |
| Chain-of-Table | LLM greedily picks `f_select_row` / `f_select_column` | Policy issues the select operations | Greedy, with no value estimate ([Chain-of-Table](https://arxiv.org/abs/2401.04398)) |
| TableRAG | Embedding top-k columns and cells under a token budget | Replace the top-k | 49.2 vs 37.7 for row-and-column retrieval on ArcadeQA ([TableRAG](https://arxiv.org/abs/2410.04739)) |
| Dater / Binder | LLM sub-table prompt / implicit in the program | Pre-filter the table the LLM sees | No selection-error figure found |
| MURRE / MMQA's MTR | Cosine similarity + LLM removal with beam and LLM stop / stop at zero column overlap | Choose the next table, and STOP | 3-hop R@2 on MMQA: MURRE 34.05, JAR 44.70 ([PEARL](https://arxiv.org/abs/2608.30291)) |

### Graph walkers contributed the pointer head and the STOP score

The policy architecture this project needs was worked out on knowledge graphs.

**MINERVA** casts query answering as a finite-horizon, partially observed MDP ([MINERVA](https://arxiv.org/abs/1711.05851)).

- **Policy:** an LSTM encodes the history. Each outgoing edge is stacked as [relation; entity], and the edges are scored with d_t = softmax(A_t(W2 ReLU(W1[h_t; o_t; r_q]))). This is an order-invariant pointer over a variable-size set.
- **STOP:** emulated with a NO_OP self-loop under a fixed horizon. The authors write that "we could have allowed the agent to take a special 'STOP' action, but we found the current setup to work sufficiently well".
- **Training:** REINFORCE with 20 rollouts per example and a moving-average baseline. A learned baseline "performed similarly".

**Lin et al.** keep that MDP and add three devices this project can borrow ([Lin et al.](https://arxiv.org/abs/1808.10568)):

- **Reward shaping:** R = R_b + (1−R_b)·f, where f is a frozen embedding scorer.
- **Action dropout:** a random fraction of edges is masked during sampling.
- **Fan-out caps:** each entity keeps its top 256-512 neighbours by PageRank.

They lift FB15k-237 MRR from MINERVA's 29.3 to 40.7. Their ablation shows that action dropout matters most: dev MRR is 38.2 for the full model, 31.0 without dropout and 35.1 without shaping. Their warning transfers directly to tables full of repeated values: "there are usually more spurious paths than correct ones, spurious paths are often found first", and high fan-out "often exacerbate[s] this problem".

**M-Walk** is the only classic walker with an explicit STOP ([M-Walk](https://arxiv.org/abs/1802.04394)).

- **Actions:** the current node's edges plus STOP. Neighbours are scored by inner product, and the STOP score comes from a separate MLP over a summary of the neighbourhood.
- **Shared logits:** the policy and the Q-values use the same scores.
- **State:** all traversed nodes and their neighbourhoods. This is the nearest analogue to an evidence set.
- **Results:** 89.9 MAP on NELL-995 against MINERVA's 87.6, using MCTS and Q-learning rather than policy gradients.
- **Ceiling:** the authors note that "when the ground truth is not in the candidate set, M-Walk always makes mistakes".

Three more knowledge-graph papers supply one ingredient each:

- **SRN** carries the idea to natural-language KBQA with answer-only supervision. It adds potential-based reward shaping to tackle "the delayed and sparse reward problem caused by weak supervision". Its full method could not be verified beyond the abstract ([SRN](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc)).
- **DeepPath** is the warm-start precedent. It imitates BFS paths with supervision before REINFORCE, without which "the RL model will show very poor convergence properties". Its efficiency reward of 1/length anticipates a per-hop cost ([DeepPath](https://arxiv.org/abs/1707.06690)).
- **GraphRAG-R1** brings the cost idea into the LLM era with a cost-aware F1 reward, so that "responses achieving identical F1 score receive higher rewards when using fewer retrievals" ([GraphRAG-R1](https://arxiv.org/abs/2507.23581)).

These walkers do not transfer unchanged, for two reasons:

- **Fixed vocabularies.** They assume a fixed relation vocabulary and entity embeddings indexed by ID. MINERVA had to drop NELL-995 test triples with unseen entities, shrinking the test set from 3,992 to 2,818 queries ([MINERVA](https://arxiv.org/abs/1711.05851)). In HybridQA, every test table is new and every column header is a new "relation". A table-graph policy must therefore encode candidates from their content (header, cell text, type, position), the direction KG-R1's schema-agnostic interface points to ([KG-R1](https://arxiv.org/abs/2509.26383)).
- **Single location.** Every classic walker keeps one current location. This project's state is an accumulated evidence set, whose frontier spans rows, passages and joinable tables.

### Multi-table retrievers hard-code the stop and join decisions

For actions that join tables, the precedents are retrievers, not RL.

- **MMQA's MTR** decomposes the question with GPT-4-turbo and retrieves the top-K tables per sub-question. From the second round on, it multiplies relevance by a binary column-overlap score, and stops when the overlap is 0. Its best retrieval F1 is 68.3 ([MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)).
- **MURRE** retrieves by cosine similarity and has an LLM rewrite the question to "remove" what the retrieved tables already cover ([MURRE](https://aclanthology.org/2025.coling-main.386.pdf)). It searches with a beam "to reduce the error cascades", caps hops at 3 (over 98% of questions need no more), and lets the LLM emit "None" to stop. It gains 5.7% on average over the previous best on SpiderUnion and BirdUnion. **MURRE's authors explicitly leave "the retrieval recall improvement leveraging the results of text-to-SQL" to future work.** That is the downstream-feedback signal an RL selector would use.
- **JAR** instead solves a mixed-integer program over relevance and join compatibility ([JAR](https://arxiv.org/abs/2404.09889)). A greedy join-aware search matches it while running 4-400 times faster ([Greedy-JAR](https://arxiv.org/abs/2511.13418)), so a sequential formulation is not inherently worse than global optimisation.

Every method degrades with depth. On a unified MMQA corpus, R@2 falls from 2-hop to 3-hop questions from 55.90 to 44.70 for JAR, and from 49.15 to 34.05 for MURRE. The offline join-path index PEARL reaches 58.09 on 3-hop questions ([PEARL](https://arxiv.org/abs/2608.30291)). REaR, an LLM-free retrieve-expand-refine retriever, names as future work "adaptive expansion that balances recall and latency by adjusting depth to query complexity via query-aware heuristics or reinforcement learning" ([REaR](https://arxiv.org/abs/2511.00805)).

### Search-R1 and a controlled study frame the PPO and reward debates

**Search-R1** offers the most direct PPO-versus-GRPO comparison for retrieval agents ([Search-R1](https://arxiv.org/html/2503.09516)):

- "GRPO converges faster than PPO across all cases. This is because PPO relies on a critic model, which requires several warm-up steps."
- But "GRPO leads to reward collapse after training for many steps, whereas PPO remains stable".
- Final rewards are comparable, so PPO is Search-R1's default.
- Masking retrieved tokens out of the loss "consistently outperforms the variant without masking".

**The companion study by the same group** is the main counter-evidence to an evidence reward ([Jin et al.](https://arxiv.org/html/2505.15117)). It finds that "adding intermediate retrieval rewards does not significantly improve final performance for either PPO or GRPO", and that "increasing [the retrieval-reward weight] consistently leads to degraded performance". The authors attribute this to a substring-match reward that "may overly constrain the retrieval trajectory".

The opposite evidence is just as specific:

- StepSearch's coverage-based gain helps.
- IGPO shows that outcome-only GRPO suffers "advantage collapse" when every rollout in a group scores the same. Its turn-level information-gain reward adds 16.6 points for a 3B model and 8.3 for a 7B model ([IGPO](https://arxiv.org/html/2510.14967)).
- KG-R1 loses Hits@1 without its retrieval reward.

The evidence on PPO also cuts both ways:

- KG-R1 reports that replacing GRPO with PPO dropped WebQSP from 82.75 to 59.62 and CWQ from 65.31 to 35.66, with reward hacking in which "the agent fabricates 'retrieved' content" ([KG-R1](https://arxiv.org/abs/2509.26383)).
- Graph-R1 finds that "GRPO significantly outperforms REINFORCE++ ... and PPO" ([Graph-R1](https://arxiv.org/abs/2507.21892)).

All of these are LLM token policies, whose critic must read long text. None tests a compact pointer policy with a learned critic.

## Seven gaps a project can fill, each with a falsifiable test

Each gap below comes with its evidence and an experiment that can close it. Each claim that something is absent rests on searches through October 2026 (arXiv, ACL Anthology, Semantic Scholar and a community index), not on proof.

### Gap 1: no RL method selects table entities sequentially with a learned STOP

No published work trains a policy, with an RL signal derived from answer correctness, whose actions are individual cells, rows, columns, linked passages or joinable tables, chosen one per step, with an explicit STOP. The evidence, paper by paper:

- José et al. act on modules ([José et al.](https://arxiv.org/abs/2407.04858)).
- Learning to Reduce, Table-R1 and Reasoning-Table each select once, inside one table.
- OpenTable-R1 selects whole tables through search queries, with one gold table per question ([OpenTable-R1](https://arxiv.org/abs/2507.03018)).
- The knowledge-graph walkers operate on curated graphs with ID embeddings.
- Every supervised table-text selector learns from answer-location labels, through stages trained independently ([HybridQA](https://arxiv.org/abs/2004.07347); [MITQA](https://arxiv.org/abs/2112.07337)).

**Test that closes it:** an entity-level PPO policy on HybridQA's full dev set, reporting EM/F1 and evidence recall, against the same frozen reader fed by HYBRIDER-style greedy selection.

### Gap 2: no RL selector has been shown to beat the best fixed heuristic in table-text QA

The only attempt tied the "always retrieve text" baseline. It used a single run, only 100k BM25 steps and no seed variance ([José et al.](https://arxiv.org/abs/2407.04858)). The LLM graph frameworks evaluate on samples of 200-500 questions ([ODYSSEY](https://arxiv.org/html/2501.17767); [N2N-GQA](https://arxiv.org/html/2601.06603)).

**Test that closes it:** compare PPO, over at least five seeds, against four baselines:

- the best fixed action sequence;
- greedy rank-then-hop;
- the behaviour-cloned policy alone;
- one-shot top-k selection in the style of Learning to Reduce.

Report stratified bootstrap confidence intervals and the probability of improvement ([Agarwal et al. 2021](https://arxiv.org/abs/2108.13264)). A negative result is publishable here, because it would extend José et al.'s finding from module-level to entity-level actions.

### Gap 3: stopping is hard-coded everywhere in table QA

Every system reviewed fixes its stopping rule by hand:

- HYBRIDER takes two greedy steps.
- MITQA keeps five rows. Moving from one row to five already adds 8-9% gold-row recall, so the budget matters.
- COS fixes its budgets at 100 tables, 200 rows and 100 passages.
- ODYSSEY stops at BFS depth 3.
- N2N-GQA fixes its node budgets.
- MTR stops at zero column overlap.
- MURRE lets an LLM say "None".

Work on text agents shows that the stop decision is a real failure mode. Search Wisely finds one model "could have avoided searching in 27.7% of its search steps", and its β-GRPO cuts under-search from 42.04% to 34.71% ([Search Wisely](https://arxiv.org/html/2505.17281)). No table paper reports how accuracy trades against evidence size, or against the number of hops, under a learned STOP.

**Test that closes it:** sweep the per-hop cost c. Report the distribution of hops, the rates of premature STOP and over-selection, and an EM-versus-tokens Pareto curve against fixed-k baselines.

### Gap 4: the value of an evidence-recall reward is contested and untested on tables

Whether evidence-recall rewards help is disputed, and the two possible forms of the reward have never been compared:

- **No help:** Jin et al. find no benefit, and harm at large weights ([Jin et al.](https://arxiv.org/html/2505.15117)).
- **Help:** StepSearch, IGPO and KG-R1 report gains. KG-R1 loses 4.03 and 2.68 Hits@1 without its retrieval reward ([KG-R1](https://arxiv.org/abs/2509.26383)).
- **Hedged designs:** Table-R1 decays its region reward over training. Reasoning-Table pays for evidence only when the answer is correct. Learning to Reduce rewards the selection alone.

**None separates potential-based shaping, which cannot change the optimal policy, from a terminal recall bonus, which can** ([Ng et al.](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)).

HybridQA's gold evidence complicates the question in three ways ([OTT-QA](https://arxiv.org/abs/2010.10439); [HybridQA traced dev file](https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev.traced.json)):

- It is noisy: a manual study found about 15% error.
- It is often multi-valued: only 54.5% of dev questions have exactly one answer node.
- It locates answers, not reasoning paths.

No paper reports how often a selector gets the right answer from the wrong evidence. This is the table analogue of the spurious paths that Lin et al. and MAPO describe.

**Test that closes it:** a 2×3 ablation crossing the reward form (potential-based shaping vs terminal bonus) with the recall weight λ ∈ {0, small, large}, using any-of recall. Report EM, recall and the rate of right answers from wrong evidence separately.

### Gap 5: PPO with a learned critic and GAE is untested on small pointer policies

PPO with a learned critic and GAE has never been tested on a small pointer policy over a variable candidate set. Across the literature:

- **Every PPO-versus-GRPO comparison uses LLM token policies:** Search-R1, KG-R1 and Graph-R1.
- **The classic walkers use other methods:** REINFORCE, or MCTS with Q-learning.
- **The 2025-2026 table RL papers almost all use the GRPO family:** both Table-R1 papers, Reasoning-Table and TableMind.
- **Formula-R1 is the lone PPO user,** and it runs PPO over formula tokens.

Critic-free methods have an additional blind spot, derived below: they cannot see potential-based shaping at all.

**Test that closes it:** train PPO+GAE, REINFORCE with a moving-average or learned baseline, and RLOO or Dr. GRPO on identical rewards and identical seeds. Report sample efficiency and the fraction of degenerate groups, i.e. groups in which every rollout scores the same.

### Gap 6: no paper swaps a learned selector into an existing table-reasoning framework

No paper replaces a framework's selection step with a learned selector while holding the reasoner fixed. Each framework's selection step is something simpler:

- a greedy LLM choice (Chain-of-Table);
- embedding top-k (TableRAG);
- a prompt (Dater);
- left implicit in a program (Binder);
- BFS (ODYSSEY);
- greedy classifiers (HYBRIDER).

Learning to Reduce comes nearest, as a one-shot reduction for a frozen GPT-4. Two numbers show the step matters: ODYSSEY's 14 selection errors out of 41, and TableRAG's 11.5-point lead over row-and-column retrieval on ArcadeQA ([ODYSSEY](https://arxiv.org/html/2501.17767); [TableRAG](https://arxiv.org/abs/2410.04739)).

**Test that closes it:** keep the reasoner frozen and swap only the selection step. Report accuracy, tokens and latency.

Chain-of-Table, TableRAG, Dater and Binder are single-table frameworks, so there the selector acts as a sub-table pre-filter. Gold evidence for those experiments must be derived as Learning to Reduce did, from SQUALL SQL, because WikiTableQuestions ships answers only ([WikiTableQuestions](https://aclanthology.org/P15-1142/)).

### Gap 7: selecting tables to join has never been trained on downstream feedback

The literature keeps pointing at this gap without filling it:

- MURRE lists downstream feedback as future work.
- REaR suggests RL for adaptive expansion depth.
- TableMind and Table-R1 list multi-table settings as future work or unvalidated.
- Reasoning-SQL's schema-linking reward (Jaccard over the tables and columns used, weight 1 against 3 for execution) is the only RL credit for table choice found. It is implicit in SQL tokens and given in a single shot ([Reasoning-SQL](https://arxiv.org/abs/2503.23157)).

On MMQA, going from 2-hop to 3-hop questions costs JAR 11.2 R@2 points and MURRE 15.1 ([PEARL](https://arxiv.org/abs/2608.30291)).

A data barrier makes this the riskiest gap ([MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf); [HF mirror](https://huggingface.co/datasets/table-benchmark/mmqa)):

- MMQA's official data link has expired.
- The only mirror has no licence.
- There is no training split.
- Tables average 1,369-1,833 rows.

**Test that closes it:** use row- and column-level nodes over MMQA's 721-question 3-table subset, with self-made splits, and a reward from executing the gold SQL. Position this as a stretch experiment.

## HybridQA anchors evaluation, but its gold evidence is noisy

**HybridQA** has 62,682 training, 3,466 dev and 3,463 test questions. Each question is grounded in one of about 13,000 Wikipedia tables: about 15.7 rows by 4.4 columns, 70 cells and 44 hyperlinked passages of about 103 words ([HybridQA](https://aclanthology.org/2020.findings-emnlp.91/)). Answers lie in passages for 56.4% of questions, in table cells for 42.3%, and are computed for 1.1%; human performance is 88.2/93.5 EM/F1. With those averages, a per-question entity graph has roughly 130-140 nodes (about 70 cells, 16 rows, 4 columns and 44 passages). That is small enough for PPO on a CPU.

**Gold evidence exists only in the auto-generated `train.traced.json` and `dev.traced.json` files.** Each answer node has the form `[cell_text, [row, col], link_or_null, "table"|"passage"]`. The human-annotated `train/dev.json` files do not include these nodes ([README](https://raw.githubusercontent.com/wenhuchen/HybridQA/master/README.md)). Three properties of the nodes shape the reward:

- **Noise.** The dataset's authors estimate the string-matching tracer "roughly contains 15% error" ([OTT-QA](https://arxiv.org/abs/2010.10439)).
- **Multiplicity.** In dev, only 1,890 of 3,466 questions (54.5%) have exactly one node, 2,102 (60.6%) have all their nodes in a single row, and 92 have none ([dev.traced.json](https://raw.githubusercontent.com/wenhuchen/HybridQA/master/released_data/dev.traced.json)).
- **No reasoning path.** Only answer locations are traced, so "evidence recall" on HybridQA means recall of answer nodes or of their rows. It does not cover the full reasoning path.

**Access and scoring.** Test answers are hidden and scored on CodaLab competition 7979; whether submissions are still accepted was not confirmed. The official `evaluate_script.py` applies SQuAD-style normalisation and reports EM and F1 for in-table, in-passage and total questions ([evaluate_script.py](https://github.com/wenhuchen/HybridQA/blob/master/evaluate_script.py)). The data should be loaded from GitHub plus a clone of WikiTables-WithLinks, not from Hugging Face:

- The Hugging Face copy is a loading script with no answer nodes ([hybrid_qa.py](https://huggingface.co/datasets/wenhu/hybrid_qa/blob/main/hybrid_qa.py)).
- `datasets` 4.0 removed support for loading scripts ([datasets 4.0.0](https://github.com/huggingface/datasets/releases/tag/4.0.0)).
- The README's S3 zip returned HTTP 404 on 2026-10-06.

| HybridQA system | Type | Dev EM/F1 | Test EM/F1 | Source |
|---|---|---|---|---|
| Table-only / passage-only baselines | Trained | 8.4/12.1 ; 19.5/25.1 | 8.3/11.7 ; 19.1/25.0 | [HybridQA](https://aclanthology.org/2020.findings-emnlp.91/) |
| HYBRIDER (BERT-base, τ=0.8) | Weakly supervised greedy pipeline | 43.5/50.6 | 42.2/49.9 | [HybridQA](https://aclanthology.org/2020.findings-emnlp.91/) |
| HYBRIDER-Large | Same | 44.0/50.7 | 43.8/50.6 | [HybridQA](https://aclanthology.org/2020.findings-emnlp.91/) |
| DocHopper | Iterative attention over evidence | 47.7/55.0 | 46.3/53.3 | [DocHopper](https://arxiv.org/abs/2106.00200) |
| MuGER² (large) | Multi-granularity retriever + rule selector | 57.1/67.3 | 56.3/66.2 | [MuGER²](https://aclanthology.org/2022.findings-emnlp.498/) |
| MITQA | Multi-instance row retriever, top-5 | 65.5/72.7 | 64.3/71.9 | [MITQA](https://aclanthology.org/2023.acl-long.449/) |
| TACR | Alignment-based cell selection | 64.5/71.6 | 66.2/70.2 | [TACR](https://aclanthology.org/2023.findings-acl.409/) |
| S³HQA (mean of 5 runs) | Three-stage, rule-based selector | 68.4/75.3 | 67.9/75.5 | [S³HQA](https://aclanthology.org/2023.acl-short.147/) |
| HPROPRO (GPT-4) | Program prompting, no fine-tuning | 48.0/54.6 (200-question sample) | 48.7/57.7 (full test) | [HPROPRO](https://aclanthology.org/2024.acl-long.594/) |
| GPT-4, full table and passages | Zero-shot | 55.40/68.84 (500-question sample) | — | [ODYSSEY](https://arxiv.org/abs/2501.17767) |
| ODYSSEY (GPT-4-1106) | Hybrid graph, zero-shot | 58.40/71.80 (500-question sample) | — | [ODYSSEY](https://arxiv.org/abs/2501.17767) |
| N2N-GQA | Graph pruning, zero-shot | 41.50/48.17 (200-question sample) | — | [N2N-GQA](https://arxiv.org/html/2601.06603) |

The closest baselines for a frozen-reader selector are HYBRIDER and DocHopper. HYBRIDER uses the same pick-a-cell, hop, read pipeline, and its split between in-table and in-passage answers (51.3 vs 40.1 dev EM) is the natural place to show where a policy helps ([HybridQA](https://aclanthology.org/2020.findings-emnlp.91/)). DocHopper is an iterative evidence selector. The fully fine-tuned systems at 57-68 EM are upper references, not like-for-like comparisons.

Four comparability caveats apply:

- **Sample sizes.** All LLM numbers except HPROPRO's full-test 48.7 come from 200- or 500-question dev samples.
- **TACR.** Its test EM exceeds its dev EM, and it compares itself against the base rather than large MuGER².
- **No RL or frozen-reader baseline.** No published HybridQA result exists for an RL-trained selector or for a frozen SQuAD-style reader. The project must therefore measure its own upper bound: the frozen reader given oracle answer-node evidence.
- **ODYSSEY's sample.** ODYSSEY's 500 dev questions are a uniform sample, while its comparison with HPROPRO uses HPROPRO's 200-question subset ([ODYSSEY](https://aclanthology.org/2025.naacl-long.39/)).

**OTT-QA** is the open-domain extension: 41,469 training, 2,214 dev and 2,158 test questions over more than 400K tables and 5M passages. Hyperlinks are provided only for training ([OTT-QA](https://arxiv.org/abs/2010.10439)). That makes it a poor primary benchmark for a CPU project. It is still the only setting where José et al. can be compared directly. The authors of OTT-QA themselves abandoned iterative retrieval because "early mistakes can propagate to later retrieval stages" ([OTT-QA](https://arxiv.org/abs/2010.10439)).

| OTT-QA system | Dev EM/F1 | Test EM/F1 | Source |
|---|---|---|---|
| José et al. PPO (BM25) | 14.81/19.03 | — | [José et al.](https://arxiv.org/abs/2407.04858) |
| Iterative retrieval + cross-block reader | 17.1/20.7 | 16.9/20.9 | [OTT-QA](https://arxiv.org/abs/2010.10439) |
| Fusion retrieval + cross-block reader | 28.1/32.5 | 27.2/31.5 | [OTT-QA](https://arxiv.org/abs/2010.10439) |
| CORE | 49.0/55.7 | 47.3/54.1 | [CORE](https://arxiv.org/abs/2210.12338) |
| Chain-of-Skills + FiE | 56.9/63.2 | 54.9/61.5 | [COS](https://arxiv.org/abs/2305.03130) |
| N2N-GQA (500-question sample) | 48.80/57.26 | — | [N2N-GQA](https://arxiv.org/html/2601.06603) |
| ODYSSEY, GPT-4 (500-question sample) | 62.02/73.02 | — | [ODYSSEY](https://arxiv.org/html/2501.17767) |

**MMQA** is the only multi-table, multi-hop benchmark with gold SQL and primary/foreign-key labels ([MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)).

- **Construction:** 2,591 two-table and 721 three-table questions, built from Spider with 45 SQL templates and natural-language questions generated by GPT-4-turbo.
- **Table size:** averages of 1,833 and 1,369 rows per table for the two subsets.
- **Results:** o1-preview reaches 50.78 EM on two-table questions and 48.28 on three-table questions; GPT-4 reaches 28.88 and 28.58. The human reference is 89.8, though the paper does not say which metric each of its four human figures corresponds to.
- **Availability:** the official anonymous data link now returns HTTP 410 ([anonymous repo](https://anonymous.4open.science/r/MMQA-34B1)). The only copy found is a third-party Hugging Face mirror with 3,313 examples, a single test split and no licence ([HF mirror](https://huggingface.co/datasets/table-benchmark/mmqa)).
- **Quality flags:** the paper's 3-table GPT-4 rows duplicate its 2-table Gemini-pro rows in every column except one. Its data-source column says "Wikipedia", although the method section says Spider.

MMQA therefore suits only an optional transfer experiment.

Three other datasets should be rejected as training data:

- **WikiTableQuestions** ships answers only, with no evidence annotation ([WikiTableQuestions](https://aclanthology.org/P15-1142/)).
- **TAT-QA** is finance arithmetic over small tables, with paragraph-level evidence ([TAT-QA](https://aclanthology.org/2021.acl-long.254/)).
- **MultiTabQA** produces table-valued answers from Spider, ATIS and GeoQuery (6,715 training and 985 validation examples in its Spider set), though its gold SQL is a possible source of join supervision ([MultiTabQA](https://aclanthology.org/2023.acl-long.348/)).

For the frozen reader, deepset's SQuAD 2.0 models are the practical choice. They can abstain, which gives a usable "insufficient evidence" signal. Published HybridQA systems fine-tune their own BERT readers instead.

| Model | Parameters | SQuAD 2.0 dev EM/F1 | Use |
|---|---|---|---|
| [`deepset/minilm-uncased-squad2`](https://huggingface.co/deepset/minilm-uncased-squad2) | 33M | 76.19/79.55 | Smallest |
| [`deepset/tinyroberta-squad2`](https://huggingface.co/deepset/tinyroberta-squad2) | 82M | 78.86/82.04 | When rollout speed dominates |
| [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) | 124M | 79.93/82.95 | Safest default |
| [`deepset/deberta-v3-base-squad2`](https://huggingface.co/deepset/deberta-v3-base-squad2) | 184M | 83.82/87.41 | Best accuracy at base size |

The reader should also be able to copy a cell's value directly, because about 42% of answers are table cells. HYBRIDER returns answers from non-hyperlinked cells directly for the same reason ([README](https://raw.githubusercontent.com/wenhuchen/HybridQA/master/README.md)).

The protocol that follows from all of this:

1. **Training data.** Subsample 5-10K training questions for CPU budgets. Drop the zero-node "computed" questions from behaviour cloning, but keep them in evaluation, because the official total includes them.
2. **Main evaluation.** Score the full 3,466-question dev set with the official script, broken down into in-table, in-passage and total.
3. **Evidence and cost metrics.** Report any-of answer-node recall and row recall, the distribution of hops and STOP timing, and reader input tokens.
4. **Upper bound.** Report the frozen reader's score with oracle evidence.
5. **Seeds.** Use at least five seeds, with bootstrap confidence intervals.
6. **Test set.** Optionally, submit once to CodaLab.

## A correct implementation hinges on replayed masks, potential-form shaping and seeds

### PPO and GAE: use the clipped objective and fix the exponent

PPO maximises $L^{CLIP}(\theta)=\hat{\mathbb E}_t[\min(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)]$, where $r_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$ ([PPO](https://arxiv.org/abs/1707.06347)). When the policy and value share parameters, this is combined as $L^{CLIP}-c_1(V_\theta-V^{targ})^2+c_2S[\pi_\theta]$.

The PPO paper's own ablations set the defaults ([PPO](https://arxiv.org/abs/1707.06347)):

- **Clip value.** In its ablation over 7 tasks and 3 seeds, **ε=0.2 scored best (0.82 normalised)**, against 0.76 at 0.1, 0.70 at 0.3 and −0.39 with no clipping.
- **KL penalty.** An adaptive KL penalty "performed worse than the clipped surrogate objective".
- **Defaults.** γ=0.99 and λ=0.95. On Atari it used c1=1 and c2=0.01; its MuJoCo runs did not share parameters and used no entropy bonus.

GAE defines $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ with $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ ([GAE](https://arxiv.org/abs/1506.02438)). Its authors stress an asymmetry: "Taking γ<1 introduces bias into the policy gradient estimate, regardless of the value function's accuracy. On the other hand, λ<1 introduces bias only when the value function is inaccurate."

**The truncated estimator printed in the PPO paper carries an exponent T−t+1 on its last term, which is inconsistent with the summation index.** The consistent form is $\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$. Implement it as the backward recursion $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$, reset to $\hat A_t=\delta_t$ with $V(s_{t+1})=0$ whenever step t ends the episode.

For episodes of 2-5 hops, three settings follow:

- **Discount.** Set γ=1 (or 0.99). It costs almost nothing in variance and removes γ-bias.
- **λ.** Use something between 0.9 and 1.0. Andrychowicz et al. recommend 0.9 ([Andrychowicz et al.](https://openreview.net/forum?id=nIAxjsniDzg)), and Ahmadian et al. found λ=1 best with sparse, sequence-level reward ([RLOO](https://aclanthology.org/2024.acl-long.662/)).
- **Value targets.** Use $V^{targ}_t=\hat A_t+V(s_t)$, bootstrapping with V=0 at a true terminal. A true terminal is STOP, or the hop cap if the cap is part of the task. The reference PPO code does not bootstrap at time-limit truncation, although "we (intuitively) should" ([37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)).

Three implementation studies agree that code-level choices matter as much as the algorithm. Engstrom et al. found the PPO "code-optimizations are more important in terms of final reward achieved than the choice of general training algorithm (TRPO vs. PPO)" ([Engstrom et al.](https://arxiv.org/abs/2005.12729)). Andrychowicz et al. add specific guidance ([Andrychowicz et al.](https://arxiv.org/abs/2006.05990)):

- **Value clipping:** "PPO-style value loss clipping ... hurts the performance regardless of the clipping threshold."
- **Separate networks:** separate value and policy networks won on four of five environments.
- **Advantage normalisation:** per-minibatch normalisation "seems not to affect the performance too much".

The table below synthesises starting values from these sources. They have not been tuned for table QA.

| Setting | Starting value | Basis |
|---|---|---|
| Clip ε | 0.2; sweep 0.1-0.3 | [PPO](https://arxiv.org/abs/1707.06347) Table 1; [Andrychowicz et al.](https://arxiv.org/abs/2006.05990) start at 0.25 |
| Epochs and minibatches | 4-10 epochs; shuffled minibatches; recompute advantages each pass | [Andrychowicz et al.](https://arxiv.org/abs/2006.05990) |
| Optimiser | Adam; lr 3×10⁻⁴ for new heads; ε=10⁻⁵; linear decay | [Andrychowicz et al.](https://arxiv.org/abs/2006.05990); [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |
| Gradient clipping | Global norm 0.5 | [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); [Engstrom et al.](https://arxiv.org/abs/2005.12729) |
| γ, λ | γ=1.0 (or 0.99); λ=0.95, sweep 0.9-1.0 | [GAE](https://arxiv.org/abs/1506.02438); [PPO](https://arxiv.org/abs/1707.06347) |
| Critic | Separate network, or a shared trunk with c1 reported; no value-loss clipping | [Andrychowicz et al.](https://arxiv.org/abs/2006.05990) |
| Advantage normalisation | Per minibatch, with ε in the denominator (sparse binary rewards make std≈0 common) | [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |
| Entropy | c2=0.01, computed over valid candidates only | [A3C](https://arxiv.org/abs/1602.01783); [Huang & Ontañón](https://arxiv.org/abs/2006.14171) |
| Policy output layer | Small initialisation (gain 0.01, or 100× smaller weights) | [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/); [Andrychowicz et al.](https://arxiv.org/abs/2006.05990) |
| Diagnostics | Approximate KL, clip fraction, entropy, hop count | [37 details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |

### Masked pointer policies give valid gradients only if the mask is replayed

The policy head is Pointer Networks' scoring function, used as a policy over the current candidate set ([Pointer Networks](https://arxiv.org/abs/1506.03134); [Chandak et al.](https://arxiv.org/abs/1906.01772)):

$$\pi_\theta(a|s_t,\mathcal A_t)=\frac{\exp(u_\theta(s_t,e_a))\,\mathbb 1[a\in\mathcal A_t]}{\sum_{a'\in\mathcal A_t}\exp(u_\theta(s_t,e_{a'}))},\qquad u_\theta(s,e)=v^\top\tanh(W_1e+W_2h_s)$$

- **Pointer score:** $u_\theta$ is the Pointer Networks scoring function.
- **Masked softmax:** this exact masked softmax is the parameterisation Chandak et al. use for MDPs whose available actions vary by state. They also show that Q-learning can diverge in that setting with function approximation, which motivates policy gradients.
- **STOP:** following M-Walk, STOP gets its own score computed from a pooled summary of the candidate set ([M-Walk](https://arxiv.org/abs/1802.04394)).

Huang and Ontañón prove that masking is a "state-dependent differentiable function" of the logits, so the policy-gradient theorem still holds, and masked logits receive exactly zero gradient ([Huang & Ontañón](https://arxiv.org/abs/2006.14171)). Their experiments make three practical points:

- **Penalties do not scale.** Agents trained with an invalid-action penalty of −0.01 spent 3.43% of training just finding the first reward on 10×10 maps, against 0.06% with masking. A penalty of −1 performed consistently worst.
- **Naive masking breaks PPO.** Sampling from the masked distribution but computing the update with unmasked probabilities makes the KL "significantly higher than that of any other experiments".
- **Use a large finite value.** Their worked example masks with −10⁸.

Three implementation rules follow:

- **Store the mask in the rollout buffer.** The mask covers valid, not-yet-visited candidates plus STOP. Re-apply it when recomputing log-probabilities for the ratio and for the entropy.
- **Use a large finite negative logit rather than −∞.** Otherwise the entropy term produces NaN.
- **Do not reward or penalise invalid actions under masking.** An invalid-action penalty is inert when masks are in place, and belongs only in a no-mask ablation.

Two further devices come from the graph-walking literature ([Lin et al.](https://arxiv.org/abs/1808.10568)):

- **Action dropout** is a mask-based exploration device that composes with validity masks.
- **Fan-out caps** suit hub nodes such as columns and tables.

Because the number of candidates varies by state, a fixed entropy coefficient pushes hardest where there are many candidates. Normalising the entropy by log|𝒜_t| is a reasonable choice that no source has tested.

### Recall shaping preserves the optimal policy only in potential form

Ng, Harada and Russell prove that a shaping reward F leaves every optimal policy unchanged if and only if $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ for some state potential Φ. For undiscounted episodic tasks, the absorbing terminal state must have a fixed potential, taken as 0. They also trace known shaping "bugs", where agents cycle to farm reward, to rewards that are not potential-based ([Ng et al.](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)). SRN applied the same principle to KBQA ([SRN](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc)).

For this project, the theorem has four consequences:

- **The invariant form.** Let Φ(s_t) be the recall of the evidence visited so far against the gold evidence, which is fixed per question. Paying $\gamma\Phi(s_{t+1})-\Phi(s_t)$ per hop, with γ=1 and a final payment of −Φ(s_T) on the transition into the absorbing state, telescopes to a constant. The optimal policy is therefore unchanged, and shaping only moves credit to earlier hops.
- **The non-invariant forms.** Two common implementations change the objective: paying Δrecall per hop without the terminal correction, and adding a terminal recall bonus. Both reward recall for its own sake. That is legitimate if faithfulness is a goal, but it is a change to the objective, not shaping.
- **Cost and penalty terms.** The per-hop cost is not potential-based for episodes of varying length; it deliberately shifts the optimum toward shorter chains. An invalid-action penalty is not potential-based either.
- **Discounting.** If γ<1, the shaping must use exactly γΦ(s′). Otherwise a stray term (1−γ)Φ(s′) creeps in.

**One consequence is easy to miss.** With potential-based shaping, every episode in a group receives the same shaping total, −Φ(s_init). GRPO and RLOO subtract the group mean, so they see no shaping signal at all; only PPO, through per-step GAE, benefits. A comparison of PPO and critic-free methods "with identical rewards" therefore conflates reward design with credit assignment, unless both reward forms are run.

### Critic-free baselines need per-question baselines and sum-over-hops losses

The critic-free alternatives all estimate a baseline from other rollouts of the same question:

- **GRPO** drops the critic. It normalises each sampled output's reward within a group, $(r_i-\mathrm{mean}(\mathbf r))/\mathrm{std}(\mathbf r)$, averages the clipped loss over tokens with a 1/|o_i| factor, and adds a KL penalty of 0.04 against a reference model ([DeepSeekMath](https://arxiv.org/abs/2402.03300)).
- **RLOO** uses the leave-one-out mean as an unbiased baseline ([RLOO](https://aclanthology.org/2024.acl-long.662/)). Its estimator equals $\frac{k}{k-1}(R_i-\bar R)$ algebraically, so it matches mean-centred GRPO up to scale, apart from clipping and the KL term.
- **Dr. GRPO** shows that GRPO's length normalisation gives "greater gradient updates for shorter responses" among correct outputs and penalises long incorrect ones less, and that its std normalisation up-weights questions with low reward variance. It removes both normalisations ([Dr. GRPO](https://arxiv.org/abs/2503.20783)).
- **REINFORCE** is the baseline-subtracted likelihood-ratio update underlying all three ([Williams](https://link.springer.com/article/10.1007/BF00992696)).

Mapped onto this project, an output is an episode and a token is a hop. Three consequences follow:

- **Length bias at hop level.** Averaging each episode's loss over its hops recreates Dr. GRPO's length bias, which would favour long wrong chains and fight the per-hop cost. Sum over hops, or divide by the maximum hop count instead.
- **Degenerate groups.** With binary correctness and a small group size, groups where every rollout is right, or every rollout is wrong, give zero advantage. This is IGPO's "advantage collapse" ([IGPO](https://arxiv.org/html/2510.14967)). Report the fraction of such groups.
- **Avoid a batch-wide baseline.** Das et al. found that a minibatch-average reward baseline "significantly degrades the final performance" because questions differ in difficulty ([Das et al.](https://arxiv.org/pdf/1905.05733)). The REINFORCE comparison should use a per-question baseline or a learned critic.

### Warm starts are mandatory, and the critic starts cold

The canonical supervised-then-RL pipeline is AlphaGo ([AlphaGo](https://www.nature.com/articles/nature16961)):

- the supervised policy reached 57.0% move accuracy;
- the RL policy was initialised to the same weights (ρ=σ) and improved by policy gradients on a terminal ±1 reward;
- the RL policy then "won more than 80% of games against the SL policy network" ([DeepMind PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)).

DAgger explains why behaviour cloning alone is not enough. A cloned policy that errs with probability ε "can make as many as T²ε mistakes" once its own errors move it off the expert's state distribution ([DAgger](https://arxiv.org/abs/1011.0686)). With 2-5 hops, T² is small, but one wrong hop still changes the candidate set.

The retrieval evidence agrees:

- DeepPath "will show very poor convergence properties" without imitation ([DeepPath](https://arxiv.org/abs/1707.06690)).
- SoG-R1 reaches CWQ 88.7 with SFT plus RL, but only 63.0 with RL alone ([SoG-R1](https://arxiv.org/abs/2607.18481)).
- José et al., the one RL attempt on table-text QA, used no warm start at all.

Behaviour cloning on HybridQA should maximise the log of the summed probability over the acceptable answer-node set. STOP should be labelled positive only after the gold chain is complete, following Asai et al.'s end-of-evidence scheme ([Asai et al.](https://arxiv.org/pdf/1911.10470)).

After cloning, two things need watching. First, the critic is untrained, so a short critic-only warm-up before policy updates is sensible; no source tests this. Second, the cloned policy's entropy may be too low for PPO to explore, so watch the entropy curve and c2.

### Five seeds and bootstrap intervals are the minimum credible report

Henderson et al. showed that two sets of five seeds for the same algorithm and hyperparameters can differ significantly (t=−9.0916, p=0.0016). They decline to fix a seed count, recommending significance testing and power analysis instead, and baselines tuned until they "match the original reported performance" ([Henderson et al.](https://arxiv.org/abs/1709.06560)). Agarwal et al. accept that "3-10 runs are prevalent" and recommend three things in place of point estimates ([Agarwal et al. 2021](https://arxiv.org/abs/2108.13264)):

- **Stratified bootstrap confidence intervals.**
- **The interquartile mean (IQM)**, which "discards the bottom and top 25% of the runs and calculates the mean score of the remaining 50% runs".
- **Probability of improvement.**

The protocol that follows:

1. Use at least five seeds per method, preferably ten.
2. Report the IQM and mean of dev EM and evidence recall, with 95% bootstrap confidence intervals.
3. Run a paired bootstrap over questions, to separate seed variance from question sampling.
4. Show learning curves with confidence bands, not the best seed.
5. Give every method the same tuning budget.

## Conclusion

The literature turns this project from "a new architecture" into a test of a hypothesis that already has a negative precedent. José et al. showed that PPO with coarse actions and a terminal reward collapses to the best fixed policy. The project's three changes — entity-level actions, dense evidence reward and a behaviour-cloning warm start — are exactly the levers that paper left unused. So the most valuable result is an ablation that says which lever matters, measured against fixed and greedy baselines with confidence intervals. Even a null result would extend the record, because no published work has tried any of the three on tables.

The deepest design decision is what "evidence recall" is supposed to do:

- **As potential-based shaping,** it can only speed learning. It cannot change what PPO converges to, and it is invisible to GRPO and RLOO, so a "same reward" comparison of PPO with critic-free methods would quietly favour PPO's credit assignment.
- **As a terminal bonus,** it really does change the objective. It is the only form under which faithfulness is optimised. But it then trains the policy to imitate a string-matching tracer that is wrong about 15% of the time.

The project should run both forms and say which claim each supports.

The "drop-in" claim is cheapest to establish on HYBRIDER- and ODYSSEY-style hybrid graphs, where selection errors are already quantified and the reader can be held fixed. Integration into the single-table frameworks (Chain-of-Table, TableRAG, Dater, Binder) is a separate experiment, with gold evidence derived from SQUALL.

## Appendix A. Consolidated BibTeX

Entries are copied from the verified research notes and de-duplicated. Where the notes held two versions of an entry, the ACL Anthology or OpenReview export was kept. Lines beginning with `% NOTE` flag metadata uncertainty for the entry directly below.

```bibtex
% ===================== RL for table QA and evidence selection =====================

% NOTE: Venue (BRACIS 2024) from the arXiv comment; Springer LNCS/LNAI volume, pages and DOI not verified. First author is Marcos M. Jose (not "Ribeiro et al.").
@inproceedings{jose2024qatexttablesdrl,
  title     = {Question Answering with Texts and Tables through Deep Reinforcement Learning},
  author    = {Jos{\'e}, Marcos M. and Ca{\c{c}}{\~a}o, Fl{\'a}vio N. and Ribeiro, Maria F. and Cheang, Rafael M. and Pirozelli, Paulo and Cozman, Fabio G.},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS 2024)},
  year      = {2024},
  note      = {arXiv:2407.04858}
}

% NOTE: arXiv preprint (5-page short paper); no peer-reviewed venue found.
@article{lee2024learningtoreduce,
  title   = {Learning to Reduce: Optimal Representations of Structured Data in Prompting Large Language Models},
  author  = {Lee, Younghun and Kim, Sungchul and Yu, Tong and Rossi, Ryan A. and Chen, Xiang},
  journal = {arXiv preprint arXiv:2402.14195},
  year    = {2024}
}

% NOTE: arXiv preprint, no venue as of Oct 2026 check. One of three distinct papers titled "Table-R1"/"Table-r1"; this is the region-based one (Wu et al.).
@article{wu2025tabler1region,
  title   = {Table-R1: Region-based Reinforcement Learning for Table Understanding},
  author  = {Wu, Zhenhe and Yang, Jian and He, Zhongjiang and Pan, Changzai and Zhang, Jie and Liu, Jiaheng and Wu, Xianjie and Zhao, Yu and Song, Shuangyong and Li, Yongxiang and Li, Zhoujun and Li, Xueling},
  journal = {arXiv preprint arXiv:2505.12415},
  year    = {2025}
}

% NOTE: Title mismatch: the arXiv version (2505.23621) omits the final word "Tasks"; the Anthology title is used here.
@inproceedings{yang-etal-2025-table,
  title     = {Table-R1: Inference-Time Scaling for Table Reasoning Tasks},
  author    = {Yang, Zheyuan and Chen, Lyuhao and Cohan, Arman and Zhao, Yilun},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages     = {20605--20624},
  year      = {2025},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.emnlp-main.1040},
  url       = {https://aclanthology.org/2025.emnlp-main.1040/},
  note      = {arXiv:2505.23621 (arXiv title omits ``Tasks'')}
}

% NOTE: arXiv preprint self-described as "work in progress"; no venue found.
@article{lei2025reasoningtable,
  title   = {Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning},
  author  = {Lei, Fangyu and Meng, Jinxiang and Huang, Yiming and Chen, Tinghong and Zhang, Yun and He, Shizhu and Zhao, Jun and Liu, Kang},
  journal = {arXiv preprint arXiv:2506.01710},
  year    = {2025}
}

% NOTE: Retitled on arXiv (v1 was "Fortune: ..."); no venue listed on Semantic Scholar; treat as preprint.
@article{cao2025formular1,
  title   = {Formula-R1: Incentivizing LLM Reasoning over Complex Tables with Numerical Computation via Formula-Driven Reinforcement Learning},
  author  = {Cao, Lang and Xu, Jingxian and Liu, Hanbing and Wang, Jinyu and Zhou, Mengyu and Dong, Haoyu and Han, Shi and Zhang, Dongmei},
  journal = {arXiv preprint arXiv:2505.23667},
  year    = {2025},
  note    = {v1 titled ``Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models''}
}

% NOTE: WSDM '26 from the paper's ACM reference block; DOI and page numbers not verified.
@inproceedings{jiang2026tablemind,
  title     = {TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning},
  author    = {Jiang, Chuang and Cheng, Mingyue and Tao, Xiaoyu and Mao, Qingyang and Ouyang, Jie and Liu, Qi},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining (WSDM '26)},
  year      = {2026},
  note      = {arXiv:2509.06278}
}

% NOTE: arXiv preprint; no venue found.
@article{qiu2025opentabler1,
  title   = {OpenTable-R1: A Reinforcement Learning Augmented Tool Agent for Open-Domain Table Question Answering},
  author  = {Qiu, Zipeng},
  journal = {arXiv preprint arXiv:2507.03018},
  year    = {2025}
}

% NOTE: arXiv preprint; no venue found.
@article{pourreza2025reasoningsql,
  title   = {Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL},
  author  = {Pourreza, Mohammadreza and Talaei, Shayan and Sun, Ruoxi and Wan, Xingchen and Li, Hailong and Mirhoseini, Azalia and Saberi, Amin and Ar{\i}k, Sercan {\"O}.},
  journal = {arXiv preprint arXiv:2503.23157},
  year    = {2025}
}

% NOTE: arXiv preprint; only the abstract-level description was verified.
@article{kwok2026retab,
  title   = {Enhancing Table Reasoning with Deterministic Table-State Rewards},
  author  = {Kwok, Tung Sum Thomas and Wang, Xinyu and He, Hengzhi and Lin, Xiaofeng and Lu, Peng and Ma, Liheng and Wang, Chunhe and Mak, Chun Ho and Luo, Yuyu and Wu, Ying Nian and Ding, Lei and Cheng, Guang},
  journal = {arXiv preprint arXiv:2601.22530},
  year    = {2026}
}

% NOTE: NeurIPS 2018 per the arXiv comment; proceedings page numbers not verified.
@inproceedings{liang2018mapo,
  title     = {Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing},
  author    = {Liang, Chen and Norouzi, Mohammad and Berant, Jonathan and Le, Quoc and Lao, Ni},
  booktitle = {Advances in Neural Information Processing Systems 31 (NeurIPS 2018)},
  year      = {2018},
  note      = {arXiv:1807.02322}
}

@inproceedings{agarwal2019merl,
  title     = {Learning to Generalize from Sparse and Underspecified Rewards},
  author    = {Agarwal, Rishabh and Liang, Chen and Schuurmans, Dale and Norouzi, Mohammad},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {130--140},
  year      = {2019},
  note      = {arXiv:1902.07198}
}

% ===================== Table-text QA: datasets, pipelines, graph frameworks =====================

@inproceedings{chen-etal-2020-hybridqa,
    title = "{H}ybrid{QA}: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data",
    author = "Chen, Wenhu and Zha, Hanwen and Chen, Zhiyu and Xiong, Wenhan and Wang, Hong and Wang, William Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov, year = "2020", address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.91/",
    doi = "10.18653/v1/2020.findings-emnlp.91",
    pages = "1026--1036"
}

@inproceedings{chen2021open,
    title = {Open Question Answering over Tables and Text},
    author = {Wenhu Chen and Ming-Wei Chang and Eva Schlinger and William Yang Wang and William W. Cohen},
    booktitle = {International Conference on Learning Representations},
    year = {2021},
    url = {https://openreview.net/forum?id=MmCRswl1UYl}
}

@inproceedings{wang-etal-2022-muger2,
    title = "{M}u{GER}2: Multi-Granularity Evidence Retrieval and Reasoning for Hybrid Question Answering",
    author = "Wang, Yingyao and Bao, Junwei and Duan, Chaoqun and Wu, Youzheng and He, Xiaodong and Zhao, Tiejun",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec, year = "2022", address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.498/",
    doi = "10.18653/v1/2022.findings-emnlp.498",
    pages = "6687--6697"
}

% NOTE: arXiv version (2112.07337) was originally titled "Multi-Instance Training for Question Answering Across Table and Linked Text".
@inproceedings{kumar-etal-2023-multi,
    title = "Multi-Row, Multi-Span Distant Supervision For {T}able+{T}ext Question Answering",
    author = "Kumar, Vishwajeet and Gupta, Yash and Chemmengath, Saneem and Sen, Jaydeep and Chakrabarti, Soumen and Bharadwaj, Samarth and Pan, Feifei",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul, year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.449/",
    doi = "10.18653/v1/2023.acl-long.449",
    pages = "8080--8094"
}

% NOTE: Anthology lists this as ACL 2023 Volume 2 (Short Papers); one research note had recorded only "ACL 2023" without the track.
@inproceedings{lei-etal-2023-s3hqa,
    title = "{S}3{HQA}: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering",
    author = "Lei, Fangyu and Li, Xiang and Wei, Yifan and He, Shizhu and Huang, Yiming and Zhao, Jun and Liu, Kang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul, year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.147/",
    doi = "10.18653/v1/2023.acl-short.147",
    pages = "1731--1740"
}

% NOTE: Published title differs from the arXiv title (2305.14682: "A Table-alignment-based Cell-selection and Reasoning Model for Hybrid Question-Answering"); numbers identical in both.
@inproceedings{wu-etal-2023-tacr,
    title = "{TACR}: A Table Alignment-based Cell Selection Method for {H}ybrid{QA}",
    author = {Wu, Jian and Xu, Yicheng and Gao, Yan and Lou, Jian-Guang and Karlsson, B{\"o}rje F. and Okumura, Manabu},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul, year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.409/",
    doi = "10.18653/v1/2023.findings-acl.409",
    pages = "6535--6549"
}

% NOTE: Listed on OpenReview as an ICLR 2022 submission that was not accepted; cite as arXiv preprint.
@article{sun2021iterative,
    title = {Iterative Hierarchical Attention for Answering Complex Questions over Long Documents},
    author = {Sun, Haitian and Cohen, William W. and Salakhutdinov, Ruslan},
    journal = {arXiv preprint arXiv:2106.00200},
    year = {2021},
    url = {https://arxiv.org/abs/2106.00200}
}

@inproceedings{ma2022open,
  title     = {Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge},
  author    = {Ma, Kaixin and Cheng, Hao and Liu, Xiaodong and Nyberg, Eric and Gao, Jianfeng},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  pages     = {5360--5374},
  year      = {2022}
}

@inproceedings{ma2023chain,
  title     = {Chain-of-Skills: A Configurable Model for Open-domain Question Answering},
  author    = {Ma, Kaixin and Cheng, Hao and Zhang, Yu and Liu, Xiaodong and Nyberg, Eric and Gao, Jianfeng},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {1599--1618},
  year      = {2023}
}

@inproceedings{shi-etal-2024-exploring,
    title = "Exploring Hybrid Question Answering via Program-based Prompting",
    author = "Shi, Qi and Cui, Han and Wang, Haofeng and Zhu, Qingfu and Che, Wanxiang and Liu, Ting",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug, year = "2024", address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.594/",
    doi = "10.18653/v1/2024.acl-long.594",
    pages = "11035--11046"
}

% NOTE: Author order differs between sources: the Anthology export (used here) gives Agarwal, Devaguptapu, S; an arXiv-derived record gives Agarwal, S, Devaguptapu. "ODYSSEY" is the method name inside this paper.
@inproceedings{agarwal-etal-2025-hybrid,
    title = "Hybrid Graphs for Table-and-Text based Question Answering using {LLM}s",
    author = "Agarwal, Ankush and Devaguptapu, Chaitanya and S, Ganesh",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr, year = "2025", address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.39/",
    doi = "10.18653/v1/2025.naacl-long.39",
    pages = "858--875"
}

% NOTE: arXiv preprint (Jan 2026); no venue found.
@article{sharafath2026n2ngqa,
  title   = {{N2N-GQA}: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using {LLMs}},
  author  = {Sharafath, Mohamed and Annamalai, Aravindh and Murugan, Ganesh and Venugopalan, Aravindakumar},
  journal = {arXiv preprint arXiv:2601.06603},
  year    = {2026}
}

% ===================== LLM table-reasoning frameworks (integration targets) =====================

@inproceedings{wang2024chainoftable,
  title     = {Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding},
  author    = {Wang, Zilong and Zhang, Hao and Li, Chun-Liang and Eisenschlos, Julian Martin and Perot, Vincent and Wang, Zifeng and Miculicich, Lesly and Fujii, Yasuhisa and Shang, Jingbo and Lee, Chen-Yu and Pfister, Tomas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

% NOTE: NeurIPS 2024 proceedings pages not recorded. Not to be confused with the unrelated 2025 paper of the same name (Yu, Jian & Chen, arXiv 2506.10380).
@inproceedings{chen2024tablerag,
  title     = {{TableRAG}: Million-Token Table Understanding with Language Models},
  author    = {Chen, Si-An and Miculicich, Lesly and Eisenschlos, Julian Martin and Wang, Zifeng and Wang, Zilong and Chen, Yanfei and Fujii, Yasuhisa and Lin, Hsuan-Tien and Lee, Chen-Yu and Pfister, Tomas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

% NOTE: The SIGIR 2023 published title may differ slightly from the arXiv title ("...: Decompose Evidence and Questions for Table-based Reasoning"); unverified.
@inproceedings{ye2023dater,
  title     = {Large Language Models are Versatile Decomposers: Decomposing Evidence and Questions for Table-based Reasoning},
  author    = {Ye, Yunhu and Hui, Binyuan and Yang, Min and Li, Binhua and Huang, Fei and Li, Yongbin},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year      = {2023}
}

@inproceedings{cheng2023binder,
  title     = {Binding Language Models in Symbolic Languages},
  author    = {Cheng, Zhoujun and Xie, Tianbao and Shi, Peng and Li, Chengzu and Nadkarni, Rahul and Hu, Yushi and Xiong, Caiming and Radev, Dragomir and Ostendorf, Mari and Zettlemoyer, Luke and Smith, Noah A. and Yu, Tao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

% ===================== Multi-table retrieval and QA =====================

% NOTE: No arXiv version; the paper's official data link (anonymous.4open.science) has expired (HTTP 410).
@inproceedings{wu2025mmqa,
    title = {{MMQA}: Evaluating {LLM}s with Multi-Table Multi-Hop Complex Questions},
    author = {Jian Wu and Linyi Yang and Dongyuan Li and Yuliang Ji and Manabu Okumura and Yue Zhang},
    booktitle = {The Thirteenth International Conference on Learning Representations},
    year = {2025},
    url = {https://openreview.net/forum?id=GGlpykXDCa}
}

@inproceedings{zhang2025murre,
  title     = {{MURRE}: Multi-Hop Table Retrieval with Removal for Open-Domain Text-to-{SQL}},
  author    = {Zhang, Xuanliang and Wang, Dingzirui and Dou, Longxu and Zhu, Qingfu and Che, Wanxiang},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics (COLING)},
  year      = {2025}
}

@inproceedings{chen2024jar,
  title     = {Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval},
  author    = {Chen, Peter Baile and Zhang, Yi and Roth, Dan},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

% NOTE: Workshop paper (AI for Tabular Data Workshop, EurIPS 2025); cited via arXiv.
@article{boutaleb2025exploring,
  title   = {Exploring Multi-Table Retrieval Through Iterative Search},
  author  = {Boutaleb, Allaa and Amann, Bernd and Angarita, Rafael and Naacke, Hubert},
  journal = {arXiv preprint arXiv:2511.13418},
  year    = {2025},
  note    = {AI for Tabular Data Workshop, EurIPS 2025}
}

% NOTE: EMNLP 2026 per the arXiv comment (arXiv 2608.30291); main vs Findings track not stated.
@inproceedings{ho2026pearl,
  title     = {{PEARL}: Front-Loading Relational Chains for Multi-Hop Table Retrieval},
  author    = {Ho, Subeen and Kang, Hyeongu and Kang, SeongKu and Yoon, Susik},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}

% NOTE: arXiv preprint; venue unknown.
@article{agarwal2025rear,
  title   = {{REaR}: Retrieve, Expand and Refine for Effective Multitable Retrieval},
  author  = {Agarwal, Rishita and Singhal, Himanshu and Chen, Peter Baile and Roy Choudhury, Manan and Roth, Dan and Gupta, Vivek},
  journal = {arXiv preprint arXiv:2511.00805},
  year    = {2025}
}

% ===================== RL evidence selection and search agents (text) =====================

% NOTE: EMNLP 2025 per the arXiv comment ("camera-ready") and Semantic Scholar pages; main vs Findings track not confirmed.
@inproceedings{jiang2025s3,
  title     = {s3: You Don't Need That Much Data to Train a Search Agent via {RL}},
  author    = {Jiang, Pengcheng and Xu, Xueqiang and Lin, Jiacheng and Xiao, Jinfeng and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {21599--21617},
  year      = {2025},
  note      = {arXiv:2505.14146}
}

% NOTE: EMNLP 2025 main; Anthology ID 2025.emnlp-main.1106 seen via search index and Semantic Scholar, not the Anthology bib export.
@inproceedings{wang2025stepsearch,
  title     = {{StepSearch}: Igniting {LLMs} Search Ability via Step-Wise Proximal Policy Optimization},
  author    = {Wang, Ziliang and Zheng, Xuhui and An, Kang and Ouyang, Cijun and Cai, Jialu and Wang, Yuhang and Wu, Yichao},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  note      = {arXiv:2505.15107}
}

% NOTE: Venue COLM 2025 per OpenReview. Reported improvement over RAG baselines is internally inconsistent across arXiv versions (24% vs 41% for 7B).
@inproceedings{jin2025searchr1,
  title     = {Search-{R1}: Training {LLMs} to Reason and Leverage Search Engines with Reinforcement Learning},
  author    = {Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  note      = {arXiv:2503.09516}
}

% NOTE: arXiv preprint; no venue found.
@article{jin2025empirical,
  title   = {An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved {LLM} Agents},
  author  = {Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O. and Han, Jiawei},
  journal = {arXiv preprint arXiv:2505.15117},
  year    = {2025}
}

% NOTE: NeurIPS 2025 per the arXiv comment only.
@inproceedings{chen2025mmoarag,
  title     = {Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning},
  author    = {Chen, Yiqun and Yan, Lingyong and Sun, Weiwei and Ma, Xinyu and Zhang, Yi and Wang, Shuaiqiang and Yin, Dawei and Yang, Yiming and Mao, Jiaxin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2501.15228}
}

% NOTE: ICLR 2026 per the arXiv comment ("Accepted by ICLR 2026").
@inproceedings{wang2026igpo,
  title     = {Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author    = {Wang, Guoqing and Dai, Sunhao and Ye, Guangze and Gan, Zeyu and Yao, Wei and Deng, Yong and Wu, Xiaofeng and Ying, Zhenzhe},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2510.14967}
}

% NOTE: EMNLP 2025 Main per the arXiv comment.
@inproceedings{wu2025searchwisely,
  title     = {Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty},
  author    = {Wu, Peilin and Zhang, Mian and Zhang, Xinlu and Du, Xinya and Chen, Zhiyu Zoey},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  note      = {arXiv:2505.17281}
}

@inproceedings{asai2020learning,
  title     = {Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering},
  author    = {Asai, Akari and Hashimoto, Kazuma and Hajishirzi, Hannaneh and Socher, Richard and Xiong, Caiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  note      = {arXiv:1911.10470}
}

@inproceedings{das2019multistep,
  title     = {Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering},
  author    = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and McCallum, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  note      = {arXiv:1905.05733}
}

% ===================== RL graph walking and RL-based KBQA =====================

% NOTE: ICLR 2018 venue confirmed from the arXiv PDF header; the OpenReview forum page could not be loaded, so the arXiv URL is used.
@inproceedings{das2018minerva,
  title     = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author    = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1711.05851}
}

@inproceedings{lin2018multihop,
  title     = {Multi-Hop Knowledge Graph Reasoning with Reward Shaping},
  author    = {Lin, Xi Victoria and Socher, Richard and Xiong, Caiming},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages     = {3243--3253},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/D18-1362},
  url       = {https://aclanthology.org/D18-1362/}
}

% NOTE: Page numbers in the NeurIPS volume not verified (the ACM DL record gives pp. 6787-6798).
@inproceedings{shen2018mwalk,
  title     = {{M-Walk}: Learning to Walk over Graphs using {Monte Carlo} Tree Search},
  author    = {Shen, Yelong and Chen, Jianshu and Huang, Po-Sen and Guo, Yuqing and Gao, Jianfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018},
  url       = {http://papers.neurips.cc/paper/7912-m-walk-learning-to-walk-over-graphs-using-monte-carlo-tree-search.pdf}
}

% NOTE: No arXiv version; full text paywalled (HTTP 403). Author spelling from Semantic Scholar and a citing paper's reference list; method details beyond the abstract unverified.
@inproceedings{qiu2020srn,
  title     = {Stepwise Reasoning for Multi-Relation Question Answering over Knowledge Graph with Weak Supervision},
  author    = {Qiu, Yunqi and Wang, Yuanzhuo and Jin, Xiaolong and Zhang, Kun},
  booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM '20)},
  pages     = {474--482},
  year      = {2020},
  publisher = {ACM},
  doi       = {10.1145/3336191.3371812}
}

@inproceedings{xiong2017deeppath,
  title     = {{DeepPath}: A Reinforcement Learning Method for Knowledge Graph Reasoning},
  author    = {Xiong, Wenhan and Hoang, Thien and Wang, William Yang},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages     = {564--573},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/D17-1060},
  url       = {https://aclanthology.org/D17-1060/}
}

% NOTE: arXiv preprint (v5, May 2026); no venue in the arXiv comments. "KG-R1" is the system name.
@article{lin2025kgr1,
  title   = {Efficient and Transferable Agentic Knowledge Graph {RAG} via Reinforcement Learning},
  author  = {Lin, Junhong and Liu, Shicheng and Song, Jinyeop and Wang, Song and Shun, Julian and Zhu, Yada},
  journal = {arXiv preprint arXiv:2509.26383},
  year    = {2025}
}

% NOTE: arXiv preprint (Jul 2026); no venue. The paper's table and prose give slightly different SFT-only and RL-only numbers.
@article{sun2026sogr1,
  title   = {{Search-on-Graph-R1}: Training Large Language Models to Search Knowledge Graphs with Reinforcement Learning},
  author  = {Sun, Jia Ao and Yu, Hao and Mo, Fengran and Su, Zhan and Hui, Yuchen and Liu, Bang and Nie, Jian-Yun},
  journal = {arXiv preprint arXiv:2607.18481},
  year    = {2026}
}

% NOTE: WWW 2026 per the arXiv comment; proceedings pages and DOI not found.
@inproceedings{yu2026graphragr1,
  title     = {{GraphRAG-R1}: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning},
  author    = {Yu, Chuanyue and Zhao, Kuo and Li, Yuhan and Chang, Heng and Feng, Mingjian and Jiang, Xiangzhe and Sun, Yufei and Li, Jia and Zhang, Yuzhi and Li, Jianxin and Zhang, Ziwei},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  note      = {arXiv:2507.23581}
}

@inproceedings{luo2026graphr1,
  title     = {{Graph-R1}: Towards Agentic {GraphRAG} Framework via End-to-end Reinforcement Learning},
  author    = {Luo, Haoran and E, Haihong and Chen, Guanting and Lin, Qika and Guo, Yikai and Xu, Fangzhi and Kuang, Zemin and Song, Meina and Wu, Xiaobao and Zhu, Yifan and Tuan, Luu Anh},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  note      = {arXiv:2507.21892}
}

% ===================== Optimisation foundations =====================

% NOTE: Springer metadata gives issue 3 (often cited as 8(3-4)); some details of the equations verified only via a third-party transcription.
@article{williams1992simple,
  title     = {Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning},
  author    = {Williams, Ronald J.},
  journal   = {Machine Learning},
  volume    = {8},
  number    = {3},
  pages     = {229--256},
  year      = {1992},
  publisher = {Kluwer Academic Publishers},
  doi       = {10.1007/BF00992696}
}

@inproceedings{schulman2016gae,
  title     = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  author    = {Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael I. and Abbeel, Pieter},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2016},
  eprint    = {1506.02438},
  archivePrefix = {arXiv}
}

@inproceedings{mnih2016asynchronous,
  title     = {Asynchronous Methods for Deep Reinforcement Learning},
  author    = {Mnih, Volodymyr and Badia, Adri{\`a} Puigdom{\`e}nech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P. and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {48},
  pages     = {1928--1937},
  year      = {2016},
  publisher = {PMLR}
}

% NOTE: arXiv preprint only; no peer-reviewed venue in the arXiv record.
@article{schulman2017ppo,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017}
}

% NOTE: Proceedings title differs from the arXiv title (2005.12729); proceedings title used.
@inproceedings{engstrom2020implementation,
  title     = {Implementation Matters in Deep {RL}: A Case Study on {PPO} and {TRPO}},
  author    = {Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Janoos, Firdaus and Rudolph, Larry and Madry, Aleksander},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  note      = {arXiv:2005.12729, titled ``Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO''}
}

% NOTE: Proceedings title differs from the arXiv title (2006.05990: "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"); venue verified via listings, not the OpenReview forum page itself.
@inproceedings{andrychowicz2021matters,
  title     = {What Matters for On-Policy Deep Actor-Critic Methods? {A} Large-Scale Study},
  author    = {Andrychowicz, Marcin and Raichuk, Anton and Sta{\'n}czyk, Piotr and Orsini, Manu and Girgin, Sertan and Marinier, Rapha{\"e}l and Hussenot, L{\'e}onard and Geist, Matthieu and Pietquin, Olivier and Michalski, Marcin and Gelly, Sylvain and Bachem, Olivier},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=nIAxjsniDzg}
}

% NOTE: Non-archival ICLR Blog Track post; cite only alongside the archival implementation studies.
@inproceedings{huang2022the37implementation,
  author    = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  title     = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year      = {2022},
  url       = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}

% NOTE: The FLAIRS article page lists no page range.
@article{huang2022closer,
  title   = {A Closer Look at Invalid Action Masking in Policy Gradient Algorithms},
  author  = {Huang, Shengyi and Onta{\~n}{\'o}n, Santiago},
  journal = {The International FLAIRS Conference Proceedings},
  volume  = {35},
  year    = {2022},
  doi     = {10.32473/flairs.v35i.130584}
}

@inproceedings{vinyals2015pointer,
  title     = {Pointer Networks},
  author    = {Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {28},
  pages     = {2692--2700},
  year      = {2015}
}

@inproceedings{chandak2020sas,
  title     = {Reinforcement Learning When All Actions Are Not Always Available},
  author    = {Chandak, Yash and Theocharous, Georgios and Metevier, Blossom and Thomas, Philip S.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {34},
  number    = {04},
  pages     = {3381--3388},
  year      = {2020},
  doi       = {10.1609/aaai.v34i04.5740}
}

@inproceedings{ng1999policy,
  title     = {Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping},
  author    = {Ng, Andrew Y. and Harada, Daishi and Russell, Stuart J.},
  booktitle = {Proceedings of the Sixteenth International Conference on Machine Learning (ICML)},
  pages     = {278--287},
  year      = {1999},
  publisher = {Morgan Kaufmann},
  address   = {San Francisco, CA, USA}
}

% NOTE: arXiv preprint only (introduces GRPO).
@article{shao2024deepseekmath,
  title   = {{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author  = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal = {arXiv preprint arXiv:2402.03300},
  year    = {2024}
}

@inproceedings{ahmadian-etal-2024-back,
  title     = {Back to Basics: Revisiting {REINFORCE}-Style Optimization for Learning from Human Feedback in {LLM}s},
  author    = {Ahmadian, Arash and Cremer, Chris and Gall{\'e}, Matthias and Fadaee, Marzieh and Kreutzer, Julia and Pietquin, Olivier and {\"U}st{\"u}n, Ahmet and Hooker, Sara},
  editor    = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = aug,
  year      = {2024},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.acl-long.662/},
  doi       = {10.18653/v1/2024.acl-long.662},
  pages     = {12248--12267}
}

% NOTE: COLM 2025 venue from the PDF header and an OpenReview listing; author list taken from arXiv, not checked separately against the COLM proceedings.
@inproceedings{liu2025understanding,
  title     = {Understanding {R1-Zero}-Like Training: A Critical Perspective},
  author    = {Liu, Zichen and Chen, Changyu and Li, Wenjun and Qi, Penghui and Pang, Tianyu and Du, Chao and Lee, Wee Sun and Lin, Min},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=5PAF7PAY2Y}
}

@article{silver2016mastering,
  title   = {Mastering the Game of {Go} with Deep Neural Networks and Tree Search},
  author  = {Silver, David and Huang, Aja and Maddison, Chris J. and Guez, Arthur and Sifre, Laurent and van den Driessche, George and Schrittwieser, Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot, Marc and Dieleman, Sander and Grewe, Dominik and Nham, John and Kalchbrenner, Nal and Sutskever, Ilya and Lillicrap, Timothy and Leach, Madeleine and Kavukcuoglu, Koray and Graepel, Thore and Hassabis, Demis},
  journal = {Nature},
  volume  = {529},
  number  = {7587},
  pages   = {484--489},
  year    = {2016},
  doi     = {10.1038/nature16961}
}

@inproceedings{ross2011reduction,
  title     = {A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning},
  author    = {Ross, St{\'e}phane and Gordon, Geoffrey J. and Bagnell, J. Andrew},
  booktitle = {Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {15},
  pages     = {627--635},
  year      = {2011},
  publisher = {PMLR}
}

@inproceedings{henderson2018deep,
  title     = {Deep Reinforcement Learning That Matters},
  author    = {Henderson, Peter and Islam, Riashat and Bachman, Philip and Pineau, Joelle and Precup, Doina and Meger, David},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  pages     = {3207--3214},
  year      = {2018},
  doi       = {10.1609/aaai.v32i1.11694}
}

@inproceedings{agarwal2021deep,
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  author    = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron and Bellemare, Marc G.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {29304--29320},
  year      = {2021}
}

% ===================== Datasets considered and rejected as primary benchmarks =====================

@inproceedings{pasupat-liang-2015-compositional,
    title = "Compositional Semantic Parsing on Semi-Structured Tables",
    author = "Pasupat, Panupong and Liang, Percy",
    booktitle = "Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = jul, year = "2015", address = "Beijing, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P15-1142/",
    doi = "10.3115/v1/P15-1142",
    pages = "1470--1480"
}

@inproceedings{zhu-etal-2021-tat,
    title = "{TAT}-{QA}: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance",
    author = "Zhu, Fengbin and Lei, Wenqiang and Huang, Youcheng and Wang, Chao and Zhang, Shuo and Lv, Jiancheng and Feng, Fuli and Chua, Tat-Seng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug, year = "2021", address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.254/",
    doi = "10.18653/v1/2021.acl-long.254",
    pages = "3277--3287"
}

@inproceedings{pal-etal-2023-multitabqa,
    title = "{M}ulti{T}ab{QA}: Generating Tabular Answers for Multi-Table Question Answering",
    author = "Pal, Vaishali and Yates, Andrew and Kanoulas, Evangelos and de Rijke, Maarten",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul, year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.348/",
    doi = "10.18653/v1/2023.acl-long.348",
    pages = "6322--6334"
}
```
