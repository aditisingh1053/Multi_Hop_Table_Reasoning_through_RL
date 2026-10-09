# Reinforcement Learning for Table QA and Table Reasoning (2017-2026), focused on RL that learns WHAT to look at

Scope: weakly-supervised semantic parsing with RL (2017-2019); RL evidence selection and table/text controllers (2024); LLM RL (GRPO/PPO) for table reasoning (2025-2026); brief notes on RL text-to-SQL. Every paper listed here was checked on its arXiv abstract page (most through the arXiv export API), ACL Anthology, or the paper PDF/HTML. Numbers marked (PDF) were read directly from the paper text. Numbers marked (HTML-extract) came from an LLM summary of the arXiv HTML and should be spot-checked before they go in the report.

Verification note: WebFetch summaries of three PDFs (MeRL, Seq2SQL, José et al.) were **wrong**. One expanded MAPO as "Modular Adversarial Policy Optimization", and another got Seq2SQL's reward values wrong. All classic-era numbers below were re-read from `pdftotext` output of the arXiv PDFs.

---

## Q1. Classic RL for table QA (Seq2SQL, NSM, MAPO, MeRL): what was learned with RL, how sparse reward and spurious programs were handled, and the reported numbers

### Takeaway
In the 2017-2019 era, RL learned to **generate programs or queries** (logical forms) over a single table, not to choose evidence. The shared problems were sparse binary reward and spurious programs. The shared fixes are the direct ancestors of this project's BC warm start and evidence-recall reward: supervised or pseudo-gold pretraining before policy gradient (Seq2SQL teacher forcing, NSM iterative ML), a memory buffer of high-reward trajectories (MAPO), and a learned auxiliary reward that discounts accidentally-correct trajectories (MeRL). Headline WikiTableQuestions accuracy rose from 43.7 to 46.3 (MAPO, 10-model ensemble) and then to 46.9 (MeRL ensemble).

### Cited Findings
**Seq2SQL (Zhong, Xiong, Socher; arXiv 1709.00103, 2017; arXiv preprint, never formally published)**
- RL trains only the **WHERE clause**, whose conditions are unordered, so cross-entropy against one gold order is a poor fit. Aggregation and SELECT column are trained with cross-entropy, and the model uses a mixed objective. (PDF) — [arXiv 1709.00103](https://arxiv.org/abs/1709.00103)
- Execution reward (PDF, Eq. 7): **-2** if the query is not valid SQL, **-1** if it is valid but returns the wrong result, **+1** if correct. The loss is the negative expected reward, optimized with policy gradient. The WHERE clause is first supervised by teacher forcing ("the policy is not learned from scratch") and then trained with RL. — [arXiv 1709.00103](https://arxiv.org/abs/1709.00103)
- WikiSQL test (PDF Table): Seq2SQL without RL reached 47.4% logical-form / 57.1% execution accuracy; **Seq2SQL with RL reached 48.3% / 59.4%**. The attentional seq2seq baseline reached 23.4% / 35.9%. The paper also released WikiSQL (80,654 questions over 24,241 tables). — [arXiv 1709.00103](https://arxiv.org/abs/1709.00103)

**Neural Symbolic Machines (Liang, Berant, Le, Forbus, Lao; ACL 2017)**
- A seq2seq "programmer" with a key-variable memory, plus a symbolic Lisp "computer" that prunes invalid programs. Trained with **REINFORCE** on the task reward. — [arXiv 1611.00020](https://arxiv.org/abs/1611.00020); [ACL Anthology P17-1003](https://aclanthology.org/P17-1003/)
- Sparse reward (PDF): the reward is the F1 of the final answer, given only at the last step. Iterative maximum likelihood finds **pseudo-gold programs**. "Augmented REINFORCE" places those programs in the beam with a fixed, reasonably large probability, which the paper compares to imitation-learning proposal distributions. The paper notes that a pseudo-gold program "could be a spurious program that accidentally produces the correct answer". — [arXiv 1611.00020](https://arxiv.org/abs/1611.00020)
- **NSM was evaluated on Freebase (WebQuestionsSP), not tables.** Test F1 was **69.0** vs STAGG 66.8 under weak supervision; STAGG with full supervision reaches 71.7 (PDF Table 2). Cite NSM as the method lineage, not as a table result. — [arXiv 1611.00020](https://arxiv.org/abs/1611.00020)

**MAPO: Memory Augmented Policy Optimization (Liang, Norouzi, Berant, Le, Lao; NeurIPS 2018 spotlight)**
- Writes the expected return as a weighted sum of an expectation over a **memory buffer of high-reward trajectories** and an expectation over trajectories outside it. Adds (1) memory weight clipping, (2) systematic exploration that never resamples the same trajectory, and (3) distributed sampling. The method targets deterministic environments with discrete actions and sparse rewards. — [arXiv 1807.02322](https://arxiv.org/abs/1807.02322)
- On spurious programs (PDF): "high-reward but spurious programs will bias the gradient during training". The paper contrasts MAPO with methods that give every buffered trajectory the same reward, which are "prone to spurious programs". — [arXiv 1807.02322](https://arxiv.org/abs/1807.02322)
- WikiTableQuestions test (PDF): single model 43.8 (dev 42.7), mean of 5 runs 43.1, **10-model ensemble 46.3**; the previous SOTA was 43.7. WikiSQL with weak supervision: single model 72.6 test, **10-model ensemble 74.9**. — [arXiv 1807.02322](https://arxiv.org/abs/1807.02322)

**MeRL: Learning to Generalize from Sparse and Underspecified Rewards (Agarwal, Liang, Schuurmans, Norouzi; ICML 2019, PMLR 97:130-140)**
- Binary success rewards are "underspecified" because they do not separate purposeful from accidental success. MeRL explores with a **mode-covering KL** objective to collect diverse successful trajectories (MAPOX = IML exploration + MAPO), then trains with a mode-seeking objective. It **meta-learns an auxiliary terminal reward** over trajectory-level features so that the trained policy does well on a validation set. A Bayesian-optimization variant (BoRL) serves as the comparison. — [arXiv 1902.07198](https://arxiv.org/abs/1902.07198)
- WikiTableQuestions test, single model (PDF Table 2): MAPO 42.9, MAPOX 43.3, BoRL 43.8, **MeRL 44.1** (+1.2 over MAPO). 10-model ensemble: **MeRL 46.9** vs MAPO 46.3 (PDF Table 4). — [arXiv 1902.07198](https://arxiv.org/abs/1902.07198)
- WikiSQL with weak supervision, test (PDF Table 3): MAPO 72.4, MAPOX 74.2, BoRL 74.2, **MeRL 74.8** (+2.4). Ensemble of 5: MeRL 76.9 vs MAPO 74.2. — [arXiv 1902.07198](https://arxiv.org/abs/1902.07198)

### Inferences
- None of these methods learns *evidence selection*. The "action" is a program token. The selection of cells, rows, and columns happens implicitly inside the program (WHERE conditions, `filter` and `hop` operators). This line of work can be framed as RL that selects table content indirectly, through executable programs, over one given table.
- The tricks map onto the project's design. Seq2SQL teacher forcing, NSM iterative ML, and the MAPO buffer are forms of warm start / demonstration replay that justify a **behaviour-cloning warm start before PPO**. MeRL's auxiliary reward that discounts spurious successes is the closest classic analogue to adding a **gold-evidence-recall term** so that the policy is not rewarded for lucky answers from wrong evidence.
- All four use REINFORCE-family estimators without a learned critic. The project's PPO + GAE actor-critic is a methodological departure worth stating.

### Gaps
- I did not verify MAPO's NeurIPS proceedings page numbers, so they are left out of the BibTeX.
- I did not check whether NSM or MAPO report how often their programs are spurious on WTQ. Pasupat & Liang's spurious-program analysis is outside this scope.

---

## Q2. 2025-2026 LLM RL for tables: disambiguating the "Table-R1"s and checking Reasoning-Table, TableGPT-R1, Fortune, TableMind, Table-Critic, other agents, and RL text-to-SQL

### Takeaway
There are **three distinct papers spelled "Table-R1"/"Table-r1"**, plus a multimodal "V-tableR1".

| Paper | arXiv | Authors | Venue | What it is |
|---|---|---|---|---|
| "Table-R1: Region-based Reinforcement Learning for Table Understanding" | 2505.12415 | Wu et al. (Beihang / TeleAI) | arXiv | RL that learns which **row/column region** to look at |
| "Table-R1: Inference-Time Scaling for Table Reasoning" | 2505.23621 | Yang, Chen, Cohan, Zhao (Yale) | EMNLP 2025 | RLVR over full serialized tables |
| "Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models" | 2506.06137 | Jin et al. | arXiv | Program-based reasoning with GRPO |

Almost all 2025-2026 LLM work uses **GRPO-family** critic-free RL with outcome rewards over a **single, fully serialized table**. Only the region-based Table-R1 rewards *where the model looks*, through row/column IoU. Reasoning-Table adds a cell-level position reward. Text-to-SQL RL (SQL-R1, Arctic) gives the full schema with no learned table selection. Reasoning-SQL's schema-linking Jaccard reward is the only implicit table/column-selection reward found. Table-Critic is **not RL**.

### Cited Findings
**Disambiguation (exact titles and authors from arXiv API metadata)**
- **2505.12415**: "Table-R1: Region-based Reinforcement Learning for Table Understanding". Zhenhe Wu, Jian Yang, Zhongjiang He, Changzai Pan, Jie Zhang, Jiaheng Liu, Xianjie Wu, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li, Xueling Li. First submitted 2025-05-18, v3 in 2026. No venue stated. — [arXiv 2505.12415](https://arxiv.org/abs/2505.12415)
- **2505.23621**: "Table-R1: Inference-Time Scaling for Table Reasoning". Zheyuan Yang, Lyuhao Chen, Arman Cohan, Yilun Zhao. The arXiv comment says EMNLP 2025. **Title mismatch:** the ACL Anthology lists it as "Table-R1: Inference-Time Scaling for Table Reasoning **Tasks**" (EMNLP 2025 main, pp. 20605-20624). — [arXiv 2505.23621](https://arxiv.org/abs/2505.23621); [ACL Anthology 2025.emnlp-main.1040](https://aclanthology.org/2025.emnlp-main.1040/)
- **2506.06137**: "Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models". Rihui Jin, Zheyu Xin, Xing Xie, Zuoyi Li, Guilin Qi, Yongrui Chen, Xinbang Dai, Tongtong Wu, Gholamreza Haffari. arXiv preprint. — [arXiv 2506.06137](https://arxiv.org/abs/2506.06137)
- An arXiv title search for "Table-R1" / "TableR1" / "Table-r1" returned only these three, plus **V-tableR1** (2604.20755, process-supervised *multimodal* table reasoning, Jiang et al. 2026), which is out of scope. — [arXiv 2604.20755](https://arxiv.org/abs/2604.20755)

**Table-R1 (region-based, 2505.12415): the most relevant LLM-era paper**
- A table region is **T_reg = {[column indices], [row indices]}**, a row and column sub-table emitted inside the chain of thought before answering. Training uses Region-Enhanced SFT (RE-SFT) and then **TARPO** (Table-Aware GRPO). (HTML-extract) — [arXiv HTML v3](https://arxiv.org/html/2505.12415v3)
- Reward: r = α_t · r_region + (1 − α_t) · r_answer. The region reward is the IoU of predicted rows and columns against the annotation, and the answer reward is binary. α_t = γ·e^(−ρτ) **decays over training**, shifting weight from region accuracy to the answer. A **consistency penalty** applies when region and answer advantages have opposite signs. (HTML-extract) — [arXiv HTML v3](https://arxiv.org/html/2505.12415v3); abstract: "decaying region rewards and consistency penalties" — [arXiv 2505.12415](https://arxiv.org/abs/2505.12415)
- Models: Qwen2-7B, CodeQwen-7B, DeepSeek-Coder-7B, Llama3.1-8B, TeleChat2-7B, Qwen2.5-3B, Qwen3-8B. Training data is TableInstruct with new region annotations (TableInstruct-RE). Evaluation is on TableBench, WikiTQ, and WikiSQL. (HTML-extract) — [arXiv HTML v3](https://arxiv.org/html/2505.12415v3)
- Numbers: an average gain of **+14.36 points** across base models and the three benchmarks. TARPO uses **67.5% fewer response tokens than GRPO** (abstract). With Qwen3-8B, Table-R1 scores WikiTQ 73.87, WikiSQL 72.50, overall 55.33, vs GPT-4o at 68.40 / 47.60 / 49.17 (PDF Table 2). — [arXiv 2505.12415](https://arxiv.org/abs/2505.12415)
- Stated limitations: only 3B-8B models; 8192/2048 input/output tokens; the region is inserted midway into the chain of thought without refining it; validated only on academic datasets, **"not real-world complex tables ... or multi-table QA scenarios"**. (HTML-extract) — [arXiv HTML v3](https://arxiv.org/html/2505.12415v3)

**Table-R1 (inference-time scaling, 2505.23621, EMNLP 2025)**
- Two tracks: SFT distillation from DeepSeek-R1 traces (Table-R1-SFT) and **RLVR with GRPO** (Table-R1-Zero). The GRPO variant uses token-level loss, asymmetric clipping, and no KL term. Rewards: exact match for short-form TQA, label accuracy for fact verification, BLEU + ROUGE-L average for free-form QA, and a format reward for `<think>`/`<answer>`. (HTML-extract) — [arXiv HTML v2](https://arxiv.org/html/2505.23621v2)
- Base models: Qwen2.5-7B and Llama-3.1-8B. RLVR data: 48,563 samples from WTQ, HiTab, TabFact, and FeTaQA. Thirteen benchmarks, nine of them out-of-domain. Tables are fully serialized with a 4,096-token prompt limit, and there is **no region selection**. (HTML-extract) — [arXiv HTML v2](https://arxiv.org/html/2505.23621v2)
- From the EMNLP PDF Table 2: Table-R1-Zero (Llama-3.1-8B init) scores FeTaQA 32.7 BLEU, TabFact 87.6, WTQ 81.2, vs GPT-4.1 at 25.1 / 86.5 / 68.0. The abstract claims the model "matches or exceeds ... GPT-4.1 and DeepSeek-R1, while using only a 7B-parameter LLM". Limitation: Llama-3.1-8B shows unstable RLVR and can fail to acquire the format. — [ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.1040.pdf); [arXiv 2505.23621](https://arxiv.org/abs/2505.23621)

**Table-r1 (program-based, small LMs, 2506.06137)**
- Stage 1 is self-supervised **Layout Transformation Inference**: the model infers the code that maps a table to a transformed layout (swap, delete, extract, transpose). Stage 2 is **mix-paradigm GRPO**, which favours program-based reasoning but allows a dynamic fallback to text-based reasoning. Rewards (HTML-extract): format up to 0.75, answer up to 1.5, compilation, a short-code penalty, and a comment ratio. — [arXiv 2506.06137](https://arxiv.org/abs/2506.06137); [HTML](https://arxiv.org/html/2506.06137v1)
- Models: Qwen2.5-Coder-7B and LLaMA3.1-8B. The abstract reports "at least a 15% accuracy improvement over the base model (LLaMA-8B)" across four benchmarks. HTML-extract for Qwen-7B: WikiTQ 76.5, TabFact 85.3, HiTab 76.7, AIT-QA 89.9. It uses only the **first 10 rows** during RL because of token limits (truncation, not learned selection). Single table. — [arXiv 2506.06137](https://arxiv.org/abs/2506.06137); [HTML](https://arxiv.org/html/2506.06137v1)

**Reasoning-Table (Lei, Meng, Huang, Chen, Zhang, He, Zhao, Liu; arXiv 2506.01710, "work in progress")**
- GRPO with rule-based outcome rewards, unified across TQA, fact verification, table-to-text, and text-to-SQL. The abstract claims it is "the first application of reinforcement learning (RL) to table reasoning". That claim is inaccurate given the 2017-2019 work in Q1 and arguably concurrent with 2505.12415. — [arXiv 2506.01710](https://arxiv.org/abs/2506.01710)
- Reward (HTML-extract): R = R_ans · (1 + λ1 · R_pos) + λ2 · R_fmt. **R_pos is the Jaccard similarity between cells the model cites (`<|cell|><|column|>` tags) and ground-truth evidence cells**, obtained by intersecting cell sets across 5 correct LLM samples. Because it is multiplicative, the position reward pays only when the answer is correct. — [arXiv HTML v1](https://arxiv.org/html/2506.01710v1)
- Base model Qwen2.5-7B-Instruct, plus Qwen2.5-Coder-7B and OmniSQL-7B for SQL. Data includes WikiTQ, HybridQA, MultiHiertt, OTT-QA, FinQA, TAT-QA, HiTab, Spider, BIRD, TabFact, FEVEROUS, and ToTTo. **For OTT-QA, the oracle table is supplied (no learned table selection).** (HTML-extract) — [arXiv HTML v1](https://arxiv.org/html/2506.01710v1)
- Abstract numbers: it surpasses **Claude-3.7-Sonnet by 4.0%** on table reasoning benchmarks and reaches **68.3% on BIRD dev** with a 7B model. **Conflict:** the HTML-extract instead said "surpasses Claude-3.5-Sonnet by ~2.5%"; cite the abstract. Stated limitations: no code-execution module for TQA, no control of data mixing proportions, limited data scaling. — [arXiv 2506.01710](https://arxiv.org/abs/2506.01710)

**Fortune / Formula-R1 (Cao, Xu, Liu, Wang, Zhou, Dong, Han, Zhang; arXiv 2505.23667; MSR plus universities)**
- **Retitled.** v1 was "Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models". The current version is "Formula-R1: Incentivizing LLM Reasoning over Complex Tables with Numerical Computation via Formula-Driven Reinforcement Learning", where the training method is called "Formula Tuning (Fortune)". — [arXiv 2505.23667v1](https://arxiv.org/abs/2505.23667v1); [arXiv 2505.23667](https://arxiv.org/abs/2505.23667)
- The model generates spreadsheet formulas that reference cell ranges. Trained with **PPO**, not GRPO (HTML-extract). Reward: 1 if the formula executes and the answer is correct, **0.2 if executable but wrong**, 0 if not executable. Base models: Llama-3.1-8B and Qwen2.5-Coder-7B with a cold-start SFT. Seven benchmarks, e.g. Formula-R1-Plus scores WikiTQ 82.54, TabFact 95.06, HiTab 87.24, FinQA 80.47, MultiHiertt 51.73, TableBench 44.96. (HTML-extract) — [arXiv HTML v3](https://arxiv.org/html/2505.23667v3)
- No explicit limitations section (HTML-extract). Semantic Scholar lists no venue, so treat it as an arXiv preprint.

**TableMind (Jiang, Cheng, Tao, Mao, Ouyang, Liu; WSDM '26)**
- An autonomous programmatic agent with a plan-action-reflect loop that runs Python in a sandbox, with at most 3 tool turns. Trained with SFT and then **RAPO (Rank-Aware Policy Optimization)**, a GRPO variant that up-weights misaligned trajectories. Rewards are format, accuracy, and a **tool reward with an implicit curriculum** (encourages exploration early, efficiency later). Base model Qwen3-8B. (HTML-extract) — [arXiv HTML v4](https://arxiv.org/html/2509.06278v4); [arXiv 2509.06278](https://arxiv.org/abs/2509.06278)
- Results: WikiTQ 76.82, TabMWP 99.27, TabFact 91.85, HiTab 71.95, FinQA 42.02. Stated future work: **"extending it to multi-table reasoning"** and improving the sample efficiency of RL. Single-table only. (HTML-extract) — [arXiv HTML v4](https://arxiv.org/html/2509.06278v4)
- Follow-up **TableMind++** (2603.07528, Cheng et al., 2026) adds uncertainty awareness. I did not inspect it further. — [arXiv 2603.07528](https://arxiv.org/abs/2603.07528)

**TableGPT-R1 (Yang et al., 16 authors, Zhejiang Univ.; arXiv 2512.20312, Dec 2025)**
- Qwen3-8B with a three-stage **GRPO++** curriculum (DAPO/GSPO-style asymmetric clipping). Rewards: rule-based verification, a criteria-injected reward model, process step rewards (+0.1 for successful execution, −0.1/−0.2 for failures), and penalties for verbosity and repetition. Given a table *file path*, the agent "must autonomously load the table, inspect its structure, and determine what information to retrieve". Spider 86.73, BIRD 63.17. Stated limitations: RL beyond about 120-200 steps causes general-capability decline and reward hacking; future work targets "ultra-large database interactions". (HTML-extract) — [arXiv HTML v2](https://arxiv.org/html/2512.20312v2); [arXiv 2512.20312](https://arxiv.org/abs/2512.20312)

**Mixture-of-Minds (Zhou et al., 15 authors, Meta; ACL 2026 Long)**
- Three agents: a planner (at most 4 operations), a pandas coder, and an answerer. Each is trained with **GRPO in sequence**, using MCTS-style rollouts to build pseudo-gold intermediate supervision. Rewards: planning = 10% format + 90% BLEU to the gold plan; coding = format + execution + pandas-operation F1 + output BLEU; answering = format + EM. Qwen3-32B with test-time scaling reaches 62.13% on TableBench vs o4-mini-high at 61.69%. Row and column filtering emerges inside the code rather than through an explicit selection action, and multi-table input is not addressed. (HTML-extract) — [arXiv 2510.20176](https://arxiv.org/abs/2510.20176); [ACL Anthology 2026.acl-long.112](https://aclanthology.org/2026.acl-long.112/)

**Table-Critic (Yu, Chen, Wang; ACL 2025 Main): not RL**
- A multi-agent Judge / Critic / Refiner / Curator framework with a self-evolving template tree, i.e. prompting-based refinement. **Exclude** from an RL list, or mention only as a non-RL baseline. — [arXiv 2502.11799](https://arxiv.org/abs/2502.11799)

**Other 2026 RL items on evidence or grounding (secondary)**
- **RSAT** (Gajjar, Subramaniakuppusamy; arXiv 2605.00199; the comment says SURGeLLM Workshop at ACL 2026): SFT then GRPO for **cell-level citations**. The reward combines NLI-based faithfulness, citation validity, and parsimony. Models: Qwen2.5 1.5B-7B and Llama 3 1B-8B. Faithfulness rises from 0.224 to 0.826 and citation validity reaches 0.992. Datasets were not identified. (summary of abstract page) — [arXiv 2605.00199](https://arxiv.org/abs/2605.00199)
- **RE-Tab / TabROUGE** (Kwok et al.; arXiv 2601.22530, 2026): a deterministic, LCS-based reward that scores **intermediate table states** (sub-tables), applicable to stepwise reasoning in the style of Chain-of-Table. The paper reports +26.7 pp average accuracy, up to 33% fewer test-time-scaling samples, and +8.34 pp from preliminary GRPO post-training. Only the abstract-level description was verified. — [arXiv 2601.22530](https://arxiv.org/abs/2601.22530)

**RL text-to-SQL (mention briefly)**
- **Reasoning-SQL** (Pourreza, Talaei, Sun, Wan, Li, Mirhoseini, Saberi, Arık; arXiv 2503.23157): GRPO with partial rewards. Weights: execution 3, LLM-judge 2, syntax 1, **schema-linking 1** (Jaccard over tables and columns used in predicted vs gold SQL), n-gram 1, format 1. Models are Qwen2.5-Coder 3B/7B/14B. The 14B model beats o3-mini by 4% and Gemini-1.5-Pro-002 by 3% on BIRD (HTML-extract: 72.29 dev / 72.78 test). — [arXiv 2503.23157](https://arxiv.org/abs/2503.23157); [HTML](https://arxiv.org/html/2503.23157v2)
- **SQL-R1** (Ma, Zhuang, Xu, Jiang, Chen, Guo; arXiv 2504.08600; comment: accepted by NeurIPS 2025): GRPO. Rewards are format ±1, execution +2/−2, result +3/−3, plus a length reward. Qwen2.5-Coder 3B/7B/14B. **Full schema given** (CREATE TABLE statements), with no learned table selection. Spider 88.6% and BIRD 67.1% (abstract). — [arXiv 2504.08600](https://arxiv.org/abs/2504.08600); [HTML](https://arxiv.org/html/2504.08600v5)
- **Arctic-Text2SQL-R1** (Yao, Sun, Borchmann, Nuti, Shen, Deng, Zhai, Zhang, Li, He; arXiv 2505.20315): GRPO with reward **1** for a correct execution result, **0.1** for syntactically valid and executable SQL, 0 otherwise. 7B/14B/32B models, full schema. BIRD dev/test 70.5/71.8 (32B). The 7B model beats prior 70B-class systems. — [arXiv 2505.20315](https://arxiv.org/abs/2505.20315); [HTML](https://arxiv.org/html/2505.20315v2)

### Inferences
- **"RL learns where to look" exists only in single-table, single-shot form.** The region-based Table-R1 is the clearest precedent: an explicit region action supervised by an IoU reward that decays toward an answer reward. Reasoning-Table's cell-Jaccard position reward, gated by answer correctness, is a second precedent. Both are close analogues of the project's "gold-evidence recall + answer correctness" reward. Both emit the region as text in one pass and do not choose sequentially among candidate entities with STOP.
- Nearly all LLM-era work uses critic-free GRPO variants. Formula-R1's PPO is the exception. The project's **PPO + GAE with a small pointer-head policy** over a variable candidate set is therefore a different design point: a cheap, structured selector rather than a fine-tuned LLM. That is a defensible contrast, and probably needed for compute on a course project.
- Several outcome-only papers report token or turn efficiency gains (TARPO −67.5% tokens; TableMind tool-reward curriculum). The project's **per-hop cost** term echoes these.
- Text-to-SQL RL is relevant only through **Reasoning-SQL's schema-linking reward**, which is effectively a table/column-selection recall reward over a multi-table DB, but implicit in SQL tokens. SQL-R1 and Arctic give the full schema and should be mentioned at most in one sentence.

### Gaps
- Table-R1-region, Table-r1, Formula-R1, Reasoning-Table, TableGPT-R1, and OpenTable-R1 list no peer-reviewed venue on arXiv or Semantic Scholar as of this check. Treat them as preprints. They may have later venues I did not find.
- Some per-benchmark numbers for Table-r1, Formula-R1, TableMind, TableGPT-R1, and Mixture-of-Minds are HTML-extracts and should be spot-checked in the PDFs before they go into tables.
- The datasets for RSAT and the benchmarks and models for RE-Tab were not verified beyond abstract-level summaries.
- The ReasonTabQA (2601.07280) RL method "TabCodeRL" runs on an industrial multi-table benchmark, but the abstract does not say whether RL learns table selection. Unresolved. — [arXiv 2601.07280](https://arxiv.org/abs/2601.07280)

---

## Q3. Per-paper comparison: selection unit, RL algorithm, reward, model size, datasets, headline numbers, limitations

### Takeaway
Selection granularity differs across the field:
- Program tokens (2017-2019)
- One-shot row/column subsets (Learning to Reduce, region-based Table-R1)
- Cited cells (Reasoning-Table, RSAT)
- Tool/code calls (TableMind, TableGPT-R1, Mixture-of-Minds)
- Retriever-module choice per hop (José et al. 2024)
- Table-search queries (OpenTable-R1)

No paper uses entity-level sequential selection across tables and passages with a STOP action.

### Cited Findings
| Paper | Selection / action unit | RL algo | Reward components | Model | Data | Headline | Key limitation |
|---|---|---|---|---|---|---|---|
| Seq2SQL 2017 [src](https://arxiv.org/abs/1709.00103) | WHERE-clause tokens (pointer over columns, ops, question words) | REINFORCE + CE (mixed) | Execution: −2 invalid / −1 wrong / +1 correct | Seq2seq LSTM pointer | WikiSQL | Execution 59.4, logical form 48.3 (test) | Needs gold SQL for CE parts |
| NSM 2017 [src](https://arxiv.org/abs/1611.00020) | Program tokens (Lisp) | REINFORCE + iterative ML pseudo-gold | Final-answer F1 | Seq2seq + key-variable memory | WebQSP (Freebase, not tables) | F1 69.0 | Spurious pseudo-gold programs |
| MAPO 2018 [src](https://arxiv.org/abs/1807.02322) | Program tokens | Policy gradient with memory buffer | Binary answer match | NSM-style | WTQ, WikiSQL | WTQ 46.3 (ens. 10) / 43.8 single; WikiSQL 74.9 (ens.) | Spurious programs bias gradient |
| MeRL 2019 [src](https://arxiv.org/abs/1902.07198) | Program tokens | MAPOX + meta-learned auxiliary reward | Binary + learned trajectory-feature reward | MAPO arch. | WTQ, WikiSQL | WTQ 44.1 single / 46.9 ens.; WikiSQL 74.8 | Needs a validation set and hand-designed trajectory features |
| Learning to Reduce 2024 [src](https://arxiv.org/abs/2402.14195) | Relevant **columns, then rows** (generated as text; two separate models) | PPO (+KL penalty; NLPO top-p masking p=0.9) | +reward for relevant items; small −reward for irrelevant; large −reward for missing necessary items | FLAN-T5-Large; frozen GPT-4 reader | WTQ (SQUALL-derived evidence); HybridQA (unseen) | Column recall 87% on HybridQA vs RoBERTa 70%, GPT-4 0-shot ~71.5% | 5-page short paper; single table; reward on selection, not on downstream answer |
| José et al. 2024 [src](https://arxiv.org/abs/2407.04858) | **Per hop: Retrieve Texts / Retrieve Tables / Generate Answer** | DQN and PPO (Stable-Baselines3) | 2 for exact match, F1 for partial, −0.5 for no overlap, **−0.02 per retrieval** | MLP/LSTM/GRU/Transformer policy over frozen BM25/Tri-encoder + FiE reader | OTT-QA | Best F1 19.03 (PPO + Transformer, BM25), equal to the fixed "always retrieve text" baseline | Coarse module-level actions; single run; 100k steps; did not beat heuristic |
| Table-R1 region 2025 [src](https://arxiv.org/abs/2505.12415) | **Row + column index sets** (sub-table) inside CoT | TARPO (GRPO variant) | α_t·IoU(region) + (1−α_t)·answer; decaying α; consistency penalty | 3B-8B (Qwen3-8B best) | TableBench, WTQ, WikiSQL | +14.36 avg; −67.5% tokens vs GRPO; Qwen3-8B overall 55.33 vs GPT-4o 49.17 | Single table; region not refined; 8B max |
| Table-R1 ITS 2025 [src](https://aclanthology.org/2025.emnlp-main.1040/) | None (full serialized table) | GRPO (no KL, token-level loss) | EM / label accuracy / BLEU+ROUGE + format | Qwen2.5-7B, Llama-3.1-8B | 13 benchmarks | WTQ 81.2, TabFact 87.6 (Zero, Llama) vs GPT-4.1 68.0 / 86.5 | Llama RL instability |
| Table-r1 2025 [src](https://arxiv.org/abs/2506.06137) | Program vs text mode (implicit) | Mix-paradigm GRPO | Format, answer, compile, code-length, comments | Qwen2.5-Coder-7B, Llama3.1-8B | WTQ, TabFact, HiTab, AIT-QA | ≥15% over Llama-8B base | Hand-tuned reward weights; first 10 rows only |
| Reasoning-Table 2025 [src](https://arxiv.org/abs/2506.01710) | Cited cells (tags) | GRPO | R_ans·(1+λ·Jaccard(cells)) + λ·format | Qwen2.5-7B | 12+ datasets incl. HybridQA, OTT-QA (oracle table) | +4.0% over Claude-3.7-Sonnet; BIRD dev 68.3 | No code execution; WIP |
| Formula-R1 / Fortune 2025 [src](https://arxiv.org/abs/2505.23667) | Spreadsheet formula (cell ranges) | PPO | 1 correct / 0.2 executable / 0 fail | Llama-3.1-8B, Qwen2.5-Coder-7B | 7 benchmarks | WTQ 82.54, TabFact 95.06 (Plus) | No limitations section |
| TableMind 2025-26 [src](https://arxiv.org/abs/2509.06278) | Python tool calls (≤3 turns) | RAPO (GRPO variant) | Format, accuracy, tool-use curriculum | Qwen3-8B | WTQ, TabMWP, TabFact, HiTab, FinQA | WTQ 76.82 | Multi-table left to future work |
| TableGPT-R1 2025 [src](https://arxiv.org/abs/2512.20312) | Code/tool calls; agent loads files | GRPO++ (3 stages) | Rules + reward model + step rewards + regularizers | Qwen3-8B | Spider, BIRD, RealHitBench, internal | Spider 86.73, BIRD 63.17 | Reward hacking after long RL |
| OpenTable-R1 2025 [src](https://arxiv.org/abs/2507.03018) | **Table search (BM25+, top_k) + SQL calls** over a table corpus | Async GRPO + LoRA | Episode reward (components not detailed) | Qwen3-4B | Open WikiTable | EM 12.0 → 72.1 (cold start) → **86.2** (RL); turns 8.8 → 6.8 | Sampled 1,024 test questions; BM25 only; one gold table per question |
| Mixture-of-Minds 2025-26 [src](https://aclanthology.org/2026.acl-long.112/) | Plan / code / answer per agent | Sequential GRPO | BLEU to plan; exec + pandas-op F1; EM | 8B-70B | TableBench, FinQA | 62.13 TableBench (Qwen3-32B + TTS) | No multi-table |

Additional verified details for Learning to Reduce (Lee, Kim, Yu, Rossi, Chen; Purdue / Adobe; arXiv 2402.14195, read from the PDF):
- Gold relevant rows and columns are auto-derived by iteratively deleting rows and columns from WTQ tables while the SQUALL SQL still returns the answer.
- PPO uses an adaptive KL penalty, following Ziegler et al., and NLPO-style top-p masking (p=0.9) of the token action space.
- The motivation cites StructGPT's finding that ~30% of table-QA errors come from incorrect data selection.
- Source: [arXiv 2402.14195](https://arxiv.org/abs/2402.14195)

### Inferences
- The **asymmetric penalty in Learning to Reduce** (missing evidence is critical, extra evidence is mild) is a ready-made argument for weighting gold-evidence *recall* over precision in the project's reward. The **per-hop cost** then plays the precision-control role.
- **NLPO top-p masking in Learning to Reduce** is the nearest precedent for **invalid-action masking** in a table-selection policy.

### Gaps
- Learning to Reduce's Table 2 row-reduction numbers were garbled in PDF extraction. Only the column-reduction numbers corroborated by the prose are reported (87% on HybridQA; RoBERTa 93% on WTQ falling to 70% on HybridQA; GPT-4 74% falling to 71.5%).
- OpenTable-R1's reward function was not described in the extracted content.

---

## Q4. Is there RL-learned selection over MULTIPLE tables, or over table+text, with multi-hop chaining? (counter-evidence for the project's gap claim)

### Takeaway
**Partial counter-evidence exists, so the gap claim must be narrowed.**
- **José et al. (BRACIS 2024)** train a **PPO/DQN controller that chooses, per hop, between retrieving texts, retrieving tables, or answering (STOP)** on OTT-QA, with an answer-F1 reward and a −0.02 per-retrieval cost. This is structurally very close to the project's MDP. But the action is a *module choice*, not an entity choice, and it **failed to beat the best fixed heuristic** (F1 19.03 = "always retrieve text").
- **OpenTable-R1** uses GRPO to train an LLM agent that learns *which tables to fetch* via multi-turn BM25 search over a table corpus, but each question has one gold table and there is no join or multi-hop chaining.

No paper found does **entity-level** (cell/row/column/passage/table) sequential selection with a pointer policy, invalid-action masking, PPO+GAE, and an evidence-recall reward across multiple tables or table+linked text. Several 2025-2026 papers list multi-table as explicit future work.

### Cited Findings
- **José, Cação, Ribeiro, Cheang, Pirozelli, Cozman (BRACIS 2024; arXiv 2407.04858)**, all read from the PDF:
  - Actions: "Retrieve Texts (A1) ... Retrieve Tables (A2) ... Generate Answer (A3)". At most three retrieval steps. The first retrieval fetches 10 passages (text or tabular) that form 10 blocks, and each later step retrieves 4 new passages per block.
  - Observation: 11 embedding vectors (the question plus 10 block averages).
  - Reward at the end of the episode: 2 if exact match, F1 if partial, −0.5 otherwise, plus "a minor penalty of −0.02 for each retrieval action".
  - DQN and PPO from Stable-Baselines3 with MLP, LSTM, GRU, and Transformer extractors. Retrievers BM25 and Tri-encoder; reader FiE from COS.
  - Best: PPO + Transformer with BM25, EM 14.81 / F1 19.03, which "achieved the same level of performance as the best baseline, which involved consistently selecting the A1 action regardless of the question". The non-RL Iterative-Retrieval + Cross-Block Reader reaches 20.7.
  - Limitations: single runs (each training run took over a week), only 100k steps, and an untuned Tri-encoder.
  - Source: [arXiv 2407.04858](https://arxiv.org/abs/2407.04858)
- **OpenTable-R1 (Zipeng Qiu; arXiv 2507.03018)**:
  - The agent calls a BM25+ table-search API (keywords, top_k default 8) and a SQLite executor across multiple turns, trained with cold-start SFT then Async GRPO with LoRA (HTML-extract).
  - On Open WikiTable (67,023 question-table pairs), EM rose from 12.0% (zero-shot) to 72.1% (cold start) to **86.2%** (RL), and average turns fell from 8.8 to 6.8.
  - The paper notes top_k can be used to "access [multiple tables] and union them together", but complex cross-table joins are not demonstrated.
  - Sources: [arXiv 2507.03018](https://arxiv.org/abs/2507.03018); [HTML](https://arxiv.org/html/2507.03018)
- **Reasoning-Table** trains on HybridQA and OTT-QA but gives the **oracle table** for OTT-QA, so table selection is not learned (HTML-extract). — [arXiv HTML v1](https://arxiv.org/html/2506.01710v1)
- **Learning to Reduce** evaluates its PPO row/column selector *zero-shot on HybridQA tables* (87% column recall), but it selects within one table and ignores linked passages. — [arXiv 2402.14195](https://arxiv.org/abs/2402.14195)
- **Explicit future-work statements** that support the gap:
  - TableMind: "extending it to multi-table reasoning ... remain[s] important" (HTML-extract). — [arXiv HTML v4](https://arxiv.org/html/2509.06278v4)
  - Region-based Table-R1: not validated on "multi-table QA scenarios" (HTML-extract). — [arXiv HTML v3](https://arxiv.org/html/2505.12415v3)
  - REAR, a non-RL multi-table retriever, proposes "adaptive expansion that balances recall and latency by adjusting depth to query complexity via query-aware heuristics or reinforcement learning" as future work (read from the PDF). — [arXiv 2511.00805](https://arxiv.org/abs/2511.00805)
- **Reasoning-SQL** rewards schema-linking (Jaccard over tables and columns) on multi-table BIRD databases. That is RL credit for *which tables/columns* are used, but implicit in the SQL and single-shot. — [arXiv 2503.23157](https://arxiv.org/abs/2503.23157)
- **MoRE (Peng et al.; arXiv 2505.22095; the awesome list says SIGIR 2026)** trains, with "Step-GRPO", an MLLM to choose which retrieval expert to call at each reasoning step. I **could not verify** whether tables are one of the expert sources. — [arXiv 2505.22095](https://arxiv.org/abs/2505.22095)
- A community list of 42 text-table and 12 multimodal-table RL papers flags multi-table items such as ReasonTabQA, M3TQA, and MTabVQA. Its only multi-hop text+table RL entry is José et al. 2024, and its only region-selection entry is the region-based Table-R1. This list is a pointer only; its "MULTI-TABLE" flags often mean multi-dataset training, not table selection. — [Awesome-RL-Table-Understanding](https://github.com/lyc61c/Awesome-RL-Table-Understanding)

### Inferences
- **Recommended gap wording:** "Prior RL for table QA either (a) learns programs over a single given table (NSM/MAPO/MeRL, Seq2SQL, Formula-R1), (b) selects a sub-table in one shot within a single table (Learning to Reduce, region-based Table-R1, Reasoning-Table's cell reward), (c) learns coarse module- or tool-level control (José et al. 2024 on OTT-QA; OpenTable-R1's table search; TableMind and TableGPT-R1 code tools), or (d) learns implicit schema linking through SQL (Reasoning-SQL). None learns an entity-level, multi-hop selection policy over cells, rows, columns, linked passages and joinable tables with explicit STOP and evidence-recall reward."
- José et al. is **both the closest prior work and a cautionary result**. A PPO controller with outcome reward and per-step cost, but with coarse actions and no warm start, collapsed to the heuristic policy. This motivates three of the project's choices:
  - Finer-grained candidate actions with a pointer head.
  - A BC warm start.
  - A dense gold-evidence-recall reward, which José et al. lacked.
- OpenTable-R1 shows that RL can teach an LLM *which tables to retrieve* in open-domain single-table QA. Any claim that "RL table selection is unexplored" would be false. The novelty must rest on **multi-hop chaining across multiple tables or table+text with entity-level actions** and on **plugging the selector into existing frameworks** (TableRAG, Chain-of-Table, Dater).

### Gaps
- My searches (arXiv API title and abstract queries, web search, and the awesome list) found no RL method on MMQA, MultiTableQA, Spider-style multi-table QA, or HybridQA that learns sequential evidence selection. Absence of evidence is not proof. Specialized 2026 venues (ACL/EMNLP 2026 Findings, KDD, SIGIR) may contain papers I missed.
- The arXiv API returned nothing for several abstract-level queries combining "reinforcement learning" with "HybridQA/OTT-QA/table-text", possibly due to query syntax or rate limiting (HTTP 429 was observed). Those queries were covered only through web search.
- The table coverage of MoRE (2505.22095) and the multi-table RL in ReasonTabQA (TabCodeRL) remain unverified.

---

## Q5. Ranking: top 3-5 papers for this project ("PPO-Based Entity Selection for Multi-Hop Table Reasoning")

### Takeaway
1. **José et al. 2024**: closest MDP, plus a negative result to beat.
2. **Table-R1 region-based (Wu et al. 2025)**: RL reward on the selected row/column region.
3. **Learning to Reduce (Lee et al. 2024)**: PPO-trained evidence selector feeding a frozen LLM.
4. **Reasoning-Table (Lei et al. 2025)**: answer-gated cell-evidence reward; oracle-table gap on OTT-QA.
5. **MAPO (Liang et al. 2018)**, with NSM and MeRL as one classic cluster: sparse reward and spurious-success handling.

Honourable mention: **OpenTable-R1**, as counter-evidence that RL can learn table retrieval.

### Cited Findings
- #1 José et al.: PPO/DQN choose Retrieve-Text / Retrieve-Table / Answer per hop on OTT-QA. Reward: answer EM/F1 with −0.02 per retrieval. Best F1 19.03 equals the fixed baseline. — [arXiv 2407.04858](https://arxiv.org/abs/2407.04858)
- #2 Table-R1 (region): the row and column index region is an explicit action. The reward mixes region IoU and answer correctness with decaying weight and a consistency penalty. +14.36 avg; −67.5% tokens vs GRPO. — [arXiv 2505.12415](https://arxiv.org/abs/2505.12415)
- #3 Learning to Reduce: a PPO-trained FLAN-T5 selects relevant columns and rows for a frozen GPT-4. The reward is asymmetric: missing evidence is penalized heavily. It generalizes to HybridQA (87% column recall). — [arXiv 2402.14195](https://arxiv.org/abs/2402.14195)
- #4 Reasoning-Table: R_ans·(1+λ·cell-Jaccard) + format. Trained on HybridQA/OTT-QA with the oracle table. — [arXiv 2506.01710](https://arxiv.org/abs/2506.01710)
- #5 MAPO: a memory buffer of high-reward trajectories plus systematic exploration under sparse reward, WTQ 46.3. MeRL adds a learned auxiliary reward against spurious success. NSM bootstraps REINFORCE with iterative-ML pseudo-gold programs. — [arXiv 1807.02322](https://arxiv.org/abs/1807.02322); [arXiv 1902.07198](https://arxiv.org/abs/1902.07198); [arXiv 1611.00020](https://arxiv.org/abs/1611.00020)

### Inferences
- #1 to #3 cover the project's three pillars: the multi-hop MDP with STOP and cost, the evidence-accuracy reward, and a learned selector feeding a frozen reasoner. #4 and #5 back the reward design and warm start.
- Suggested baselines and positioning:
  - Compare against a **fixed or greedy heuristic policy**, as José et al. did; their RL agent could not beat it.
  - Compare against **one-shot region selection**, in the style of Table-R1 region or Learning to Reduce.
  - Present PPO+GAE with a critic as the alternative to critic-free GRPO used by most 2025-2026 work.

### Gaps
- The ranking is a judgement based on how closely each method matches the project. No paper evaluates the exact setting, so there is no shared benchmark number for direct comparison.

---

## Q6. Verified bibliographic records and BibTeX

### Takeaway
Ready-to-use BibTeX follows. Peer-reviewed venues are confirmed for NSM (ACL 2017, Anthology bib), MAPO (NeurIPS 2018, arXiv comment), MeRL (ICML 2019, arXiv journal_ref), Table-R1 ITS (EMNLP 2025, Anthology bib), Mixture-of-Minds (ACL 2026, Anthology bib), TableMind (WSDM '26, the paper's ACM reference block), and José et al. (BRACIS 2024, arXiv comment). SQL-R1's NeurIPS 2025 comes only from the arXiv comment. Everything else is an arXiv preprint.

### Cited Findings
- Anthology BibTeX was fetched directly for P17-1003, 2025.emnlp-main.1040, and 2026.acl-long.112. — [P17-1003](https://aclanthology.org/P17-1003/); [2025.emnlp-main.1040](https://aclanthology.org/2025.emnlp-main.1040/); [2026.acl-long.112](https://aclanthology.org/2026.acl-long.112/)
- Author lists for arXiv items are copied from arXiv API metadata or the arXiv abstract page. — [arXiv](https://arxiv.org)

```bibtex
% ---------- Classic era ----------
@article{zhong2017seq2sql,
  title   = {Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning},
  author  = {Zhong, Victor and Xiong, Caiming and Socher, Richard},
  journal = {arXiv preprint arXiv:1709.00103},
  year    = {2017}
}

@inproceedings{liang-etal-2017-neural,
  title     = {Neural Symbolic Machines: Learning Semantic Parsers on {F}reebase with Weak Supervision},
  author    = {Liang, Chen and Berant, Jonathan and Le, Quoc and Forbus, Kenneth D. and Lao, Ni},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {23--33},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/P17-1003},
  url       = {https://aclanthology.org/P17-1003/}
}

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

% ---------- Learned evidence selection / table+text controllers ----------
@article{lee2024learningtoreduce,
  title   = {Learning to Reduce: Optimal Representations of Structured Data in Prompting Large Language Models},
  author  = {Lee, Younghun and Kim, Sungchul and Yu, Tong and Rossi, Ryan A. and Chen, Xiang},
  journal = {arXiv preprint arXiv:2402.14195},
  year    = {2024}
}

@inproceedings{jose2024qatexttablesdrl,
  title     = {Question Answering with Texts and Tables through Deep Reinforcement Learning},
  author    = {Jos{\'e}, Marcos M. and Ca{\c{c}}{\~a}o, Fl{\'a}vio N. and Ribeiro, Maria F. and Cheang, Rafael M. and Pirozelli, Paulo and Cozman, Fabio G.},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS 2024)},
  year      = {2024},
  note      = {arXiv:2407.04858}
}

% ---------- LLM-era RL for tables (2025-2026) ----------
@article{wu2025tabler1region,
  title   = {Table-R1: Region-based Reinforcement Learning for Table Understanding},
  author  = {Wu, Zhenhe and Yang, Jian and He, Zhongjiang and Pan, Changzai and Zhang, Jie and Liu, Jiaheng and Wu, Xianjie and Zhao, Yu and Song, Shuangyong and Li, Yongxiang and Li, Zhoujun and Li, Xueling},
  journal = {arXiv preprint arXiv:2505.12415},
  year    = {2025}
}

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

@article{jin2025tabler1program,
  title   = {Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models},
  author  = {Jin, Rihui and Xin, Zheyu and Xie, Xing and Li, Zuoyi and Qi, Guilin and Chen, Yongrui and Dai, Xinbang and Wu, Tongtong and Haffari, Gholamreza},
  journal = {arXiv preprint arXiv:2506.06137},
  year    = {2025}
}

@article{lei2025reasoningtable,
  title   = {Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning},
  author  = {Lei, Fangyu and Meng, Jinxiang and Huang, Yiming and Chen, Tinghong and Zhang, Yun and He, Shizhu and Zhao, Jun and Liu, Kang},
  journal = {arXiv preprint arXiv:2506.01710},
  year    = {2025}
}

@article{cao2025formular1,
  title   = {Formula-R1: Incentivizing LLM Reasoning over Complex Tables with Numerical Computation via Formula-Driven Reinforcement Learning},
  author  = {Cao, Lang and Xu, Jingxian and Liu, Hanbing and Wang, Jinyu and Zhou, Mengyu and Dong, Haoyu and Han, Shi and Zhang, Dongmei},
  journal = {arXiv preprint arXiv:2505.23667},
  year    = {2025},
  note    = {v1 titled ``Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models''}
}

@inproceedings{jiang2026tablemind,
  title     = {TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning},
  author    = {Jiang, Chuang and Cheng, Mingyue and Tao, Xiaoyu and Mao, Qingyang and Ouyang, Jie and Liu, Qi},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining (WSDM '26)},
  year      = {2026},
  note      = {arXiv:2509.06278}
}

@article{yang2025tablegptr1,
  title   = {TableGPT-R1: Advancing Tabular Reasoning Through Reinforcement Learning},
  author  = {Yang, Saisai and Huang, Qingyi and Yuan, Jing and Zha, Liangyu and Tang, Kai and Yang, Yuhang and Wang, Ning and Wei, Yucheng and Li, Liyao and Ye, Wentao and Chen, Hao and Zhang, Tao and Zhou, Junlin and Wang, Haobo and Chen, Gang and Zhao, Junbo},
  journal = {arXiv preprint arXiv:2512.20312},
  year    = {2025}
}

@article{qiu2025opentabler1,
  title   = {OpenTable-R1: A Reinforcement Learning Augmented Tool Agent for Open-Domain Table Question Answering},
  author  = {Qiu, Zipeng},
  journal = {arXiv preprint arXiv:2507.03018},
  year    = {2025}
}

@inproceedings{zhou-etal-2026-mixture,
  title     = {Mixture-of-Minds: Multi-Agent Reinforcement Learning for Table Understanding},
  author    = {Zhou, Yuhang and Zhang, Mingrui and Li, Ke and Wang, Mingyi and Liu, Qiao and Wang, Qifei and Liu, Jiayi and Liu, Fei and Li, Serena and Li, Weiwei and Gao, Mingze and Kumar, Abhishek and Fan, Xiangjun and Zhao, Zhuokai and Zhang, Lizhu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {2424--2439},
  year      = {2026},
  address   = {San Diego, California, United States},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2026.acl-long.112},
  url       = {https://aclanthology.org/2026.acl-long.112/}
}

% ---------- Brief mentions (RL text-to-SQL; evidence/grounding rewards) ----------
@article{pourreza2025reasoningsql,
  title   = {Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL},
  author  = {Pourreza, Mohammadreza and Talaei, Shayan and Sun, Ruoxi and Wan, Xingchen and Li, Hailong and Mirhoseini, Azalia and Saberi, Amin and Ar{\i}k, Sercan {\"O}.},
  journal = {arXiv preprint arXiv:2503.23157},
  year    = {2025}
}

@inproceedings{ma2025sqlr1,
  title     = {SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning},
  author    = {Ma, Peixian and Zhuang, Xialie and Xu, Chengjin and Jiang, Xuhui and Chen, Ran and Guo, Jian},
  booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS 2025)},
  year      = {2025},
  note      = {arXiv:2504.08600; venue per arXiv comment ``Accepted by NeurIPS 2025''}
}

@article{yao2025arctictext2sqlr1,
  title   = {Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL},
  author  = {Yao, Zhewei and Sun, Guoheng and Borchmann, Lukasz and Nuti, Gaurav and Shen, Zheyu and Deng, Minghang and Zhai, Bohan and Zhang, Hao and Li, Ang and He, Yuxiong},
  journal = {arXiv preprint arXiv:2505.20315},
  year    = {2025}
}

@article{gajjar2026rsat,
  title   = {RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners},
  author  = {Gajjar, Jugal and Subramaniakuppusamy, Kamalasankari},
  journal = {arXiv preprint arXiv:2605.00199},
  year    = {2026},
  note    = {arXiv comment reports SURGeLLM Workshop at ACL 2026 (unconfirmed on Anthology)}
}

@article{kwok2026retab,
  title   = {Enhancing Table Reasoning with Deterministic Table-State Rewards},
  author  = {Kwok, Tung Sum Thomas and Wang, Xinyu and He, Hengzhi and Lin, Xiaofeng and Lu, Peng and Ma, Liheng and Wang, Chunhe and Mak, Chun Ho and Luo, Yuyu and Wu, Ying Nian and Ding, Lei and Cheng, Guang},
  journal = {arXiv preprint arXiv:2601.22530},
  year    = {2026}
}
```

### Inferences
- **Recommended core set for the final report (12 papers):** Seq2SQL, NSM, MAPO, MeRL, Learning to Reduce, José et al. 2024, Table-R1 (region), Table-R1 (ITS), Reasoning-Table, Formula-R1/Fortune, TableMind, OpenTable-R1.
- **Optional (pick 1-3):** Table-r1 (program), TableGPT-R1, Mixture-of-Minds, and Reasoning-SQL as the one text-to-SQL mention.
- **Exclude:** Table-Critic (not RL), V-tableR1 (multimodal), SQL-R1/Arctic (no table selection) unless one sentence on text-to-SQL is wanted.

### Gaps
- The DOI and page numbers for TableMind (WSDM '26) and José et al. (BRACIS 2024, Springer LNCS/LNAI volume) were not verified, so they are omitted from the BibTeX.
- The NeurIPS proceedings page numbers for MAPO and SQL-R1 were not verified.
- The RSAT workshop venue comes from an arXiv-comment summary and was not confirmed on the ACL Anthology.
