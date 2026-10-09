# Evidence-Selection Frameworks in Multi-Hop Table-Text QA and Multi-Table QA (what a PPO entity selector plugs into or competes with)

Scope note for the report writer: every paper below was checked against its arXiv abstract page / arXiv API metadata, ACL Anthology, IJCAI or ICLR proceedings. Method details and numbers come from the full PDFs (text extracted locally), unless marked otherwise. Papers are labelled **[CORE]** (directly relevant, cite) or **[SECONDARY]** (verified, cite only if space allows). Uncertain details are marked **(UNCERTAIN)**.

**Important correction to the brief:** arXiv 2407.04858 is **not** "Ribeiro et al.". Its first author is **Marcos M. José**. The full list is Marcos M. José, Flávio N. Cação, Maria F. Ribeiro, Rafael M. Cheang, Paulo Pirozelli, Fabio G. Cozman, and it appeared at BRACIS 2024 — [arXiv 2407.04858](https://arxiv.org/abs/2407.04858). Cite it as "José et al. (2024)".

---

## Q1. Table-text multi-hop QA (HybridQA / OTT-QA): how does each system select cells, rows, linked passages and tables at each hop? (incl. close reading of the 2024 deep-RL paper)

### Takeaway
Pre-LLM HybridQA/OTT-QA systems choose evidence with **separately trained rankers or classifiers** (cell ranker, row retriever, passage filter, chain extractor). These are trained with **distant or weak supervision built from where the answer string appears**, never against final answer correctness. At inference they use **greedy top-1** selection (HYBRIDER) or **fixed top-k** retention (MITQA, CORE, COS). The only RL work on table-text QA, José et al. (2024), learns *which module to call* (retrieve text / retrieve table / answer). It does not learn *which entity to select*. Its best PPO agent collapsed to the trivial "always retrieve text" policy (F1 19.03, equal to the best fixed action sequence).

### Cited Findings

**[CORE] José et al. 2024 — "Question Answering with Texts and Tables through Deep Reinforcement Learning" (BRACIS 2024; arXiv 2407.04858). Close reading:**
- **Dataset:** OTT-QA (open-domain). Training sampled questions from the 41,469-question train set. Results are reported on the OTT-QA validation set — [José et al.](https://arxiv.org/abs/2407.04858)
- **Actions (3, module-level):** A1 = Retrieve Texts, A2 = Retrieve Tables, A3 = Generate Answer. The agent is "an action selector to determine which pre-trained component should be activated", following MSCQA — [José et al.](https://arxiv.org/abs/2407.04858)
- **Retrieval mechanics (fixed, not learned):**
  - The first retrieval step fetches 10 passages (text or table) and builds 10 "blocks".
  - Each further retrieval step concatenates the items in each block and retrieves 4 new passages per block.
  - There are at most 3 retrieval steps (an episode is at most 4 timesteps).
  - When the agent answers, the reader receives all non-repeated passages, capped at 50 — [José et al.](https://arxiv.org/abs/2407.04858)
- **Modules:** retriever = BM25 or a Tri-encoder; reader = the public Fusion-in-Encoder (FiE) from Chain-of-Skills. "We opted not to retrain any models from the literature" — [José et al.](https://arxiv.org/abs/2407.04858)
- **State/observation:** a sequence of 11 vectors: one question embedding plus one per block. Each block vector is the *average* of its passage embeddings (Tri-encoder 512-d, or MPNet multi-qa 768-d for BM25) — [José et al.](https://arxiv.org/abs/2407.04858)
- **Reward:** delayed (terminal) reward.
  - R = 2 if the prediction exactly matches the gold answer.
  - R = F1 if there is partial token overlap.
  - R = −0.5 otherwise.
  - Each retrieval action costs −0.02.
  - The no-overlap penalty was softened from −1 (R3-style) to −0.5 "due to the complexity of the problem" — [José et al.](https://arxiv.org/abs/2407.04858)
- **RL algorithms:**
  - DQN and PPO from Stable-Baselines3, with MLP, LSTM, GRU and Transformer feature extractors.
  - PPO settings: n_steps 128, batch 32, 60 epochs. Actor and critic heads (64 and 32 units) sit on a shared feature extractor.
  - Training ran 1M timesteps with the Tri-encoder but only 100k with BM25, because BM25 inference was up to 20× slower.
  - "Each training execution took over a week", so there were no multiple seeds — [José et al.](https://arxiv.org/abs/2407.04858)
- **Results on the OTT-QA dev set:**
  - The best fixed baseline was BM25 with A1→A1→A1→A3 (always retrieve text) = EM 14.81 / F1 19.03.
  - The best DRL agent (PPO + Transformer, BM25) also scored **EM 14.81 / F1 19.03**. The authors say it was "converging to the trivial solution of only picking the best action on average (retrieving texts)".
  - The best DQN agent (GRU) scored 13.23/17.16.
  - The best Tri-encoder fixed baseline scored F1 8.24, and the best Tri-encoder agent scored F1 6.44 (PPO-MLP in the text; Table 2 lists 6.13 for DQN-GRU) — [José et al.](https://arxiv.org/abs/2407.04858)
- **Comparison to prior work:** 19.03 F1 vs 20.7 for the non-RL Iterative-Retrieval (sparse) + Cross-Block Reader, and vs COS at 63.2 F1 on dev — [José et al.](https://arxiv.org/abs/2407.04858)
- **Authors' own diagnosis:**
  - "Table retrieval proved noisy, especially in iterative search, as tables often contain excessive data with relevant information typically in a single row". They suggest breaking tables into rows.
  - Network architecture had "minimal impact".
  - 100k BM25 steps were "potentially insufficient".
  - Future work: add a BERT joint reranker and per-passage (not averaged) observations, and integrate the agent with COS — [José et al.](https://arxiv.org/abs/2407.04858)
- **What it does NOT do (derived from the method description):**
  - No entity-level action: it cannot pick a cell, row, column, specific passage or join.
  - No evidence-set state: blocks are averaged embeddings.
  - No evidence-recall reward term, only answer F1/EM plus a step cost.
  - The PPO critic is the Stable-Baselines3 default value head. No GAE design or reward-shaping analysis is reported.
  - No HybridQA (closed-setting) or multi-table experiments.
  - Single run — [José et al.](https://arxiv.org/abs/2407.04858)
- Code: [github.com/MMenonJ/DRL_QA_TT](https://github.com/MMenonJ/DRL_QA_TT) (as stated in [José et al.](https://arxiv.org/abs/2407.04858))

**[CORE] HYBRIDER — Chen et al. 2020, "HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data" (Findings of EMNLP 2020; arXiv 2004.07347)**
- **Pipeline:** Linking, then Reasoning = Ranking model p_f(c|q,C), Hop model p_h(c′|q,c), and RC model p_r(a|P,q).
- **Linking step:** cell matching plus a passage retriever (TF-IDF and longest-substring ensemble) with distance threshold τ. Linked passages are mapped back to cells — [HybridQA](https://arxiv.org/abs/2004.07347)
- **Ranking model:** a BERT classifier over each retrieved cell plus its same-row neighbours. Each cell is encoded as (content, location, description, source, score).
- **Hop model:** chooses the neighbouring cell in the row that holds the answer.
- **Training:** the three models are trained *independently* with weak supervision from "synthesized reasoning paths" inferred from where the answer appears. The authors say these paths are "somewhat noisy".
- **Inference:** "we use greedy search at first two steps to remain only the highest probably cell" — [HybridQA](https://arxiv.org/abs/2004.07347)
- **Results:** dev 44.0/50.7 and test 43.8/50.6 EM/F1 (BERT-large, τ=0.8), vs 8.3 (table-only) and 19.1 (passage-only) test EM. Human performance is 88.2/93.5 — [HybridQA](https://arxiv.org/abs/2004.07347)
- **Error analysis (full dev set), per-stage accuracies:** linking 87.4%, ranking 87.9%, hop 89.2%, RC 62%. Errors are "quite uniformly distributed" except RC — [HybridQA](https://arxiv.org/abs/2004.07347)
- **Open domain:** in OTT-QA, HYBRIDER on top of BM25 drops to 8.4/10.6 test EM/F1 with top-1 retrieval, and 9.7/12.8 with the best top-K — [OTT-QA](https://arxiv.org/abs/2010.10439)

**[CORE] OTT-QA — Chen et al. 2021, "Open Question Answering over Tables and Text" (ICLR 2021; arXiv 2010.10439)**
- **Iterative retriever (sparse/dense):** retrieves 10 texts and 10 table segments, then expands each hop by retriever similarity (as summarised by [José et al.](https://arxiv.org/abs/2407.04858)).
- **Why the authors replace it with "fusion retrieval"** (early-fusing table segments with linked passages, then single-shot dense retrieval), quoted directly:
  - Iterative retrieval "often requires having supervision signals for every retrieval step ... which is not available in OTT-QA".
  - It "suffers from the problem of error propagation, as early mistakes can propagate to later retrieval stages".
  - It requires re-encoding the query with the entire history at every hop — [OTT-QA](https://arxiv.org/abs/2010.10439)
- **Results (EM/F1):** Iterative-Retrieval + Cross-Block Reader dev 17.1/20.7 (sparse), test 16.9/20.9. Fusion-Retrieval + Cross-Block Reader dev 28.1/32.5, test 27.2/31.5 — [OTT-QA](https://arxiv.org/abs/2010.10439)

**[CORE] MuGER² — Wang et al. 2022 (Findings of EMNLP 2022; arXiv 2210.10350)**
- **Unified retriever:** scores multi-granularity evidence units (question-to-header/column, cell, and link/passage), trained with distant supervision plus contrastive learning — [MuGER2](https://arxiv.org/abs/2210.10350)
- **Evidence selector (E-SEL):** compares aggregated table-cell vs passage scores. It declares the answer "In-Table" if s_tab > s_pass, else "In-Passage", then navigates to the single fine-grained cell or link for the reader. This is a fixed decision rule over retriever scores.
- E-SEL adds +7.8 EM / +7.9 F1 — [MuGER2](https://arxiv.org/abs/2210.10350)
- **HybridQA results:** MuGER²-large dev 57.1/67.3, test 56.3/66.2 — [MuGER2](https://arxiv.org/abs/2210.10350)

**[CORE] MITQA — Kumar et al. 2023, "Multi-Row, Multi-Span Distant Supervision For Table+Text Question Answering" (ACL 2023 long, pp. 8080–8094; arXiv 2112.07337, originally titled "Multi-Instance Training for Question Answering Across Table and Linked Text")**
- **Retrieval unit:** a table row together with its linked passages.
- **Modules:** TableRetriever, RowPassageLinker, PassageFilter, RowRetriever and AnswerExtractor, plus a joint Row+Span Reranker.
- RowRetriever uses a multi-instance loss because the gold answer can match several rows or spans (noisy distant supervision) — [MITQA (ACL Anthology)](https://aclanthology.org/2023.acl-long.449/); [arXiv](https://arxiv.org/abs/2112.07337)
- **Evidence that top-1 selection is lossy:** "if we retain the top five rows from RowRetriever, gold row recall jumps 8–9% compared to using only the top one row". MITQA therefore keeps the top 5 and reranks jointly — [MITQA](https://arxiv.org/abs/2112.07337)
- **HybridQA results:** dev 65.5/72.7, test 64.3/71.9 EM/F1 (vs POINTR+MATE test 62.8/70.2) — [MITQA](https://arxiv.org/abs/2112.07337)

**[CORE] S³HQA — Lei et al. 2023 (ACL 2023; arXiv 2305.11725)**
- **Three stages:**
  - A row-based retriever with *refinement training*: train first on unambiguous instances, then use that model to denoise the noisy distant labels.
  - A passage retriever.
  - A **rule-based "hybrid selector"** keyed on question type: for bridge questions it keeps "a single row and its linked passage"; for comparison/count questions it keeps multiple rows and deletes low-scoring linked passages.
  - A generation-based reasoner (row-wise generator or LLM prompting) — [S3HQA](https://arxiv.org/abs/2305.11725)
- **Results:** HybridQA dev 68.4/75.3, test 67.9/75.5 EM/F1. It ranked first on the HybridQA leaderboard at publication — [S3HQA](https://arxiv.org/abs/2305.11725)
- **Ablations:** row retriever top-1 accuracy is 88.0 (DeBERTa) vs 84.1 without refinement training. Removing the hybrid selector costs −2.9 EM / −1.6 F1 — [S3HQA](https://arxiv.org/abs/2305.11725)

**[CORE] TACR — Wu et al. 2023, "A Table-alignment-based Cell-selection and Reasoning Model for Hybrid Question-Answering" (Findings of ACL 2023; arXiv 2305.14682)**
- **Motivation:** "it is generally unrealistic to iterate over all table rows, columns, and linked passages to retrieve evidence".
- **Method:** a multi-task cell-selection model with a table–question alignment objective returns the top-k cells. A QA module then reads the row containing the selected cells — [TACR](https://arxiv.org/abs/2305.14682)
- **Results:** alignment improves cell-selection accuracy by 2.5/3.9/4.3 points at top-1/3/5 (ALBERT) and MRR by 3.7. Row/column selection accuracy exceeds 90% — [TACR](https://arxiv.org/abs/2305.14682)
- **Stated limitations:** no numerical reasoning across cells (counting/comparing). "Strong ability in column selection while performing relatively worse in row selection" — [TACR](https://arxiv.org/abs/2305.14682)

**[CORE] DocHopper — Sun, Cohen & Salakhutdinov 2021, "Iterative Hierarchical Attention for Answering Complex Questions over Long Documents" (arXiv 2106.00200; no venue found)**
- **Method:** at each hop it attends over paragraph- or sentence-level embeddings, retrieves a summary or sentence, and **updates the query embedding** with it. The model is end-to-end differentiable.
- Attention is supervised when (distant) labels exist. Tables are flattened for HybridQA — [DocHopper](https://arxiv.org/abs/2106.00200)
- **HybridQA results:** dev 47.7/55.0, test 46.3/53.3. Ablations (dev) **(UNCERTAIN — column alignment in the PDF text extraction)**: without query update about 44.2/50.9; single-hop about 27.8/34.1, i.e. multi-hop iteration matters — [DocHopper](https://arxiv.org/abs/2106.00200)

**[CORE] CORE — Ma et al. 2022, "Open-domain Question Answering via Chain of Reasoning over Heterogeneous Knowledge" (Findings of EMNLP 2022; arXiv 2210.12338)**
- **Pipeline:**
  - A DPR retriever fetches 100 tables.
  - A BERT **linker** retrieves passages per row.
  - A **chainer** selects the top-50 (row, passage) chains by **zero-shot T0 question-generation likelihood**: the higher the probability of generating the question from the chain, the higher its rank.
  - A FiD reader answers — [CORE](https://arxiv.org/abs/2210.12338); summarised also in [José et al.](https://arxiv.org/abs/2407.04858)
- **OTT-QA results:** dev 49.0/55.7, test 47.3/54.1 (joint).
- **Diagnostic:** giving the FR+CBR baseline oracle links raises dev EM from 28.1 to 35.2, so linking errors cost about 7 EM — [CORE](https://arxiv.org/abs/2210.12338)

**[CORE] Chain-of-Skills (COS) — Ma et al. 2023 (ACL 2023, pp. 1599–1618; arXiv 2305.03130)**
- **Retrieval skills:** modular, multi-task-trained skills (single retrieval, expanded-query retrieval, entity span proposal, entity linking, reranking) — [COS](https://arxiv.org/abs/2305.03130)
- **OTT-QA inference:**
  - 100 tables are retrieved and split into rows.
  - The reranker keeps the top 200 rows.
  - Each row gets 10 expanded-query passages plus 1 linked-entity passage.
  - CORE's chainer keeps the top 100 passages.
  - A FiE reader answers ([José et al.](https://arxiv.org/abs/2407.04858) description; [COS](https://arxiv.org/abs/2305.03130))
- **Results:** COS+FiE on OTT-QA dev 56.9/63.2, test 54.9/61.5 EM/F1 — [COS](https://arxiv.org/abs/2305.03130)

**[SECONDARY] CARP — Zhong et al. 2022, "Reasoning over Hybrid Chain for Table-and-Text Open Domain QA" (IJCAI 2022, pp. 4531–4537; arXiv 2201.05880)**
- **Method:** early-fused table+passage blocks are retrieved by a DPR-style dense retriever. A trained hybrid-chain extractor then identifies the reasoning chain fed to the reader — [CARP](https://arxiv.org/abs/2201.05880); [IJCAI](https://www.ijcai.org/proceedings/2022/0629.pdf)
- **OTT-QA results:** dev 33.2/38.6, test 32.5/38.5 — [CORE Table 1](https://arxiv.org/abs/2210.12338)

**[SECONDARY] OTTeR — Huang et al. 2022, "Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA" (Findings of EMNLP 2022; arXiv 2210.05197)**
- **Method:** a retriever-only improvement using modality-enhanced representations, mixed-modality hard negatives and synthetic pre-training — [OTTeR](https://arxiv.org/abs/2210.05197)
- **Results:** with the Cross-Block Reader, OTT-QA dev 37.1/42.8, test 37.3/43.1 — [OTTeR](https://arxiv.org/abs/2210.05197)

**[SECONDARY] DEHG — Feng et al. 2022, "Multi-Hop Open-Domain Question Answering over Structured and Unstructured Knowledge" (Findings of NAACL 2022, pp. 151–156)**
- **Method:** a Document-Entity Heterogeneous Graph with cell–passage, cell–question and passage–question subgraphs, BERT node encoding, message passing and an LSTM answer decoder — [ACL Anthology](https://aclanthology.org/2022.findings-naacl.12/)
- **Results:** HybridQA dev 65.2/76.3, test 63.9/75.5 as tabulated in [S3HQA](https://arxiv.org/abs/2305.11725) **(UNCERTAIN — read from a multi-column PDF table; confirm before citing)**.
- Not on arXiv.

### Inferences
- Every supervised HybridQA/OTT-QA selector (HYBRIDER's ranker and hop model, MITQA's RowRetriever, S3HQA's retriever, MuGER²'s unified retriever, TACR's cell selector) is trained on **labels derived from where the answer string appears**, not on final answer correctness. HYBRIDER explicitly trains its stages *independently* to avoid marginalising over paths. This is the gap an RL selector with an answer-correctness reward fills.
- The selection steps a PPO policy could replace are:
  - HYBRIDER's greedy rank-then-hop (top-1 twice);
  - S3HQA's rule-based hybrid selector (question-type rules);
  - MuGER²'s s_tab vs s_pass threshold rule;
  - MITQA's fixed top-5 retention;
  - CORE/COS's fixed top-50/100 chain budgets.
- These are all fixed-budget or greedy decisions with no learned STOP.
- José et al. is the closest prior RL work but operates at the **module** level with averaged-embedding states. Its collapse to the best fixed policy suggests that coarse actions plus a terminal-only reward give too little signal. That supports the project's design choices: entity-level actions, an evidence-recall reward term, a per-hop cost, and GAE.
- **Likely numeric targets:**
  - HybridQA (closed setting): HYBRIDER 43.8, MITQA 64.3, S3HQA 67.9 test EM.
  - OTT-QA: COS 54.9 test EM; José et al. 19.03 F1 (dev).

### Gaps
- TACR's end-to-end HybridQA EM/F1 could not be reliably read from the PDF text (garbled multi-column table). Check Table 2 of the paper.
- S3HQA's Anthology volume (main vs short paper) and pages were not verified. The arXiv comment says only "ACL 2023".
- DocHopper's publication venue was not found; treat it as an arXiv preprint.
- MITQA's OTT-QA numbers (about 40.0/45.1 dev, 36.4/41.9 test) were seen in the PDF but the column alignment is uncertain.

### BibTeX (Q1)
```bibtex
@inproceedings{jose2024question,
  title     = {Question Answering with Texts and Tables through Deep Reinforcement Learning},
  author    = {Jos{\'e}, Marcos M. and Ca{\c{c}}{\~a}o, Fl{\'a}vio N. and Ribeiro, Maria F. and Cheang, Rafael M. and Pirozelli, Paulo and Cozman, Fabio G.},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2024},
  note      = {arXiv:2407.04858}
}

@inproceedings{chen2020hybridqa,
  title     = {{HybridQA}: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data},
  author    = {Chen, Wenhu and Zha, Hanwen and Chen, Zhiyu and Xiong, Wenhan and Wang, Hong and Wang, William Yang},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1026--1036},
  year      = {2020}
}

@inproceedings{chen2021open,
  title     = {Open Question Answering over Tables and Text},
  author    = {Chen, Wenhu and Chang, Ming-Wei and Schlinger, Eva and Wang, William Yang and Cohen, William W.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@inproceedings{wang2022muger2,
  title     = {{MuGER}$^2$: Multi-Granularity Evidence Retrieval and Reasoning for Hybrid Question Answering},
  author    = {Wang, Yingyao and Bao, Junwei and Duan, Chaoqun and Wu, Youzheng and He, Xiaodong and Zhao, Tiejun},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year      = {2022}
}

@inproceedings{kumar2023multirow,
  title     = {Multi-Row, Multi-Span Distant Supervision For Table+Text Question Answering},
  author    = {Kumar, Vishwajeet and Gupta, Yash and Chemmengath, Saneem and Sen, Jaydeep and Chakrabarti, Soumen and Bharadwaj, Samarth and Pan, Feifei},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {8080--8094},
  year      = {2023}
}

@inproceedings{lei2023s3hqa,
  title     = {{S}$^3${HQA}: A Three-Stage Approach for Multi-hop Text-Table Hybrid Question Answering},
  author    = {Lei, Fangyu and Li, Xiang and Wei, Yifan and He, Shizhu and Huang, Yiming and Zhao, Jun and Liu, Kang},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2023}
}

@inproceedings{wu2023tacr,
  title     = {{TACR}: A Table-alignment-based Cell-selection and Reasoning Model for Hybrid Question-Answering},
  author    = {Wu, Jian and Xu, Yicheng and Gao, Yan and Lou, Jian-Guang and Karlsson, B{\"o}rje F. and Okumura, Manabu},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023}
}

@article{sun2021iterative,
  title   = {Iterative Hierarchical Attention for Answering Complex Questions over Long Documents},
  author  = {Sun, Haitian and Cohen, William W. and Salakhutdinov, Ruslan},
  journal = {arXiv preprint arXiv:2106.00200},
  year    = {2021}
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

@inproceedings{zhong2022reasoning,
  title     = {Reasoning over Hybrid Chain for Table-and-Text Open Domain Question Answering},
  author    = {Zhong, Wanjun and Huang, Junjie and Liu, Qian and Zhou, Ming and Wang, Jiahai and Yin, Jian and Duan, Nan},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI)},
  pages     = {4531--4537},
  year      = {2022}
}

@inproceedings{huang2022mixed,
  title     = {Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in {OpenQA}},
  author    = {Huang, Junjie and Zhong, Wanjun and Liu, Qian and Gong, Ming and Jiang, Daxin and Duan, Nan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  pages     = {4117--4129},
  year      = {2022}
}

@inproceedings{feng2022multihop,
  title     = {Multi-Hop Open-Domain Question Answering over Structured and Unstructured Knowledge},
  author    = {Feng, Yue and Han, Zhen and Sun, Mingming and Li, Ping},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2022},
  pages     = {151--156},
  year      = {2022}
}
```

---

## Q2. LLM-era graph / agent frameworks with explicit selection steps (ODYSSEY, TableRAG, Chain-of-Table, Dater, Binder, ReAcTable, H-STAR, 2025–2026 frameworks)

### Takeaway
LLM-era frameworks do the selection with **zero-shot LLM prompts plus embedding-similarity thresholds plus fixed graph-traversal heuristics**. None is trained against answer correctness:
- ODYSSEY uses BFS with 3-hop pruning and a 0.8 similarity threshold.
- N2N-GQA uses GraphRank centrality pruning.
- Chain-of-Table uses greedy LLM operation choice (f_select_row / f_select_column).
- TableRAG uses LM-expanded schema and cell retrieval.

These are the "heuristic / similarity / greedy-LLM" selection steps the project's learned selector targets.

### Cited Findings

**[CORE] ODYSSEY — Agarwal, S & Devaguptapu 2025, "Hybrid Graphs for Table-and-Text based Question Answering using LLMs" (NAACL 2025 main; arXiv 2501.17767).** "ODYSSEY" is the method name inside this paper, not a separate system.
- **Graph construction:**
  - The table component is the retrieved sub-table restricted to question-relevant headers.
  - A document–entity bipartite graph links NER entities to their documents.
  - Entities are connected to matching sub-table cells — [arXiv HTML](https://arxiv.org/html/2501.17767); [ACL Anthology](https://aclanthology.org/2025.naacl-long.39/)
- **Selection mechanism:**
  - GPT-3.5 extracts question entities and types and maps entities to headers via a prompt.
  - instructor-xl embeddings match question entities to graph nodes, keeping "the highest-ranked entity surpassing a 0.8 threshold".
  - A **BFS** then performs "a 3-hop traversal, which results in a pruned graph".
  - If nothing matches, the whole table and passages are kept (about 10% / 8% of cases) — [arXiv HTML](https://arxiv.org/html/2501.17767)
- **Training:** none (zero-shot) — [arXiv HTML](https://arxiv.org/html/2501.17767)
- **Results (500-question dev samples, GPT-4):**
  - HybridQA EM 58.4 / F1 71.8 (vs 55.4 EM with full context).
  - OTT-QA EM 62.02 / F1 73.02.
  - Token reduction of 45% (HybridQA) and 53% (OTT-QA) — [arXiv HTML](https://arxiv.org/html/2501.17767)
- **Error analysis (100 HybridQA samples, 41 wrong):**
  - 17 expression mismatch;
  - **14 semantic-module (selection) errors**: 8 entity-matching, 2 entity-extraction, 4 entity–header mapping;
  - 3 LLM errors;
  - 7 dataset issues — [arXiv HTML](https://arxiv.org/html/2501.17767)
- **Latency:** 3.6–5 s per instance vs 1–1.5 s for the zero-shot baseline — [arXiv HTML](https://arxiv.org/html/2501.17767)

**[CORE] N2N-GQA — Sharafath, Annamalai, Murugan & Venugopalan 2026, "N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs" (arXiv 2601.06603, Jan 2026; preprint)**
- **Graph:** ColBERTv2 retrieves top-k items. Nodes are retrieved passages or serialized table rows; edges are TF-IDF-weighted shared terms — [N2N-GQA](https://arxiv.org/html/2601.06603)
- **Selection:**
  - "GraphRank" score = S_sem_norm(v) × (1 + (1−α) × S_struct_norm(v)), with α = 0.85 (semantic similarity times weighted degree centrality).
  - Graphs are pruned to 5–10 nodes at intermediate hops and 12–25 at the final hop.
  - "No training required" — [N2N-GQA](https://arxiv.org/html/2601.06603)
- **Results:**
  - OTT-QA (500-question sample): 48.80 EM / 57.26 F1, vs vanilla RAG 8.00 EM and query-decomposition RAG 28.60 EM.
  - HybridQA (200 questions): 41.50 / 48.17.
  - Still 8.1 EM below fine-tuned COS (56.9) — [N2N-GQA](https://arxiv.org/html/2601.06603)
- **Stated limitations:** evidence absent from the top-k; entity ambiguity; loss of structure in table serialization — [N2N-GQA](https://arxiv.org/html/2601.06603)

**[CORE] Chain-of-Table — Wang et al. 2024, "Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding" (ICLR 2024; arXiv 2401.04398)**
- **Method:**
  - `DynamicPlan(T, Q, chain)` prompts the LLM to pick the next atomic operation (e.g., `f_select_row`, `f_select_column`, `f_add_col`, `f_group_by`, `f_sort`) from an operation pool.
  - `GenerateArgs` fills in the arguments.
  - The loop runs until the end tag [E] is produced — [Chain-of-Table](https://arxiv.org/abs/2401.04398)
- **Selection strategy:** "employs a greedy search strategy ... instead of relying on self-consistency sampling". It uses 50% fewer queries than Binder and 75% fewer than Dater — [Chain-of-Table](https://arxiv.org/abs/2401.04398)
- **Results (PaLM 2):** TabFact 86.61, WikiTQ 67.31 (Binder 76.98 / 61.48; Dater 84.63 / —) — [Chain-of-Table](https://arxiv.org/abs/2401.04398)
- **Scope:** single table only.

**[CORE] TableRAG — Chen et al. 2024, "TableRAG: Million-Token Table Understanding with Language Models" (NeurIPS 2024; arXiv 2410.04739)**
- **Method:** LM query expansion feeds (a) **schema retrieval**, which selects relevant column names, and (b) **cell retrieval**, which selects relevant distinct or frequent cell values from an encoded cell database. Selection is by embedding similarity under a token budget. A program-aided (ReAct-style) LM agent then solves the query — [TableRAG](https://arxiv.org/abs/2410.04739)
- **Results (GPT-3.5-Turbo):** ArcadeQA 49.2 / BirdQA 45.5 accuracy, vs RowColRetrieval 37.7 / 39.6 and ReadSchema 43.1 / 40.3 — [TableRAG](https://arxiv.org/abs/2410.04739)
- **Scope:** single (large) table.

**[SECONDARY] H-STAR — Abhyankar, Gupta, Roth & Reddy 2025 (NAACL 2025; arXiv 2407.05952)**
- Two stages:
  - (1) step-wise table extraction: "multi-view" column retrieval followed by row extraction;
  - (2) adaptive semantic/symbolic reasoning by question type.
- Selection is LLM-driven (SQL plus text) — [H-STAR](https://arxiv.org/abs/2407.05952)

**[SECONDARY] Dater — Ye et al. 2023, "Large Language Models are Versatile Decomposers" (SIGIR 2023; arXiv 2301.13808)**
- The LLM decomposes "huge evidence (a huge table) into sub-evidence (a small table)" by retaining relevant rows and columns via prompting, and decomposes the question into sub-questions.
- Uses a "parsing-execution-filling" strategy — [Dater](https://arxiv.org/abs/2301.13808)

**[SECONDARY] Binder — Cheng et al. 2023, "Binding Language Models in Symbolic Languages" (ICLR 2023; arXiv 2210.02875)**
- Codex generates a SQL/Python program with neural LM API calls in a single pass. Selection is implicit in the program — [Binder](https://arxiv.org/abs/2210.02875); characterisation in [Chain-of-Table](https://arxiv.org/abs/2401.04398)

**[SECONDARY] ReAcTable — Zhang et al. 2023 (arXiv 2310.00815)**
- A ReAct framework for table QA that iteratively generates SQL/Python to produce intermediate tables — [ReAcTable](https://arxiv.org/abs/2310.00815)
- **(UNCERTAIN venue:** believed to be PVLDB 2024 but not verified; cite as arXiv.)

**[SECONDARY] Agentic DAG-Orchestrated Planner — Kirushikesh D B et al. 2026, "Agentic DAG-Orchestrated Planner Framework for Multi-Modal, Multi-Hop Question Answering in Hybrid Data Lakes" (arXiv 2603.14229; preprint)**
- An LLM planner compiles NL queries into DAG execution plans over table and document stores, with schema-aware reasoning and structural/semantic validation.
- Reports +14.8% absolute correctness and +10.7% completeness over baselines — [arXiv 2603.14229](https://arxiv.org/abs/2603.14229)

### Inferences
- ODYSSEY and N2N-GQA are the most natural **plug-in points** for a learned selector in the LLM era. Both build an explicit heterogeneous graph (cells/rows, entities, passages) and then prune it with a fixed heuristic: BFS depth 3 with a 0.8 threshold, or GraphRank with α = 0.85 and fixed node budgets.
- A PPO policy that adds nodes one at a time with a STOP action is a drop-in replacement for the pruning step. ODYSSEY's own error analysis (14/41 errors from the entity-matching/selection module) quantifies the head-room.
- Chain-of-Table's greedy LLM choice of `f_select_row`/`f_select_column` is the clearest "greedy-LLM selection" foil. It is single-table and has no learned value estimate.
- None of these LLM frameworks reports a learned critic or a per-hop cost. Cost appears only as measured token or latency savings (ODYSSEY −45/−53% tokens; Chain-of-Table −50/−75% queries).

### Gaps
- TableRAG (Yu, Jian & Chen 2025, "TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning", EMNLP 2025, [arXiv 2506.10380](https://arxiv.org/abs/2506.10380)) was verified on arXiv but its selection mechanism was not read. It shares a name with Chen et al. 2024, so avoid confusing them.
- ODYSSEY and N2N-GQA evaluate on 500/200-question samples, not the full dev/test sets, so the numbers are not directly comparable to supervised systems.
- No 2025–2026 table-text framework was found that trains its hop-wise evidence selector with RL against answer correctness. This is an absence-of-evidence statement from a limited search.

### BibTeX (Q2)
```bibtex
@inproceedings{agarwal2025hybrid,
  title     = {Hybrid Graphs for Table-and-Text based Question Answering using {LLMs}},
  author    = {Agarwal, Ankush and S, Ganesh and Devaguptapu, Chaitanya},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL)},
  year      = {2025}
}

@article{sharafath2026n2ngqa,
  title   = {{N2N-GQA}: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using {LLMs}},
  author  = {Sharafath, Mohamed and Annamalai, Aravindh and Murugan, Ganesh and Venugopalan, Aravindakumar},
  journal = {arXiv preprint arXiv:2601.06603},
  year    = {2026}
}

@inproceedings{wang2024chainoftable,
  title     = {Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding},
  author    = {Wang, Zilong and Zhang, Hao and Li, Chun-Liang and Eisenschlos, Julian Martin and Perot, Vincent and Wang, Zifeng and Miculicich, Lesly and Fujii, Yasuhisa and Shang, Jingbo and Lee, Chen-Yu and Pfister, Tomas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@inproceedings{chen2024tablerag,
  title     = {{TableRAG}: Million-Token Table Understanding with Language Models},
  author    = {Chen, Si-An and Miculicich, Lesly and Eisenschlos, Julian Martin and Wang, Zifeng and Wang, Zilong and Chen, Yanfei and Fujii, Yasuhisa and Lin, Hsuan-Tien and Lee, Chen-Yu and Pfister, Tomas},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{abhyankar2025hstar,
  title     = {{H-STAR}: {LLM}-driven Hybrid {SQL}-Text Adaptive Reasoning on Tables},
  author    = {Abhyankar, Nikhil and Gupta, Vivek and Roth, Dan and Reddy, Chandan K.},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL)},
  year      = {2025}
}

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

@article{zhang2023reactable,
  title   = {{ReAcTable}: Enhancing {ReAct} for Table Question Answering},
  author  = {Zhang, Yunjia and Henkel, Jordan and Floratou, Avrilia and Cahoon, Joyce and Deep, Shaleen and Patel, Jignesh M.},
  journal = {arXiv preprint arXiv:2310.00815},
  year    = {2023}
}

@article{db2026agentic,
  title   = {Agentic {DAG}-Orchestrated Planner Framework for Multi-Modal, Multi-Hop Question Answering in Hybrid Data Lakes},
  author  = {D B, Kirushikesh and Kesarwani, Manish and Madaan, Nishtha and Mehta, Sameep and Dennis, Aldrin and Ajay, Siddarth and B R, Rakesh and Rajagopal, Renu and Kairali, Sudheesh},
  journal = {arXiv preprint arXiv:2603.14229},
  year    = {2026}
}
```
(The Dater title in the published SIGIR version may differ slightly from the arXiv title, which is "...: Decompose Evidence and Questions for Table-based Reasoning". **UNCERTAIN.** Use the arXiv title if unsure.)

---

## Q3. Multi-table retrieval and QA (MMQA, JAR, MURRE, 2025–2026): how is the next table / join chosen?

### Takeaway
Multi-table selection splits into four styles:
- **iterative greedy** hop-by-hop selection: MMQA's MTR, MURRE (beam), Greedy-JAR;
- **global one-shot optimisation**: JAR's mixed-integer program, ARM's retrieve-all-at-once, CORE-T's single-LLM-call subset choice;
- **offline join-path indexing**: PEARL;
- **expand-then-refine**: REaR.

All are inference-time heuristics, similarity models or LLM prompts. None is trained on downstream answer or SQL correctness; MURRE lists this explicitly as future work. Recent results show greedy/iterative approaches are **not worse than global MIP** (Greedy-JAR ≈ JAR, 4–400× faster). Hop count is where everyone degrades: on 3-hop MMQA, R@2 is only 34–58%.

### Cited Findings

**[CORE] MMQA — Wu, Yang, Li, Ji, Okumura & Zhang 2025, "MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions" (ICLR 2025)**
- **Benchmark:** multi-table, multi-hop questions built from Spider. It totals 3,312 tables with question, SQL, gold answer and primary/foreign-key annotations.
- **Tasks evaluated:** multi-table retrieval, Text-to-SQL, multi-table QA, primary-key selection and foreign-key selection — [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)
- **MTR (proposed retriever):**
  - GPT-4-turbo decomposes the question (one-shot) into sub-questions q1..qn.
  - Over n rounds, each round retrieves the top-K (K = 2, 5, 10) tables with a fine-tuned single-table retriever (SGPT-5.8B or TableLlama-7B).
  - From round 2 onward, score = question-relevance × **table-relevance**, where table-relevance is **binary column overlap** with tables retrieved in the previous round ("assign a score of 1 and 0. When 0 is assigned, stop iterations").
  - Scores are summed across rounds — [MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)
- **MTR results:** best retrieval F1 is 68.3%. Question decomposition raises top-2 precision from 65.3 to 72.3 and recall from 62.3 to 64.7 — [MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)
- **LLM results:** O1-preview reaches 50.78 EM on multi-table QA vs GPT-4 at 28.88 and humans at 89.8. Foreign-key selection accuracy is 34.17 (O1-preview) vs 95.3 (human) — [MMQA](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)
- No arXiv version found; cite the ICLR proceedings ([mlanthology entry](https://mlanthology.org/iclr/2025/wu2025iclr-mmqa/)).

**[CORE] JAR — Chen, Zhang & Roth 2024, "Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval" (ACL 2024; arXiv 2404.09889)**
- **Method:**
  - An LLM decomposes the query into (concept, attribute) sub-queries.
  - Each table gets coarse- and fine-grained query–table relevance plus pairwise **table–table join compatibility**.
  - A **mixed-integer program** re-ranks by selecting the k-table set that maximises relevance plus compatibility, under connectivity and sub-query-coverage constraints — [JAR](https://arxiv.org/abs/2404.09889)
- **Results (Spider, Bird):** "outperforms the state-of-the-art approaches for table retrieval by up to 9.3% in F1 score and for end-to-end QA by up to 5.4% in accuracy" — [JAR](https://arxiv.org/abs/2404.09889)
- **Limitations:**
  - MIP "may have scalability issues".
  - Key–foreign-key relations "may not be the only connection".
  - Joins are single-column only.
  - Nothing is trained on answer correctness (inference-time optimisation with pretrained encoders and GPT-3.5) — [JAR (HTML)](https://arxiv.org/html/2404.09889)

**[CORE] MURRE — Zhang, Wang, Dou, Zhu & Che 2025, "MURRE: Multi-Hop Table Retrieval with Removal for Open-Domain Text-to-SQL" (COLING 2025; arXiv 2402.10666)**
- **Each hop has two phases:**
  - (i) Retrieval: cosine similarity between the embedded (rewritten) question and each table, normalised to a probability.
  - (ii) **Removal:** an LLM rewrites the question to *remove* information already covered by retrieved tables and expresses what remains in tabular form.
- **Search and stopping:** beam search ("to reduce the error cascades in multi-hop retrieval"); max hop H = 3 ("over 98% of questions" need ≤3). **LLM early stop**: the LLM emits a special mark ("None") when the retrieved tables suffice — [MURRE (Anthology PDF)](https://aclanthology.org/2025.coling-main.386.pdf)
- **Motivation:** standard multi-hop retrieval that concatenates retrieved documents to the query "tend[s] to retrieve tables similar to the retrieved ones but irrelevant to the question" — [MURRE](https://arxiv.org/abs/2402.10666)
- **Results:** +5.7% average over the previous SOTA on SpiderUnion and BirdUnion. Of the improved examples, 87.0% fix "Similar Candidate" errors (80 examples) and 14.9% fix "Semantic Gap" errors (14 examples). The authors call error cascades "minimal and significantly compensated" — [MURRE](https://aclanthology.org/2025.coling-main.386.pdf)
- **Stated limitation:** "our method does not consider the performance improvement brought by the text-to-SQL feedback ... We leave the retrieval recall improvement leveraging the results of text-to-SQL for future work" — [MURRE](https://aclanthology.org/2025.coling-main.386.pdf)

**[CORE] ARM — Chen, Zhang, Cafarella & Roth 2025, "Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method" (ACL 2025 per citing papers; arXiv 2501.18539)**
- **Argument:** agentic iterative RAG "is inefficient because successive queries depend on previous results rather than being guided by the organization of available data". ARM instead aligns the question with the data organisation and retrieves all needed objects at once — [ARM](https://arxiv.org/abs/2501.18539)
- **Results on Bird:**
  - Up to +5.2 execution accuracy vs RAG with decomposition, and +15.9 vs ReAct.
  - Recall 96.5 and perfect recall 92.7 with an average of 5.00 retrieved objects (best RAG baseline: 89.0 / 78.4).
- **Results on OTT-QA:** +5.5 / +19.3 F1 vs the same baselines — [ARM](https://arxiv.org/abs/2501.18539); [search snippet of abstract](https://arxiv.org/abs/2501.18539v1)
- **Venue:** ACL 2025, per the reference "Proceedings of the 63rd Annual Meeting..." in [REaR](https://arxiv.org/abs/2511.00805). The arXiv page shows no venue comment **(UNCERTAIN, minor)**.

**[CORE] Greedy-JAR — Boutaleb, Amann, Angarita & Naacke 2025, "Exploring Multi-Table Retrieval Through Iterative Search" (AI for Tabular Data Workshop, EurIPS 2025; arXiv 2511.13418)**
- Frames multi-table retrieval as an iterative search balancing "relevance, coverage, and joinability" with a **greedy join-aware** algorithm.
- Reports "competitive retrieval performance compared to the MIP-based approach while being 4-400x faster" on 5 NL2SQL benchmarks — [arXiv 2511.13418](https://arxiv.org/abs/2511.13418)
- **Unified-corpus results** (from PEARL's Table 1): 3-hop R@2 is Spider 60.00 (JAR 62.80), MMQA 44.50 (JAR 44.70) and Bird 54.30 (JAR 54.30) — [PEARL](https://arxiv.org/abs/2608.30291)

**[SECONDARY] CORE-T — Soliman, Gupta, Roth & Gurevych 2026, "CORE-T: COherent REtrieval of Tables for Text-to-SQL" (EMNLP 2026 main per arXiv comment; arXiv 2601.13111)**
- **Selection:** dense top-K retrieval, then "a single LLM call selects a coherent, joinable subset", then a two-step additive adjustment restores strongly compatible tables. Tables carry LLM-generated purpose metadata and a precomputed compatibility cache. Training-free.
- **Results:** up to +22.7 table-selection F1, up to 40% fewer tables, and up to +24.4 multi-table execution accuracy — [arXiv 2601.13111](https://arxiv.org/abs/2601.13111)

**[SECONDARY] PEARL — Ho, Kang, Kang & Yoon 2026, "PEARL: Front-Loading Relational Chains for Multi-Hop Table Retrieval" (EMNLP 2026 per arXiv comment, track unspecified; arXiv 2608.30291)**
- **Method:** training-free and offline. It discovers join paths, generates multi-hop queries over them, and indexes vertically partitioned sub-table units, so there is no query-time LLM inference — [PEARL](https://arxiv.org/abs/2608.30291)
- **3-hop R@2 on a unified corpus (Spider / MMQA / Bird):**

  | Method | Spider | MMQA | Bird |
  |---|---|---|---|
  | PEARL | 81.67 | 58.09 | 59.17 |
  | JAR | 62.80 | 44.70 | 54.30 |
  | MURRE | 52.78 | 34.05 | 45.83 |
  | CORE-T | 55.56 | 45.80 | 52.67 |

  PEARL's gain is +30.05% on Spider 3-hop. Its gap over baselines grows with hop count — [PEARL](https://arxiv.org/abs/2608.30291)
- **Limitation:** join paths are capped at length 3 — [PEARL](https://arxiv.org/abs/2608.30291)

**[SECONDARY] REaR — Agarwal, Singhal, Chen, Roy Choudhury, Roth & Gupta 2025, "REaR: Retrieve, Expand and Refine for Effective Multitable Retrieval" (arXiv 2511.00805; preprint)**
- **Motivation:** "Retrieval in current Text-to-SQL pipelines is fundamentally myopic: it optimizes for query–table relevance while ignoring table–table compatibility."
- **Method:** LLM-free. It retrieves, expands with joinable tables via precomputed column-embedding similarity, then refines by pruning — [REaR](https://arxiv.org/abs/2511.00805)
- **Results:**
  - Expansion raises Full Recall from 82.79% to 93.42%; refinement settles at 87.29% while keeping precision (36.09 → 36.33).
  - On Bird (UAE retriever + Gemini 2.0 Flash), execution accuracy goes 37.87 → 42.24 (+Expansion) → 43.09 (+Refine) — [REaR](https://arxiv.org/abs/2511.00805)

**[SECONDARY] Schema-graph multi-table QA — Wang, Costa, Kovaceva, Wang & Pereira 2025, "Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance" (Findings of EMNLP 2025; arXiv 2506.04427)**
- Searches a human-curated schema graph to build reasoning chains (join paths), with pruning and sub-path merging.
- Argues that semantic-similarity linking "work[s] well only on simplified hand-crafted datasets" — [arXiv 2506.04427](https://arxiv.org/abs/2506.04427)

### Inferences
- For "joined-table" actions in the project's MDP:
  - MMQA's MTR is a hand-coded greedy policy whose STOP rule is "column overlap = 0".
  - MURRE is a beam-search policy with an LLM-decided STOP.
  - JAR is a non-sequential global optimiser.
- A PPO selector can be positioned as *learning* the stop and join decisions that these methods hard-code, and as using downstream correctness (SQL/answer) as reward, which MURRE explicitly leaves open.
- PEARL's tables show that every method loses heavily from 2-hop to 3-hop (e.g., MMQA R@2: JAR 55.90 → 44.70; MURRE 49.15 → 34.05). This is a concrete, citable motivation for multi-step credit assignment (GAE) over myopic scoring.
- Greedy-JAR ≈ JAR suggests that a well-designed sequential selector need not sacrifice quality relative to global MIP. That is a useful argument that an MDP/greedy-rollout formulation is not inherently inferior.

### Gaps
- MMQA's exact question counts (2-table vs 3-table) were not extracted; only "3,312 tables" was confirmed.
- MMQA has no arXiv ID; cite ICLR/OpenReview.
- Greedy-JAR's exact scoring function was not visible in the abstract.
- PEARL's EMNLP 2026 track (main or Findings) is not stated in the arXiv comment.
- REaR's venue is unknown (arXiv only).

### BibTeX (Q3)
```bibtex
@inproceedings{wu2025mmqa,
  title     = {{MMQA}: Evaluating {LLMs} with Multi-Table Multi-Hop Complex Questions},
  author    = {Wu, Jian and Yang, Linyi and Li, Dongyuan and Ji, Yuliang and Okumura, Manabu and Zhang, Yue},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@inproceedings{chen2024jar,
  title     = {Is Table Retrieval a Solved Problem? Exploring Join-Aware Multi-Table Retrieval},
  author    = {Chen, Peter Baile and Zhang, Yi and Roth, Dan},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

@inproceedings{zhang2025murre,
  title     = {{MURRE}: Multi-Hop Table Retrieval with Removal for Open-Domain Text-to-{SQL}},
  author    = {Zhang, Xuanliang and Wang, Dingzirui and Dou, Longxu and Zhu, Qingfu and Che, Wanxiang},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics (COLING)},
  year      = {2025}
}

@inproceedings{chen2025arm,
  title     = {Can we Retrieve Everything All at Once? {ARM}: An Alignment-Oriented {LLM}-based Retrieval Method},
  author    = {Chen, Peter Baile and Zhang, Yi and Cafarella, Michael and Roth, Dan},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}

@article{boutaleb2025exploring,
  title   = {Exploring Multi-Table Retrieval Through Iterative Search},
  author  = {Boutaleb, Allaa and Amann, Bernd and Angarita, Rafael and Naacke, Hubert},
  journal = {arXiv preprint arXiv:2511.13418},
  year    = {2025},
  note    = {AI for Tabular Data Workshop, EurIPS 2025}
}

@inproceedings{soliman2026coret,
  title     = {{CORE-T}: {CO}herent {RE}trieval of Tables for Text-to-{SQL}},
  author    = {Soliman, Hassan and Gupta, Vivek and Roth, Dan and Gurevych, Iryna},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}

@inproceedings{ho2026pearl,
  title     = {{PEARL}: Front-Loading Relational Chains for Multi-Hop Table Retrieval},
  author    = {Ho, Subeen and Kang, Hyeongu and Kang, SeongKu and Yoon, Susik},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}

@article{agarwal2025rear,
  title   = {{REaR}: Retrieve, Expand and Refine for Effective Multitable Retrieval},
  author  = {Agarwal, Rishita and Singhal, Himanshu and Chen, Peter Baile and Roy Choudhury, Manan and Roth, Dan and Gupta, Vivek},
  journal = {arXiv preprint arXiv:2511.00805},
  year    = {2025}
}

@inproceedings{wang2025plugging,
  title     = {Plugging Schema Graph into Multi-Table {QA}: A Human-Guided Framework for Reducing {LLM} Reliance},
  author    = {Wang, Xixi and Costa, Miguel and Kovaceva, Jordanka and Wang, Shuai and Pereira, Francisco C.},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025}
}
```

---

## Q4. Cross-cutting: which methods are myopic or greedy, which are trained end-to-end on answer correctness, and what failure modes do they report (with numbers)? Plus a ranking for this project

### Takeaway
Almost every system reviewed selects evidence **myopically or with fixed budgets**, and **none trains an entity-level selector against final answer correctness**:
- Supervised selectors use answer-location distant supervision with independently trained stages.
- LLM and graph frameworks are zero-shot.
- Multi-table retrievers optimise similarity or joinability at inference.

The only RL attempt (José et al. 2024) is module-level and collapsed to a fixed policy. The documented failure modes are error propagation across hops, noisy table retrieval, similar-but-irrelevant candidates, and entity-matching errors. Together they give the project a clear, citable gap.

### Cited Findings

**Selection style per system:**

| System | Selection unit | Decision mechanism | Greedy / myopic? | Trained on final answer? |
|---|---|---|---|---|
| HYBRIDER ([HybridQA](https://arxiv.org/abs/2004.07347)) | cell → neighbour cell → passage | BERT ranker + hop classifier, weak supervision | Yes: "greedy search at first two steps" | No: stages trained independently |
| MITQA ([ACL 2023](https://aclanthology.org/2023.acl-long.449/)) | row + linked passages | multi-instance RowRetriever + joint reranker | Fixed top-5 | No: distant supervision |
| S3HQA ([arXiv](https://arxiv.org/abs/2305.11725)) | row, passages | trained retriever + rule-based selector by question type | Rule-based | No |
| MuGER² ([arXiv](https://arxiv.org/abs/2210.10350)) | header/cell/link | unified retriever + s_tab vs s_pass rule | Rule-based | No |
| TACR ([arXiv](https://arxiv.org/abs/2305.14682)) | cells → row | alignment-enhanced cell classifier | Top-k | No |
| DocHopper ([arXiv](https://arxiv.org/abs/2106.00200)) | paragraph/sentence | iterative attention with query update | Soft attention per hop | End-to-end differentiable; attention supervised when labels exist |
| OTT-QA iterative ([ICLR 2021](https://arxiv.org/abs/2010.10439)) | text/table blocks | dual-encoder similarity | Yes | No |
| CORE / COS ([CORE](https://arxiv.org/abs/2210.12338), [COS](https://arxiv.org/abs/2305.03130)) | table → rows → passages (chains) | DPR + linker + zero-shot T0 chainer | Fixed top-k budgets (50/100/200) | No |
| José et al. ([arXiv](https://arxiv.org/abs/2407.04858)) | module (text retr. / table retr. / answer) | PPO/DQN policy | Learned, but coarse | **Yes** (EM/F1 terminal reward), but only module-level |
| ODYSSEY ([arXiv](https://arxiv.org/html/2501.17767)) | graph nodes (cells, entities, docs) | LLM NER + embedding threshold 0.8 + BFS 3 hops | Heuristic | No (zero-shot) |
| N2N-GQA ([arXiv](https://arxiv.org/html/2601.06603)) | rows/passages | GraphRank (similarity × centrality) | Heuristic budgets | No (zero-shot) |
| Chain-of-Table ([arXiv](https://arxiv.org/abs/2401.04398)) | rows/columns via operations | LLM prompt | Yes: "greedy search strategy" | No |
| TableRAG ([arXiv](https://arxiv.org/abs/2410.04739)) | columns, cell values | LM query expansion + embedding retrieval | Top-k under token budget | No |
| MTR / MMQA ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf)) | tables | sub-question relevance × binary column overlap | Yes: per round; stop when overlap = 0 | No |
| JAR ([arXiv](https://arxiv.org/abs/2404.09889)) | table set | MIP over relevance + join compatibility | No (global) | No |
| Greedy-JAR ([arXiv](https://arxiv.org/abs/2511.13418)) | tables | greedy join-aware search | Yes | No |
| MURRE ([COLING 2025](https://aclanthology.org/2025.coling-main.386.pdf)) | tables | cosine similarity + LLM removal + LLM early stop | Beam search | No (explicitly future work) |
| ARM ([arXiv](https://arxiv.org/abs/2501.18539)) | tables/passages | LLM alignment, retrieve-all-at-once | No (global) | No |

**Failure modes with numbers:**
- **Error propagation across hops:**
  - OTT-QA authors abandon iterative retrieval because "early mistakes can propagate to later retrieval stages" — [OTT-QA](https://arxiv.org/abs/2010.10439)
  - HYBRIDER's stage accuracies (87.4 linking × 87.9 ranking × 89.2 hop × 62 RC) compound — [HybridQA](https://arxiv.org/abs/2004.07347)
  - MURRE adopts beam search specifically "to reduce the error cascades" — [MURRE](https://aclanthology.org/2025.coling-main.386.pdf)
- **Top-1 selection loses gold evidence:** keeping 5 rows instead of 1 raises gold-row recall by 8–9% in MITQA — [MITQA](https://arxiv.org/abs/2112.07337)
- **Wrong or missing linked passage:** oracle links raise OTT-QA dev EM from 28.1 to 35.2 (FR+CBR) — [CORE](https://arxiv.org/abs/2210.12338)
- **Selection-module errors:** ODYSSEY attributes 14 of 41 errors to entity matching, extraction or mapping — [ODYSSEY](https://arxiv.org/html/2501.17767)
- **Noisy whole-table retrieval:** "tables often contain excessive data with relevant information typically in a single row" — [José et al.](https://arxiv.org/abs/2407.04858)
  - BM25 table-first action sequences score only 4–8 F1 vs 16–19 for text-first — [José et al.](https://arxiv.org/abs/2407.04858)
- **Distractor / similar candidates:** 87.0% of MURRE's gains over single-hop come from fixing "Similar Candidate" errors (an irrelevant table sharing tokens with the question) — [MURRE](https://aclanthology.org/2025.coling-main.386.pdf)
- **Row selection weaker than column selection:** stated by TACR — [TACR](https://arxiv.org/abs/2305.14682)
- **Degradation with hop count:** on unified-corpus MMQA, R@2 falls from 2-hop to 3-hop for JAR (55.90 → 44.70) and MURRE (49.15 → 34.05) — [PEARL](https://arxiv.org/abs/2608.30291)
- **RL-specific failure:** with coarse actions and a terminal reward, the PPO agent "converg[ed] to the trivial solution of only picking the best action on average" — [José et al.](https://arxiv.org/abs/2407.04858)

### Inferences

**Ranking of the most relevant systems for this project (top 6), with rationale:**

1. **José et al. 2024 (DRL table-text QA, BRACIS).**
   - The only prior RL formulation of hop-wise selection in table-text QA. It uses PPO with an answer-F1/EM reward and a −0.02 per-step cost, which is nearly the project's reward minus the recall term.
   - It must be cited as the closest work and as a foil: module-level actions, averaged-embedding state, no evidence-recall reward, single seed, collapse to the best fixed policy at F1 19.03.
   - The project's entity-level actions plus recall shaping plus GAE directly address its stated failure.
2. **HYBRIDER / HybridQA (Chen et al. 2020).**
   - The canonical link → rank → hop → read pipeline with explicit greedy top-1 rank and hop decisions, trained on weak labels and not answer correctness.
   - It provides per-stage error rates and the natural closed-setting environment (cells, rows, linked passages) for the MDP.
   - The PPO selector replaces its rank and hop stages.
3. **S³HQA (with MITQA as companion).**
   - The strongest supervised HybridQA systems (67.9 / 64.3 test EM).
   - Their selectors are a rule-based question-type selector (S³HQA) and a fixed top-5 row reranker (MITQA). Both are exactly the "heuristic selection step" a learned policy replaces, and both are the numbers to beat or compare against on HybridQA.
4. **ODYSSEY (Hybrid Graphs, NAACL 2025).**
   - The LLM-era graph framework whose node selection is a hand-set heuristic (0.8 similarity threshold plus 3-hop BFS).
   - It is zero-shot and has a documented selection-error share (14/41).
   - A learned add-node/STOP policy plugs directly into its pruning step and can be evaluated against its token savings (45–53%).
5. **MMQA + MTR (ICLR 2025).**
   - The multi-table multi-hop benchmark the project can use for "joined table" actions.
   - Its MTR baseline is a greedy iterative selector with a hard-coded STOP (column overlap = 0).
   - LLMs remain far below humans (O1-preview 50.78 vs 89.8 EM).
6. **MURRE (COLING 2025), read together with JAR and Greedy-JAR.**
   - MURRE is the most MDP-like multi-table retriever: hop-wise, beam search, LLM-decided early stop, max 3 hops. It explicitly leaves training retrieval with downstream (SQL) feedback as future work, which is the project's contribution.
   - JAR (global MIP) and Greedy-JAR (greedy ≈ MIP, 4–400× faster) frame the global-vs-sequential trade-off.

**Honourable mentions:**
- CORE / COS: the strongest OTT-QA chain selectors (54.9 test EM). Their zero-shot T0 chainer is another replaceable heuristic.
- Chain-of-Table: the canonical "greedy LLM selection" of rows and columns, single-table.
- N2N-GQA (2026): zero-shot graph centrality pruning on OTT-QA.

**Positioning statement supported by the evidence:** existing frameworks choose evidence by (a) similarity-ranked retrieval with fixed top-k, (b) independently trained classifiers using answer-location distant supervision, (c) fixed graph-traversal heuristics, or (d) greedy LLM prompting. None learns an entity-level, multi-step selection policy with a value function from answer-correctness reward. The single RL attempt shows that module-level RL with terminal reward collapses to a fixed policy.

### Gaps
- No paper reviewed reports a learned critic or value baseline for evidence selection in table QA, and no entity-level RL selector was found. This rests on the papers examined here plus two targeted searches; adjacent RL-for-search work (e.g., R1-Searcher, StepSearch) may be covered by other researchers and was not assessed here.
- Seed variance is not reported by José et al. (single run) or by most LLM frameworks (sampled dev subsets), so statistical comparisons with them should be hedged.
- Whether ODYSSEY's and N2N-GQA's sample-based numbers transfer to the full dev sets is unknown.
