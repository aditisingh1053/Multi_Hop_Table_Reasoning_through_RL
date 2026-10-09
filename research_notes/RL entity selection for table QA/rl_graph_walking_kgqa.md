# RL Graph Walking and RL-based KBQA as Prior Art for PPO Entity Selection in Multi-Hop Table QA

Scope note: this cluster covers RL agents that pick the next entity or edge on a graph (KG path walking, RL-based KBQA, and 2025-2026 agentic KG/GraphRAG RL). Every paper below was checked against its arXiv / ACL Anthology / IJCAI / ACM / NeurIPS page. Method details and numbers come from the full PDFs: arXiv or Anthology PDFs converted to text locally, or, for SRN, secondary sources (flagged). The papers are tiered so the report writer can stay selective:
- **Core (cite and discuss):** MINERVA, Lin et al. 2018, M-Walk, SRN, DeepPath, KG-R1, Search-on-Graph-R1
- **Supporting (cite for one specific point):** DacKGR (sparse graphs), CPL (action space augmented from text, analogous to linked passages), GraphRAG-R1 (retrieval-cost reward shaping), Graph-R1 (RL over a graph built from unstructured text; GRPO vs PPO)
- **Peripheral (drop first if space is tight):** DIVA, RLH (hierarchical RL)

---

## Q1. Which papers frame multi-hop reasoning as an MDP where the agent picks the next entity/relation, and do they exist? (verified list + BibTeX)

### Takeaway
All 13 papers below exist. Their title, full author list, venue, year and identifier were checked on primary pages. Two of the names in the brief need correcting: the "hierarchical RL walker" is Wan et al., IJCAI 2020 ("RLH"), and "KG-R1" is the system name in Lin et al.'s arXiv paper "Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning", which has no stated venue. A closely related July 2026 paper, Search-on-Graph-R1, also exists and matches the project's BC-warm-start-then-RL recipe.

### Cited Findings

**Verified metadata**

1. **DeepPath.** Wenhan Xiong, Thien Hoang, William Yang Wang. EMNLP 2017, pp. 564-573. arXiv:1707.06690. DOI 10.18653/v1/D17-1060. Sources: [arXiv](https://arxiv.org/abs/1707.06690), [ACL Anthology](https://aclanthology.org/D17-1060/)
2. **MINERVA: "Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning".** Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, Andrew McCallum. ICLR 2018. arXiv:1711.05851. The PDF header reads "Published as a conference paper at ICLR 2018". Source: [arXiv](https://arxiv.org/abs/1711.05851)
3. **M-Walk: "Learning to Walk over Graphs using Monte Carlo Tree Search".** Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo, Jianfeng Gao. NeurIPS 2018. arXiv:1802.04394. Sources: [arXiv](https://arxiv.org/abs/1802.04394), [NeurIPS PDF](http://papers.neurips.cc/paper/7912-m-walk-learning-to-walk-over-graphs-using-monte-carlo-tree-search.pdf), [ACM DL record (pp. 6787-6798)](https://dl.acm.org/doi/abs/10.5555/3327757.3327784)
4. **"Multi-Hop Knowledge Graph Reasoning with Reward Shaping"** (code name MultiHopKG). Xi Victoria Lin, Richard Socher, Caiming Xiong. EMNLP 2018, pp. 3243-3253. arXiv:1808.10568. DOI 10.18653/v1/D18-1362. Sources: [arXiv](https://arxiv.org/abs/1808.10568), [ACL Anthology](https://aclanthology.org/D18-1362/)
5. **DIVA: "Variational Knowledge Graph Reasoning".** Wenhu Chen, Wenhan Xiong, Xifeng Yan, William Yang Wang. NAACL-HLT 2018, pp. 1823-1832. arXiv:1803.06581. DOI 10.18653/v1/N18-1165. Sources: [arXiv](https://arxiv.org/abs/1803.06581), [ACL Anthology](https://aclanthology.org/N18-1165/)
6. **CPL: "Collaborative Policy Learning for Open Knowledge Graph Reasoning".** Cong Fu, Tong Chen, Meng Qu, Woojeong Jin, Xiang Ren. EMNLP-IJCNLP 2019, pp. 2672-2681. arXiv:1909.00230. DOI 10.18653/v1/D19-1269. Sources: [arXiv](https://arxiv.org/abs/1909.00230), [ACL Anthology](https://aclanthology.org/D19-1269/)
7. **SRN: "Stepwise Reasoning for Multi-Relation Question Answering over Knowledge Graph with Weak Supervision".** Yunqi Qiu, Yuanzhuo Wang, Xiaolong Jin, Kun Zhang. WSDM 2020, pp. 474-482. DOI 10.1145/3336191.3371812. No arXiv version was found. Sources: [Semantic Scholar record incl. DBLP key conf/wsdm/QiuWJZ20](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc), [DOI](https://doi.org/10.1145/3336191.3371812), [author list and pages as cited in He et al. WSDM 2021](https://richardhgl.github.io/files/wsdm2021.pdf)
8. **RLH: "Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning".** Guojia Wan, Shirui Pan, Chen Gong, Chuan Zhou, Gholamreza Haffari. IJCAI 2020, pp. 1926-1932. DOI 10.24963/ijcai.2020/267. No arXiv ID was found. Sources: [IJCAI proceedings](https://www.ijcai.org/proceedings/2020/267), [PDF](https://www.ijcai.org/proceedings/2020/0267.pdf)
9. **DacKGR: "Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph".** Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, Suhui Wu. EMNLP 2020, pp. 5694-5703. arXiv:2010.01899. DOI 10.18653/v1/2020.emnlp-main.459. Sources: [arXiv](https://arxiv.org/abs/2010.01899), [ACL Anthology](https://aclanthology.org/2020.emnlp-main.459/)
10. **Graph-R1: "Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning".** Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan. ICML 2026 (PMLR 306). arXiv:2507.21892 (v1 29 Jul 2025, v2 2 Jun 2026). The arXiv comment reads "Accepted by ICML 2026 main conference", and the PDF header reads "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026". Sources: [arXiv](https://arxiv.org/abs/2507.21892), [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/63269), [GitHub](https://github.com/LHRLAB/Graph-R1)
11. **GraphRAG-R1: "Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning".** Chuanyue Yu, Kuo Zhao, Yuhan Li, Heng Chang, Mingjian Feng, Xiangzhe Jiang, Yufei Sun, Jia Li, Yuzhi Zhang, Jianxin Li, Ziwei Zhang (author order as on arXiv). The Web Conference (WWW) 2026. arXiv:2507.23581. The arXiv comment reads "Accepted by the Web Conference 2026". Source: [arXiv](https://arxiv.org/abs/2507.23581)
12. **KG-R1: "Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning".** Junhong Lin, Shicheng Liu, Jinyeop Song, Song Wang, Julian Shun, Yada Zhu. arXiv:2509.26383 (v5, 22 May 2026). No venue is given in the arXiv comments. Sources: [arXiv](https://arxiv.org/abs/2509.26383), [GitHub](https://github.com/Jinyeop3110/KG-R1)
13. **Search-on-Graph-R1 (SoG-R1): "Training Large Language Models to Search Knowledge Graphs with Reinforcement Learning".** Jia Ao Sun, Hao Yu, Fengran Mo, Zhan Su, Yuchen Hui, Bang Liu, Jian-Yun Nie. arXiv:2607.18481 (20 Jul 2026). No venue is listed. Source: [arXiv](https://arxiv.org/abs/2607.18481)

#### BibTeX (ready to use)

```bibtex
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

@inproceedings{das2018minerva,
  title     = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author    = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1711.05851}
}

@inproceedings{shen2018mwalk,
  title     = {{M-Walk}: Learning to Walk over Graphs using {Monte Carlo} Tree Search},
  author    = {Shen, Yelong and Chen, Jianshu and Huang, Po-Sen and Guo, Yuqing and Gao, Jianfeng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  year      = {2018},
  url       = {http://papers.neurips.cc/paper/7912-m-walk-learning-to-walk-over-graphs-using-monte-carlo-tree-search.pdf}
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

@inproceedings{chen2018diva,
  title     = {Variational Knowledge Graph Reasoning},
  author    = {Chen, Wenhu and Xiong, Wenhan and Yan, Xifeng and Wang, William Yang},
  booktitle = {Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  pages     = {1823--1832},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/N18-1165},
  url       = {https://aclanthology.org/N18-1165/}
}

@inproceedings{fu2019cpl,
  title     = {Collaborative Policy Learning for Open Knowledge Graph Reasoning},
  author    = {Fu, Cong and Chen, Tong and Qu, Meng and Jin, Woojeong and Ren, Xiang},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {2672--2681},
  year      = {2019},
  address   = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/D19-1269},
  url       = {https://aclanthology.org/D19-1269/}
}

@inproceedings{qiu2020srn,
  title     = {Stepwise Reasoning for Multi-Relation Question Answering over Knowledge Graph with Weak Supervision},
  author    = {Qiu, Yunqi and Wang, Yuanzhuo and Jin, Xiaolong and Zhang, Kun},
  booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM '20)},
  pages     = {474--482},
  year      = {2020},
  publisher = {ACM},
  doi       = {10.1145/3336191.3371812}
}

@inproceedings{wan2020rlh,
  title     = {Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning},
  author    = {Wan, Guojia and Pan, Shirui and Gong, Chen and Zhou, Chuan and Haffari, Gholamreza},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)},
  pages     = {1926--1932},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/267}
}

@inproceedings{lv2020dackgr,
  title     = {Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph},
  author    = {Lv, Xin and Han, Xu and Hou, Lei and Li, Juanzi and Liu, Zhiyuan and Zhang, Wei and Zhang, Yichi and Kong, Hao and Wu, Suhui},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {5694--5703},
  year      = {2020},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2020.emnlp-main.459},
  url       = {https://aclanthology.org/2020.emnlp-main.459/}
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

@inproceedings{yu2026graphragr1,
  title     = {{GraphRAG-R1}: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning},
  author    = {Yu, Chuanyue and Zhao, Kuo and Li, Yuhan and Chang, Heng and Feng, Mingjian and Jiang, Xiangzhe and Sun, Yufei and Li, Jia and Zhang, Yuzhi and Li, Jianxin and Zhang, Ziwei},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  note      = {arXiv:2507.23581}
}

@article{lin2025kgr1,
  title   = {Efficient and Transferable Agentic Knowledge Graph {RAG} via Reinforcement Learning},
  author  = {Lin, Junhong and Liu, Shicheng and Song, Jinyeop and Wang, Song and Shun, Julian and Zhu, Yada},
  journal = {arXiv preprint arXiv:2509.26383},
  year    = {2025}
}

@article{sun2026sogr1,
  title   = {{Search-on-Graph-R1}: Training Large Language Models to Search Knowledge Graphs with Reinforcement Learning},
  author  = {Sun, Jia Ao and Yu, Hao and Mo, Fengran and Su, Zhan and Hui, Yuchen and Liu, Bang and Nie, Jian-Yun},
  journal = {arXiv preprint arXiv:2607.18481},
  year    = {2026}
}
```

### Inferences
- The lineage splits into three eras:
  - **Classic REINFORCE walkers, 2017-2020:** DeepPath, MINERVA, DIVA, Lin et al., CPL, SRN, RLH and DacKGR. M-Walk is the MCTS/Q-learning outlier.
  - **A quiet period:** 2021-2024 is not covered by the verified set (see Gaps).
  - **LLM-agent RL over graphs, 2025-2026:** Graph-R1, GraphRAG-R1, KG-R1 and SoG-R1, all trained with GRPO.
- None of the 13 verified papers uses PPO with GAE on a small, non-LLM pointer policy. That combination is open territory the project can claim. Phrase it as "among the verified RL graph-walking papers", not as an absolute novelty claim.

### Gaps
- Pages and DOI for GraphRAG-R1 in the WWW '26 proceedings were not found. The BibTeX omits them.
- M-Walk's page numbers in the NeurIPS volume (as opposed to the ACM DL numbering, 6787-6798) were not verified.
- MINERVA's OpenReview forum page could not be loaded (browser check), so the arXiv URL is used in the BibTeX. The ICLR 2018 venue is confirmed by the arXiv PDF header.
- DBLP BibTeX pages were bot-blocked. The SRN author spelling comes from the Semantic Scholar record and from He et al.'s reference list.

---

## Q2. For each paper: state, action space (scoring of variable-size neighbour sets), STOP handling, RL algorithm, reward design, datasets and headline numbers

### Takeaway
The canonical design comes from MINERVA and Lin et al.:
- **State:** an LSTM history of (relation, entity) steps plus the query.
- **Actions:** each outgoing edge is embedded as [relation; entity], and a query-conditioned MLP output is dotted with every action embedding (a pointer-style, permutation-invariant head over a variable-size set).
- **STOP:** a self-loop (NO_OP) under a fixed horizon T.
- **Training:** REINFORCE with a baseline and entropy regularisation.
- **Reward:** terminal and binary, later shaped by a pretrained KG-embedding score.

M-Walk is the only classic paper with an explicit, separately scored STOP action and a shared policy/Q-value head. SRN brings potential-based reward shaping to natural-language KBQA. The 2025-2026 LLM agents use GRPO with format, answer and retrieval rewards plus cost penalties, and two of them report that PPO underperforms GRPO in their LLM setting.

### Cited Findings

**MINERVA (Das et al., ICLR 2018)**
- **MDP:** Query answering (e1, r, ?) is cast as a "finite horizon, deterministic partially observed Markov decision process". The state is (e_t, e1q, rq, e2q), but the answer e2q is hidden, so the observation is (e_t, e1q, rq). — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Actions:** All outgoing edges of the current node. Inverse edges (e2, r^-1, e1) are added, which "allows our agent to undo a potentially wrong decision". — [MINERVA](https://arxiv.org/abs/1711.05851)
- **STOP:** The authors augment each node with a "NO OP" self-loop and unroll for a fixed T. They write: "we could have allowed the agent to take a special 'STOP' action, but we found the current setup to work sufficiently well." — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Policy:**
  - History is encoded by an LSTM: h_t = LSTM(h_{t-1}, [a_{t-1}; o_t]).
  - Each candidate edge is embedded as [r_l; e_d] and stacked into A_t (|A_St| x 2d).
  - The decision is d_t = softmax(A_t (W2 ReLU(W1 [h_t; o_t; r_q]))).
  - The paper notes nodes "do not have a fixed ordering or number of edges" and the procedure is "invariant to order in which edges are presented".

  — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Reward and training:**
  - The reward is terminal only: R(S_T) = I{e_t = e2q}.
  - Training uses REINFORCE with 20 rollouts per example.
  - The baseline is a moving average of cumulative discounted reward; a learned baseline "performed similarly".
  - An entropy regulariser is added.
  - Inference uses beam search with width 50.

  — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Headline numbers (Hits@1 / Hits@10 / MRR):**

  | Dataset | MINERVA | Comparison |
  |---|---|---|
  | NELL-995 | 0.663 / 0.831 / 0.725 | ConvE 0.672 / 0.864 / 0.747 |
  | FB15K-237 | 0.217 / 0.456 / 0.293 | ConvE 0.313 / 0.600 / 0.410 |
  | WN18RR | 0.413 / 0.513 / 0.448 | — |
  | WikiMovies (accuracy) | 96.7 | NeuralLP 94.6, KV-MemNN 93.9 |

  On Countries S3 (AUC-PR), MINERVA scores 95.10. — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Single model vs per-relation models:** One MINERVA model for all query relations beats per-relation DeepPath models on NELL-995 MAP, e.g. worksFor 0.825 vs 0.711. — [MINERVA](https://arxiv.org/abs/1711.05851)

**Lin, Socher, Xiong (EMNLP 2018), "MultiHopKG"**
- **MDP:** Same as MINERVA. The state is s_t = (e_t, (e_s, r_q)) and actions are the outgoing edges of e_t. "A self-loop edge is added to every A_t. Because search is unrolled for a fixed number of steps T, the self-loop acts similarly to a 'stop' action." — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Policy:** π(a_t|s_t) = softmax(A_t × W2 ReLU(W1[e_t; h_t; r_q])), with a 3-layer LSTM path encoder. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Training:** REINFORCE with an entropy term. Multi-answer queries are treated as n single-answer queries. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Reward shaping (counters false negatives):** R(s_T) = R_b(s_T) + (1 - R_b(s_T)) · f(e_s, r_q, e_T). Here f is a frozen, pretrained ConvE or ComplEx fact scorer that gives a soft reward to unobserved but plausible answers. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Action dropout (counters spurious paths):**
  - Sampling uses π~ ∝ (π · m + ε), with m_i ~ Bernoulli(1 - α).
  - The gradient is still computed with the unmasked π.
  - The tuned rate α ranges from 0.1 to 0.9, and "larger action dropout rates are necessary for KGs that are densely connected" (the optimal rate correlates positively with average fan-out).

  — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Fan-out cap:** Each entity keeps only its top-η neighbours by PageRank (η in 256-512) to avoid GPU memory overflow. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Headline numbers (Ours(ConvE), ×100):**
  - FB15k-237: Hits@1 32.7, Hits@10 56.4, MRR 40.7, versus MINERVA 21.7 / 45.6 / 29.3.
  - UMLS: 90.2 / 99.2 / 94.0.
  - Kinship: 78.9 / 98.2 / 86.5.
  - The authors claim path-based state-of-the-art gains of 4%, 9% and 39% on UMLS, Kinship and FB15k-237.

  — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Ablation (dev MRR):** "In general, removing action dropout has a greater impact." On WN18RR, removing reward shaping actually helps.

  | Dataset | Full | without reward shaping | without action dropout |
  |---|---|---|---|
  | FB15k-237 | 38.2 | 35.1 | 31.0 |
  | UMLS | 73.0 | 67.7 | 61.3 |
  | WN18RR | 43.8 | 45.7 | 39.1 |

  — [Lin et al.](https://arxiv.org/abs/1808.10568)

**M-Walk (Shen et al., NeurIPS 2018)**
- **State:** s_t = s_{t-1} ∪ {a_{t-1}, n_t, E_nt, N_nt}. This is the full history of traversed nodes, their edges and their neighbourhoods, plus the query. — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Actions and explicit STOP:**
  - A_t = E_nt ∪ {STOP}. Selecting STOP ends the walk and outputs the current node as the prediction.
  - Neighbour scores are inner products, u_n = <h_S,t, h_n',t>.
  - The STOP score u_0 = f_θ(h_S,t, h_A,t) comes from a separate MLP over a summary vector of the neighbourhood.
  - The policy is a temperature softmax over (u_0, u_n1, ...), and the Q-values are an element-wise sigmoid of the same scores, so the policy and the Q-network share parameters.

  — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Algorithm:**
  - MCTS with PUCT, guided by the policy, generates trajectories with more positive reward.
  - The network is then trained off-policy by Q-learning, which updates the policy indirectly through the shared parameters.
  - MCTS is also used at test time.

  — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Reward:** +1 if the STOP node equals the target, otherwise 0. The authors say "The rewards are sparse." — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Horizons:** Maximum walk length is 8 on NELL-995 and 5 on WN18RR. — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Headline numbers:**
  - NELL-995 overall MAP (DeepPath per-relation protocol): 89.9, versus MINERVA 87.6 and DeepPath 78.8.
  - WN18RR: Hits@1 41.4, Hits@3 44.5, MRR 43.7, versus MINERVA (their run) 35.1 / 44.5 / 40.9 and ConvE 39.6 / 44.7 / 43.3.
  - Three Glass Puzzle: 99.0% versus 49.0% for the best REINFORCE method.

  — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Ablation:** PG-Walk (the same network trained with policy gradient) loses about 1 point on NELL-995, and Q-Walk (Q-learning without MCTS) loses about 2 points. — [M-Walk](https://arxiv.org/abs/1802.04394)

**DeepPath (Xiong et al., EMNLP 2017)**
- **State:** s_t = (e_t, e_target - e_t), using TransE embeddings. The target entity is part of the state, so this is a fact-prediction or path-finding setting, not query answering. — [DeepPath](https://aclanthology.org/D17-1060/)
- **Actions:** "To keep the output dimension of the policy network consistent, the action space is defined as all the relations in the KG." On NELL that is 400 actions including inverses. A relation with no matching edge is a failed step: reward -1, and the agent stays in place. — [DeepPath](https://arxiv.org/abs/1707.06690)
- **Reward:** Three parts:
  - global accuracy: +1 if the target is reached, -1 otherwise;
  - efficiency: 1/length(p);
  - diversity: -(1/|F|) Σ cos(p, p_i) over previously found paths.

  — [DeepPath](https://arxiv.org/abs/1707.06690)
- **Training:**
  - Warm start: supervised policy learning on paths found by a randomized two-sided BFS (through a random intermediate node), "inspired by the imitation learning pipeline used by AlphaGo".
  - Then REINFORCE retraining with a maximum episode length.
  - The found paths become binary features for per-relation PRA-style classifiers.

  — [DeepPath](https://arxiv.org/abs/1707.06690)
- **Headline numbers:**
  - Link-prediction overall MAP: FB15K-237 0.572 (PRA 0.541, TransE 0.532, TransR 0.540) and NELL-995 0.796 (PRA 0.675, TransR 0.789).
  - Fact-prediction MAP: FB15K-237 0.311 and NELL-995 0.493.

  — [DeepPath](https://arxiv.org/abs/1707.06690)

**DIVA (Chen et al., NAACL 2018)**
- **Task and model:** Relation inference (e_s, ?, e_d). Paths are latent variables, and the model has three parts:
  - a path-finder prior p(L|e_s,e_d), an LSTM with p((a,e)|h_t) = softmax(A_t c_t) over the variable-size outgoing-edge matrix;
  - a relation-aware posterior q(L|e_s,e_d,r);
  - a CNN path-reasoner likelihood p(r|L).

  Search terminates when the target is reached or at the step limit. — [DIVA](https://arxiv.org/abs/1803.06581)
- **Training:** The posterior update is "a special case of REINFORCE" with a model-based reward log p(r|L) instead of a heuristic reward. The beam size is 5. — [DIVA](https://arxiv.org/abs/1803.06581)
- **Headline numbers:**
  - NELL-995, 12 relations: MAP 88.6 (DeepPath 79.6 in their table).
  - NELL-995, 9 relations: 87.9, against MINERVA's 88.2, so DIVA does not beat MINERVA there.
  - FB15k-237, 20 relations: MAP 59.8 (MML 58.7, DeepPath 57.2, MINERVA 55.2 from their own implementation).

  — [DIVA](https://arxiv.org/abs/1803.06581)

**CPL (Fu et al., EMNLP-IJCNLP 2019)**
- **Architecture:** Two agents, a path reasoner and a fact extractor, both trained with REINFORCE in alternation. — [CPL](https://arxiv.org/abs/1909.00230)
  - The reasoner's action space is the union of KG edges and corpus-extracted edges at e_t.
  - Its state is [e_s, r_q, h_t] and actions are [r, e] embeddings scored MINERVA-style.
  - It gets a terminal reward of 1/0 after a fixed number of steps.
  - The extractor gets a "step-wise delayed reward" when an edge it suggested is used on a successful path.
- **Pre-training:** The reasoner is pre-trained on the KG, and the extractor (PCNN-ATT) on distant supervision. — [CPL](https://arxiv.org/abs/1909.00230)
- **Results (UMLS-PubMed and FB60K-NYT10):**
  - "CPL and MINERVA perform worse than triple-ranking methods when the size of KGs is small, while outperforms them significantly when adding more triples."
  - "MINERVA is better than CPL on full FB60K-NYT10 because the alignment between FB60K and NYT10 is very limited."

  — [CPL](https://aclanthology.org/D19-1269/)

**SRN (Qiu et al., WSDM 2020)**
- **Method:** SRN "formulates multi-relation question answering as a sequential decision problem", "performs effective path search over the knowledge graph", and "leverages beam search to reduce the number of candidates". Its policy uses "the attention mechanism ... [to] enhance the unique impact of different parts of a given question over triple selection". — [SRN abstract via Semantic Scholar](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc)
- **Reward:** "to alleviate the delayed and sparse reward problem caused by weak supervision [final answer only], we propose a potential-based reward shaping strategy". — [SRN abstract via Semantic Scholar](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc)
- **Shaping formula:** R*(S_t, A_t, S_{t+1}) = R + F_φ*(S_t, A_t, S_{t+1}), with F = γφ*(S_{t+1}) - φ*(S_t). This comes from a search-result excerpt of the paper and could not be checked against the PDF. — [ResearchGate excerpt of SRN](https://www.researchgate.net/publication/338754985_Stepwise_Reasoning_for_Multi-Relation_Question_Answering_over_Knowledge_Graph_with_Weak_Supervision)
- **Headline numbers (Hits@1):**
  - MetaQA 1-hop / 2-hop / 3-hop: 97.0 / 95.1 / 75.2. WebQSP and CWQ were not reported.

    — [He et al. WSDM 2021 (NSM), Table 2](https://richardhgl.github.io/files/wsdm2021.pdf)
  - PathQuestion PQ-2H / PQ-3H: 96.3 / 89.2. PathQuestion-Large PQL-2H / PQL-3H: 78.6 / 77.5. These numbers are consistent with QAGCN's text: QAGCN is "-8.5% in comparison with SRN" on PQL-3hop and "1.6% higher" on PQ-3hop.

    — [QAGCN](https://arxiv.org/abs/2206.01818)
  - Non-RL label-propagation and GNN methods later surpassed SRN on MetaQA-3hop; QAGCN reports itself "29.8% ... higher than the best-performing RL-based method SRN on MetaQA 3-hop". — [QAGCN](https://arxiv.org/abs/2206.01818)

**RLH, hierarchical RL (Wan et al., IJCAI 2020)**
- **Method:** Each reasoning step is decomposed into two policies, both trained with REINFORCE:
  - a high-level policy that encodes history;
  - a low-level policy that walks a hierarchy of relation clusters, built by k-means over TransE relation embeddings, to pick the relation.

  The low-level policy is pre-trained before joint training. — [RLH](https://www.ijcai.org/proceedings/2020/0267.pdf)
- **Settings:** High-level horizon 4, low-level horizon 2, and a "reward factor" of 1.2 that weights the per-hop terminal reward. The exact reward formula was garbled in text extraction. — [RLH](https://www.ijcai.org/proceedings/2020/0267.pdf)
- **Headline numbers:**
  - FB15K-237: Hits@1 0.342, Hits@10 0.648, MRR 0.460, versus MINERVA 0.217 / 0.456 / 0.293 and ConvE 0.313 / 0.601 / 0.410.
  - NELL995: Hits@1 0.692, MRR 0.723, versus MINERVA 0.663 / 0.725.
  - WN18RR: Hits@1 0.453, MRR 0.481.

  — [RLH](https://www.ijcai.org/proceedings/2020/0267.pdf)

**DacKGR (Lv et al., EMNLP 2020)**
- **Base setup:** The REINFORCE agent follows Lin et al., and a self-loop action (r_LOOP, e_t) is "similar to a 'STOP' action". — [DacKGR](https://arxiv.org/abs/2010.01899)
- **Dynamic anticipation:** A pretrained KG-embedding model's answer-probability vector is injected into the state. — [DacKGR](https://arxiv.org/abs/2010.01899)
- **Dynamic completion:** Additional (r, e) actions predicted by the embedding model are added to the action space, controlled by a proportion α and a cap M. — [DacKGR](https://arxiv.org/abs/2010.01899)
- **Headline numbers (MRR):**
  - FB15K-237-10%: 21.8 versus MultiHopKG 13.6.
  - FB15K-237-20%: 24.7 versus 23.0.
  - The embedding model ConvE still scores higher on these sparse splits: 24.5 and 26.1.

  — [DacKGR](https://aclanthology.org/2020.emnlp-main.459/)

**Graph-R1 (Luo et al., ICML 2026)**
- **Environment:** A knowledge hypergraph built by an LLM that extracts n-ary facts from text chunks. — [Graph-R1](https://arxiv.org/abs/2507.21892)
- **Actions:** Each action has four sub-actions:
  - think, which decides whether to continue or terminate;
  - query generation;
  - graph retrieval (entity-based plus hyperedge retrieval with rank fusion);
  - answer.

  The state is the action history. — [Graph-R1](https://arxiv.org/abs/2507.21892)
- **Training and reward:** GRPO with R = -1.0 + R_format + I{R_format = 1.0} · R_answer. R_format gives 0.5 per well-formed step, capped at 1.0, and R_answer is token-level F1. — [Graph-R1](https://arxiv.org/abs/2507.21892)
- **Headline numbers:** Average F1 over six RAG datasets (2Wiki, HotpotQA, MuSiQue, NQ, PopQA, TriviaQA) with Qwen2.5-7B is 57.28, versus StandardRAG 32.05, HyperGraphRAG 29.40 and Search-R1 46.19. The next sentence of the paper gives 57.82 for the same 7B setting, an internal inconsistency. — [Graph-R1](https://arxiv.org/abs/2507.21892)
- **RL algorithm comparison:** "GRPO significantly outperforms REINFORCE++ ... and PPO." — [Graph-R1](https://arxiv.org/abs/2507.21892)

**GraphRAG-R1 (Yu et al., WWW 2026)**
- **Algorithm:** A modified GRPO with "rollout-with-thinking" and a retrieval-masked loss, both taken from R1-Searcher. The backbone is Qwen2.5-7B with LoRA, and retrieval runs through HippoRAG2. — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)
- **Progressive Retrieval Attenuation (PRA) reward:**
  - R_1 = r_0, and R_n = R_{n-1} + r_0 · β^{n-1} for n > 1.
  - It is bounded by r_0/(1-β), with r_0 = 0.5 and β = 0.5.

  — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)
- **Cost-Aware F1 (CAF) reward:** Answer F1 scaled by a factor that decreases with the number of retrieval calls n (hyperparameters 2 and 0.1), so "responses achieving identical F1 score receive higher rewards when using fewer retrievals". — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)
- **Phase-dependent training:**
  1. cold-start SFT;
  2. format plus PRA;
  3. CAF plus format.

  — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)
- **Headline numbers:** F1 gains over the strongest baseline of up to 38.08% (HotpotQA), 62.43% (MuSiQue) and 83.81% (2Wiki). Removing PRA or CAF causes "substantial performance drops". — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)

**KG-R1 (Lin et al., arXiv 2025/2026)**
- **Setup:** A single LLM agent (Qwen2.5-3B-it) and a "schema-agnostic" KG server with four 1-hop operations: get_tail_relations, get_head_relations, get_tail_entities, get_head_entities. — [KG-R1](https://arxiv.org/abs/2509.26383)
- **Actions and STOP:** At each turn the agent emits either a `<kg-query>` or an `<answer>`, and the answer ends the episode. The turn limit is H = 5. — [KG-R1](https://arxiv.org/abs/2509.26383)
- **Training:**
  - A GRPO-style objective with N = 16 rollouts per query.
  - Group-relative, turn-level credit assignment, with G_t = r_t^turn + λ·R^global.
  - Turn rewards cover format validity, KG-query validity, and answer format.
  - The global reward combines answer correctness (Hits@1) with a retrieval score of 1 "if any ground-truth entity appears anywhere in the retrieved information".

  — [KG-R1](https://arxiv.org/abs/2509.26383)
- **Headline numbers:**
  - WebQSP Hits@1 82.75 and CWQ 65.31, using 2.2-3x fewer generated tokens than ToG and ReKnoS.
  - Zero-shot cross-KG transfer, averaged: 77.50 when trained on WebQSP and 72.82 when trained on CWQ.

  — [KG-R1](https://arxiv.org/abs/2509.26383)
- **Ablations:**
  - Removing turn rewards: WebQSP 82.75 → 69.29, CWQ 65.31 → 52.03.
  - Removing the retrieval reward: -4.03 / -2.68 Hits@1, and "significantly lowers retrieval success".
  - Replacing GRPO with PPO: WebQSP 82.75 → 59.62, CWQ 65.31 → 35.66, with observed reward hacking ("the agent fabricates 'retrieved' content").

  — [KG-R1](https://arxiv.org/abs/2509.26383)

**Search-on-Graph-R1 (Sun et al., arXiv July 2026)**
- **Pipeline (three stages):**
  1. A frontier teacher (Qwen3-235B) is scaffolded with each question's gold SPARQL and walks the answer path with a live Search(entities, direction, properties) tool on Freebase.
  2. A Llama-3.1-8B student is fine-tuned on these traces with LoRA SFT.
  3. The student is trained further with GRPO in verl.

  — [SoG-R1](https://arxiv.org/abs/2607.18481)
- **Reward:** R = g(τ) · r_em(τ) · f(n_t), where g is a format gate requiring at least one tool call, r_em is exact match with no partial credit, and f(n_t) = max(f_min, 1 - λ·max(0, n_t - T_d)) is a multiplicative penalty on excess turns. — [SoG-R1](https://arxiv.org/abs/2607.18481)
- **Headline numbers (WebQSP / CWQ / GrailQA):**

  | Configuration | Accuracy |
  |---|---|
  | Full SFT + RL | 91.7 / 88.7 / 90.0 |
  | SFT only (table) | 89.6 / 86.1 / 88.8 |
  | RL only, no warm start (table) | 80.9 / 63.0 / 60.7 |

  The prose gives slightly different numbers for the last two rows: 90.9 / 86.3 / 88.8 and 81.9 / 63.0 / 60.0, an internal inconsistency. — [SoG-R1](https://arxiv.org/abs/2607.18481)
- **Efficiency:** RL cuts the average number of Search calls per correctly answered question from 2.68 to 2.28 (WebQSP), 4.98 to 4.80 (CWQ), and 2.08 to 1.88 (GrailQA). — [SoG-R1](https://arxiv.org/abs/2607.18481)
- **Tool batching:** Letting one Search call take a list of entities fixes a failure where the agent "often issues calls for only a subset before moving on, dropping the rest". — [SoG-R1](https://arxiv.org/abs/2607.18481)

### Inferences
- **The project's pointer head is the MINERVA, Lin and DIVA scoring rule generalised.** Those papers dot a query- and history-conditioned vector with stacked candidate embeddings. The project changes two things: candidates become heterogeneous (cells, rows, columns, passages, tables), and the source is the frontier of an evidence set rather than one node's outgoing edges.
- **M-Walk's u_0 = f(h_S, h_A) is the closest precedent for an explicit STOP logit** computed from a summary of the candidate set. MINERVA and Lin instead emulate STOP with a self-loop and a fixed horizon.
- **Every classic walker keeps a single current location (a path).** The project's state is an accumulated evidence set. M-Walk's state is the nearest analogue, since it keeps all traversed nodes and neighbourhoods, but its actions are still limited to the current node's neighbours.
- **Reward-design lineage maps directly onto the project's reward:**
  - per-hop cost ← DeepPath's 1/length efficiency term, GraphRAG-R1's CAF, and SoG-R1's turn penalty;
  - gold-evidence recall ← KG-R1's retrieval score (binary "any gold entity retrieved") and Lin's soft reward for plausible unobserved answers;
  - dense shaping ← SRN's potential-based shaping, which keeps the optimal policy unchanged under Ng et al. 1999.
- **Training-recipe lineage maps onto BC warm start + RL:**
  - DeepPath: supervised imitation of BFS paths, then REINFORCE.
  - SoG-R1: SFT on gold-path traces, then GRPO. It also shows RL-only is far worse (CWQ 63.0 vs 88.7).
  - CPL: pre-trains both agents.

### Gaps
- **SRN details missing:** The ACM PDF and the ResearchGate copy were blocked (HTTP 403). SRN's exact state definition, how it scores and sizes the beam over triples, its STOP/termination rule, the potential function φ*, and its policy-gradient variant (REINFORCE or otherwise) could not be confirmed from the paper. Only the abstract, one quoted formula, and secondary-source numbers are available.
- **CPL numbers:** The per-split Hits@K tables did not extract cleanly (misaligned columns), so only the authors' qualitative conclusions are reported.
- **GraphRAG-R1's CAF formula:** The exact functional form was garbled in extraction; only its described behaviour and hyperparameters are reported.
- **RLH reward:** The exact per-hop reward formula was garbled in extraction.

---

## Q3. Limitations that matter when moving from curated KGs to graphs induced from tables

### Takeaway
The classic walkers assume a fixed relation vocabulary, learned embeddings for each entity ID (transductive), and a fairly dense, clean graph. Table-induced graphs violate all three: column headers form an open relation vocabulary, every test table is new, and graphs mix low-degree cells with hub-like row and column nodes. The papers' own analyses document the failure modes the project must design against:
- spurious paths (Lin);
- false negatives and multi-answer queries (MINERVA, Lin);
- sparsity (DacKGR, CPL);
- graph noise such as duplicate or missing entities (DIVA);
- sparse terminal reward (M-Walk, SRN);
- schema mismatch and costly or unstable RL in the LLM agents (KG-R1, Graph-R1).

### Cited Findings
- **Fixed relation vocabulary:**
  - DeepPath's policy outputs a softmax over all KG relations "to keep the output dimension of the policy network consistent". — [DeepPath](https://arxiv.org/abs/1707.06690)
  - MINERVA and Lin et al. use learned embedding matrices r ∈ R^{|R|×d} and e ∈ R^{|E|×d}. — [MINERVA](https://arxiv.org/abs/1711.05851); [Lin et al.](https://arxiv.org/abs/1808.10568)
  - RLH clusters a fixed set of TransE relation embeddings. — [RLH](https://www.ijcai.org/proceedings/2020/0267.pdf)
- **Transductive entities:** MINERVA removed NELL-995 test triples whose entities "never appeared in the graph", since "there will be no trained embeddings for those entities". This cut the test set from 3,992 to 2,818 queries. — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Path patterns must recur:** On FB15K-237 "path types do not repeat often", so MINERVA "finds it hard to learn path types that generalize". 54% of FB15K-237 validation queries are 1-to-M, and the authors judge the single-target evaluation unsuitable for such queries. — [MINERVA](https://arxiv.org/abs/1711.05851)
- **Spurious paths:** "Since there are usually more spurious paths than correct ones, spurious paths are often found first, and following exploration can be increasingly biased towards them." Also: "Entities with larger fan-in (in-degree) and fan-out (out-degree) often exacerbate this problem." — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **False negatives:** The binary reward "rewards the false negative search results identically to true negatives". Lin et al.'s dev-set MRR is "significantly lower" than test MRR because multi-answer queries split across train, dev and test. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Gap to embedding models:** "The performance gap between RL-based approaches and the embedding-based approaches for KGQA remains" (Lin et al.). M-Walk on FB15k-237 "is still worse than the embedding-based methods". — [Lin et al.](https://arxiv.org/abs/1808.10568); [M-Walk](https://arxiv.org/abs/1802.04394)
- **Sparsity:**
  - DacKGR: path-based models "cannot work well on those sparse KGs that only contain sparse paths". Its own added edges carry "some noise ... In the beginning, the agent cannot distinguish the noise part". — [DacKGR](https://arxiv.org/abs/2010.01899)
  - CPL: path-based models lose to triple-ranking models on small or sparse subgraphs "because the general and evidential paths for reasoning on sparse KGs are not enough". — [CPL](https://arxiv.org/abs/1909.00230)
  - DeepPath: performs poorly "when there are not enough paths between entities", e.g. only 4 unique paths for filmWrittenBy. — [DeepPath](https://arxiv.org/abs/1707.06690)
- **Graph noise:** DIVA's error analysis on NELL attributes 39% of errors to "KG noise", 48% to the path finder and 13% to the path reasoner. The named noise types are "duplicate entity" (e.g. "Mavericks" vs "DallasMavericks") and "missing entity". — [DIVA](https://arxiv.org/abs/1803.06581)
- **Candidate-set recall ceiling:** M-Walk only ranks the nodes it visits, about 20-60 unique candidates on WN18RR. "When the ground truth is not in the candidate set, M-Walk always makes mistakes." — [M-Walk](https://arxiv.org/abs/1802.04394)
- **Sparse terminal reward:**
  - M-Walk is motivated by "the challenge of sparse rewards"; with MCTS, its share of positive-reward trajectories exceeds both PG-Walk and Q-Walk. — [M-Walk](https://arxiv.org/abs/1802.04394)
  - SRN targets "the delayed and sparse reward problem caused by weak supervision". — [SRN](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc)
  - DeepPath: without supervised warm-start, "the RL model will show very poor convergence properties". — [DeepPath](https://arxiv.org/abs/1707.06690)
- **Exploration on dense or hub-heavy graphs:** The optimal action-dropout rate correlates positively with average node fan-out. — [Lin et al.](https://arxiv.org/abs/1808.10568)
- **Schema mismatch and noise (LLM era):**
  - KG-R1's limitations: performance "depends on the quality and schema consistency of the KG, and transferability may degrade under severe schema mismatch or noisy graph structures", and RL training "remains computationally expensive". — [KG-R1](https://arxiv.org/abs/2509.26383)
  - SoG-R1 is evaluated only on Freebase, needs gold SPARQL at training time, and uses oracle entity linking, so "performance under self-linking is presumed to be lower". — [SoG-R1](https://arxiv.org/abs/2607.18481)
- **Credit assignment and RL stability (LLM era):**
  - KG-R1: removing turn-level rewards costs about 13.3-13.5 Hits@1 points, and PPO caused reward hacking and large drops relative to GRPO. — [KG-R1](https://arxiv.org/abs/2509.26383)
  - Graph-R1 likewise reports GRPO beating PPO and REINFORCE++. — [Graph-R1](https://arxiv.org/abs/2507.21892)
  - GraphRAG-R1 cites "reward hacking" with answer-only rewards and a "shallow retrieval" failure (retrieving once) as its motivation for process rewards. — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)
- **Text-augmented graphs:** CPL found that the benefit of adding text-extracted edges depends on corpus-KG alignment. With low alignment (FB60K-NYT10), plain MINERVA beat CPL on the full graph. — [CPL](https://aclanthology.org/D19-1269/)

### Inferences
- **Open relation vocabulary:** A table-graph policy cannot use a softmax over relation IDs (DeepPath) or ID-indexed relation and entity embeddings (MINERVA, Lin). Candidates must be encoded from their content, e.g. header text, cell text, row/column position and type, as KG-R1's schema-agnostic interface and SRN's question-attention over triples point toward. This is a concrete gap to name.
- **Table-specific spurious paths:** In tables, many cells share values (repeated entities, numbers, dates), so a walker can reach a correct-looking answer cell through an unrelated row. This is Lin et al.'s spurious-path problem, sharpened. The project's gold-evidence recall reward is a direct, supervised remedy that the KG papers lacked: Lin relied on random action dropout, DIVA on a learned path reasoner.
- **Hubs and sparsity together:** Column and table nodes are hubs (high fan-out) while individual cells are sparse. That points to combining Lin-style fan-out capping and action dropout with the project's invalid-action masking, while remembering that DacKGR and CPL show sparsity hurts path-based RL most.
- **False positives as well as false negatives:** For table QA with answer-only correctness, the bigger risk is false positives (right answer, wrong evidence), not Lin's false negatives. The evidence-recall term addresses this. Potential-based shaping (SRN) is a principled way to add dense evidence signals without changing the optimal policy.
- **PPO caution is regime-specific:** The PPO-vs-GRPO findings (KG-R1, Graph-R1) come from LLM token-level policies, where a value critic over long text contexts is hard to learn. They do not directly show that PPO is a bad choice for a small pointer policy with a learned critic and GAE. The report should still cite them as a risk (reward hacking, instability) and justify PPO with a BC warm start, invalid-action masking, and per-step rewards.

### Gaps
- No paper in this cluster evaluates RL walking on graphs induced from tables or semi-structured web tables. The transfer claims above are inferences.
- A systematic search for 2021-2024 RL KG walkers that address inductive or unseen-entity settings or use PPO/A2C was not done within this task's budget. Such work may exist and should be checked before claiming novelty for "PPO on graph walking".
- SoG-R1 names two concurrent SFT-then-RL KG reasoners, "EoG (Yan et al., 2026)" and "KG-Hopper (Wang and Yu, 2026)". Neither was verified here. — [SoG-R1](https://arxiv.org/abs/2607.18481)

---

## Q4. Closest ancestors of a PPO pointer policy that selects entities on a table-induced graph, and top-5 ranking

### Takeaway
The closest ancestors, in order:
1. **MINERVA:** the MDP template and pointer-style edge scoring.
2. **Lin et al. 2018:** reward shaping, action dropout and fan-out capping, which address spurious paths and false negatives.
3. **M-Walk:** an explicit STOP logit plus a shared policy/value head, against sparse reward.
4. **SRN:** natural-language-question KBQA with weak supervision and potential-based reward shaping.
5. **KG-R1, paired with Search-on-Graph-R1:** the modern agentic counterpart, with schema-agnostic actions, a retrieval-recall reward, turn-level credit assignment, a cost penalty, and SFT warm start before RL. Both also warn about PPO instability in LLM settings.

DeepPath is the historical precedent for warm-starting from imitation, but its fixed relation action space makes it a poor architectural ancestor.

### Cited Findings (ranked)
1. **MINERVA (ICLR 2018).**
   - Query-conditioned walk from the topic entity under a finite-horizon POMDP.
   - Variable-size, order-invariant candidate scoring: softmax(A_t · MLP([h_t; o_t; r_q])).
   - NO_OP self-loop as STOP; REINFORCE with a baseline and entropy; terminal reward.
   - Also handles "partially structured natural language queries" (WikiMovies 96.7).

   It is the template the project's pointer head and STOP generalise. — [MINERVA](https://arxiv.org/abs/1711.05851)
2. **Lin, Socher, Xiong (EMNLP 2018).** Same MDP, plus three things:
   - reward shaping R = R_b + (1 - R_b)·f from a pretrained scorer, a template for blending answer correctness with a learned or auxiliary evidence signal;
   - action dropout, a mask-based exploration device that pairs naturally with invalid-action masking;
   - PageRank fan-out capping.

   It improves FB15k-237 MRR from 29.3 (MINERVA) to 40.7 and explicitly analyses spurious paths and unseen queries. — [Lin et al.](https://arxiv.org/abs/1808.10568)
3. **M-Walk (NeurIPS 2018).**
   - Explicit STOP action with its own score u_0 = f(h_S, h_A) beside inner-product neighbour scores.
   - Policy and Q share the same logits, an actor-critic-like coupling.
   - State includes all traversed nodes and neighbourhoods, the nearest analogue to an evidence set.
   - MCTS and off-policy Q-learning address sparse reward.

   — [M-Walk](https://arxiv.org/abs/1802.04394)
4. **SRN (WSDM 2020).**
   - The RL-KBQA ancestor: natural-language multi-relation questions, final-answer-only ("weak") supervision.
   - Question-attention for triple selection and beam search over candidates.
   - Potential-based reward shaping against delayed and sparse reward.
   - MetaQA 3-hop Hits@1 75.2; PQL-3H 77.5.

   — [SRN abstract](https://www.semanticscholar.org/paper/8241481fd3832d9086677942e8dc825266f79adc); [He et al. 2021](https://richardhgl.github.io/files/wsdm2021.pdf); [QAGCN](https://arxiv.org/abs/2206.01818)
5. **KG-R1 (arXiv 2025/2026), with Search-on-Graph-R1 (arXiv 2026) as companion.**
   - KG-R1:
     - schema-agnostic 1-hop action set enabling zero-shot transfer to unseen KGs (77.50 average Hits@1);
     - a retrieval reward for getting any gold entity into the retrieved evidence, a binary cousin of gold-evidence recall;
     - turn-level group-relative credit assignment;
     - PPO underperformed GRPO with reward hacking.

     — [KG-R1](https://arxiv.org/abs/2509.26383)
   - SoG-R1:
     - SFT on gold-path traces, then GRPO;
     - a multiplicative turn-count penalty;
     - RL-only far below SFT+RL (CWQ 63.0 vs 88.7);
     - RL shortens trajectories (CWQ 4.98 → 4.80 Search calls).

     This mirrors behaviour cloning, then RL, with a per-hop cost. — [SoG-R1](https://arxiv.org/abs/2607.18481)

**Honourable mentions**
- **DeepPath:** BC warm start from BFS paths, plus efficiency and diversity rewards. — [DeepPath](https://arxiv.org/abs/1707.06690)
- **DacKGR:** sparse-graph remedies that add candidate actions from a scorer. — [DacKGR](https://arxiv.org/abs/2010.01899)
- **CPL:** the action space is augmented on the fly with edges extracted from text, structurally like adding linked passages as candidates next to table cells. — [CPL](https://arxiv.org/abs/1909.00230)
- **GraphRAG-R1:** retrieval-count-aware rewards and phase-dependent training. — [GraphRAG-R1](https://arxiv.org/abs/2507.23581)

### Inferences
**Gap analysis.** Compared against this ancestry, the project's distinctive combination is:
- **(a) A set-valued evidence state.** It is not a single-location path; MINERVA, Lin and DacKGR all walk from e_t.
- **(b) Heterogeneous candidates of variable size.** Cells, rows, columns, passages and joined tables, with type-aware encodings instead of ID embeddings. That removes the fixed-vocabulary and transductive limits documented in Q3.
- **(c) An explicit STOP in the pointer head.** Only M-Walk does this among the classic walkers; MINERVA explicitly declined it.
- **(d) PPO actor-critic with GAE on a compact pointer policy.** The verified walkers use REINFORCE, MCTS/Q-learning or (LLM) GRPO.
- **(e) A graded gold-evidence recall reward plus per-hop cost.** The ancestors have binary retrieval reward (KG-R1), efficiency 1/length (DeepPath), retrieval-count decay (GraphRAG-R1) and turn penalty (SoG-R1).
- **(f) Use as a plug-in selector inside existing table-reasoning frameworks**, rather than as a standalone answerer.

**What to borrow:**
- Lin's action dropout and fan-out cap, for hub nodes like columns.
- SRN-style potential-based shaping, to make the evidence-recall signal dense without changing the optimal policy.
- M-Walk's separate STOP logit computed from a pooled candidate-set summary.
- SoG-R1's evidence that a warm start is critical, and that multi-entity selection per step (batched entities) avoids "dropped entity" errors.

**Risks to pre-empt in the report:**
- Spurious paths through repeated cell values (Lin).
- A candidate-set recall ceiling: if the gold evidence is never offered as a candidate, the policy cannot recover (M-Walk's out-of-candidate-set error).
- RL instability or reward hacking (KG-R1, Graph-R1), although these were observed in LLM-token regimes.
- Path-based RL losing to non-path baselines on sparse graphs (DacKGR, CPL, MINERVA vs ConvE).

### Gaps
- No verified paper combines PPO+GAE with a pointer head over a variable candidate set on any graph. This is the core novelty claim, but it rests on an unexhaustive search (see Q3 Gaps).
- "Learning to Retrieve Reasoning Paths over Wikipedia Graph" (Asai et al., ICLR 2020) is a relevant non-RL analogue: a recurrent retriever with an end-of-evidence token over a hyperlink graph. It is outside this RL cluster and was not verified here; it is noted for researchers covering table/text evidence retrieval.
- Table-specific RL selectors (e.g. RL over table cells in table-QA frameworks) belong to another research thread and were deliberately excluded here.
