# RL for Multi-Hop Evidence Retrieval / Evidence Selection in QA (text QA and search agents): selection actions, STOP decisions and reward design

Scope note: I checked metadata (title, authors, date, comments/venue) for every paper below. I used the arXiv API, Semantic Scholar's batch API, OpenReview and ACL Anthology. Technical details come from the papers' full text: arXiv HTML for the 2025-2026 papers, and the arXiv/ACL PDFs for the older ones. I flag any venue I could not confirm on a primary page. Retrieval date: 2026-10-06.

## Q0. Verified paper list (core 15 + secondary), exact metadata, annotations, BibTeX

### Takeaway
Fifteen core papers are directly relevant. Six predate LLMs: three train evidence selection or retrieval with RL (R^3, Choi et al., Das et al.) and three are the closest supervised path retrievers (Asai et al., GoldEn, MDR). The other nine are 2025-2026 RL search agents chosen for how they design the selection action, the STOP decision and the reward. Ten more papers are verified and listed as secondary.

### Cited Findings

#### Core papers (metadata verified)

1. **R^3: Reinforced Ranker-Reader for Open-Domain Question Answering.** Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerald Tesauro, Bowen Zhou, Jing Jiang. AAAI 2018. arXiv:1709.00023. The arXiv comment reads "accepted by AAAI 2018". Title flag: the arXiv metadata says "Reinforced *Reader-Ranker*", while the PDF and Semantic Scholar say "Reinforced *Ranker-Reader*". — [arXiv](https://arxiv.org/abs/1709.00023)
   - The ranker makes a hard selection of a passage, so it is trained with REINFORCE. Its reward comes from how well the reader extracts the answer from the selected passage: 2 for an exact match, F1 for partial overlap, -1 for no overlap. The reader is trained with SGD at the same time, so this is joint training rather than a frozen reader. R^3 is first pretrained as the Simple Ranker-Reader (SR2), "to encourage convergence". — [arXiv PDF](https://arxiv.org/pdf/1709.00023)
   - Quasar-T test results (F1/EM). Same Single Reader with the SR2 ranker: 38.9/31.8. With the R^3 ranker: 40.0/33.1. Full SR2: 38.7/31.9. Full R^3: 40.8/34.1. Upper bound if the reader got the top-3 or top-5 passages: 51.7/43.7 and 58.7/49.2. That gap shows how much headroom better ranking would leave. — [arXiv PDF, Tables 4-5](https://arxiv.org/pdf/1709.00023)
   - It selects exactly one passage, so there is no STOP decision.

2. **Coarse-to-Fine Question Answering for Long Documents.** Eunsol Choi, Daniel Hewlett, Jakob Uszkoreit, Illia Polosukhin, Alexandre Lacoste, Jonathan Berant. ACL 2017 (long), pp. 209-220, doi:10.18653/v1/P17-1020. The arXiv version (1611.01839) is titled "Hierarchical Question Answering for Long Documents" and lists the authors in a different order. — [ACL Anthology](https://aclanthology.org/P17-1020/); [arXiv](https://arxiv.org/abs/1611.01839)
   - Sentence selection is a latent variable trained "from the answer only using reinforcement learning". The reward is the log-probability of the correct answer given the selected sentence. Because REINFORCE has high variance, training starts with distant supervision and gradually switches to RL, a curriculum the authors liken to DAGGER: the distant-supervision objective is used with probability r^e. — [ACL PDF](https://aclanthology.org/P17-1020.pdf)
   - REINFORCE beats the distant-supervision PIPELINE baseline, "which relies on a noisy supervision signal". It is 3.5x-6.7x faster at test time (K=1 or 2 sentences). The full-document BASE model is still slightly better on datasets where answers sit in the first few sentences. Sampling a second sentence improves every dataset. — [ACL PDF](https://aclanthology.org/P17-1020.pdf)
   - K is fixed at 1 or 2, so there is no learned STOP.

3. **Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering.** Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Andrew McCallum. ICLR 2019. arXiv:1905.05733. — [arXiv](https://arxiv.org/abs/1905.05733)
   - The reader is trained with supervision first. "After the training converges, we freeze the parameters of the reader model and train the parameters of the GRU network using policy gradients." The GRU query reformulator is rewarded with the reader's F1 at each step, so this is a **frozen-reader RL setup**. Using the minibatch-average reward as a baseline "significantly degrades the final performance", because questions in a batch vary in difficulty. — [arXiv PDF](https://arxiv.org/pdf/1905.05733)
   - Quasar-T retrieval P@1/P@3/P@5. Initial retriever: 35.7/49.6/56.3. After 7 multi-step reformulation steps: 42.9/55.5/59.3. R^3's ranker: 40.3/51.3/54.5. The number of steps is a fixed hyperparameter; there is no learned STOP. — [arXiv PDF, Table 3](https://arxiv.org/pdf/1905.05733)
   - With few steps, the policy explores: about 2 unique paragraphs across 3 steps. With more steps it shifts to exploiting paragraphs it has already found. — [arXiv PDF](https://arxiv.org/pdf/1905.05733)

4. **Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering.** Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, Caiming Xiong. ICLR 2020. arXiv:1911.10470. — [arXiv](https://arxiv.org/abs/1911.10470)
   - Recurrent retrieval over the hyperlink graph. "The process is terminated when [EOE], the end-of-evidence symbol, is selected, to allow it to capture reasoning paths with arbitrary length." [EOE] is a learned vector, and it is a negative candidate at every step t < |gold path|. That makes this **supervised STOP learning**; there is no RL. — [arXiv PDF](https://arxiv.org/pdf/1911.10470)
   - Training uses TF-IDF and hyperlink negatives, plus beam search at inference. On HotpotQA full wiki it gains +14.5 F1 / +14.0 EM over Semantic Retrieval and +10.9 F1 over Transformer-XH. Ablations: greedy search instead of beam costs about 4 EM; a reader trained without negatives loses more than 16 EM. — [arXiv PDF](https://arxiv.org/pdf/1911.10470)

5. **Answering Complex Open-domain Questions Through Iterative Query Generation (GoldEn Retriever).** Peng Qi, Xiaowen Lin, Leo Mehr, Zijian Wang, Christopher D. Manning. EMNLP-IJCNLP 2019, doi:10.18653/v1/D19-1261. arXiv:1910.07000. — [arXiv](https://arxiv.org/abs/1910.07000)
   - It alternates between reading context and generating natural-language search queries for off-the-shelf IR. It beats the best published HotpotQA model without using BERT. — [arXiv abstract](https://arxiv.org/abs/1910.07000)

6. **Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval (MDR).** Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei Du, Patrick Lewis, William Yang Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe Kiela, Barlas Oğuz. ICLR 2021 (Semantic Scholar lists ICLR; arXiv v1 Sep 2020). arXiv:2009.12756. — [arXiv](https://arxiv.org/abs/2009.12756); [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2009.12756?fields=venue,year)
   - It needs no hyperlinks or entity linking. "Both datasets assume 2 hops, so we fix n = 2", so the hop count is fixed and there is no STOP. It matches the best published HotpotQA accuracy while being 10x faster at inference, and is state of the art on HotpotQA and multi-evidence FEVER. — [arXiv PDF](https://arxiv.org/pdf/2009.12756)

7. **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning.** Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han. COLM 2025 (OpenReview). arXiv:2503.09516 (v5). — [arXiv](https://arxiv.org/abs/2503.09516); [OpenReview](https://openreview.net/forum?id=Rwhi91ideu)
   - The reward is an outcome-only exact-match reward. PPO is the default algorithm, and GRPO is also reported. Retrieved tokens are masked out of the loss, which "stabilizes training". STOP is implicit: the policy ends the trajectory by emitting `<answer>`. — [arXiv HTML](https://arxiv.org/html/2503.09516)
   - Reported improvement over RAG baselines, by source:
     - The v5 abstract says 24% (Qwen2.5-7B) and 20% (3B).
     - The introduction of the same version says 41% and 20%.
     - The OpenReview and search-index summary also says 41% and 20%.
     - **Flag: the paper is internally inconsistent across versions.** — [arXiv HTML](https://arxiv.org/html/2503.09516)

8. **An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents.** Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, Jiawei Han. arXiv preprint, May 2025, arXiv:2505.15117. I found no venue. — [arXiv](https://arxiv.org/abs/2505.15117)
   - This is the main controlled study of reward design and of PPO vs GRPO. Findings are in Q3 and Q5.

9. **R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning.** Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen. arXiv preprint, Mar 2025, arXiv:2503.05592. I found no venue. — [arXiv](https://arxiv.org/abs/2503.05592)
   - Two-stage, outcome-based RL with a modified Reinforce++:
     - Stage 1 rewards retrieval plus format only, with no answer reward, to teach the model to call search.
     - Stage 2 removes the retrieval reward and adds the answer reward.
   - With Qwen-2.5-7B-Base it beats ReARTeR (GPT-4o-mini) by up to 48.22% on HotpotQA and 21.72% on 2Wiki. — [arXiv HTML](https://arxiv.org/html/2503.05592)

10. **StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization.** Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu. EMNLP 2025 main. The ACL Anthology ID 2025.emnlp-main.1106 appears in the search index, and Semantic Scholar lists EMNLP. arXiv:2505.15107. — [arXiv](https://arxiv.org/abs/2505.15107); [ACL Anthology PDF](https://aclanthology.org/2025.emnlp-main.1106v2.pdf)
    - **PPO with GAE plus token-level step rewards.** Each search round's reward is information gain minus a redundancy penalty:
      - Information gain is the marginal improvement in TF-IDF cosine coverage of the gold documents, a submodular measure against a running max-similarity memory.
      - The redundancy penalty is the fraction of documents in this round that were already retrieved earlier.
    - The global reward at the last token is answer F1 plus a "search-key" reward (keyword F1 against reference subquestion queries). Retrieved tokens are masked. — [arXiv HTML](https://arxiv.org/html/2505.15107)
    - It gains 5.7, 9.1, 10.0 and 15.2 absolute points over standard RL baselines on the multi-hop benchmarks. — [arXiv HTML](https://arxiv.org/html/2505.15107)

11. **s3: You Don't Need That Much Data to Train a Search Agent via RL.** Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han. EMNLP 2025. The arXiv comment says "EMNLP 2025 camera-ready"; Semantic Scholar gives pp. 21599-21617. arXiv:2505.14146. — [arXiv](https://arxiv.org/abs/2505.14146); [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2505.14146?fields=venue,journal)
    - **Frozen generator.** The searcher's action space covers queries, selecting documents into the evidence set, and a stop decision. It is trained with PPO on the Gain Beyond RAG reward: generation accuracy with the s3 context minus accuracy with naive top-k RAG context. Details in Q4.

12. **Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning (MMOA-RAG).** Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, Jiaxin Mao. NeurIPS 2025 (arXiv comment). arXiv:2501.15228. — [arXiv](https://arxiv.org/abs/2501.15228)
    - Four modules: query rewriter, fixed retriever, **document Selector**, and generator. Three are trainable LLM agents optimized with MAPPO on a shared F1 reward, **after an SFT warm start**. That mirrors this project's behaviour-cloning-then-PPO setup. Details in Q3.

13. **DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning.** Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, Jiawei Han. COLM 2025 (OpenReview PDF header: "Published as a conference paper at COLM 2025"). arXiv:2503.00223. — [arXiv](https://arxiv.org/abs/2503.00223); [OpenReview](https://openreview.net/forum?id=u9JXu4L17I)
    - RL query generation where retrieval metrics such as recall and NDCG are used directly as the reward, against a frozen search engine. Recall: 65.07% vs 24.68% previous SOTA on publication search, and 63.18% vs 32.11% on trial search. — [arXiv HTML](https://arxiv.org/html/2503.00223)

14. **Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents (IGPO).** Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, Zhenzhe Ying. ICLR 2026 (arXiv comment "Accepted by ICLR 2026"). arXiv:2510.14967. — [arXiv](https://arxiv.org/abs/2510.14967)
    - The turn-level reward is the marginal increase in the policy's probability of the ground-truth answer. Details in Q3.

15. **Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty.** Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen. EMNLP 2025 Main (arXiv comment). arXiv:2505.17281. — [arXiv](https://arxiv.org/abs/2505.17281)
    - It defines over-search and under-search, which are the failure modes of the STOP/continue decision, and proposes β-GRPO with a confidence-threshold reward. Details in Q3.

#### Secondary papers (verified metadata; less central, or only partially examined)

- **ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning.** Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, Weipeng Chen. NeurIPS 2025 (arXiv comment). arXiv:2503.19470. GRPO; no supervised data on reasoning steps; reflection and self-correction emerge during RL. — [arXiv HTML](https://arxiv.org/html/2503.19470)
- **ZeroSearch: Incentivize the Search Capability of LLMs without Searching.** Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, Jingren Zhou. arXiv:2505.04588 (v3, May 2026). During training, an LLM fine-tuned with lightweight SFT simulates the search engine, which avoids noisy documents and API costs. **Venue unverified:** Semantic Scholar lists only arXiv. A search-engine summary claimed EMNLP 2025, but I could not confirm that on a primary page. — [arXiv](https://arxiv.org/abs/2505.04588)
- **DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments.** Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu. EMNLP 2025 per Semantic Scholar (pp. 414-431; track not confirmed). arXiv:2504.03160. — [arXiv](https://arxiv.org/abs/2504.03160); [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2504.03160?fields=venue,journal)
- **Search and Refine During Think: Facilitating Knowledge Refinement for Improved Retrieval-Augmented Reasoning (AutoRefine).** Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, Xiang Wang. NeurIPS 2025 (Semantic Scholar: "Advances in Neural Information Processing Systems 38", doi:10.52202/085713-5211). arXiv:2505.11277. GRPO with an outcome reward plus a "retrieval-specific reward" on the content of the `<refine>` block, which is an explicit evidence-distillation step. — [arXiv HTML](https://arxiv.org/html/2505.11277)
- **Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning (ReasonRAG).** Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Xiangyu Zhao. arXiv:2505.14069. Semantic Scholar lists NeurIPS; the arXiv page does not state a venue (**flag**). — [arXiv](https://arxiv.org/abs/2505.14069)
- **Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward.** Yong Deng, Guoqing Wang, Zhenzhe Ying, Xiaofeng Wu, Jinzhen Lin, Wenwen Xiong, Yuqin Dai, Shuo Yang, Zhanwei Zhang, Qiwen Wang, Yang Qin, Yuan Wang, Quanxing Zha, Sunhao Dai, Changhua Meng. arXiv:2508.12800, preprint. Not examined in depth. — [arXiv](https://arxiv.org/abs/2508.12800)
- **Ask the Right Questions: Active Question Reformulation with Reinforcement Learning.** Christian Buck, Jannis Bulian, Massimiliano Ciaramita, Wojciech Gajewski, Andrea Gesmundo, Neil Houlsby, Wei Wang. ICLR 2018. arXiv:1705.07830. An agent "sits between the user and a black box QA system" and is trained with policy gradient to maximize answer quality: an early frozen-reader RL design. — [arXiv](https://arxiv.org/abs/1705.07830)
- **ReasoNet: Learning to Stop Reading in Machine Comprehension.** Yelong Shen, Po-Sen Huang, Jianfeng Gao, Weizhu Chen. KDD 2017. arXiv:1609.05284. A termination state is learned with RL so the model "can dynamically determine whether to continue ... or to terminate reading when it concludes that existing information is adequate". This is the canonical early learned-STOP design. — [arXiv](https://arxiv.org/abs/1609.05284)
- **Learning to Search in Long Documents Using Document Structure.** Mor Geva, Jonathan Berant. COLING 2018 (ACL Anthology C18-1014 per Semantic Scholar). arXiv:1806.03529. RL navigation over a document tree. Metadata only; content not examined. — [arXiv](https://arxiv.org/abs/1806.03529)
- **Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning (MINERVA).** Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy, Alex Smola, Andrew McCallum. ICLR 2018. arXiv:1711.05851. Hop-by-hop RL entity selection over a KB. Metadata only; content not examined this session. — [arXiv](https://arxiv.org/abs/1711.05851)

#### BibTeX (core)

```bibtex
@inproceedings{wang2018r3,
  title     = {{R}$^3$: Reinforced Ranker-Reader for Open-Domain Question Answering},
  author    = {Wang, Shuohang and Yu, Mo and Guo, Xiaoxiao and Wang, Zhiguo and Klinger, Tim and Zhang, Wei and Chang, Shiyu and Tesauro, Gerald and Zhou, Bowen and Jiang, Jing},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2018},
  note      = {arXiv:1709.00023}
}

@inproceedings{choi2017coarse,
  title     = {Coarse-to-Fine Question Answering for Long Documents},
  author    = {Choi, Eunsol and Hewlett, Daniel and Uszkoreit, Jakob and Polosukhin, Illia and Lacoste, Alexandre and Berant, Jonathan},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {209--220},
  year      = {2017},
  doi       = {10.18653/v1/P17-1020}
}

@inproceedings{das2019multistep,
  title     = {Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering},
  author    = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and McCallum, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  note      = {arXiv:1905.05733}
}

@inproceedings{asai2020learning,
  title     = {Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering},
  author    = {Asai, Akari and Hashimoto, Kazuma and Hajishirzi, Hannaneh and Socher, Richard and Xiong, Caiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  note      = {arXiv:1911.10470}
}

@inproceedings{qi2019golden,
  title     = {Answering Complex Open-domain Questions Through Iterative Query Generation},
  author    = {Qi, Peng and Lin, Xiaowen and Mehr, Leo and Wang, Zijian and Manning, Christopher D.},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year      = {2019},
  doi       = {10.18653/v1/D19-1261}
}

@inproceedings{xiong2021mdr,
  title     = {Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval},
  author    = {Xiong, Wenhan and Li, Xiang Lorraine and Iyer, Srini and Du, Jingfei and Lewis, Patrick and Wang, William Yang and Mehdad, Yashar and Yih, Wen-tau and Riedel, Sebastian and Kiela, Douwe and O{\u{g}}uz, Barlas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  note      = {arXiv:2009.12756}
}

@inproceedings{jin2025searchr1,
  title     = {Search-{R1}: Training {LLMs} to Reason and Leverage Search Engines with Reinforcement Learning},
  author    = {Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  note      = {arXiv:2503.09516}
}

@article{jin2025empirical,
  title   = {An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved {LLM} Agents},
  author  = {Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O. and Han, Jiawei},
  journal = {arXiv preprint arXiv:2505.15117},
  year    = {2025}
}

@article{song2025r1searcher,
  title   = {{R1-Searcher}: Incentivizing the Search Capability in {LLMs} via Reinforcement Learning},
  author  = {Song, Huatong and Jiang, Jinhao and Min, Yingqian and Chen, Jie and Chen, Zhipeng and Zhao, Wayne Xin and Fang, Lei and Wen, Ji-Rong},
  journal = {arXiv preprint arXiv:2503.05592},
  year    = {2025}
}

@inproceedings{wang2025stepsearch,
  title     = {{StepSearch}: Igniting {LLMs} Search Ability via Step-Wise Proximal Policy Optimization},
  author    = {Wang, Ziliang and Zheng, Xuhui and An, Kang and Ouyang, Cijun and Cai, Jialu and Wang, Yuhang and Wu, Yichao},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  note      = {arXiv:2505.15107}
}

@inproceedings{jiang2025s3,
  title     = {s3: You Don't Need That Much Data to Train a Search Agent via {RL}},
  author    = {Jiang, Pengcheng and Xu, Xueqiang and Lin, Jiacheng and Xiao, Jinfeng and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {21599--21617},
  year      = {2025},
  note      = {arXiv:2505.14146}
}

@inproceedings{chen2025mmoarag,
  title     = {Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning},
  author    = {Chen, Yiqun and Yan, Lingyong and Sun, Weiwei and Ma, Xinyu and Zhang, Yi and Wang, Shuaiqiang and Yin, Dawei and Yang, Yiming and Mao, Jiaxin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2501.15228}
}

@inproceedings{jiang2025deepretrieval,
  title     = {{DeepRetrieval}: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning},
  author    = {Jiang, Pengcheng and Lin, Jiacheng and Cao, Lang and Tian, Runchu and Kang, SeongKu and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025},
  note      = {arXiv:2503.00223}
}

@inproceedings{wang2026igpo,
  title     = {Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
  author    = {Wang, Guoqing and Dai, Sunhao and Ye, Guangze and Gan, Zeyu and Yao, Wei and Deng, Yong and Wu, Xiaofeng and Ying, Zhenzhe},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {arXiv:2510.14967}
}

@inproceedings{wu2025searchwisely,
  title     = {Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty},
  author    = {Wu, Peilin and Zhang, Mian and Zhang, Xinlu and Du, Xinya and Chen, Zhiyu Zoey},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  note      = {arXiv:2505.17281}
}
```

#### BibTeX (secondary)

```bibtex
@inproceedings{chen2025research,
  title     = {{ReSearch}: Learning to Reason with Search for {LLMs} via Reinforcement Learning},
  author    = {Chen, Mingyang and Sun, Linzhuang and Li, Tianpeng and Sun, Haoze and Zhou, Yijie and Zhu, Chenzheng and Wang, Haofen and Pan, Jeff Z. and Zhang, Wen and Chen, Huajun and Yang, Fan and Zhou, Zenan and Chen, Weipeng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2503.19470}
}

@article{sun2025zerosearch,
  title   = {{ZeroSearch}: Incentivize the Search Capability of {LLMs} without Searching},
  author  = {Sun, Hao and Qiao, Zile and Guo, Jiayan and Fan, Xuanbo and Hou, Yingyan and Jiang, Yong and Xie, Pengjun and Zhang, Yan and Huang, Fei and Zhou, Jingren},
  journal = {arXiv preprint arXiv:2505.04588},
  year    = {2025}
}

@inproceedings{zheng2025deepresearcher,
  title     = {{DeepResearcher}: Scaling Deep Research via Reinforcement Learning in Real-world Environments},
  author    = {Zheng, Yuxiang and Fu, Dayuan and Hu, Xiangkun and Cai, Xiaojie and Ye, Lyumanshan and Lu, Pengrui and Liu, Pengfei},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {414--431},
  year      = {2025},
  note      = {arXiv:2504.03160; track not verified}
}

@inproceedings{shi2025autorefine,
  title     = {Search and Refine During Think: Facilitating Knowledge Refinement for Improved Retrieval-Augmented Reasoning},
  author    = {Shi, Yaorui and Li, Sihang and Wu, Chang and Liu, Zhiyuan and Fang, Junfeng and Cai, Hengxing and Zhang, An and Wang, Xiang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2505.11277}
}

@article{zhang2025reasonrag,
  title   = {Process vs. Outcome Reward: Which is Better for Agentic {RAG} Reinforcement Learning},
  author  = {Zhang, Wenlin and Li, Xiangyang and Dong, Kuicai and Wang, Yichao and Jia, Pengyue and Li, Xiaopeng and Zhang, Yingyi and Xu, Derong and Du, Zhaocheng and Guo, Huifeng and Tang, Ruiming and Zhao, Xiangyu},
  journal = {arXiv preprint arXiv:2505.14069},
  year    = {2025}
}

@article{deng2025atomsearcher,
  title   = {{Atom-Searcher}: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward},
  author  = {Deng, Yong and Wang, Guoqing and Ying, Zhenzhe and Wu, Xiaofeng and Lin, Jinzhen and Xiong, Wenwen and Dai, Yuqin and Yang, Shuo and Zhang, Zhanwei and Wang, Qiwen and Qin, Yang and Wang, Yuan and Zha, Quanxing and Dai, Sunhao and Meng, Changhua},
  journal = {arXiv preprint arXiv:2508.12800},
  year    = {2025}
}

@inproceedings{buck2018ask,
  title     = {Ask the Right Questions: Active Question Reformulation with Reinforcement Learning},
  author    = {Buck, Christian and Bulian, Jannis and Ciaramita, Massimiliano and Gajewski, Wojciech and Gesmundo, Andrea and Houlsby, Neil and Wang, Wei},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  note      = {arXiv:1705.07830}
}

@inproceedings{shen2017reasonet,
  title     = {{ReasoNet}: Learning to Stop Reading in Machine Comprehension},
  author    = {Shen, Yelong and Huang, Po-Sen and Gao, Jianfeng and Chen, Weizhu},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2017},
  note      = {arXiv:1609.05284}
}

@inproceedings{geva2018learning,
  title     = {Learning to Search in Long Documents Using Document Structure},
  author    = {Geva, Mor and Berant, Jonathan},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics (COLING)},
  year      = {2018},
  note      = {arXiv:1806.03529}
}

@inproceedings{das2018minerva,
  title     = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author    = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  note      = {arXiv:1711.05851}
}
```

### Inferences
- No verified paper here applies RL evidence selection to tables or table+text. Every one works over text passages or sentences, or over a KB in the case of MINERVA. That is a clear gap the project can claim.
- Two of the core papers can be cited as "frozen reader + RL selector" precedents: Das et al. 2019 and s3. A third, Buck et al., is a looser precedent: a black-box QA environment with a query reformulator rather than an evidence selector.

### Gaps
- GoldEn Retriever: I verified only the abstract this session. Its per-hop training signal (it is believed to use supervised "oracle" queries) and its hop count were not confirmed from full text.
- MINERVA and Geva & Berant: metadata only. Their STOP and action designs were not checked.
- Track (main vs Findings vs Industry) is unconfirmed for DeepResearcher and s3. Venues are unconfirmed for ZeroSearch, ReasonRAG and the Jin et al. empirical study.

## Q1. Pre-LLM RL evidence selection vs. supervised path retrieval: what did RL buy?

### Takeaway
Before LLMs, RL selection (REINFORCE) gave modest but consistent gains when gold evidence labels were missing or noisy: about +2 F1/EM on Quasar-T for R^3, and better than noisy distant supervision for Choi et al. All of these designs fixed the number of selections and none learned STOP. Supervised path retrievers trained on gold evidence chains (Asai et al., MDR) produced much larger jumps on HotpotQA. Asai et al. also learned STOP with supervision, via an [EOE] symbol.

### Cited Findings
- R^3 trains the ranker with REINFORCE using a bounded reader-based reward (2 for EM, F1 for partial overlap, -1 for none). Training needs SR2 pretraining "to encourage convergence". — [R^3 PDF](https://arxiv.org/pdf/1709.00023)
- On Quasar-T test, R^3 reaches 40.8 F1 / 34.1 EM vs 38.7/31.9 for the jointly supervised SR2. Swapping the RL-trained ranker in front of a fixed Single Reader gives 40.0/33.1 vs 38.9/31.8 for the SR2 ranker. — [R^3 PDF, Table 4](https://arxiv.org/pdf/1709.00023)
- Choi et al.: "Jointly learning answer generation and sentence selection, REINFORCE outperforms PIPELINE, which relies on a noisy supervision signal". Training needed a curriculum from distant supervision to RL because "Training with REINFORCE is known to be unstable due to the high variance". — [ACL 2017 PDF](https://aclanthology.org/P17-1020.pdf)
- Das et al.: RL query reformulation against a frozen reader raised Quasar-T P@1 from 35.7 to 42.9 after 7 steps, beating R^3's ranker at 40.3. Steps are fixed, not learned. — [arXiv PDF](https://arxiv.org/pdf/1905.05733)
- Asai et al. (supervised; [EOE] STOP; beam search) reports +14.5 F1 / +14.0 EM over the prior SOTA on HotpotQA full wiki. Greedy decoding instead of beam search costs about 4 EM. — [arXiv PDF](https://arxiv.org/pdf/1911.10470)
- MDR fixes n=2 hops, needs no hyperlinks, and matches the best published HotpotQA accuracy at 10x faster inference. — [arXiv PDF](https://arxiv.org/pdf/2009.12756)
- ReasoNet (KDD 2017) learns a termination state with RL, "to terminate reading when it concludes that existing information is adequate". It is the earliest verified learned-STOP design in this list. — [arXiv](https://arxiv.org/abs/1609.05284)

### Inferences
- When gold evidence paths exist (HotpotQA supporting facts; the project's gold cells, rows and passages), supervised path learning gives the bulk of the gains. That pattern supports the project's BC-then-PPO design: BC on gold evidence chains, then PPO to align with what the frozen reader actually needs and to learn cost-aware stopping. RL alone from answer reward needed pretraining or curricula to converge (R^3, Choi).
- Asai's [EOE]-as-a-candidate trick is the natural supervised precursor of the project's STOP action. The BC phase can label STOP as positive at t = |gold chain| + 1 and negative before that.

### Gaps
- I found no pre-LLM paper that directly compares RL vs supervised training of a *multi-hop* path retriever on the same dataset with a learned STOP. Comparisons above cross papers and datasets (Quasar-T vs HotpotQA).
- ReasoNet's exact RL algorithm and numbers were not extracted; I saw only the abstract.

## Q2. 2025-2026 RL search agents: which exist, and how they design the selection action and the STOP decision

### Takeaway
All 2025-2026 agents listed in the brief exist and are verified: Search-R1, R1-Searcher, ReSearch, DeepResearcher, ZeroSearch, StepSearch, s3, MMOA-RAG and DeepRetrieval. Later additions include IGPO (ICLR 2026), AutoRefine, Search Wisely and ReasonRAG. Most treat STOP implicitly: the trajectory ends when the policy emits an answer, under a maximum-turn cap. Only s3 and MMOA-RAG have an explicit document-selection action that builds an evidence set for a separate reader.

### Cited Findings
- Search-R1 uses `<search>`, `<information>` and `<answer>` tokens. The model decides when to search and ends the episode by answering. During training, "LLMs learn to call the search engine more times as the training proceeds". — [arXiv HTML](https://arxiv.org/html/2503.09516)
- s3's loop: "it generates queries, retrieves documents, selects a subset of useful evidence, and decides whether to continue searching. A frozen generator LLM then consumes the accumulated evidence". Each rollout consists of "emitted queries, document selections, and a stop decision". — [arXiv HTML](https://arxiv.org/html/2505.14146)
- MMOA-RAG's Selector "output[s] the IDs of candidate documents helpful to answering". Its action space is restricted to document-ID words, which "can significantly reduce the exploration space of the Selector and provide a more stable training process". — [arXiv HTML](https://arxiv.org/html/2501.15228)
- AutoRefine adds an explicit `<refine>` step that distills evidence from retrieved documents. It is trained with GRPO plus a retrieval-specific reward on the refined content. — [arXiv HTML](https://arxiv.org/html/2505.11277)
- ReSearch uses GRPO with no supervised reasoning-step data. — [arXiv HTML](https://arxiv.org/html/2503.19470)
- ZeroSearch replaces live search with an LLM that simulates search during training, to avoid uncontrolled document quality and API cost. — [arXiv](https://arxiv.org/abs/2505.04588)
- DeepResearcher scales RL to real-world web environments. — [arXiv](https://arxiv.org/abs/2504.03160)
- For fair comparison, s3 caps all multi-turn baselines at 4 turns and 3 documents per turn. — [arXiv HTML](https://arxiv.org/html/2505.14146)

### Inferences
- The project's design is a discrete entity-selection action plus an explicit STOP action, with a frozen reader. Structurally it is closer to s3 and the MMOA-RAG Selector than to Search-R1-style token-generating agents. It should be positioned that way: "a small-action-space, explicit evidence-set MDP" rather than a free-text search agent.
- An explicit STOP action plus an explicit per-hop cost is rare among 2025 agents. Most rely on a maximum-turn cap and implicit answer-emission.

### Gaps
- I did not extract DeepResearcher's and ZeroSearch's reward formulas or numbers in detail.

## Q3. Reward design evidence: outcome-only vs process or evidence-level rewards, reward hacking, and effects on retrieval-step counts

### Takeaway
The evidence conflicts:
- Jin et al.'s controlled study found that an intermediate retrieval reward (substring-EM of retrieved documents) gives no consistent benefit under either PPO or GRPO, and that larger weights *hurt*.
- Papers with better-designed step rewards report clear gains: StepSearch (gold-document information gain minus redundancy), IGPO (gain in the ground-truth answer's probability) and ReasonRAG (process rewards via MCTS).

Penalties on step counts or format errors are common and small: MMOA-RAG's -0.5 and -1, and StepSearch's redundancy term. Explicit reports of reward hacking are scarce.

### Cited Findings
- **Jin et al. (empirical study):**
  - "format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact."
  - "Adding intermediate retrieval rewards does not significantly improve final performance for either PPO or GRPO."
  - "Increasing [the retrieval-reward weight] consistently leads to degraded performance." The authors say the substring-EM retrieval reward "may overly constrain the retrieval trajectory".
  - An excessively large format-reward weight "may cause overfitting to format reward". — [arXiv HTML](https://arxiv.org/html/2505.15117)
- **Jin et al., effect of search-engine quality:** "Stronger search engines ... lead to more stable training and higher final performance". "With BM25, the agent gradually increases the number of search engine calls". — [arXiv HTML](https://arxiv.org/html/2505.15117)
- **StepSearch:**
  - Global reward = answer F1 plus a keyword-match search-key reward.
  - Per-round reward = information gain minus redundancy penalty, placed on the last token of each round, with GAE.
  - Information gain is measured against gold documents via TF-IDF cosine coverage with a running maximum, which gives diminishing returns. The redundancy penalty is the fraction of documents already retrieved in earlier rounds.
  - Result: +5.7, +9.1, +10.0 and +15.2 absolute points over standard RL baselines. — [arXiv HTML](https://arxiv.org/html/2505.15107)
- **IGPO:**
  - Outcome-only rewards cause "advantage collapse": when all rollouts in a group get the same reward, the group-relative advantage is zero.
  - Turn reward = the marginal increase in the policy's probability of producing the correct answer. IG and outcome rewards are normalized separately and combined as discounted returns.
  - Average score 60.2, +6.3 over DeepResearcher.
  - vs standard GRPO: +16.6 for 3B (32.3 → 48.9) and +8.3 for 7B (51.9 → 60.2). — [arXiv HTML](https://arxiv.org/html/2510.14967)
- **ReasonRAG:**
  - Uses the Shortest Path Reward Estimation (SPRE) process reward, which "penalizes unnecessarily long trajectories", together with MCTS exploration. Training is preference optimization on process-level pairs, not online PPO.
  - Trained on 5k queries, it beats Search-R1 trained on 90k, with average EM 34.4% and F1 42.3%. — [arXiv HTML](https://arxiv.org/html/2505.14069)
- **R1-Searcher:** curriculum by reward. Stage 1 uses retrieval + format rewards only, so the model learns to invoke search. Stage 2 drops the retrieval reward and adds the answer reward. — [arXiv HTML](https://arxiv.org/html/2503.05592)
- **MMOA-RAG penalties:**
  - Query rewriter: -0.5 if it produces more than 4 sub-questions.
  - Selector: -1 for duplicate or ill-formatted document IDs.
  - Generator: -0.5 for overly long answers.
  - "the values of each penalty are mostly 0". — [arXiv HTML](https://arxiv.org/html/2501.15228)
- **s3:** its GBR reward is relative to a naive-RAG baseline, which "sidesteps answer token overfitting". Performance peaks at k=8 documents and 3 turns. "Adding more turns or broader retrieval brings limited improvement", and the policy "learns to emit focused and early queries". — [arXiv HTML](https://arxiv.org/html/2505.14146)
- **Search Wisely:**
  - Over- and under-search are widespread: one model "could have avoided searching in 27.7% of its search steps"; another showed error rates as high as 63% in non-search steps.
  - β-GRPO (a confidence threshold in the reward) lowers over-search from 21.10% to 19.89% and under-search from 42.04% to 34.71% vs Search-R1-GRPO (Qwen2.5-3B), and gives "4% higher average exact match". — [arXiv HTML](https://arxiv.org/html/2505.17281)
- **DeepRetrieval:** uses retrieval metrics (recall, NDCG) directly as the reward and argues this is "more direct and robust" than downstream-generator rewards. — [arXiv HTML](https://arxiv.org/html/2503.00223)
- **AutoRefine:** combines outcome reward with a retrieval-specific reward. Its ablations "confirm the contribution of both the retrieval-specific reward and the refinement module". — [arXiv HTML](https://arxiv.org/html/2505.11277)

### Inferences
- The project's reward has three parts: answer correctness, gold-evidence recall and per-hop cost. It sits between StepSearch (gold-document information gain plus redundancy penalty) and s3 (gain over a baseline with a frozen reader).
- Jin et al.'s negative result means the recall weight should be ablated (λ_recall ∈ {0, small, large}). A coverage-style, diminishing-returns recall (as in StepSearch) is preferable to a binary "contains answer" signal.
- Making recall *marginal*, rewarding newly covered gold entities per hop, also penalizes redundant selections implicitly, the way StepSearch does.
- Per-hop costs in this literature are small relative to the outcome reward (MMOA uses -0.5 against an F1 in [0,1]). Search Wisely shows that both over- and under-search are real failure modes. So the project should report the distribution of hops and STOP timing, not just accuracy.

### Gaps
- I found no paper with an explicit reward-hacking case study, such as a selector gaming a recall term or stopping early to avoid cost penalties. Jin et al.'s format-reward overfitting is the closest. This is a gap the project can probe empirically.
- I did not extract numeric tables for the Jin et al. reward ablations; only the qualitative conclusions above.

## Q4. Frozen-reader setups: which papers train only the selector or retriever, and what did they get?

### Takeaway
Four verified papers train only the selector, retriever or query policy against a frozen downstream model:
- s3 (frozen generator; selection + STOP; PPO).
- Das et al. 2019 (frozen reader; RL reformulator).
- DeepRetrieval (frozen search engine; recall reward).
- Buck et al. 2018 (black-box QA environment).

s3 is the strongest and most recent evidence that the frozen-reader setting works with very little data.

### Cited Findings
- **s3:**
  - "The generator remains frozen; gradients are backpropagated only through the search policy."
  - GBR = Acc(G(q, D_s3), A) − Acc(G(q, D_RAG), A).
  - Training data is filtered to questions the frozen generator gets wrong under naive RAG (169,615 → 70,286).
  - It needs 20 PPO steps (≈2.4k examples) vs 2,100 steps / 170k examples for Search-R1, and wall-clock time drops 33×.
  - Best average generation accuracy on six general QA datasets: 58.9%. — [arXiv HTML](https://arxiv.org/html/2505.14146)
- **s3 ablation:** removing document selection degrades 4 of 6 datasets but slightly helps NQ and 2Wiki. Removing "Begin with Search" (seeding with the original question) causes a significant drop. — [arXiv HTML](https://arxiv.org/html/2505.14146)
- **s3 algorithm:** PPO is used "due to its strong empirical stability". — [arXiv HTML](https://arxiv.org/html/2505.14146)
- **Das et al.:** the reader is frozen after supervised training, and the GRU reformulator is trained with policy gradient on reader F1. Quasar-T P@1 rises from 35.7 to 42.9. — [arXiv PDF](https://arxiv.org/pdf/1905.05733)
- **DeepRetrieval:** a 3B query generator, trained against real or frozen search engines with recall as the reward, reaches 65.07% recall (vs 24.68%) on publication search. — [arXiv HTML](https://arxiv.org/html/2503.00223)
- **Buck et al. (AQA):** the agent "sits between the user and a black box QA system", trained with policy gradient on answer quality. — [arXiv](https://arxiv.org/abs/1705.07830)
- **Not frozen (contrast):** R^3 and Choi et al. train selector and reader jointly. MMOA-RAG trains the generator as an agent too. Search-R1, R1-Searcher, ReSearch and StepSearch train a single LLM that both searches and answers. — [R^3](https://arxiv.org/pdf/1709.00023); [MMOA-RAG](https://arxiv.org/html/2501.15228); [Search-R1](https://arxiv.org/html/2503.09516)
- **MMOA-RAG module configurations:** the paper studies subsets of agents, e.g. S+G with no query rewriter. Its "Table 2" shows the gain from the MAPPO stage over the SFT stage for each configuration. — [arXiv HTML](https://arxiv.org/html/2501.15228)

### Inferences
- s3's "gain over a fixed baseline" reward is a ready-made variance-reduction idea for the project. The reward for answer correctness could be defined relative to the frozen reader's accuracy with a heuristic evidence set, such as BM25 top-k cells or rows.
- s3's training-data filter (drop questions the reader already solves with naive retrieval) carries over directly.
- With a frozen reader, gains must come entirely from *which* evidence is selected. That makes evidence-recall and STOP-timing analyses natural central results.

### Gaps
- I did not extract MMOA-RAG's per-module ablation numbers (e.g. Selector-only training).
- No verified paper uses a frozen reader with a *table* or hybrid evidence set.

## Q5. PPO vs GRPO in this literature: stability findings

### Takeaway
Search-R1 reports the most direct comparison: GRPO converges faster, but PPO is more stable (GRPO showed reward collapse), and final rewards are comparable. Search-R1 therefore uses PPO by default. s3 and StepSearch choose PPO for stability. IGPO diagnoses "advantage collapse" in outcome-reward GRPO. Several other agents use critic-free methods anyway: ReSearch, AutoRefine and β-GRPO use GRPO; R1-Searcher uses Reinforce++.

### Cited Findings
- **Search-R1:**
  - "(1) GRPO converges faster than PPO across all cases. This is because PPO relies on a critic model, which requires several warm-up steps".
  - "(2) PPO demonstrates greater training stability ... GRPO leads to reward collapse after training for many steps, whereas PPO remains stable".
  - "(3) The final training rewards of PPO and GRPO are comparable".
  - Conclusion: "PPO exhibits greater training stability, making it a preferable choice in this setting." — [arXiv HTML](https://arxiv.org/html/2503.09516)
- **Search-R1, retrieved-token loss masking:** "consistently outperforms the variant without masking". This applies under PPO. — [arXiv HTML](https://arxiv.org/html/2503.09516)
- **s3:** chooses PPO "due to its strong empirical stability", citing Jin et al. 2025 and Jiang et al. 2025. — [arXiv HTML](https://arxiv.org/html/2505.14146)
- **StepSearch:** builds "StePPO" on PPO with GAE, so per-round rewards can be credited through a learned value function. — [arXiv HTML](https://arxiv.org/html/2505.15107)
- **IGPO:** "standard GRPO leads to advantage collapse" under outcome-only rewards. The fix is dense turn-level rewards inside a GRPO-style objective with discounted returns. The benefit is larger for weaker (3B) models. — [arXiv HTML](https://arxiv.org/html/2510.14967)
- **Jin et al. (empirical study):** evaluates both PPO and GRPO. Intermediate retrieval rewards fail to help under either. General-purpose LLMs outperform reasoning LLMs as the policy under both. — [arXiv HTML](https://arxiv.org/html/2505.15117)
- **Critic-free choices:** R1-Searcher uses a modified Reinforce++; ReSearch, AutoRefine and Search Wisely use GRPO variants. — [R1-Searcher](https://arxiv.org/html/2503.05592); [ReSearch](https://arxiv.org/html/2503.19470); [AutoRefine](https://arxiv.org/html/2505.11277); [Search Wisely](https://arxiv.org/html/2505.17281)
- **MMOA-RAG:** uses MAPPO (actor, critic and an SFT reference model) after an SFT warm start. — [arXiv HTML](https://arxiv.org/html/2501.15228)

### Inferences
- The project's choice of PPO + GAE (with a critic) after behaviour cloning is well supported:
  - Search-R1 found PPO more stable.
  - StepSearch shows GAE is the natural way to credit per-hop rewards.
  - MMOA-RAG demonstrates the SFT-warm-start → PPO recipe for a document selector.
- A GRPO baseline would be a cheap and well-motivated ablation. The literature predicts faster early convergence but possible collapse, plus advantage collapse when all rollouts for a question score identically, which is likely with exact-match on easy questions.

### Gaps
- I found no PPO vs GRPO comparison for a *small discrete action space* (entity or document selection). All comparisons above are for token-level LLM policies.
- Search-R1's PPO/GRPO numbers per dataset (Table 3) were not extracted, only the qualitative conclusions.

## Q6. Ranking: top 3-5 papers for this project, plus gap statement

### Takeaway
Top 5 for "PPO-based entity selection with STOP and a frozen reader":
1. s3
2. Search-R1, with the Jin et al. empirical study as its companion
3. StepSearch
4. MMOA-RAG
5. Asai et al. 2020

Honourable mentions: Das et al. 2019 and R^3, the classic frozen- or joint-reader RL selectors.

### Cited Findings
1. **s3 (EMNLP 2025):** frozen generator. Explicit document-selection and stop actions. PPO. Reward is gain over a naive-RAG baseline. Very data-efficient (2.4k examples). This is the closest analogue to the project's MDP. — [arXiv HTML](https://arxiv.org/html/2505.14146)
2. **Search-R1 (COLM 2025) + Jin et al. (2025):** the reference PPO-vs-GRPO stability result, retrieved-token masking, outcome-only reward, and a controlled study showing format rewards help while naive intermediate retrieval rewards do not. — [Search-R1](https://arxiv.org/html/2503.09516); [Empirical study](https://arxiv.org/html/2505.15117)
3. **StepSearch (EMNLP 2025):** PPO + GAE with per-step rewards from gold-document information gain and redundancy penalty, plus a global answer-F1 reward. Nearly the same reward decomposition as the project's correctness + recall + per-hop-cost reward. — [arXiv HTML](https://arxiv.org/html/2505.15107)
4. **MMOA-RAG (NeurIPS 2025):** a document-Selector agent with a constrained ID action space, SFT warm start then (MA)PPO, and small penalty terms. It mirrors the project's BC → PPO pipeline. — [arXiv HTML](https://arxiv.org/html/2501.15228)
5. **Asai et al. (ICLR 2020):** a supervised sequential evidence retriever with a learned [EOE] STOP and beam search. It is the right non-RL baseline and the template for supervised STOP labels in the BC phase. — [arXiv PDF](https://arxiv.org/pdf/1911.10470)

### Inferences
- **Gap statement for the report.** No verified paper combines all of the following:
  - (a) structured (table/hybrid) evidence;
  - (b) an explicit discrete entity-selection + STOP MDP;
  - (c) a frozen reader;
  - (d) a composite reward of answer correctness + gold-evidence recall + per-hop cost;
  - (e) PPO + GAE after behaviour cloning.

  Closest partial overlaps: s3 has (b), (c) and (e) but its reward has no recall or cost term. StepSearch has (d) and (e) but no frozen reader. MMOA-RAG has a selector and SFT → PPO but trains the generator too.
- **Contested points the project can test empirically:**
  - Whether evidence-level (recall) rewards help: Jin et al. say no; StepSearch, IGPO and AutoRefine say yes.
  - How per-hop cost shapes STOP timing: over- and under-search, as in Search Wisely.

### Gaps
- The ranking is my judgement based on structural similarity to the project. No source ranks these papers for table QA.
