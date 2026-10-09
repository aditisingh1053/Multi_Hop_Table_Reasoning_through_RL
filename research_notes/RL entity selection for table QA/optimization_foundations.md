# Optimization Foundations for a PPO Actor-Critic Entity-Selection Policy (variable discrete action set, sparse terminal reward)

Scope note: These notes cover the optimisation and RL foundations only: PPO/GAE/REINFORCE/A2C, implementation details, invalid-action masking, variable action sets, reward shaping, critic-free baselines (GRPO/RLOO/Dr. GRPO), imitation warm start, and statistical reporting. I checked every bibliographic detail against an arXiv API record, the publisher or proceedings page (PMLR, ACL Anthology, AAAI OJS, NeurIPS, Springer, Nature, FLAIRS OJS), OpenReview, or ML Anthology. I extracted the equations from the papers' own PDF or HTML text (pdftotext on the arXiv PDFs). Where I could not read the primary text directly, the note says so. Notation is LaTeX. The recommended core list has 16 papers, plus 3 optional and 2 "do not cite" papers; see the last section.

---

## Q1. PPO, GAE, REINFORCE with baseline, and A3C/A2C: exact equations and recommended hyper-parameters

### Takeaway
PPO maximises the clipped surrogate $L^{CLIP}$. It is combined with a squared-error value loss and an entropy bonus when policy and value share parameters, and it is normally paired with truncated GAE advantages. The paper's best clip value is $\epsilon=0.2$, and its default GAE setting is $\gamma=0.99,\ \lambda=0.95$. GAE's $\lambda$ trades bias for variance: $\lambda=1$ is unbiased but high variance, $\lambda=0$ is low variance but biased unless $V$ is exact. REINFORCE (Williams 1992) is the baseline-subtracted likelihood-ratio update. A3C is the canonical reference for an actor-critic with a shared trunk, a softmax policy head, a linear value head and an entropy bonus with $\beta=0.01$.

### Cited Findings

**PPO (Schulman, Wolski, Dhariwal, Radford, Klimov, 2017; arXiv:1707.06347, preprint only, no peer-reviewed venue in the arXiv record)** — [arXiv](https://arxiv.org/abs/1707.06347)
- Vanilla policy-gradient estimator and loss (eqs. 1–2): $\hat g=\hat{\mathbb E}_t\!\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\hat A_t\right]$, $L^{PG}(\theta)=\hat{\mathbb E}_t\!\left[\log\pi_\theta(a_t|s_t)\hat A_t\right]$. The paper states that running multiple optimisation steps on $L^{PG}$ with the same trajectory "is not well-justified, and empirically it often leads to destructively large policy updates" — [arXiv](https://arxiv.org/abs/1707.06347)
- Probability ratio and CPI surrogate (eq. 6): $r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$, $L^{CPI}(\theta)=\hat{\mathbb E}_t\!\left[r_t(\theta)\hat A_t\right]$ — [arXiv](https://arxiv.org/abs/1707.06347)
- Clipped surrogate (eq. 7): $L^{CLIP}(\theta)=\hat{\mathbb E}_t\!\left[\min\!\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\right]$, "where epsilon is a hyperparameter, say, $\epsilon=0.2$". It is "a lower bound (i.e., a pessimistic bound) on the unclipped objective". It equals $L^{CPI}$ to first order around $\theta_{old}$ — [arXiv](https://arxiv.org/abs/1707.06347)
- Adaptive-KL alternative (eq. 8): $L^{KLPEN}(\theta)=\hat{\mathbb E}_t\!\left[r_t(\theta)\hat A_t-\beta\,\mathrm{KL}[\pi_{\theta_{old}}(\cdot|s_t),\pi_\theta(\cdot|s_t)]\right]$. With $d=\hat{\mathbb E}_t[\mathrm{KL}]$: if $d<d_{targ}/1.5$ then $\beta\leftarrow\beta/2$; if $d>d_{targ}\times1.5$ then $\beta\leftarrow\beta\times2$. The authors "found that the KL penalty performed worse than the clipped surrogate objective" — [arXiv](https://arxiv.org/abs/1707.06347)
- Combined objective for shared policy/value parameters (eq. 9): $L_t^{CLIP+VF+S}(\theta)=\hat{\mathbb E}_t\!\left[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta](s_t)\right]$, with $L_t^{VF}=(V_\theta(s_t)-V_t^{targ})^2$ and $S$ the entropy bonus — [arXiv](https://arxiv.org/abs/1707.06347)
- Truncated advantage estimators (eqs. 10–12): the A3C-style estimator is $\hat A_t=-V(s_t)+r_t+\gamma r_{t+1}+\dots+\gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)$. Truncated GAE is $\hat A_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\dots+(\gamma\lambda)^{T-t+1}\delta_{T-1}$ with $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. Both are printed exactly like this in the paper. **Pitfall:** the printed exponent $T-t+1$ is inconsistent with the last index $T-1$; the consistent exponent is $T-t-1$ (my reading; see Inferences) — [arXiv](https://arxiv.org/abs/1707.06347)
- Surrogate ablation, Table 1 (7 MuJoCo tasks × 3 seeds, normalised score): no clipping or penalty −0.39; clip $\epsilon=0.1$: 0.76; **$\epsilon=0.2$: 0.82 (best)**; $\epsilon=0.3$: 0.70; adaptive KL $d_{targ}=0.003/0.01/0.03$: 0.68/0.74/0.71; fixed KL $\beta=0.3/1/3/10$: 0.62/0.71/0.72/0.69 — [arXiv](https://arxiv.org/abs/1707.06347)
- MuJoCo hyper-parameters (Table 3): horizon $T=2048$, Adam stepsize $3\times10^{-4}$, 10 epochs, minibatch 64, $\gamma=0.99$, $\lambda=0.95$. In these experiments "We don't share parameters between the policy and value function (so coefficient $c_1$ is irrelevant), and we don't use an entropy bonus." The network was an MLP with 2×64 tanh units — [arXiv](https://arxiv.org/abs/1707.06347)
- Atari hyper-parameters (Table 5): $T=128$, Adam stepsize $2.5\times10^{-4}\times\alpha$, 3 epochs, minibatch $32\times8$, $\gamma=0.99$, $\lambda=0.95$, 8 actors, clip $0.1\times\alpha$, VF coeff $c_1=1$, entropy coeff $c_2=0.01$, with "$\alpha$ is linearly annealed from 1 to 0 over the course of learning". The Atari runs use the same shared-trunk architecture as Mnih et al. 2016 — [arXiv](https://arxiv.org/abs/1707.06347)
- Diagnostic: in Figure 2, $L^{CLIP}$ peaks where the updated policy has a KL of about 0.02 from the old one (Hopper, first update) — [arXiv](https://arxiv.org/abs/1707.06347)

**GAE (Schulman, Moritz, Levine, Jordan, Abbeel; ICLR 2016; arXiv:1506.02438)** — the PDF header reads "Published as a conference paper at ICLR 2016" — [arXiv](https://arxiv.org/abs/1506.02438)
- TD residual: $\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$. This is $\gamma$-just (unbiased for $A^{\pi,\gamma}$) only when $V=V^{\pi,\gamma}$; "otherwise it will yield biased policy gradient estimates" — [arXiv](https://arxiv.org/abs/1506.02438)
- Estimator (eq. 16): $\hat A_t^{GAE(\gamma,\lambda)}:=(1-\lambda)\big(\hat A_t^{(1)}+\lambda\hat A_t^{(2)}+\lambda^2\hat A_t^{(3)}+\dots\big)=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V$ — [arXiv](https://arxiv.org/abs/1506.02438)
- Special cases (eqs. 17–18): $GAE(\gamma,0):\ \hat A_t=\delta_t$, and $GAE(\gamma,1):\ \hat A_t=\sum_{l\ge0}\gamma^l r_{t+l}-V(s_t)$. "GAE($\gamma$,1) is $\gamma$-just regardless of the accuracy of $V$, but it has high variance… GAE($\gamma$,0) is $\gamma$-just for $V=V^{\pi,\gamma}$ and otherwise induces bias, but it typically has much lower variance." — [arXiv](https://arxiv.org/abs/1506.02438)
- Roles of $\gamma$ and $\lambda$: "Taking $\gamma<1$ introduces bias into the policy gradient estimate, regardless of the value function's accuracy. On the other hand, $\lambda<1$ introduces bias only when the value function is inaccurate. Empirically, we find that the best value of $\lambda$ is much lower than the best value of $\gamma$" — [arXiv](https://arxiv.org/abs/1506.02438)
- Empirical ranges: cart-pole "best results are obtained at intermediate values of the parameters: $\gamma\in[0.96,0.99]$ and $\lambda\in[0.92,0.99]$", with the fastest improvement at $\lambda\in[0.92,0.98]$ (for $\gamma=0.99$). For 3D biped locomotion the best range was $\gamma\in[0.99,0.995]$, $\lambda\in[0.96,0.99]$ — [arXiv](https://arxiv.org/abs/1506.02438)
- GAE fits the value function with a trust-region method, to "avoid overfitting to the most recent batch of data" — [arXiv](https://arxiv.org/abs/1506.02438)

**REINFORCE (Williams 1992, *Machine Learning* 8(3):229–256, DOI 10.1007/BF00992696)** — [Springer](https://link.springer.com/article/10.1007/BF00992696)
- Weight update: $\Delta w_{ij}=\alpha_{ij}(r-b_{ij})e_{ij}$. Here $e_{ij}=\partial\ln g_i/\partial w_{ij}$ is the "characteristic eligibility", $b_{ij}$ is a "reinforcement baseline" that is conditionally independent of the unit output given $W$ and the input, and $\alpha_{ij}\ge0$. The name stands for "REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility" — [Springer PDF (search-indexed text)](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf); [transcription](https://github.com/bay3s/reinforce-rl/blob/main/original-paper.md)
- Theorem 1: "For any REINFORCE algorithm the average update vector in weight space lies in a direction for which this performance measure is increasing." Episodic, delayed-reward form: $\Delta w_{ij}=\alpha_{ij}(r-b_{ij})\sum_{t=1}^{k}e_{ij}(t)$ — [transcription](https://github.com/bay3s/reinforce-rl/blob/main/original-paper.md) (secondary; see Gaps)
- The Springer abstract confirms that the algorithms "make weight adjustments in a direction that lies along the gradient of expected reinforcement" — [Springer](https://link.springer.com/article/10.1007/BF00992696)

**A3C (Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver, Kavukcuoglu; ICML 2016, PMLR 48:1928–1937)** — [PMLR](https://proceedings.mlr.press/v48/mniha16.html)
- Actor-critic update: $\nabla_{\theta'}\log\pi(a_t|s_t;\theta')A(s_t,a_t;\theta,\theta_v)$ with the $k$-step advantage $A=\sum_{i=0}^{k-1}\gamma^i r_{t+i}+\gamma^kV(s_{t+k};\theta_v)-V(s_t;\theta_v)$, where $k\le t_{max}$ — [arXiv](https://arxiv.org/abs/1602.01783)
- Shared trunk: "we always share some of the parameters in practice. We typically use a convolutional neural network that has one softmax output for the policy $\pi(a_t|s_t;\theta)$ and one linear output for the value function $V(s_t;\theta_v)$, with all non-output layers shared." — [arXiv](https://arxiv.org/abs/1602.01783)
- Entropy regularisation: the gradient is $\nabla_{\theta'}\log\pi(a_t|s_t;\theta')(R_t-V(s_t;\theta_v))+\beta\nabla_{\theta'}H(\pi(s_t;\theta'))$. Entropy "improved exploration by discouraging premature convergence to suboptimal deterministic policies". The paper used "entropy regularization with a weight $\beta=0.01$ for all Atari and TORCS experiments" — [arXiv](https://arxiv.org/abs/1602.01783)
- "A2C" (the synchronous variant) is the name the PPO paper uses for its tuned baseline ("well-tuned implementations of A2C [Mni+16]"), citing Mnih et al. 2016 — [arXiv PPO](https://arxiv.org/abs/1707.06347)

### Inferences
- **Eq. (10)/(11) exponent typo.** The PPO paper prints $\gamma^{T-t+1}r_{T-1}$ and $(\gamma\lambda)^{T-t+1}\delta_{T-1}$. Since the sum starts at exponent 0 for index $t$, the term for index $T-1$ must carry exponent $T-1-t$. The report should write $\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$ and note the correction. Implement it as the standard backward recursion $\hat A_t=\delta_t+\gamma\lambda(1-d_{t+1})\hat A_{t+1}$, where $d$ is the done flag.
- **Short episodes (2–5 hops), terminal reward.** With so few steps, $\gamma=1$ (or 0.99) costs almost nothing in variance and removes the $\gamma$-bias GAE warns about. $\lambda$ can be high (0.95–1.0), because the bias from an imperfect critic matters more than the variance of a 5-step return. Ahmadian et al. (Q6) found that $\lambda=1$ worked best in a sparse, sequence-level-reward setting. This is a configuration recommendation, not a measured result for this task.
- **Value target.** Set $V_t^{targ}=\hat A_t+V(s_t)$ (the TD($\lambda$) return). At a true terminal (STOP chosen, or the hop cap if the cap is part of the task), bootstrap with $V=0$.
- **Write-up.** Present the policy loss as $-L^{CLIP}$, the critic loss as $c_1(V_\theta-V^{targ})^2$, and the entropy as $-c_2S$, matching eq. (9). State explicitly whether the trunk is shared, because $c_1$ only matters when it is.

### Gaps
- I could not machine-read Williams (1992) directly: the UMass mirror PDF has a broken font encoding and Springer is paywalled. The update rule and acronym are confirmed by the search-indexed Springer PDF text. The definition $e_{ij}=\partial\ln g_i/\partial w_{ij}$, the Theorem 1 wording and the episodic form come from a third-party transcription. They match the standard reading, but are not checked against the page image.
- The PPO paper gives no recommended $c_1$ for a shared trunk outside the Atari table ($c_1=1$). The commonly used $c_1=0.5$ (an OpenAI-baselines default) was not checked in any primary source here.

---

## Q2. Implementation studies: which PPO details matter most?

### Takeaway
Three studies agree on these points:
- Code-level details can matter as much as the PPO clipping itself.
- PPO-style value-loss clipping does not help; one study found it hurts.
- Per-minibatch advantage normalisation has little effect, at least in continuous control.
- Learning-rate annealing and gradient clipping help a little.
- Initialisation and normalisation (of rewards, observations and value targets) matter substantially.
- In the large study, separate policy and value networks beat a shared trunk.

### Cited Findings

**Engstrom et al. (ICLR 2020).** The paper's header reads "Published as a conference paper at ICLR 2020". The proceedings/OpenReview title is "Implementation Matters in Deep RL: A Case Study on PPO and TRPO", while the arXiv title is "…in Deep Policy Gradients…" — [arXiv](https://arxiv.org/abs/2005.12729); [ML Anthology](https://mlanthology.org/iclr/2020/engstrom2020iclr-implementation/)
- The nine "code-level optimizations" in the reference PPO code:
  1. value-function clipping
  2. reward scaling ("rewards are divided through by the standard deviation of a rolling discounted sum of the rewards")
  3. orthogonal initialisation and layer scaling
  4. Adam learning-rate annealing
  5. reward clipping (to $[-5,5]$ or $[-10,10]$)
  6. observation normalisation
  7. observation clipping (to $[-10,10]$)
  8. tanh activations
  9. global gradient clipping (global $\ell_2$ norm $\le0.5$)

  — [arXiv](https://arxiv.org/abs/2005.12729)
- Clipped value loss, as written: $L^V=\max\!\big[(V_{\theta_t}-V_{targ})^2,\ (\mathrm{clip}(V_{\theta_t},V_{\theta_{t-1}}-\varepsilon,V_{\theta_{t-1}}+\varepsilon)-V_{targ})^2\big]$, with $\varepsilon$ tied to the ratio-clip value — [arXiv](https://arxiv.org/abs/2005.12729)
- Ablation of the first four optimisations (Humanoid and Walker2d, 5 seeds, LR grid): "reward normalization, Adam annealing, and network initialization each significantly impact the rewards landscape… and were necessary for attaining the highest PPO reward" — [arXiv](https://arxiv.org/abs/2005.12729)
- "PPO code-optimizations are more important in terms of final reward achieved than the choice of general training algorithm (TRPO vs. PPO)". PPO without clipping, but with the code-level optimisations, "often matches the performance of standard PPO" — [arXiv](https://arxiv.org/abs/2005.12729)
- Trust region: "all three algorithms fail to maintain a ratio-based trust region, despite PPO and PPO-M being trained directly with a ratio-clipping objective". All three do keep a KL-based trust region at their optimal parameters — [arXiv](https://arxiv.org/abs/2005.12729)

**Andrychowicz et al. (ICLR 2021).** The proceedings title is "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study"; the arXiv title is "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study". The study covers more than 50 design choices — [OpenReview](https://openreview.net/forum?id=nIAxjsniDzg); [arXiv](https://arxiv.org/abs/2006.05990). Its recommendations, verbatim from the arXiv v1 PDF:
- "Use the PPO policy loss. Start with the clipping threshold set to 0.25 but also try lower and higher values if possible." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Initialize the last policy layer with 100× smaller weights… Use a wide value MLP (no layers shared with the policy) but tune the policy width". "Separate value and policy networks (C47) appear to lead to better performance on four out of five environments." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Always use observation normalization and check if value function normalization improves performance. Gradient clipping might slightly help but is of secondary importance." Also: "per-minibatch advantage normalization (C67) seems not to affect the performance too much." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Use GAE with $\lambda=0.9$ but neither Huber loss nor PPO-style value loss clipping." Also: "PPO-style value loss clipping (C13) hurts the performance regardless of the clipping threshold." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Go over experience multiple times. Shuffle individual transitions before assigning them to minibatches and recompute advantages once per data pass." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Discount factor $\gamma$ is one of the most important hyperparameters and should be tuned per environment (start with $\gamma=0.99$)." — [arXiv](https://arxiv.org/abs/2006.05990)
- "Use Adam optimizer with momentum $\beta_1=0.9$ and a tuned learning rate (0.0003 is a safe default). Linearly decaying the learning rate may slightly improve performance but is of secondary importance." — [arXiv](https://arxiv.org/abs/2006.05990)
- Regularisation (entropy, KL penalties): "We do not find evidence that any of the investigated regularizers helps significantly" except on HalfCheetah. The suggested reasons are that the PPO loss already enforces a trust region and that careful policy initialisation already ensures exploration — [arXiv](https://arxiv.org/abs/2006.05990)

**Huang, Dossa, Raffin, Kanervisto, Wang, "The 37 Implementation Details of Proximal Policy Optimization" (ICLR Blog Track, 25 Mar 2022).** This is citable: the page generates `@inproceedings{…, booktitle = {ICLR Blog Track}, year = {2022}, url = …}` — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- The 13 core details include:
  - orthogonal initialisation with gain $\sqrt2$ for hidden layers, 0.01 for the policy output and 1 for the value output, with constant zero biases
  - Adam $\epsilon=10^{-5}$ (not the PyTorch default $10^{-8}$)
  - learning rate annealed linearly to 0 (from $2.5\times10^{-4}$ on Atari, $3\times10^{-4}$ on MuJoCo)
  - GAE with value bootstrapping and returns $=$ advantages $+$ values
  - shuffled minibatches
  - advantage normalisation "at the minibatch level instead of the whole batch level"
  - clipped surrogate and clipped value loss
  - `loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient`, with policy and value sharing one optimiser
  - global gradient-norm clipping at 0.5
  - debug variables (approximate KL, clipfrac)
  - shared vs separate networks

  — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- On value clipping, the post says: "Engstrom, Ilyas, et al., (2020) find no evidence that the value function loss clipping helps with the performance. Andrychowicz, et al. (2021) suggest value function loss clipping even hurts performance" — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- On truncation: the reference PPO code does not bootstrap at time-limit truncations, although "we (intuitively) should" — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### Inferences
- **Starting configuration for this project** (synthesised from the sources above, not tuned for table QA):

  | Setting | Value |
  |---|---|
  | Clip $\epsilon$ | 0.2 (sweep {0.1, 0.2, 0.3}) |
  | Epochs per batch | 4–10 |
  | Minibatches | shuffled |
  | Optimiser | Adam, $\beta_1=0.9$, $\epsilon=10^{-5}$ |
  | Learning rate (new heads) | $3\times10^{-4}$, linearly annealed |
  | Learning rate (pretrained encoder) | much smaller, if fine-tuned at all (my judgement) |
  | Global gradient-norm clip | 0.5 |
  | $\gamma$ | 1.0 or 0.99 |
  | GAE $\lambda$ | 0.95 (sweep 0.9–1.0) |
  | Advantage normalisation | per minibatch, with an $\epsilon$ in the denominator |
  | Value-loss clipping | none |
  | Value targets / rewards | normalised if reward scales differ |
  | Entropy coefficient | $c_2=0.01$ (A3C/PPO-Atari) |
  | Policy output-layer init | small (×0.01) |
  | Diagnostics | log approximate KL and clipfrac |

- **Shared vs separate critic.** The evidence favours separate networks (Andrychowicz). If a shared encoder is used, as A3C does for efficiency, report $c_1$ and consider detaching or down-weighting the critic gradient into the encoder. After a behaviour-cloning warm start, the critic is untrained, so a short critic-only warm-up before PPO policy updates is sensible. This is my recommendation, not something the sources tested.
- **Advantage normalisation with sparse binary terminal reward.** A minibatch can contain all-equal returns, giving std ≈ 0. Add an $\epsilon$, or normalise at batch level. Andrychowicz found the choice matters little in their setting.
- The studies are about MuJoCo and Atari. Their transfer to a 2–5-step discrete pointer policy is plausible but untested; present them as "recommended defaults", not as evidence about this task.

### Gaps
- None of the three studies tests discrete, variable-size action sets with very short horizons, so no source can say which detail matters most in exactly this regime.
- I did not list all 37 details from the blog post, only the 13 core ones relevant here. The Atari- and continuous-control-specific details are irrelevant to this project.

---

## Q3. Invalid-action masking: is it a valid policy gradient, and how does it compare with penalties?

### Takeaway
Yes, masking gives a valid policy gradient. Huang & Ontañón show that masking is a state-dependent differentiable transformation of the logits, so the policy-gradient theorem still applies. Masked logits receive exactly zero gradient. Masking scales to large invalid-action spaces, whereas negative-reward penalties fail to scale. "Naive" masking (masking at sampling time but not in the update) makes the PPO KL explode.

### Cited Findings
- Venue: FLAIRS-35 (2022), *The International FLAIRS Conference Proceedings* vol. 35, DOI 10.32473/flairs.v35i.130584 — [FLAIRS OJS](https://journals.flvc.org/FLAIRS/article/view/130584); [arXiv](https://arxiv.org/abs/2006.14171)
- Masking function (Proposition 1): $\pi_\theta(\cdot|s)=\mathrm{softmax}(\mathrm{mask}(l(s)))$, where $\mathrm{mask}(l(s))_i=l_i$ if $a_i$ is valid in $s$ and $M$ otherwise. The worked example uses $M=-1\times10^{8}$. "If M is chosen to be sufficiently negative, the probability of choosing the masked invalid action… will be virtually zero." — [arXiv](https://arxiv.org/abs/2006.14171)
- Gradient: in their example, $\nabla\log\pi(a_0|s_0)=[0.67,-0.33,0.0000,-0.33]$. Masking "appears to do more than just 'renormalizing the probability distribution'; it in fact makes the gradient corresponding to the logits of the invalid action to zero." — [arXiv](https://arxiv.org/abs/2006.14171)
- Validity: mask "is either an identity function or a constant function for elements in the logits… $\pi_\theta$ is differentiable to its parameters… which satisfies the assumption of policy gradient theorem… We therefore call mask a state-dependent differentiable function." — [arXiv](https://arxiv.org/abs/2006.14171)
- Penalty comparison: they tried $r_{invalid}\in\{0,-0.01,-0.1,-1\}$. "Invalid action penalty is able to achieve good results in 4×4 maps, but it does not scale to larger maps." On 10×10 with $r_{invalid}=-0.01$, agents "spent 3.43% of the entire training time just discovering the first reward, while agents trained with invalid action masking take roughly 0.06%." Setting $r_{invalid}=-1$ "seems to have an adverse effect of discouraging exploration… achieving consistently the worst performance across maps." — [arXiv](https://arxiv.org/abs/2006.14171)
- Naive masking (sample from the masked distribution, but compute the gradient and ratio with unmasked probabilities): the KL between target and current policy "is significantly higher than that of any other experiments", and time-to-solve is volatile (49.14% of training on 24×24) — [arXiv](https://arxiv.org/abs/2006.14171)
- Masking removed at evaluation: the agent "is still able to perform well to a certain degree", and still beats the penalty-trained agents — [arXiv](https://arxiv.org/abs/2006.14171)
- Their PPO settings: $\gamma=0.99$, $\lambda=0.97$, clip 0.2, entropy 0.01, gradient-norm threshold 0.5, 10 update epochs, LR $3\times10^{-4}$ for both policy and value — [arXiv](https://arxiv.org/abs/2006.14171)
- The 37-details post describes the same technique: replace invalid logits with $-\infty$ before the softmax. It is used in AlphaStar and OpenAI Five — [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### Inferences
- **Implementation rules for this project:**
  - Store the candidate mask (valid and not-yet-visited, plus STOP) in the rollout buffer.
  - Re-apply the same mask when recomputing $\log\pi_\theta$ for the PPO ratio and the entropy. Otherwise you get the "naive masking" failure mode.
  - Use a large finite negative value (e.g. $-10^8$ or the dtype minimum) rather than $-\infty$, so that the entropy term $\sum p\log p$ does not produce NaN.
  - Compute the entropy over valid candidates only.
- **Invalid-action penalty.** With masking in place, an invalid-action penalty is redundant for sampled actions, because invalid actions are never executed. If the report keeps the penalty term "for completeness", say that it is inactive under masking, or that it applies only to a no-mask ablation. Huang & Ontañón suggest that a no-mask, penalty-only ablation will likely explore worse.
- **Entropy normalisation.** Because $|\mathcal A_t|$ varies, the maximum entropy $\log|\mathcal A_t|$ varies by state. A fixed $c_2$ therefore pushes harder in states with many candidates. Normalising the entropy by $\log|\mathcal A_t|$ is a reasonable design choice, but I found no source that tests it.

### Gaps
- The paper's experiments are on µRTS with fixed-size, multi-discrete action heads, not on pointer-scored variable candidate sets. The theory (Proposition 1) transfers directly; the empirical scaling numbers do not transfer quantitatively.

---

## Q4. Variable or large discrete action sets: which paper to cite for scoring candidates by embeddings?

### Takeaway
For the per-hop pointer head over a variable candidate set, cite Pointer Networks (Vinyals et al. 2015) for the attention-as-pointer scoring function. Cite Chandak et al. 2020, "Reinforcement Learning When All Actions Are Not Always Available" (AAAI 2020), for the RL formulation with a state-dependent available-action set, a masked-softmax policy and policy-gradient convergence. Jain et al. 2020 (ICML) is a good optional citation if the report stresses generalising to unseen candidates through action embeddings. Do not cite Dulac-Arnold et al. 2015 or Chandak et al.'s "Lifelong Learning with a Changing Action Set" as the main justification; they address different problems.

### Cited Findings
- **Pointer Networks (Vinyals, Fortunato, Jaitly; NeurIPS 2015, vol. 28, pp. 2692–2700).**
  - The paper targets problems where "the number of target classes in each step of the output depends on the length of the input, which is variable" — [NeurIPS](https://papers.nips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html); [ML Anthology](https://mlanthology.org/neurips/2015/vinyals2015neurips-pointer/)
  - Pointer distribution: $u_j^i=v^\top\tanh(W_1e_j+W_2d_i)$ for $j\in(1,\dots,n)$, and $p(C_i|C_1,\dots,C_{i-1},\mathcal P)=\mathrm{softmax}(u^i)$, "where softmax normalizes the vector $u^i$ (of length $n$) to be an output distribution over the dictionary of inputs". It "uses attention as a pointer to select a member of the input sequence as the output" — [arXiv](https://arxiv.org/abs/1506.03134)
- **Chandak, Theocharous, Metevier, Thomas, "Reinforcement Learning When All Actions Are Not Always Available" (AAAI 2020, vol. 34(04), pp. 3381–3388, DOI 10.1609/aaai.v34i04.5740).**
  - The work studies stochastic-action-set MDPs (SAS-MDPs, from Boutilier et al. 2018), where "the set of available decisions (actions) at each time step is stochastic", and presents "new policy gradient algorithms for SAS-MDPs that incorporate variance reduction techniques unique to this setting, and provide conditions for their convergence" — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/5740); [arXiv](https://arxiv.org/abs/1906.01772)
  - Policy parameterisation: "To make the policy handle stochastic action sets, we make use of a mask which indicates the available actions… the masked softmax, i.e., $\pi(s,\alpha,a):=\dfrac{\exp(y_a)\cdot\mathbb 1_{\{a\in\alpha\}}}{\sum_{a'}\exp(y_{a'})\cdot\mathbb 1_{\{a'\in\alpha\}}}$", where $y_a$ is the score of action $a$ and $\alpha$ is the available set — [arXiv](https://arxiv.org/abs/1906.01772)
  - The paper argues that SAS Q-learning can diverge with function approximation, motivating policy-gradient methods. It proposes several baselines (beyond the state-dependent $b(s)$) that are "easy to compute" and combined optimally to reduce variance. It also notes "the policy can be parameterized such that it is not required to embed the available actions as a part of the state" — [arXiv](https://arxiv.org/abs/1906.01772)
- **Jain, Szot, Lim, "Generalization to New Actions in Reinforcement Learning" (ICML 2020, PMLR 119:4661–4672).**
  - The policy "computes each given action's utility using its representation and outputs a distribution". The policy receives "a set of available actions $\mathcal A=\{a_1,\dots,a_k\}$", and utility scores are converted by softmax: $\pi(a_i|s,\mathcal A)=\dfrac{e^{f_\omega(c_i,f_\nu(s))}}{\sum_{j=1}^{k}e^{f_\omega(c_j,f_\nu(s))}}$ (symbol subscripts approximate; see Gaps). The utility "estimates the score of an action at the current state, through its action representation, just like a Q-function" — [PMLR](https://proceedings.mlr.press/v119/jain20b.html); [arXiv](https://arxiv.org/abs/2011.01928)
  - To avoid over-exploiting a subset of training actions, they sample action subsets during training and add maximum-entropy regularisation — [arXiv](https://arxiv.org/abs/2011.01928)
- **Dulac-Arnold et al. 2015, "Deep Reinforcement Learning in Large Discrete Action Spaces" (arXiv:1512.07679; the arXiv record lists no venue).** Its Wolpertinger architecture produces a continuous "proto-action" in embedding space, takes $k$ nearest neighbours, and refines them with a critic. It targets "sub-linear complexity relative to the size of the set" for very large, fixed action sets — [arXiv](https://arxiv.org/abs/1512.07679)
- **Chandak, Theocharous, Nota, Thomas, "Lifelong Learning with a Changing Action Set" (AAAI 2020 per the arXiv comment; arXiv:1906.01770).** It addresses settings where "the set of possible actions (decisions) varies over time", across a lifetime, not per state — [arXiv](https://arxiv.org/abs/1906.01770)

### Inferences
- The project's head (pointer scores over candidate entity embeddings, masked softmax including STOP) is exactly the Ptr-Net scoring function used as a policy, inside a SAS-MDP-style setting. Write it in the report as $\pi_\theta(a|s_t,\mathcal A_t)=\dfrac{\exp(u_\theta(s_t,e_a))\mathbb 1[a\in\mathcal A_t]}{\sum_{a'\in\mathcal A_t}\exp(u_\theta(s_t,e_{a'}))}$ with $u_\theta(s,e)=v^\top\tanh(W_1e+W_2h_s)$, or a bilinear or dot-product variant. Cite Vinyals et al. 2015 for $u$, and Chandak et al. 2020 (SAS) together with Huang & Ontañón for the mask.
- Dulac-Arnold's kNN proto-action method suits huge fixed action spaces with a deterministic actor and critic. It does not match a few dozen per-state candidates scored exhaustively. The "Lifelong" paper addresses action-set growth over time. Both are weaker fits than the SAS paper.

### Gaps
- Greek-letter subscripts in Jain et al.'s softmax equation were lost in text extraction. The structure (a utility of action representation $c_i$ and state encoding, softmaxed over the given set) is verified; the exact symbols $\omega,\nu$ are my reconstruction.
- I did not verify page numbers for the "Lifelong Learning with a Changing Action Set" AAAI version. Since it is not recommended, I did not pursue this further.

---

## Q5. Reward shaping (Ng, Harada & Russell 1999): exact form, invariance theorem, and whether evidence-recall shaping is potential-based

### Takeaway
Policy invariance holds if and only if the shaping reward has the form $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ for a state-only potential $\Phi$. For undiscounted episodic tasks, the potential of the terminal (absorbing) state must be a fixed constant, taken as 0 without loss of generality. Evidence-recall shaping is potential-based only if implemented as a per-hop difference $\gamma\Phi(s_{t+1})-\Phi(s_t)$, with $\Phi(s)$ the recall of the evidence visited so far and $\Phi(\text{terminal})$ fixed. A final "recall bonus", or per-hop Δrecall without the terminal correction, changes the optimal policy. The per-hop cost and the invalid-action penalty are not potential-based either.

### Cited Findings
- Venue: Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999), Morgan Kaufmann, pp. 278–287 — [UT Austin bib entry](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html); [DBLP record](https://dblp.org/rec/conf/icml/NgHR99.html); [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Setting: the shaped MDP is $M'=(S,A,T,\gamma,R+F)$. "Theorem 1: Let any $S$, $A$, $\gamma$, and any shaping reward function $F:S\times A\times S\mapsto\mathbb R$ be given. We say $F$ is a potential-based shaping function if there exists a real-valued function $\Phi:S\mapsto\mathbb R$ such that for all $s\in S-\{s_0\}, a\in A, s'\in S$: $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$." It is "a necessary and sufficient condition for it to guarantee consistency with the optimal policy" — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Sufficiency: every optimal policy in $M'$ is optimal in $M$, and vice versa. Necessity: if $F$ is not potential-based, then without knowledge of $T$ and $R$ there exist transition and reward functions for which no optimal policy in $M'$ is optimal in $M$ — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Undiscounted case: "For undiscounted ($\gamma=1$) MDPs, we assume… a distinguished absorbing state $s_0$… with no further rewards thereafter", and all policies are assumed proper. The proof shifts $\Phi$ so that "$\Phi(s_0)=0$" without loss of generality — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Corollary 2 and remarks: $Q^*_{M'}(s,a)=Q^*_M(s,a)-\Phi(s)$ and $V^*_{M'}(s)=V^*_M(s)-\Phi(s)$. Near-optimal policies are also preserved (Remark 1). With $\Phi=V^*_M$, the shaped value function becomes trivially easy to learn. Potential-based $F$ "also work when trying to learn a good policy from within a restricted class of policies" (e.g. neural-network policies) — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Known shaping "bugs", where the agent learns to cycle and collect shaping reward, "are shown to arise from non-potential-based rewards" — [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

### Inferences
- **Potential-based recall shaping** (inference derived from Theorem 1 and the absorbing-state condition):
  - Let $\Phi(s_t)=\mathrm{recall}(\text{visited}_t,\ \text{gold evidence})$. The gold evidence is fixed per question, so $\Phi$ is a function of the (question-augmented) state.
  - Per-hop shaping $F_t=\gamma\Phi(s_{t+1})-\Phi(s_t)$ is potential-based.
  - With $\gamma=1$, the shaping telescopes to $\Phi(s_0^{abs})-\Phi(s_{init})$, which is a constant when the transition into the absorbing state also pays $-\Phi(s_T)$ (i.e. $\Phi(\text{absorbing})=0$).
  - So the optimal policy is unchanged. Shaping only redistributes credit to earlier hops, which is what helps PPO/GAE with sparse terminal rewards.
- **Non-invariant variants.** Two common implementations change the objective:
  - paying per-hop Δrecall without the terminal $-\Phi(s_T)$ correction, which nets to $+\mathrm{recall}_T-\mathrm{recall}_0$
  - adding a terminal "evidence recall" bonus

  Both reward reaching high recall in itself. This may be desirable, for faithfulness, but the report should call it a reward-design change rather than invariance-preserving shaping. It can also make the policy trade answer accuracy for recall.
- **Per-hop cost and invalid-action penalty.** A constant $-c$ per hop is not of the form $\gamma\Phi(s')-\Phi(s)$ for a bounded state potential on episodes of varying length. It deliberately changes the optimum, toward shorter chains. The invalid-action penalty is also not potential-based, and is irrelevant under masking (Q3).
- **With $\gamma<1$,** use exactly $\gamma\Phi(s')$, not $\Phi(s')$. Otherwise you get a small non-potential term $(1-\gamma)\Phi(s')$.
- **Effect on actor-critic training.** For exact value functions, Corollary 2 implies $A^*_{M'}(s,a)=A^*_M(s,a)$. Potential-based shaping therefore leaves the true advantage unchanged and changes only what the learned critic must fit ($V'=V-\Phi$) and the variance and bias of GAE estimates. This is my extension to $\pi$-values, which the paper notes in Remark 1 holds for arbitrary policies.
- **Fair comparison across methods.** GRPO and RLOO use the episode total. If the shaping is potential-based with $\Phi(\text{absorbing})=0$, the shaping's contribution to every episode total is the same constant, $-\Phi(s_{init})$. So group-relative methods see no shaping signal at all. Only PPO, which uses per-step rewards through GAE, benefits from the densification. This matters when interpreting "identical rewards" comparisons.

### Gaps
- The Ng et al. PDF text was partly garbled in extraction (two-column layout). The theorem statement, the necessity and sufficiency claims and the absorbing-state assumption were readable. The exact wording of Corollary 2 is reconstructed from the readable fragments and matches the standard statement.
- I did not fetch a source that treats terminal-state potentials in episodic shaping beyond Ng et al. Grzes 2017 (AAMAS) is the usual reference but was not verified here, so it is not cited.

---

## Q6. Critic-free alternatives: GRPO, RLOO, Dr. GRPO (exact advantages and known biases)

### Takeaway
- **GRPO** drops the critic. It normalises each sampled output's reward within a group, $(r_i-\mathrm{mean})/\mathrm{std}$, and plugs this into a PPO-style clipped token-averaged objective with a KL penalty.
- **RLOO** uses a leave-one-out mean baseline, which is unbiased.
- **Dr. GRPO** shows that GRPO's $1/|o_i|$ length normalisation and its std normalisation introduce a response-length bias and a question-difficulty bias, and removes both.

### Cited Findings
**DeepSeekMath / GRPO (Shao et al. 2024; arXiv:2402.03300, preprint)** — [arXiv](https://arxiv.org/abs/2402.03300); [HTML](https://arxiv.org/html/2402.03300v3)
- Motivation: "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources". In PPO, "the value function… is typically another model of comparable size as the policy model" — [arXiv](https://arxiv.org/abs/2402.03300)
- Objective (eq. 3): $\mathcal J_{GRPO}(\theta)=\mathbb E_{q\sim P(Q),\{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(O|q)}\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big[\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big]-\beta\,\mathbb D_{KL}[\pi_\theta\|\pi_{ref}]\Big\}$, where $\rho_{i,t}=\dfrac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$ — [HTML](https://arxiv.org/html/2402.03300v3)
- KL estimator (eq. 4): $\mathbb D_{KL}[\pi_\theta\|\pi_{ref}]=\dfrac{\pi_{ref}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)}-\log\dfrac{\pi_{ref}(o_{i,t}|\cdot)}{\pi_\theta(o_{i,t}|\cdot)}-1$ — [HTML](https://arxiv.org/html/2402.03300v3)
- Outcome-supervision advantage: $\hat A_{i,t}=\tilde r_i=\dfrac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)}$ for all tokens $t$ of output $i$. With process supervision, $\hat A_{i,t}=\sum_{\mathrm{index}(j)\ge t}\tilde r_i^{\mathrm{index}(j)}$, using normalised step rewards — [HTML](https://arxiv.org/html/2402.03300v3)
- Settings: "learning rate of the policy model as 1e-6. The KL coefficient is 0.04. For each question, we sample 64 outputs. The max length is set to 1024, and the training batch size is 1024. The policy model only has a single update following each exploration stage." — [arXiv](https://arxiv.org/abs/2402.03300)

**RLOO / "Back to Basics" (Ahmadian, Cremer, Gallé, Fadaee, Kreutzer, Pietquin, Üstün, Hooker; ACL 2024 Long, pp. 12248–12267)** — [ACL Anthology](https://aclanthology.org/2024.acl-long.662/); [arXiv](https://arxiv.org/abs/2402.14740)
- REINFORCE with baseline (eq. 7): $\mathbb E_{x\sim\mathcal D,y\sim\pi_\theta(\cdot|x)}\big[(R(y,x)-b)\nabla_\theta\log\pi_\theta(y|x)\big]$. Moving-average baseline: $b_{MA}=\frac1S\sum_sR(x^s,y^s)$ — [HTML](https://arxiv.org/html/2402.14740v2)
- RLOO estimator: $\frac1k\sum_{i=1}^k\Big[R(y_{(i)},x)-\frac1{k-1}\sum_{j\ne i}R(y_{(j)},x)\Big]\nabla\log\pi(y_{(i)}|x)$, with $y_{(1)},\dots,y_{(k)}\stackrel{iid}{\sim}\pi_\theta(\cdot|x)$. They tested $k\in\{2,4\}$ — [HTML](https://arxiv.org/html/2402.14740v2)
- Bandit view: because reward arrives only at the end, they model the full completion as a single action, so REINFORCE can "directly optimize the full trajectory (sequence) return coupled with unbiased baselines, whereas actor-critic algorithms… such as PPO, bootstrap off intermediary state value-functions to reduce variance at the cost of introducing bias" — [arXiv](https://arxiv.org/abs/2402.14740)
- GAE-$\lambda$ finding in RLHF: "$\lambda=1.0$ is the vanilla policy gradient which has the highest variance but no bias… performs the best". Clipping is rarely active (the loss "is actually clipped on average <5% of the time per batch") — [arXiv](https://arxiv.org/abs/2402.14740); [HTML](https://arxiv.org/html/2402.14740v2)
- Results: RLOO beats PPO in their simulated win-rates; vanilla REINFORCE also beats PPO; and "RLOO$_{k=2}$ either closely matches or outperforms RAFT$_{k=4}$" — [HTML](https://arxiv.org/html/2402.14740v2)

**Dr. GRPO (Liu, Chen, Li, Qi, Pang, Du, Lee, Lin; "Understanding R1-Zero-Like Training: A Critical Perspective"; COLM 2025)** — the PDF header reads "Published as a conference paper at COLM 2025"; the OpenReview venue is "COLM 2025" — [OpenReview](https://openreview.net/forum?id=5PAF7PAY2Y); [arXiv](https://arxiv.org/abs/2503.20783)
- Biases identified: the GRPO objective's $\frac1{|o_i|}$ factor, and the advantage $\hat A_{i,t}=\dfrac{R(q,o_i)-\mathrm{mean}(\mathbf R)}{\mathrm{std}(\mathbf R)}$ — [HTML](https://arxiv.org/html/2503.20783v2)
- Length bias: "For positive advantages… this bias results in greater gradient updates for shorter responses… Conversely, for negative advantages… longer responses are penalized less due to their larger $|o_i|$, causing the policy to prefer lengthier responses among incorrect ones." — [HTML](https://arxiv.org/html/2503.20783v2)
- Difficulty bias: "Questions with lower standard deviations… are given higher weights during policy updates." — [HTML](https://arxiv.org/html/2503.20783v2)
- Fix: $\mathcal J_{DrGRPO}=\frac1G\sum_{i=1}^G\sum_{t=1}^{|o_i|}\{\dots\}$ (no $1/|o_i|$), with $\tilde A_{i,t}=R(q,o_i)-\mathrm{mean}(\{R(q,o_1),\dots,R(q,o_G)\})$ (no std). The paper says GRPO has "an optimization bias… which artificially increases response length (especially for incorrect outputs)", and that Dr. GRPO "improves token efficiency while maintaining reasoning performance" — [arXiv](https://arxiv.org/abs/2503.20783)

### Inferences
- **Mapping to this project:**
  - output $o_i$ = one multi-hop episode
  - token $t$ = one hop
  - group = $G$ rollouts of the same question
  - $R_i=\sum_t r_t$ (the episode return, including shaping, so that rewards are "identical" across methods)

  Note from Q5: potential-based shaping contributes the same constant to every episode total, so it gives no signal to GRPO or RLOO.
- **Length bias at hop level.** Averaging the loss per episode over hops ($1/|o_i|$) reproduces Dr. GRPO's length bias. Failing episodes with more hops are penalised less per hop, which would encourage long wrong chains and interact with the per-hop cost. Use a sum over hops, or divide by a constant such as the maximum hop count.
- **Degenerate groups.** With binary correctness rewards and small $G$, groups where all rollouts are correct or all are wrong are common. Then $\mathrm{std}=0$ and the advantage is 0, so the group contributes no gradient, and the std normalisation needs an $\epsilon$. Report the fraction of degenerate groups. The Dr. GRPO-style mean-only advantage avoids the division but still yields 0 for such groups.
- **Equivalences to state in the report.** RLOO's leave-one-out baseline equals $\frac{k}{k-1}(R_i-\bar R)$. So mean-centred GRPO without std, i.e. Dr. GRPO, equals RLOO up to a constant scale, apart from GRPO's clipping and KL terms. This is algebra, not a claim from the sources.

### Gaps
- None of the three papers studies short multi-step discrete episodes with shaped per-step rewards; all three are about LLM token sequences. Their quantitative findings do not transfer directly.
- I did not verify the COLM 2025 proceedings author list separately from arXiv, because OpenReview's API blocked automated access. The OpenReview search result confirmed the title and venue only.

---

## Q7. Imitation warm start then RL: which citation?

### Takeaway
Cite AlphaGo (Silver et al. 2016, *Nature*) as the canonical "supervised policy, then initialise the RL policy to the SL weights, then improve by policy gradient on a terminal win/loss reward" pipeline. It is structurally identical to behaviour cloning followed by PPO with a terminal reward. Cite DAgger (Ross, Gordon & Bagnell, AISTATS 2011) for why pure behaviour cloning is insufficient: compounding error that is quadratic in the horizon.

### Cited Findings
- **AlphaGo (Silver, Huang, Maddison, Guez, Sifre, van den Driessche, Schrittwieser, Antonoglou, Panneershelvam, Lanctot, Dieleman, Grewe, Nham, Kalchbrenner, Sutskever, Lillicrap, Leach, Kavukcuoglu, Graepel, Hassabis; *Nature* 529(7587):484–489, 2016, DOI 10.1038/nature16961).**
  - Venue and DOI: [Nature](https://www.nature.com/articles/nature16961)
  - The SL policy network predicted expert moves with 57.0% test accuracy. "The RL policy network $p_\rho$ is identical in structure to the SL policy network, and its weights $\rho$ are initialized to the same values, $\rho=\sigma$." — [DeepMind PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  - Terminal reward: $r(s)=0$ for non-terminal $t<T$, and $z_t=\pm r(s_T)$ ($+1$ win, $-1$ loss). The update is $\Delta\rho\propto\dfrac{\partial\log p_\rho(a_t|s_t)}{\partial\rho}z_t$. "Randomizing from a pool of opponents… stabilizes training by preventing overfitting to the current policy." — [DeepMind PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
  - "When played head-to-head, the RL policy network won more than 80% of games against the SL policy network." — [DeepMind PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
- **DAgger (Ross, Gordon, Bagnell; AISTATS 2011, PMLR 15:627–635).**
  - Venue: [PMLR](https://proceedings.mlr.press/v15/ross11a.html)
  - "A classifier that makes a mistake with probability $\epsilon$ under the distribution of states/observations encountered by the expert can make as many as $T^2\epsilon$ mistakes in expectation over $T$-steps under the distribution of states the classifier itself induces (Ross and Bagnell, 2010)… as soon as the learner makes a mistake, it may encounter completely different observations than those under expert demonstration, leading to a compounding of errors." — [arXiv](https://arxiv.org/abs/1011.0686)
  - DAgger learns "a stationary deterministic policy guaranteed to perform well under its induced distribution of states (number of mistakes/costs that grows linearly in $T$ and classification cost $\epsilon$)" — [arXiv](https://arxiv.org/abs/1011.0686)

### Inferences
- The project pipeline maps one-to-one onto AlphaGo's: BC on gold hop sequences ($\sigma$), then PPO initialised at $\rho=\sigma$ with a terminal correctness reward $z$. Use AlphaGo as the precedent and DAgger as the motivation for why RL fine-tuning, not more BC, is needed.
- With 2–5 hops the $T^2$ factor is small. The covariate-shift argument still applies, because one wrong hop changes the candidate set.
- If gold evidence chains can serve as an online expert, DAgger is a cheap extra baseline. This is optional and beyond the stated scope.
- **Pitfalls after a BC start:**
  - The BC policy may have low entropy, so PPO explores little; watch the entropy curve and $c_2$.
  - The critic starts untrained (see Q2).
  - A KL penalty toward the BC policy, analogous to GRPO's $\pi_{ref}$ term, is an optional stabiliser. None of the sources tests it in this setting.

### Gaps
- The DeepMind-hosted PDF I parsed did not include the full Methods section, so I could not verify whether AlphaGo's policy-gradient Methods used a baseline $v(s)$ in the RL-policy update. The main-text update ($\propto\nabla\log p_\rho\cdot z_t$) is verified.
- Other BC-then-RL citations (e.g. in dialogue or semantic parsing) were out of scope and not checked.

---

## Q8. Statistical reporting for RL: seeds and confidence intervals

### Takeaway
Henderson et al. show that two 5-seed averages of the same algorithm and hyper-parameters can differ significantly. They recommend many seeds plus significance testing (t-test, KS, bootstrap CIs, power analysis), but give no fixed seed count. Agarwal et al. accept that 3–10 runs is the practical norm and recommend reporting interval estimates via stratified bootstrap CIs, the interquartile mean (IQM), performance profiles, and probability of improvement, using the `rliable` library.

### Cited Findings
- **Henderson, Islam, Bachman, Pineau, Precup, Meger, "Deep Reinforcement Learning That Matters" (AAAI 2018, vol. 32(1), pp. 3207–3214, DOI 10.1609/aaai.v32i1.11694).**
  - Venue: [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/11694); [ML Anthology](https://mlanthology.org/aaai/2018/henderson2018aaai-deep/)
  - Seed variance: TRPO on HalfCheetah with identical hyper-parameters, averaged over two different sets of 5 random seeds, gave significantly different curves ("$t=-9.0916$, $p=0.0016$") — [arXiv](https://arxiv.org/abs/1709.06560)
  - Recommendations: "many trials must be run with different random seeds when comparing performance", and "it is important to perform proper significance testing". Bootstrap CIs (10k bootstrap iterations, pivotal method), t-tests and KS tests are demonstrated. "There can be no specific number of trials specified as a recommendation, [but] power analysis methods can be used". Baselines should "match the original reported performance" after hyper-parameter search — [arXiv](https://arxiv.org/abs/1709.06560)
- **Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS 2021, vol. 34, pp. 29304–29320; Outstanding Paper Award per the arXiv comment).**
  - Venue: [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html); [arXiv](https://arxiv.org/abs/2108.13264)
  - "3-10 runs are prevalent in deep RL". As a practical solution "easily applicable with 3-10 runs per task", they recommend "interval estimates via stratified bootstrap confidence intervals, as opposed to point estimates", performance profiles (score distributions), and robust aggregates — [arXiv](https://arxiv.org/abs/2108.13264)
  - IQM definition: "IQM discards the bottom and top 25% of the runs and calculates the mean score of the remaining 50% runs". IQM "has much smaller CIs than median". Average probability of improvement "shows how likely it is for X to outperform Y on a randomly selected task" — [arXiv](https://arxiv.org/abs/2108.13264)
  - "We recommend using confidence intervals for measuring the uncertainty in results and showing effect sizes (e.g., performance improvements over baseline) that are compatible with the given data." — [arXiv](https://arxiv.org/abs/2108.13264)

### Inferences
- **Reporting protocol for this project** (PPO vs REINFORCE+baseline vs GRPO, identical rewards):
  - Use at least 5 seeds per method, preferably 10 if compute allows.
  - Report mean and IQM over seeds of test answer accuracy and evidence recall, with 95% bootstrap CIs.
  - Report probability of improvement for PPO vs each baseline.
  - Learning curves should show the mean with a CI band, not just the best seed.
  - Tune hyper-parameters with equal budget per method, as Henderson stresses for fair baselines.
- With one dataset there is no "task" stratification. Bootstrap over seeds; optionally, use a paired bootstrap over test questions to separate seed variance from question-sampling variance.

### Gaps
- Neither paper gives a minimum seed count for short-horizon discrete tasks. Henderson explicitly declines to give a number.

---

## Q9. Final tight reading list: verification status and BibTeX

### Takeaway
I recommend 16 core citations, all verified on primary or official index pages. Three are optional (blog track, DAgger, Jain) and two are explicitly not recommended. Title discrepancies exist between the arXiv and proceedings versions of Engstrom 2020 and Andrychowicz 2021. Use the proceedings titles.

### Cited Findings
Verification table (V = verified on the linked page):

| # | Paper | Venue / identifiers verified | Source |
|---|---|---|---|
| 1 | Williams 1992 | Machine Learning 8(3):229–256, DOI 10.1007/BF00992696 (V) | [Springer](https://link.springer.com/article/10.1007/BF00992696) |
| 2 | Schulman et al. 2016 (GAE) | ICLR 2016 (PDF header), arXiv:1506.02438 (V) | [arXiv](https://arxiv.org/abs/1506.02438) |
| 3 | Mnih et al. 2016 (A3C) | ICML 2016, PMLR 48:1928–1937 (V) | [PMLR](https://proceedings.mlr.press/v48/mniha16.html) |
| 4 | Schulman et al. 2017 (PPO) | arXiv:1707.06347, preprint (V) | [arXiv](https://arxiv.org/abs/1707.06347) |
| 5 | Engstrom et al. 2020 | ICLR 2020 (PDF header + ML Anthology); title differs from arXiv (flag) | [ML Anthology](https://mlanthology.org/iclr/2020/engstrom2020iclr-implementation/) |
| 6 | Andrychowicz et al. 2021 | ICLR 2021, OpenReview nIAxjsniDzg; title differs from arXiv (flag) | [OpenReview](https://openreview.net/forum?id=nIAxjsniDzg) |
| 7 | Huang & Ontañón 2022 | FLAIRS vol. 35, DOI 10.32473/flairs.v35i.130584 (V) | [FLAIRS](https://journals.flvc.org/FLAIRS/article/view/130584) |
| 8 | Vinyals et al. 2015 (Ptr-Net) | NeurIPS 28, pp. 2692–2700 (V) | [NeurIPS](https://papers.nips.cc/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html) |
| 9 | Chandak et al. 2020 (SAS-MDP) | AAAI 34(04):3381–3388, DOI 10.1609/aaai.v34i04.5740 (V) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5740) |
| 10 | Ng, Harada, Russell 1999 | ICML 1999, pp. 278–287, Morgan Kaufmann (V via bib + DBLP) | [UT bib](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html) |
| 11 | Shao et al. 2024 (GRPO) | arXiv:2402.03300, preprint (V) | [arXiv](https://arxiv.org/abs/2402.03300) |
| 12 | Ahmadian et al. 2024 (RLOO) | ACL 2024 Long, pp. 12248–12267, DOI 10.18653/v1/2024.acl-long.662 (V) | [ACL](https://aclanthology.org/2024.acl-long.662/) |
| 13 | Liu et al. 2025 (Dr. GRPO) | COLM 2025 (PDF header + OpenReview) (V) | [OpenReview](https://openreview.net/forum?id=5PAF7PAY2Y) |
| 14 | Silver et al. 2016 (AlphaGo) | Nature 529(7587):484–489, DOI 10.1038/nature16961 (V) | [Nature](https://www.nature.com/articles/nature16961) |
| 15 | Henderson et al. 2018 | AAAI 32(1):3207–3214, DOI 10.1609/aaai.v32i1.11694 (V) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11694) |
| 16 | Agarwal et al. 2021 | NeurIPS 34, pp. 29304–29320 (V) | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) |
| opt | Huang et al. 2022 (37 details) | ICLR Blog Track 2022 (non-archival) (V) | [Blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |
| opt | Ross, Gordon, Bagnell 2011 (DAgger) | AISTATS 2011, PMLR 15:627–635 (V) | [PMLR](https://proceedings.mlr.press/v15/ross11a.html) |
| opt | Jain, Szot, Lim 2020 | ICML 2020, PMLR 119:4661–4672 (V) | [PMLR](https://proceedings.mlr.press/v119/jain20b.html) |
| no | Dulac-Arnold et al. 2015 | arXiv:1512.07679, no venue in the arXiv record | [arXiv](https://arxiv.org/abs/1512.07679) |
| no | Chandak et al. 2020 "Lifelong…" | AAAI 2020 per the arXiv comment; pages not verified | [arXiv](https://arxiv.org/abs/1906.01770) |

#### BibTeX (core 16, then optional)
```bibtex
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

@article{schulman2017ppo,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = {2017}
}

@inproceedings{engstrom2020implementation,
  title     = {Implementation Matters in Deep {RL}: A Case Study on {PPO} and {TRPO}},
  author    = {Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Janoos, Firdaus and Rudolph, Larry and Madry, Aleksander},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  note      = {arXiv:2005.12729, titled ``Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO''}
}

@inproceedings{andrychowicz2021matters,
  title     = {What Matters for On-Policy Deep Actor-Critic Methods? {A} Large-Scale Study},
  author    = {Andrychowicz, Marcin and Raichuk, Anton and Sta{\'n}czyk, Piotr and Orsini, Manu and Girgin, Sertan and Marinier, Rapha{\"e}l and Hussenot, L{\'e}onard and Geist, Matthieu and Pietquin, Olivier and Michalski, Marcin and Gelly, Sylvain and Bachem, Olivier},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=nIAxjsniDzg}
}

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

% ---------- optional ----------
@inproceedings{huang2022the37implementation,
  author    = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  title     = {The 37 Implementation Details of Proximal Policy Optimization},
  booktitle = {ICLR Blog Track},
  year      = {2022},
  url       = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
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

@inproceedings{jain2020generalization,
  title     = {Generalization to New Actions in Reinforcement Learning},
  author    = {Jain, Ayush and Szot, Andrew and Lim, Joseph J.},
  booktitle = {International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {4661--4672},
  year      = {2020},
  publisher = {PMLR}
}
```

### Inferences
- If space is tight, the A3C citation can be dropped in favour of citing PPO for the shared-trunk loss. Keep A3C, though, if the report states the entropy-bonus rationale, since PPO cites A3C and Williams for that.
- The blog post is non-archival. Cite it only alongside Engstrom and Andrychowicz, never alone.
- For BibTeX hygiene: the PPO and DeepSeekMath entries are arXiv preprints, so do not invent a venue. Williams 1992 is often cited as "8(3–4)"; Springer's metadata gives issue "3", which is used here.

### Gaps
- PMLR metadata did not show a conference sequence number for Jain et al. ICML 2020, so the booktitle omits "37th".
- AAAI OJS page numbers come from the official DC metadata (Chandak) and from ML Anthology (Henderson, consistent with the AAAI DOI). The FLAIRS article page lists no page range, so none is given.
- OpenReview API access was blocked (challenge page), so I verified the Andrychowicz and Dr. GRPO venues via OpenReview search listings, Google's ICLR 2021 listing in search results, ML Anthology, and the paper PDF header (Dr. GRPO), not via the forum page itself.
