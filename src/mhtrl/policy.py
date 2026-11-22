"""Pointer-style actor and a separate critic over a variable-size candidate set.

Actor:  h = f_s(s),  g_a = f_c([e_a ; x_a]),  score(a) = w^T ReLU(W [h ; g_a ; h * g_a])
        score(STOP) = f_stop([h ; scalar state features])
        pi(a | s) = softmax over the *valid* entries; invalid entries get a large negative
        logit, so they receive zero probability and zero gradient (masking, not penalties).
Critic: V(s) = f_v([h' ; mean_a g'_a ; max_a g'_a]) with its own parameters, since separate
        policy and value networks were found to work better than a shared trunk.
Commit (reader-in-the-loop only): in the COMMIT phase the candidates are the selected
        entities and score(e) = beta * reader_score(e) + c(h, g_e), with beta initialised so
        that beta * feature = raw reader score and c initialised near 0: the untrained head
        reproduces the reader's max-confidence aggregation, and PPO learns the residual.
"""
import math

import torch
import torch.nn as nn

from .env import (EMB_DIM, LOOP_SCORE_IDX, N_CAND_FEAT, N_STATE_SCALAR, SCORE_SCALE, STATE_DIM,
                  EnvConfig, dims)

NEG = -1e9


def _lin(i, o, gain=math.sqrt(2)):
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(i, h):
    return nn.Sequential(_lin(i, h), nn.ReLU(), _lin(h, h), nn.ReLU())


class PointerActor(nn.Module):
    def __init__(self, hidden: int = 256, state_dim: int = STATE_DIM,
                 cand_dim: int = EMB_DIM + N_CAND_FEAT, n_scalar: int = N_STATE_SCALAR,
                 commit: bool = False):
        super().__init__()
        self.n_scalar, self.commit = n_scalar, commit
        self.state_net = _mlp(state_dim, hidden)
        self.cand_net = _mlp(cand_dim, hidden)
        self.scorer = nn.Sequential(_lin(3 * hidden, hidden), nn.ReLU(), _lin(hidden, 1, 0.01))
        self.stop_head = nn.Sequential(_lin(hidden + n_scalar, hidden // 2), nn.ReLU(),
                                       _lin(hidden // 2, 1, 0.01))
        if commit:
            self.commit_scorer = nn.Sequential(_lin(3 * hidden, hidden), nn.ReLU(),
                                               _lin(hidden, 1, 0.01))
            self.commit_prior = nn.Parameter(torch.tensor(SCORE_SCALE))

    def forward(self, state, cand_emb, cand_feat, cand_mask, stop_ok):
        """Returns masked logits [B, K+1]; the last column is STOP."""
        h = self.state_net(state)
        g = self.cand_net(torch.cat([cand_emb, cand_feat], -1))
        hk = h.unsqueeze(1).expand_as(g)
        z = torch.cat([hk, g, hk * g], -1)
        cand_logit = self.scorer(z).squeeze(-1)
        if self.commit:   # the commit-phase flag is the last state scalar
            in_commit = state[:, -1:] > 0.5
            commit_logit = self.commit_scorer(z).squeeze(-1) + \
                self.commit_prior * cand_feat[..., LOOP_SCORE_IDX]
            cand_logit = torch.where(in_commit, commit_logit, cand_logit)
        stop_logit = self.stop_head(torch.cat([h, state[:, -self.n_scalar:]], -1))
        logits = torch.cat([cand_logit, stop_logit], -1)
        mask = torch.cat([cand_mask, stop_ok.unsqueeze(-1)], -1)
        return logits.masked_fill(~mask, NEG)


def make_actor(env_cfg: EnvConfig = None) -> PointerActor:
    """Actor sized for an environment; with the default config identical to PointerActor()."""
    return PointerActor(**dims(env_cfg or EnvConfig()))


def make_critic(env_cfg: EnvConfig = None, n_priv: int = 0) -> "Critic":
    d = dims(env_cfg or EnvConfig())
    return Critic(state_dim=d["state_dim"], cand_dim=d["cand_dim"], n_priv=n_priv)


class Critic(nn.Module):
    """n_priv > 0: asymmetric actor-critic (Pinto et al., 2018). The critic additionally sees
    privileged, training-only information (gold coverage, reader F1 of the evidence); the
    actor never does, so nothing privileged is needed at test time."""

    def __init__(self, hidden: int = 256, state_dim: int = STATE_DIM,
                 cand_dim: int = EMB_DIM + N_CAND_FEAT, n_priv: int = 0):
        super().__init__()
        self.n_priv = n_priv
        self.state_net = _mlp(state_dim, hidden)
        self.cand_net = _mlp(cand_dim, hidden)
        self.head = nn.Sequential(_lin(3 * hidden + n_priv, hidden), nn.ReLU(), _lin(hidden, 1, 1.0))

    def forward(self, state, cand_emb, cand_feat, cand_mask, stop_ok=None, priv=None):
        h = self.state_net(state)
        g = self.cand_net(torch.cat([cand_emb, cand_feat], -1))
        m = cand_mask.unsqueeze(-1).float()
        count = m.sum(1).clamp(min=1.0)
        mean = (g * m).sum(1) / count
        mx = g.masked_fill(m == 0, NEG).max(1).values
        mx = torch.where(cand_mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        parts = [h, mean, mx]
        if self.n_priv:
            if priv is None:
                raise ValueError("this critic needs the privileged features (priv=...)")
            parts.append(priv.to(h.dtype))
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
