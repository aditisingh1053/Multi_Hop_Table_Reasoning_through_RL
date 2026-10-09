"""Pointer-style actor and a separate critic over a variable-size candidate set.

Actor:  h = f_s(s),  g_a = f_c([e_a ; x_a]),  score(a) = w^T ReLU(W [h ; g_a ; h * g_a])
        score(STOP) = f_stop([h ; scalar state features])
        pi(a | s) = softmax over the *valid* entries; invalid entries get a large negative
        logit, so they receive zero probability and zero gradient (masking, not penalties).
Critic: V(s) = f_v([h' ; mean_a g'_a ; max_a g'_a]) with its own parameters, since separate
        policy and value networks were found to work better than a shared trunk.
"""
import math

import torch
import torch.nn as nn

from .env import EMB_DIM, N_CAND_FEAT, N_STATE_SCALAR, STATE_DIM

NEG = -1e9


def _lin(i, o, gain=math.sqrt(2)):
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(i, h):
    return nn.Sequential(_lin(i, h), nn.ReLU(), _lin(h, h), nn.ReLU())


class PointerActor(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.state_net = _mlp(STATE_DIM, hidden)
        self.cand_net = _mlp(EMB_DIM + N_CAND_FEAT, hidden)
        self.scorer = nn.Sequential(_lin(3 * hidden, hidden), nn.ReLU(), _lin(hidden, 1, 0.01))
        self.stop_head = nn.Sequential(_lin(hidden + N_STATE_SCALAR, hidden // 2), nn.ReLU(),
                                       _lin(hidden // 2, 1, 0.01))

    def forward(self, state, cand_emb, cand_feat, cand_mask, stop_ok):
        """Returns masked logits [B, K+1]; the last column is STOP."""
        h = self.state_net(state)
        g = self.cand_net(torch.cat([cand_emb, cand_feat], -1))
        hk = h.unsqueeze(1).expand_as(g)
        cand_logit = self.scorer(torch.cat([hk, g, hk * g], -1)).squeeze(-1)
        stop_logit = self.stop_head(torch.cat([h, state[:, -N_STATE_SCALAR:]], -1))
        logits = torch.cat([cand_logit, stop_logit], -1)
        mask = torch.cat([cand_mask, stop_ok.unsqueeze(-1)], -1)
        return logits.masked_fill(~mask, NEG)


class Critic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.state_net = _mlp(STATE_DIM, hidden)
        self.cand_net = _mlp(EMB_DIM + N_CAND_FEAT, hidden)
        self.head = nn.Sequential(_lin(3 * hidden, hidden), nn.ReLU(), _lin(hidden, 1, 1.0))

    def forward(self, state, cand_emb, cand_feat, cand_mask, stop_ok=None):
        h = self.state_net(state)
        g = self.cand_net(torch.cat([cand_emb, cand_feat], -1))
        m = cand_mask.unsqueeze(-1).float()
        count = m.sum(1).clamp(min=1.0)
        mean = (g * m).sum(1) / count
        mx = g.masked_fill(m == 0, NEG).max(1).values
        mx = torch.where(cand_mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return self.head(torch.cat([h, mean, mx], -1)).squeeze(-1)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
