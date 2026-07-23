# -*- coding: utf-8 -*-
"""A small PPO agent for the grid route-guidance demonstration.

The four-cell result is reported for a value-based learner. The reward-side conclusions —
the size of the recovery, the shape of the dose-response, the ablation — are therefore shown
for one learner, even though the 0% collapse itself is algorithm-free by the bound of
Section V-C. This provides a policy-gradient learner so the same four cells can be run again
with the learning rule replaced and nothing else.

Kept deliberately standard: separate actor and critic MLPs of the same width as the Q-network,
clipped surrogate objective, generalized advantage estimation, and the environment's action
mask applied to the logits so that infeasible moves carry no probability.
"""
import numpy as np
import torch
import torch.nn as nn

NEG = -1e9


class MLP(nn.Module):
    def __init__(self, state_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    def __init__(self, state_dim, n_actions, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, minibatch=64, ent_coef=0.01, seed=0):
        torch.manual_seed(seed)
        self.pi = MLP(state_dim, n_actions)
        self.v = MLP(state_dim, 1)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.ent_coef = epochs, minibatch, ent_coef
        self.n_actions = n_actions
        self.buf = []

    def _dist(self, s, mask):
        logits = self.pi(s)
        logits = logits.masked_fill(~mask, NEG)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state, mask, greedy=False):
        s = torch.from_numpy(state).unsqueeze(0)
        m = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            d = self._dist(s, m)
            a = int(torch.argmax(d.logits, dim=-1)) if greedy else int(d.sample())
            lp = float(d.log_prob(torch.tensor([a])))
            val = float(self.v(s))
        return a, lp, val

    def store(self, s, a, lp, val, r, done, mask):
        self.buf.append((s, a, lp, val, r, float(done), mask))

    def update(self):
        if not self.buf:
            return
        s, a, lp, val, r, done, mask = zip(*self.buf)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        lp_old = torch.tensor(lp, dtype=torch.float32)
        val = np.array(val, dtype=np.float32)
        r = np.array(r, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        mask = torch.tensor(np.array(mask), dtype=torch.bool)

        adv = np.zeros_like(r)
        last = 0.0
        for t in range(len(r) - 1, -1, -1):
            nextval = 0.0 if (t == len(r) - 1 or done[t]) else val[t + 1]
            delta = r[t] + self.gamma * nextval * (1.0 - done[t]) - val[t]
            last = delta + self.gamma * self.lam * (1.0 - done[t]) * last
            adv[t] = last
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.tensor(adv, dtype=torch.float32)
        ret = torch.tensor(ret, dtype=torch.float32)

        n = len(r)
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for k in range(0, n, self.minibatch):
                j = idx[k:k + self.minibatch]
                d = self._dist(s[j], mask[j])
                ratio = torch.exp(d.log_prob(a[j]) - lp_old[j])
                l1 = ratio * adv[j]
                l2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[j]
                pl = -torch.min(l1, l2).mean() - self.ent_coef * d.entropy().mean()
                vl = ((self.v(s[j]).squeeze(-1) - ret[j]) ** 2).mean()
                self.opt.zero_grad()
                (pl + 0.5 * vl).backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), 0.5)
                self.opt.step()
        self.buf = []
