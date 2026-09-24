"""A small Deep Q-Network for the grid route-guidance demonstration.

Deliberately standard (MLP Q-network, replay buffer, target network,
epsilon-greedy) so that the reported failure is a property of the reward and
boundary configuration rather than of an unusual learning algorithm. Invalid
actions are masked using the environment's action mask.
"""

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.mask2 = np.ones((capacity, 4), np.float32)
        self.i = 0
        self.full = False

    def add(self, s, a, r, s2, done, mask2):
        i = self.i
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.done[i], self.mask2[i] = s2, done, mask2
        self.i = (i + 1) % self.capacity
        self.full = self.full or self.i == 0

    def __len__(self):
        return self.capacity if self.full else self.i

    def sample(self, batch):
        n = len(self)
        idx = np.random.randint(0, n, size=batch)
        return (self.s[idx], self.a[idx], self.r[idx],
                self.s2[idx], self.done[idx], self.mask2[idx])


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 buffer=20000, batch=64, target_every=200, seed=0):
        torch.manual_seed(seed)
        self.q = QNet(state_dim, n_actions)
        self.qt = QNet(state_dim, n_actions)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buf = ReplayBuffer(buffer, state_dim)
        self.gamma = gamma
        self.batch = batch
        self.target_every = target_every
        self.n_actions = n_actions
        self.updates = 0

    def act(self, state, mask, eps):
        """Epsilon-greedy over available actions only."""
        avail = np.where(mask)[0]
        if np.random.rand() < eps:
            return int(np.random.choice(avail))
        with torch.no_grad():
            q = self.q(torch.from_numpy(state).unsqueeze(0)).numpy()[0]
        q_masked = np.where(mask, q, -1e9)
        return int(np.argmax(q_masked))

    def learn(self):
        if len(self.buf) < self.batch:
            return
        s, a, r, s2, done, mask2 = self.buf.sample(self.batch)
        s = torch.from_numpy(s); a = torch.from_numpy(a)
        r = torch.from_numpy(r); s2 = torch.from_numpy(s2)
        done = torch.from_numpy(done); mask2 = torch.from_numpy(mask2)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.qt(s2)
            q2 = q2.masked_fill(mask2 < 0.5, -1e9)
            q2max = q2.max(1)[0]
            target = r + self.gamma * (1 - done) * q2max
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.qt.load_state_dict(self.q.state_dict())
