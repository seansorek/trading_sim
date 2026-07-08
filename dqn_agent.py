"""
Enhanced DQN agent with Dueling architecture and prioritized experience replay.
"""

import random
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture: splits network into value and advantage streams.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
    """
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.advantage_stream(feat)
        
        # Dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using TD-error as priority.
    Focuses on transitions that are harder to learn.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent: how much prioritization
        self.beta = beta    # Importance-sampling exponent: how much to correct for bias
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool, td_error: float = 1.0):
        """Add transition with priority = max_priority initially, updated via learn()."""
        self.buffer.append((s, a, r, s2, done))
        # Use max_priority for new experiences (optimistic initialization)
        priority = self.max_priority if self.max_priority > 0 else 1.0
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample batch with probability proportional to priority.
        Returns: (s, a, r, s2, done, weights) where weights are importance-sampling corrections.
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha / (priorities ** self.alpha).sum()
        
        # Sample indices with replacement
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=True)
        
        # Compute importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max for stability
        weights = torch.from_numpy(weights).float().to(device)
        
        # Extract batch
        batch = [self.buffer[i] for i in indices]
        s = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
        a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        s2 = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(device)
        d = torch.tensor([b[4] for b in batch], dtype=torch.bool, device=device)
        
        return s, a, r, s2, d, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 500_000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_update_interval: int = 5000
    hidden: int = 256
    device: str = "cpu"
    use_dueling: bool = True
    use_per: bool = True  # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta: float = 0.4


class DQNAgent:
    """
    Enhanced DQN with optional Dueling architecture and Prioritized Experience Replay.
    """
    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg

        self.device = torch.device(cfg.device)
        
        # Use Dueling architecture if enabled
        if cfg.use_dueling:
            self.q = DuelingQNetwork(state_dim, action_dim, cfg.hidden).to(self.device)
            self.target = DuelingQNetwork(state_dim, action_dim, cfg.hidden).to(self.device)
        else:
            self.q = QNetwork(state_dim, action_dim, cfg.hidden).to(self.device)
            self.target = QNetwork(state_dim, action_dim, cfg.hidden).to(self.device)
        
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # Use prioritized buffer if enabled
        if cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha, beta=cfg.per_beta)
        else:
            self.buffer = deque(maxlen=cfg.buffer_size)
        
        self.steps = 0
        self.epsilon = cfg.start_epsilon
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

    def act(self, state: np.ndarray) -> int:
        self.steps += 1
        eps = self._current_epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            qvals = self.q(s)
            return int(torch.argmax(qvals, dim=1).item())

    def _current_epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.cfg.epsilon_decay_steps))
        self.epsilon = self.cfg.start_epsilon + frac * (self.cfg.end_epsilon - self.cfg.start_epsilon)
        return self.epsilon

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        if self.cfg.use_per:
            self.buffer.push(s, a, r, s2, done)
        else:
            self.buffer.append((s, a, r, s2, done))

    def can_learn(self) -> bool:
        return len(self.buffer) >= self.cfg.batch_size

    def learn(self) -> float:
        if not self.can_learn():
            return 0.0
        
        if self.cfg.use_per:
            s, a, r, s2, d, weights, indices = self.buffer.sample(self.cfg.batch_size, self.device)
        else:
            batch = random.sample(self.buffer, self.cfg.batch_size)
            s = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
            a = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
            r = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
            s2 = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
            d = torch.tensor([b[4] for b in batch], dtype=torch.bool, device=self.device)
            weights = torch.ones(self.cfg.batch_size, device=self.device)
            indices = None

        q = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            # Double DQN
            next_actions = self.q(s2).argmax(dim=1, keepdim=True)
            next_q = self.target(s2).gather(1, next_actions).squeeze(1)
            target = r + (~d).float() * self.cfg.gamma * next_q
        
        # TD loss with optional importance-sampling weighting
        td_loss = self.loss_fn(q, target)
        weighted_loss = (weights * td_loss).mean()
        
        self.opt.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.opt.step()

        # Update priorities if using PER
        if self.cfg.use_per and indices is not None:
            td_errors = (q - target).detach().cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)

        if self.steps % self.cfg.target_update_interval == 0:
            self.target.load_state_dict(self.q.state_dict())
        
        return float(weighted_loss.item())

    def save(self, path: str):
        torch.save({
            "model": self.q.state_dict(),
            "cfg": self.cfg.__dict__,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    @staticmethod
    def load(path: str) -> "DQNAgent":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        cfg_dict = blob["cfg"].copy()
        cfg = DQNConfig(**cfg_dict)
        agent = DQNAgent(blob["state_dim"], blob["action_dim"], cfg)
        agent.q.load_state_dict(blob["model"])
        agent.target.load_state_dict(agent.q.state_dict())
        return agent
