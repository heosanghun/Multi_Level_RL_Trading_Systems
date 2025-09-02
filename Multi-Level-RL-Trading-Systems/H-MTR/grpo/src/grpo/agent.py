import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class GRPOActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 3, gru_layers: int = 1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        z = F.relu(self.input_proj(x))
        out, h_n = self.gru(z, h)
        logits = self.policy(out)
        values = self.value(out).squeeze(-1)
        return logits, values, h_n

    def act(self, x, h=None):
        logits, values, h_n = self.forward(x, h)
        probs = F.softmax(logits[:, -1], dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        value = values[:, -1]
        return action, logp, value, h_n

class GRPOUpdater:
    def __init__(self, model: GRPOActorCritic, lr: float = 3e-4, clip_eps: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01, max_grad_norm: float = 0.5):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def update(self, obs, actions, old_logp, returns, advantages, epochs: int = 4, batch_size: int = 256):
        N = obs.size(0)
        idx = torch.randperm(N)
        last = {}
        for _ in range(epochs):
            for i in range(0, N, batch_size):
                b = idx[i:i+batch_size]
                logits, values, _ = self.model(obs[b])
                probs = F.softmax(logits[:, -1], dim=-1)
                dist = Categorical(probs)
                logp = dist.log_prob(actions[b])
                ratio = torch.exp(logp - old_logp[b])
                surr1 = ratio * advantages[b]
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values[:, -1], returns[b])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                last = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item()
                }
        return last
