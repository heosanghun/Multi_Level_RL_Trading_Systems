import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

class SimpleEnv:
    def __init__(self, df, commission=0.0005, slippage=0.0002, position_size=0.2):
        self.df = df.reset_index(drop=True)
        self.t = 1
        self.position = 0  # -1,0,1
        self.entry = None
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size

    def reset(self):
        self.t = 1
        self.position = 0
        self.entry = None
        return self._obs(lookback=50)

    def step(self, action: int):
        price_prev = self.df.loc[self.t-1, "close"]
        price = self.df.loc[self.t, "close"]
        reward = 0.0
        # 0: short, 1: hold, 2: long
        desired = [-1, 0, 1][action]
        if desired != self.position:
            # transaction cost
            reward -= self.commission
            self.entry = price
            self.position = desired
        # PnL component
        if self.position == 1 and self.entry is not None:
            reward += (price - price_prev) / price_prev
        elif self.position == -1 and self.entry is not None:
            reward += (price_prev - price) / price_prev
        self.t += 1
        done = self.t >= len(self.df)
        return self._obs(lookback=50), reward, done

    def _obs(self, lookback=50):
        s = self.df.iloc[max(0, self.t-lookback):self.t][["open","high","low","close","volume"]].values
        # normalize
        s = (s - s.mean(axis=0)) / (s.std(axis=0)+1e-8)
        return torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # [1,T,5]


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = torch.zeros(T)
    last = 0
    for t in reversed(range(T)):
        next_val = 0 if t==T-1 else values[t+1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        adv[t] = last = delta + gamma * lam * (1 - dones[t]) * last
    returns = adv + values
    return adv, returns

class RolloutCollector:
    def __init__(self, env: SimpleEnv, model, rollout_len=512):
        self.env = env
        self.model = model
        self.rollout_len = rollout_len

    def collect(self):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        h = None
        obs = self.env.reset()
        for _ in range(self.rollout_len):
            with torch.no_grad():
                a, logp, v, h = self.model.act(obs, h)
            next_obs, r, done = self.env.step(a.item())
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(r, dtype=torch.float32))
            val_buf.append(v.squeeze(0))
            done_buf.append(torch.tensor(float(done)))
            obs = next_obs
            if done:
                obs = self.env.reset()
                h = None
        obs = torch.cat(obs_buf, dim=0)
        acts = torch.stack(act_buf)
        logp = torch.stack(logp_buf)
        rews = torch.stack(rew_buf)
        vals = torch.stack(val_buf).squeeze(-1)
        dones = torch.stack(done_buf)
        adv, ret = compute_gae(rews, vals, dones)
        # flatten time into batch
        return obs, acts, logp, adv, ret
