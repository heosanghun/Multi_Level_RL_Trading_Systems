import argparse
import yaml
import torch
from pathlib import Path

from .data import DataLoader
from .agent import GRPOActorCritic, GRPOUpdater
from .trainer import SimpleEnv, RolloutCollector
from .backtest import SimpleBacktester


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg):
    dl = DataLoader(cfg["data"]["source"], cfg["data"]["symbol"], cfg["data"]["timeframe"])
    df = dl.load(cfg["data"]["start_date"], cfg["data"]["end_date"])
    input_dim = 5
    model = GRPOActorCritic(input_dim=input_dim, hidden_dim=cfg["model"]["hidden_dim"], action_dim=cfg["model"]["action_dim"])
    updater = GRPOUpdater(model, lr=cfg["training"]["learning_rate"], clip_eps=cfg["training"]["clip_eps"], vf_coef=cfg["training"]["vf_coef"], ent_coef=cfg["training"]["ent_coef"], max_grad_norm=cfg["training"]["max_grad_norm"])
    env = SimpleEnv(df, commission=cfg["env"]["commission"], slippage=cfg["env"]["slippage"], position_size=cfg["env"]["position_size"])
    collector = RolloutCollector(env, model, rollout_len=cfg["training"]["rollout_length"])

    total_steps = cfg["training"]["total_steps"]
    steps = 0
    while steps < total_steps:
        obs, acts, logp, adv, ret = collector.collect()
        losses = updater.update(obs, acts, logp.detach(), ret.detach(), adv.detach(), epochs=4, batch_size=cfg["training"]["batch_size"])
        steps += len(acts)
        print(f"steps={steps} losses={losses}")


def backtest(cfg):
    dl = DataLoader(cfg["data"]["source"], cfg["data"]["symbol"], cfg["data"]["timeframe"])
    df = dl.load(cfg["data"]["start_date"], cfg["data"]["end_date"])
    # 간단 룰: RSI 대체로 모멘텀 - 3연속 상승시 long, 3연속 하락시 short, else 0
    price = df["close"]
    mom = price.diff()
    streak_up = (mom>0).rolling(3).sum()
    streak_dn = (mom<0).rolling(3).sum()
    signals = (streak_up==3).astype(int) - (streak_dn==3).astype(int)
    bt = SimpleBacktester(df, commission=cfg["env"]["commission"], position_size=cfg["env"]["position_size"])
    res = bt.run_signals(signals)
    print({k:(float(v) if k!="equity" else f"len={len(v)}") for k,v in res.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2]/"configs"/"config.yaml"))
    p.add_argument("--mode", choices=["train","backtest"], default="backtest")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.mode=="train":
        train(cfg)
    else:
        backtest(cfg)

if __name__ == "__main__":
    main()
