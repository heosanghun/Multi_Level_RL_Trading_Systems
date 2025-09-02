import pandas as pd
import numpy as np

class Metrics:
    @staticmethod
    def sharpe(returns: pd.Series, rf: float = 0.0):
        ex = returns - rf/252
        std_val = ex.std()
        if std_val == 0.0:
            return 0.0
        return ex.mean() / std_val * np.sqrt(252)

    @staticmethod
    def max_drawdown(equity: pd.Series):
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd.min()

class SimpleBacktester:
    def __init__(self, df: pd.DataFrame, commission=0.0005, position_size=0.2):
        self.df = df
        self.commission = commission
        self.position_size = position_size

    def run_signals(self, signals: pd.Series):
        # signals: -1,0,1 per step
        price = self.df["close"]
        pos = signals.shift(1).fillna(0)
        ret = price.pct_change().fillna(0)
        strat_ret = pos * ret - self.commission * (signals.diff().abs().fillna(0)>0).astype(float)
        equity = (1+strat_ret).cumprod()
        return {
            "total_return": equity.iloc[-1]-1,
            "sharpe": Metrics.sharpe(strat_ret),
            "mdd": Metrics.max_drawdown(equity),
            "equity": equity
        }
