import pandas as pd
import numpy as np
from typing import Optional

try:
    import yfinance as yf
    YF = True
except Exception:
    YF = False

class DataLoader:
    def __init__(self, source: str = "yfinance", symbol: str = "BTC-USD", timeframe: str = "1h"):
        self.source = source
        self.symbol = symbol
        self.timeframe = timeframe

    def load(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        if self.source == "yfinance":
            if not YF:
                raise ImportError("yfinance 미설치")
            interval = self._map_interval(self.timeframe)
            df = yf.download(self.symbol, start=start_date, end=end_date, interval=interval, progress=False)
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            df = df[['open','high','low','close','volume']]
            df = df.dropna()
            return df
        raise ValueError(f"unsupported source: {self.source}")

    def _map_interval(self, tf: str) -> str:
        return {
            '1m':'1m','5m':'5m','15m':'15m','30m':'30m','1h':'1h','1d':'1d','1wk':'1wk'
        }.get(tf, '1h')
