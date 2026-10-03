# market_data.py
import time
import requests
import pandas as pd
from typing import Optional

from config import BINANCE_BASE_URL, BINANCE_SYMBOL_MAP


class BinanceDataClient:
    """
    Fetch OHLCV klines from Binance's public REST API.

    No API key required. Public endpoint.
    Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data

    Interval strings: 1m, 5m, 15m, 30m, 1h, 4h, 1d, ...
    """

    def __init__(self, base_url: str = BINANCE_BASE_URL, timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()

    def get_klines(
        self,
        roostoo_pair: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame indexed by UTC timestamp with columns:
        open, high, low, close, volume.

        Returns None on failure.
        """
        symbol = BINANCE_SYMBOL_MAP.get(roostoo_pair)
        if symbol is None:
            raise ValueError(f"No Binance symbol mapped for {roostoo_pair}")

        url = f"{self.base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        for attempt in range(3):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout)
                r.raise_for_status()
                raw = r.json()
                if not raw:
                    return None
                df = pd.DataFrame(
                    raw,
                    columns=[
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades",
                        "taker_buy_base", "taker_buy_quote", "ignore",
                    ],
                )
                # Binance returns strings; coerce numerics
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # open_time is ms epoch -> UTC datetime index
                df["timestamp"] = pd.to_datetime(
                    df["open_time"], unit="ms", utc=True
                )
                df = (
                    df[["timestamp", "open", "high", "low", "close", "volume"]]
                    .set_index("timestamp")
                    .sort_index()
                )

                # Drop the LAST (unclosed) candle: its close is still moving
                if len(df) > 0:
                    df = df.iloc[:-1]

                return df

            except requests.exceptions.RequestException as e:
                print(f"[binance] klines attempt {attempt+1} failed: {e}")
                time.sleep(1 + attempt)

        return None