"""
Roostoo bStock Quant Bot — Single-file version

Dependencies:
    pandas
    numpy
    requests

Environment variables:
    ROOSTOO_API_KEY
    ROOSTOO_API_SECRET
    LIVE_TRADING=false
    ROOSTOO_BASE_URL=https://mock-api.roostoo.com
    BINANCE_BASE_URL=https://data-api.binance.vision
    DRY_RUN_NAV=100000
    LOOP_SECONDS=5

IMPORTANT:
- Never hard-code or commit API keys/secrets.
- Default is DRY RUN. Set LIVE_TRADING=true only after diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. CONFIG
# ============================================================

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
ROOSTOO_BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

BINANCE_SYMBOLS = [
    "AMDBUSDT",
    "INTCBUSDT",
    "MUBUSDT",
    "NVDABUSDT",
    "SKHYBUSDT",
    "SNDKBUSDT",
]

ROOSTOO_PAIR_MAP = {
    "AMDBUSDT": "AMDB/USD",
    "INTCBUSDT": "INTCB/USD",
    "MUBUSDT": "MUB/USD",
    "NVDABUSDT": "NVDAB/USD",
    "SKHYBUSDT": "SKHYB/USD",
    "SNDKBUSDT": "SNDKB/USD",
}

LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "5"))
DRY_RUN_NAV = float(os.getenv("DRY_RUN_NAV", "100000"))

POSITION_FRACTION = 0.05
MAX_SLOTS = 4
MAX_PRICE_DIVERGENCE = 0.01

STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))
SIGNAL_LOG = Path(os.getenv("SIGNAL_LOG", "signals.csv"))
TRADE_LOG = Path(os.getenv("TRADE_LOG", "trades.csv"))
BOT_LOG = Path(os.getenv("BOT_LOG", "bot.log"))


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass
class Signal:
    strategy: str
    symbol: str
    side: str
    signal_time: str
    limit_price: float
    expiry_seconds: int
    exit_kind: str
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    time_exit_seconds: Optional[int] = None
    note: str = ""


# ============================================================
# 3. LOGGING / STATE
# ============================================================

def setup_logger():
    logger = logging.getLogger("roostoo_bot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(BOT_LOG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


LOGGER = setup_logger()


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_state():
    if not STATE_FILE.exists():
        return {
            "pending": {},
            "positions": {},
            "last_signal_times": {},
            "last_processed_1m": None,
            "last_processed_15m": None,
        }
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    state.setdefault("pending", {})
    state.setdefault("positions", {})
    state.setdefault("last_signal_times", {})
    state.setdefault("last_processed_1m", None)
    state.setdefault("last_processed_15m", None)
    return state


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ============================================================
# 4. ROOSTOO API CLIENT
# ============================================================

class RoostooError(RuntimeError):
    pass


class RoostooClient:
    def __init__(self, api_key=None, secret=None, base_url=ROOSTOO_BASE_URL, timeout=10):
        self.api_key = api_key
        self.secret = secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def timestamp():
        return str(int(time.time() * 1000))

    def signed_payload(self, payload=None):
        if not self.api_key or not self.secret:
            raise RoostooError("ROOSTOO_API_KEY / ROOSTOO_API_SECRET not configured.")

        params = dict(payload or {})
        params["timestamp"] = self.timestamp()

        body = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(
            self.secret.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, params, body

    @staticmethod
    def parse_response(response):
        response.raise_for_status()
        return response.json()

    def server_time(self):
        r = self.session.get(f"{self.base_url}/v3/serverTime", timeout=self.timeout)
        return self.parse_response(r)

    def exchange_info(self):
        r = self.session.get(f"{self.base_url}/v3/exchangeInfo", timeout=self.timeout)
        return self.parse_response(r)

    def ticker(self, pair=None):
        params = {"timestamp": self.timestamp()}
        if pair:
            params["pair"] = pair

        r = self.session.get(
            f"{self.base_url}/v3/ticker",
            params=params,
            timeout=self.timeout,
        )
        return self.parse_response(r)

    def balance(self):
        headers, params, _ = self.signed_payload({})
        r = self.session.get(
            f"{self.base_url}/v3/balance",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        return self.parse_response(r)

    def place_order(self, pair, side, quantity, order_type, price=None):
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price.")
            payload["price"] = str(price)

        headers, _, body = self.signed_payload(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        r = self.session.post(
            f"{self.base_url}/v3/place_order",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        return self.parse_response(r)

    def query_order(self, order_id=None, pair=None, pending_only=None):
        payload = {}

        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"

        headers, _, body = self.signed_payload(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        r = self.session.post(
            f"{self.base_url}/v3/query_order",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        return self.parse_response(r)

    def cancel_order(self, order_id=None, pair=None):
        payload = {}

        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair

        headers, _, body = self.signed_payload(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        r = self.session.post(
            f"{self.base_url}/v3/cancel_order",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        return self.parse_response(r)


# ============================================================
# 5. BINANCE MARKET DATA
# ============================================================

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


class BinanceMarketData:
    def __init__(self, base_url=BINANCE_BASE_URL, timeout=10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def klines(self, symbol, interval, limit=500):
        r = self.session.get(
            f"{self.base_url}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()

        df = pd.DataFrame(r.json(), columns=KLINE_COLUMNS)
        if df.empty:
            return df

        for c in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        return df

    @staticmethod
    def completed_only(df):
        if df.empty:
            return df
        now = pd.Timestamp.now(tz="UTC")
        return df[df["close_time"] < now].copy()


# ============================================================
# 6. INDICATORS
# ============================================================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def true_range(df):
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr_sma(df, period=14):
    return true_range(df).rolling(period).mean()


def atr_wilder_ewm(df, period=14):
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def resample_5m_from_1m(df):
    x = df.set_index("open_time")[["open", "high", "low", "close", "volume"]]

    return x.resample("5min", label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()


def last_return(df, periods):
    if len(df) <= periods:
        return np.nan
    c = df["close"]
    return float(c.iloc[-1] / c.iloc[-1 - periods] - 1)


# ============================================================
# 7. STRATEGIES
# ============================================================

def broad_leadlag_signals(bars_1m, now, last_signal_times):
    signals = []

    returns = {
        s: last_return(df, 1)
        for s, df in bars_1m.items()
        if len(df) >= 100
    }

    for target in BINANCE_SYMBOLS:
        if target not in returns:
            continue

        peers = [
            s for s in BINANCE_SYMBOLS
            if s != target and s in returns and np.isfinite(returns[s])
        ]

        if len(peers) < 4:
            continue

        peer_ret = float(np.mean([returns[s] for s in peers]))
        own_ret = returns[target]

        df = bars_1m[target]
        e60 = ema(df["close"], 60).iloc[-1]

        condition = (
            peer_ret > 0.004
            and peer_ret - own_ret >= 0.001
            and df["close"].iloc[-1] > e60
        )

        if not condition:
            continue

        key = f"BroadLeadLag:{target}"
        previous = last_signal_times.get(key)

        if previous is not None:
            if (now - previous).total_seconds() < 600:
                continue

        bars_5m = resample_5m_from_1m(df)
        if len(bars_5m) < 15:
            continue

        atr = atr_sma(bars_5m, 14).iloc[-1]
        if not np.isfinite(atr):
            continue

        signal_close = float(df["close"].iloc[-1])
        entry = signal_close * (1 - 0.0003)

        signals.append(
            Signal(
                strategy="BroadLeadLag",
                symbol=target,
                side="BUY",
                signal_time=now.isoformat(),
                limit_price=entry,
                expiry_seconds=60,
                exit_kind="atr",
                stop_price=entry - 0.5 * atr,
                take_profit_price=entry + 6.0 * atr,
                note=f"peer1m={peer_ret:.6f}; own1m={own_ret:.6f}",
            )
        )

        last_signal_times[key] = now

    return signals


def mu_to_amd_signal(bars_1m, now):
    mu = bars_1m.get("MUBUSDT")
    amd = bars_1m.get("AMDBUSDT")

    if mu is None or amd is None:
        return []

    r_mu = last_return(mu, 1)
    r_amd = last_return(amd, 1)

    if not (
        r_mu > 0.005
        and r_mu - r_amd >= 0.001
    ):
        return []

    # Frozen v1 research definition:
    # Signal-close limit and no explicit signal cooldown.
    entry = float(amd["close"].iloc[-1])

    return [
        Signal(
            strategy="MU→AMD",
            symbol="AMDBUSDT",
            side="BUY",
            signal_time=now.isoformat(),
            limit_price=entry,
            expiry_seconds=60,
            exit_kind="time",
            time_exit_seconds=20 * 60,
            note=f"MU1m={r_mu:.6f}; AMD1m={r_amd:.6f}",
        )
    ]


def amd_to_mu_signal(bars_1m, now, last_signal_times):
    # Exact research bucket:
    # 08:00 <= UTC < 14:00
    if not (8 <= now.hour < 14):
        return []

    amd = bars_1m.get("AMDBUSDT")
    mu = bars_1m.get("MUBUSDT")

    if amd is None or mu is None:
        return []

    r_amd = last_return(amd, 3)
    r_mu = last_return(mu, 3)

    if not (
        r_amd > 0.0075
        and r_amd - r_mu >= 0.001
    ):
        return []

    key = "AMD→MU:MUBUSDT"
    previous = last_signal_times.get(key)

    if previous is not None:
        if (now - previous).total_seconds() < 600:
            return []

    entry = float(mu["close"].iloc[-1]) * (1 - 0.0003)

    last_signal_times[key] = now

    return [
        Signal(
            strategy="AMD→MU",
            symbol="MUBUSDT",
            side="BUY",
            signal_time=now.isoformat(),
            limit_price=entry,
            expiry_seconds=60,
            exit_kind="time",
            time_exit_seconds=20 * 60,
            note=f"AMD3m={r_amd:.6f}; MU3m={r_mu:.6f}",
        )
    ]


def breakout_retest_signals(bars_15m, now):
    if any(len(df) < 390 for df in bars_15m.values()):
        return []

    above_now = 0
    above_previous = 0

    ema96 = {}

    for symbol, df in bars_15m.items():
        e = ema(df["close"], 96)
        ema96[symbol] = e

        above_now += int(df["close"].iloc[-1] > e.iloc[-1])
        above_previous += int(df["close"].iloc[-2] > e.iloc[-2])

    signals = []

    for symbol, df in bars_15m.items():
        close = df["close"]
        high = df["high"]

        r3d_now = close.iloc[-1] / close.iloc[-289] - 1
        r3d_previous = close.iloc[-2] / close.iloc[-290] - 1

        prev_24h_high_now = high.iloc[-97:-1].max()
        prev_24h_high_previous = high.iloc[-98:-2].max()

        condition_now = (
            r3d_now > 0
            and above_now >= 5
            and close.iloc[-1] > prev_24h_high_now * 1.001
        )

        condition_previous = (
            r3d_previous > 0
            and above_previous >= 5
            and close.iloc[-2] > prev_24h_high_previous * 1.001
        )

        # Only trigger when entering the full condition.
        if not (condition_now and not condition_previous):
            continue

        atr = atr_wilder_ewm(
            df[["high", "low", "close"]],
            14,
        ).iloc[-1]

        if not np.isfinite(atr):
            continue

        signal_close = float(close.iloc[-1])
        entry = signal_close * (1 - 0.005)

        signals.append(
            Signal(
                strategy="Breakout",
                symbol=symbol,
                side="BUY",
                signal_time=now.isoformat(),
                limit_price=entry,
                expiry_seconds=30 * 60,
                exit_kind="atr",
                stop_price=entry - 0.5 * atr,
                take_profit_price=entry + 4 * atr,
                note=f"r3d={r3d_now:.6f}; breadth={above_now}/6",
            )
        )

    return signals


# ============================================================
# 8. MAIN TRADING ENGINE
# ============================================================

class TradingBot:
    def __init__(self):
        self.state = load_state()

        self.market = BinanceMarketData()

        self.roostoo = RoostooClient(
            api_key=os.getenv("ROOSTOO_API_KEY"),
            secret=os.getenv("ROOSTOO_API_SECRET"),
        )

        self.exchange_info = {}
        self.pair_map = dict(ROOSTOO_PAIR_MAP)

        self.bars_1m = {}
        self.bars_15m = {}

        self.last_signal_times = {
            key: pd.Timestamp(value)
            for key, value in self.state.get("last_signal_times", {}).items()
        }

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    def startup(self):
        LOGGER.info("Starting bot. LIVE_TRADING=%s", LIVE_TRADING)

        exchange = self.roostoo.exchange_info()
        self.exchange_info = exchange.get("TradePairs", {})

        self.auto_resolve_pairs()

        unresolved = [
            s for s in BINANCE_SYMBOLS
            if self.pair_map.get(s) not in self.exchange_info
        ]

        if unresolved:
            raise RuntimeError(
                f"Could not resolve Roostoo pairs for: {unresolved}"
            )

        LOGGER.info("Pair mapping: %s", self.pair_map)

        self.refresh_market_data()

        if LIVE_TRADING:
            if not os.getenv("ROOSTOO_API_KEY") or not os.getenv("ROOSTOO_API_SECRET"):
                raise RuntimeError("LIVE_TRADING=true but API credentials are missing.")

            self.startup_reconcile_guard()

    def auto_resolve_pairs(self):
        for binance_symbol in BINANCE_SYMBOLS:
            configured = self.pair_map.get(binance_symbol)

            if configured in self.exchange_info:
                continue

            base = (
                binance_symbol[:-4]
                if binance_symbol.endswith("USDT")
                else binance_symbol
            )

            candidates = [
                pair
                for pair, meta in self.exchange_info.items()
                if str(meta.get("Coin", "")).upper() == base.upper()
            ]

            if len(candidates) == 1:
                self.pair_map[binance_symbol] = candidates[0]

    def startup_reconcile_guard(self):
        balance = self.roostoo.balance()
        wallet = balance.get("Wallet", {})

        managed_coins = {
            self.exchange_info[self.pair_map[s]].get("Coin")
            for s in BINANCE_SYMBOLS
        }

        existing = []

        for coin in managed_coins:
            w = wallet.get(coin, {})
            qty = (
                float(w.get("Free", 0) or 0)
                + float(w.get("Lock", 0) or 0)
            )

            if qty > 0:
                existing.append((coin, qty))

        if existing and not self.state.get("positions"):
            raise RuntimeError(
                "Account has bStock holdings but local state has no managed positions. "
                f"Manual reconciliation required: {existing}"
            )

    # --------------------------------------------------------
    # Market data
    # --------------------------------------------------------

    def refresh_market_data(self):
        for symbol in BINANCE_SYMBOLS:
            df1 = self.market.klines(symbol, "1m", 500)
            df15 = self.market.klines(symbol, "15m", 500)

            self.bars_1m[symbol] = (
                self.market.completed_only(df1).reset_index(drop=True)
            )

            self.bars_15m[symbol] = (
                self.market.completed_only(df15).reset_index(drop=True)
            )

    # --------------------------------------------------------
    # Exchange helpers
    # --------------------------------------------------------

    def rules(self, symbol):
        pair = self.pair_map[symbol]
        return self.exchange_info[pair]

    @staticmethod
    def floor_precision(value, decimals):
        factor = 10 ** int(decimals)
        return math.floor(value * factor) / factor

    def round_price(self, symbol, price):
        decimals = int(self.rules(symbol)["PricePrecision"])
        return self.floor_precision(price, decimals)

    def round_qty(self, symbol, qty):
        decimals = int(self.rules(symbol)["AmountPrecision"])
        return self.floor_precision(qty, decimals)

    def ticker_all(self):
        result = self.roostoo.ticker()

        if not result.get("Success", True):
            raise RoostooError(result.get("ErrMsg", "Ticker failed."))

        return result.get("Data", {})

    def nav(self, tickers):
        if not LIVE_TRADING:
            return DRY_RUN_NAV

        balance = self.roostoo.balance()
        wallet = balance.get("Wallet", {})

        nav = 0.0

        usd = wallet.get("USD", {})
        nav += (
            float(usd.get("Free", 0) or 0)
            + float(usd.get("Lock", 0) or 0)
        )

        for symbol, pair in self.pair_map.items():
            coin = self.exchange_info[pair]["Coin"]
            w = wallet.get(coin, {})

            quantity = (
                float(w.get("Free", 0) or 0)
                + float(w.get("Lock", 0) or 0)
            )

            if quantity > 0:
                nav += quantity * float(tickers[pair]["LastPrice"])

        return nav

    # --------------------------------------------------------
    # Portfolio rules
    # --------------------------------------------------------

    def symbol_busy(self, symbol):
        if symbol in self.state["positions"]:
            return True

        return any(
            p.get("symbol") == symbol
            for p in self.state["pending"].values()
        )

    def slots_used(self):
        # Pending entries reserve a slot.
        return (
            len(self.state["positions"])
            + len(self.state["pending"])
        )

    # --------------------------------------------------------
    # Signal -> entry
    # --------------------------------------------------------

    def submit_signal(self, signal, tickers):
        append_csv(
            SIGNAL_LOG,
            {
                "time": pd.Timestamp.now(tz="UTC").isoformat(),
                "strategy": signal.strategy,
                "symbol": signal.symbol,
                "side": signal.side,
                "limit_price": signal.limit_price,
                "note": signal.note,
                "action": "SEEN",
            },
        )

        if self.symbol_busy(signal.symbol):
            LOGGER.info(
                "Skip %s %s: symbol already busy.",
                signal.strategy,
                signal.symbol,
            )
            return

        if self.slots_used() >= MAX_SLOTS:
            LOGGER.info(
                "Skip %s %s: max slots reached.",
                signal.strategy,
                signal.symbol,
            )
            return

        pair = self.pair_map[signal.symbol]

        roostoo_last = float(tickers[pair]["LastPrice"])

        divergence = abs(roostoo_last / signal.limit_price - 1)

        if divergence > MAX_PRICE_DIVERGENCE:
            LOGGER.warning(
                "Skip %s %s: Binance/Roostoo divergence %.3f%%",
                signal.strategy,
                signal.symbol,
                divergence * 100,
            )
            return

        account_nav = self.nav(tickers)
        target_value = account_nav * POSITION_FRACTION

        price = self.round_price(signal.symbol, signal.limit_price)
        quantity = self.round_qty(
            signal.symbol,
            target_value / price,
        )

        if quantity <= 0:
            return

        minimum = float(self.rules(signal.symbol).get("MiniOrder", 0))

        if price * quantity <= minimum:
            LOGGER.warning(
                "Skip %s: order is below MiniOrder.",
                signal.symbol,
            )
            return

        now = pd.Timestamp.now(tz="UTC")
        expires = now + pd.Timedelta(seconds=signal.expiry_seconds)

        if not LIVE_TRADING:
            order_id = f"DRY-{int(now.timestamp() * 1000)}-{signal.symbol}"

            LOGGER.info(
                "[DRY] LIMIT BUY %s qty=%s price=%s strategy=%s",
                pair,
                quantity,
                price,
                signal.strategy,
            )

        else:
            response = self.roostoo.place_order(
                pair=pair,
                side="BUY",
                quantity=quantity,
                order_type="LIMIT",
                price=price,
            )

            if not response.get("Success"):
                LOGGER.error("Order rejected: %s", response)
                return

            detail = response.get("OrderDetail", {})
            order_id = str(detail.get("OrderID"))

            if str(detail.get("Status", "")).upper() == "FILLED":
                fill_price = float(
                    detail.get("FilledAverPrice") or price
                )

                filled_qty = float(
                    detail.get("FilledQuantity") or quantity
                )

                self.create_position(
                    signal,
                    pair,
                    order_id,
                    filled_qty,
                    fill_price,
                    now,
                )
                return

        self.state["pending"][order_id] = {
            "strategy": signal.strategy,
            "symbol": signal.symbol,
            "roostoo_pair": pair,
            "order_id": order_id,
            "quantity": quantity,
            "limit_price": price,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "signal": asdict(signal),
        }

        self.save()

    # --------------------------------------------------------
    # Filled entry -> position
    # --------------------------------------------------------

    def create_position(
        self,
        signal,
        pair,
        order_id,
        quantity,
        fill_price,
        fill_time,
    ):
        stop = signal.stop_price
        take_profit = signal.take_profit_price

        if signal.exit_kind == "atr":
            intended = signal.limit_price

            stop_distance = intended - stop
            tp_distance = take_profit - intended

            stop = fill_price - stop_distance
            take_profit = fill_price + tp_distance

        exit_deadline = None

        if signal.exit_kind == "time":
            exit_deadline = (
                fill_time
                + pd.Timedelta(seconds=signal.time_exit_seconds)
            ).isoformat()

        self.state["positions"][signal.symbol] = {
            "strategy": signal.strategy,
            "symbol": signal.symbol,
            "roostoo_pair": pair,
            "quantity": quantity,
            "entry_price": fill_price,
            "entry_time": fill_time.isoformat(),
            "stop_price": stop,
            "take_profit_price": take_profit,
            "exit_deadline": exit_deadline,
            "entry_order_id": order_id,
        }

        append_csv(
            TRADE_LOG,
            {
                "time": fill_time.isoformat(),
                "event": "ENTRY_FILLED",
                "strategy": signal.strategy,
                "symbol": signal.symbol,
                "quantity": quantity,
                "price": fill_price,
                "order_id": order_id,
            },
        )

        LOGGER.info(
            "FILLED %s %s qty=%s price=%s",
            signal.strategy,
            signal.symbol,
            quantity,
            fill_price,
        )

        self.save()

    # --------------------------------------------------------
    # Pending orders
    # --------------------------------------------------------

    def manage_pending(self):
        if not self.state["pending"]:
            return

        now = pd.Timestamp.now(tz="UTC")

        for order_id, pending in list(self.state["pending"].items()):

            if order_id.startswith("DRY-"):
                if now >= pd.Timestamp(pending["expires_at"]):
                    LOGGER.info("[DRY] expire %s", order_id)
                    del self.state["pending"][order_id]
                continue

            result = self.roostoo.query_order(order_id=order_id)
            matched = result.get("OrderMatched", [])
            detail = matched[0] if matched else None

            if detail:
                status = str(detail.get("Status", "")).upper()

                if status == "FILLED":
                    signal = Signal(**pending["signal"])

                    fill_price = float(
                        detail.get("FilledAverPrice")
                        or pending["limit_price"]
                    )

                    filled_qty = float(
                        detail.get("FilledQuantity")
                        or pending["quantity"]
                    )

                    del self.state["pending"][order_id]

                    self.create_position(
                        signal,
                        pending["roostoo_pair"],
                        order_id,
                        filled_qty,
                        fill_price,
                        now,
                    )

                    continue

                if status == "CANCELED":
                    del self.state["pending"][order_id]
                    continue

            if now >= pd.Timestamp(pending["expires_at"]):
                self.roostoo.cancel_order(order_id=order_id)

                LOGGER.info(
                    "Canceled expired order %s",
                    order_id,
                )

                del self.state["pending"][order_id]

        self.save()

    # --------------------------------------------------------
    # Exit management
    # --------------------------------------------------------

    def manage_positions(self, tickers):
        now = pd.Timestamp.now(tz="UTC")

        for symbol, position in list(self.state["positions"].items()):
            pair = position["roostoo_pair"]
            price = float(tickers[pair]["LastPrice"])

            reason = None

            if (
                position.get("stop_price") is not None
                and price <= float(position["stop_price"])
            ):
                reason = "STOP"

            elif (
                position.get("take_profit_price") is not None
                and price >= float(position["take_profit_price"])
            ):
                reason = "TAKE_PROFIT"

            elif (
                position.get("exit_deadline")
                and now >= pd.Timestamp(position["exit_deadline"])
            ):
                reason = "TIME"

            if reason is None:
                continue

            quantity = self.round_qty(
                symbol,
                float(position["quantity"]),
            )

            if not LIVE_TRADING:
                LOGGER.info(
                    "[DRY] would MARKET SELL %s qty=%s reason=%s",
                    pair,
                    quantity,
                    reason,
                )
                continue

            response = self.roostoo.place_order(
                pair=pair,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
            )

            if not response.get("Success"):
                LOGGER.error(
                    "Exit failed for %s: %s",
                    symbol,
                    response,
                )
                continue

            detail = response.get("OrderDetail", {})
            fill_price = float(
                detail.get("FilledAverPrice") or price
            )

            append_csv(
                TRADE_LOG,
                {
                    "time": now.isoformat(),
                    "event": "EXIT_FILLED",
                    "strategy": position["strategy"],
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": fill_price,
                    "order_id": detail.get("OrderID"),
                    "reason": reason,
                },
            )

            LOGGER.info(
                "EXIT %s qty=%s price=%s reason=%s",
                symbol,
                quantity,
                fill_price,
                reason,
            )

            del self.state["positions"][symbol]
            self.save()

    # --------------------------------------------------------
    # Signal evaluation
    # --------------------------------------------------------

    def evaluate_1m(self):
        latest_bar = min(
            df["open_time"].iloc[-1]
            for df in self.bars_1m.values()
        )

        key = pd.Timestamp(latest_bar).isoformat()

        if self.state["last_processed_1m"] == key:
            return []

        self.state["last_processed_1m"] = key

        now = pd.Timestamp(latest_bar)

        signals = []

        signals += broad_leadlag_signals(
            self.bars_1m,
            now,
            self.last_signal_times,
        )

        signals += mu_to_amd_signal(
            self.bars_1m,
            now,
        )

        signals += amd_to_mu_signal(
            self.bars_1m,
            now,
            self.last_signal_times,
        )

        return signals

    def evaluate_15m(self):
        latest_bar = min(
            df["open_time"].iloc[-1]
            for df in self.bars_15m.values()
        )

        key = pd.Timestamp(latest_bar).isoformat()

        if self.state["last_processed_15m"] == key:
            return []

        self.state["last_processed_15m"] = key

        return breakout_retest_signals(
            self.bars_15m,
            pd.Timestamp(latest_bar),
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    def save(self):
        self.state["last_signal_times"] = {
            key: value.isoformat()
            for key, value in self.last_signal_times.items()
        }

        save_state(self.state)

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run_once(self):
        self.manage_pending()

        tickers = self.ticker_all()

        self.manage_positions(tickers)

        self.refresh_market_data()

        signals = (
            self.evaluate_1m()
            + self.evaluate_15m()
        )

        # Frozen portfolio priority.
        priority = {
            "BroadLeadLag": 0,
            "Breakout": 1,
            "MU→AMD": 2,
            "AMD→MU": 3,
        }

        signals.sort(
            key=lambda s: priority.get(s.strategy, 99)
        )

        for signal in signals:
            self.submit_signal(signal, tickers)

        self.save()

    def run_forever(self):
        self.startup()

        while True:
            try:
                self.run_once()

            except KeyboardInterrupt:
                LOGGER.info("Stopped by user.")
                break

            except Exception:
                LOGGER.exception("Loop error.")

            time.sleep(LOOP_SECONDS)


# ============================================================
# 9. DIAGNOSTIC MODE
# ============================================================

def run_diagnostics():
    print("=== BINANCE DATA CHECK ===")

    md = BinanceMarketData()

    for symbol in BINANCE_SYMBOLS:
        df = md.klines(symbol, "1m", 5)

        print(
            symbol,
            "rows=",
            len(df),
            "last_close=",
            None if df.empty else df["close"].iloc[-1],
        )

    print("\n=== ROOSTOO PUBLIC CHECK ===")

    client = RoostooClient(
        api_key=os.getenv("ROOSTOO_API_KEY"),
        secret=os.getenv("ROOSTOO_API_SECRET"),
    )

    print("server_time:", client.server_time())

    exchange = client.exchange_info()

    print("IsRunning:", exchange.get("IsRunning"))

    pairs = exchange.get("TradePairs", {})

    for symbol, pair in ROOSTOO_PAIR_MAP.items():
        print(
            symbol,
            "->",
            pair,
            "OK" if pair in pairs else "MISSING",
        )

    print("\n=== ROOSTOO SIGNED CHECK ===")

    if os.getenv("ROOSTOO_API_KEY") and os.getenv("ROOSTOO_API_SECRET"):
        balance = client.balance()
        print("balance endpoint Success:", balance.get("Success"))
    else:
        print("Skipped because API credentials are not set.")


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    # Usage:
    #   python roostoo_bot_single.py diagnostic
    #   python roostoo_bot_single.py
    #
    # Default mode is the trading loop.
    # LIVE_TRADING=false means no real/mock orders are sent.

    if len(sys.argv) >= 2 and sys.argv[1].lower() in {"diagnostic", "diagnostics", "test"}:
        run_diagnostics()
    else:
        TradingBot().run_forever()
