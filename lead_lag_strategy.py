# lead_lag_strategy.py
"""Lead-lag / breadth-breakout strategy (teammate's bStock pool).

Ported from roostoo_bot_single.py and refactored to:
  - use the shared RoostooClient + BinanceDataClient (dependency injection),
  - size against the shared PortfolioManager equity (NOT a fixed DRY_RUN_NAV),
  - log through TradeLogger, and
  - persist a LeadLagState in the same format as the rest of the bot.

All four signals are long-only: entries are LIMIT BUY, exits are MARKET SELL.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DRY_RUN,
    MAKER_FEE,
    TAKER_FEE,
    LEADLAG_PAIRS,
    LEADLAG_POSITION_FRACTION,
    LEADLAG_MAX_SLOTS,
    LEADLAG_MAX_PRICE_DIVERGENCE,
    LEADLAG_KLINE_LIMIT,
    LEADLAG_BLL_EMA_SPAN, LEADLAG_BLL_PEER_RET, LEADLAG_BLL_LAG,
    LEADLAG_BLL_MIN_PEERS, LEADLAG_BLL_MIN_BARS, LEADLAG_BLL_COOLDOWN,
    LEADLAG_BLL_ENTRY_OFFSET, LEADLAG_BLL_ATR_PERIOD,
    LEADLAG_BLL_STOP_ATR, LEADLAG_BLL_TP_ATR, LEADLAG_BLL_EXPIRY,
    LEADLAG_MU2AMD_RET, LEADLAG_MU2AMD_LAG, LEADLAG_MU2AMD_ENTRY_OFFSET,
    LEADLAG_MU2AMD_TIME_EXIT, LEADLAG_MU2AMD_EXPIRY,
    LEADLAG_AMD2MU_WINDOW_START, LEADLAG_AMD2MU_WINDOW_END,
    LEADLAG_AMD2MU_RET, LEADLAG_AMD2MU_RET_BARS, LEADLAG_AMD2MU_LAG,
    LEADLAG_AMD2MU_COOLDOWN, LEADLAG_AMD2MU_ENTRY_OFFSET,
    LEADLAG_AMD2MU_TIME_EXIT, LEADLAG_AMD2MU_EXPIRY,
    LEADLAG_BO_EMA_SPAN, LEADLAG_BO_MIN_BARS, LEADLAG_BO_BREADTH,
    LEADLAG_BO_R3D_BARS, LEADLAG_BO_BREAK_MULT, LEADLAG_BO_ATR_PERIOD,
    LEADLAG_BO_ENTRY_OFFSET, LEADLAG_BO_STOP_ATR, LEADLAG_BO_TP_ATR,
    LEADLAG_BO_EXPIRY,
)

# Roostoo pair shorthands used by the MU<->AMD pair signals.
MU_PAIR = "MUB/USD"
AMD_PAIR = "AMDB/USD"


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


# ---------------- indicators ----------------

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


def atr_sma(df, period):
    return true_range(df).rolling(period).mean()


def atr_wilder_ewm(df, period):
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def resample_5m_from_1m(df):
    # `df` is already indexed by open_time (DatetimeIndex).
    x = df[["open", "high", "low", "close", "volume"]]
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


# ---------------- signals ----------------

def broad_leadlag_signals(bars_1m, now, last_signal_times):
    signals = []

    returns = {
        s: last_return(df, 1)
        for s, df in bars_1m.items()
        if len(df) >= LEADLAG_BLL_MIN_BARS
    }

    for target in LEADLAG_PAIRS:
        if target not in returns:
            continue

        peers = [
            s for s in LEADLAG_PAIRS
            if s != target and s in returns and np.isfinite(returns[s])
        ]
        if len(peers) < LEADLAG_BLL_MIN_PEERS:
            continue

        peer_ret = float(np.mean([returns[s] for s in peers]))
        own_ret = returns[target]

        df = bars_1m[target]
        e60 = ema(df["close"], LEADLAG_BLL_EMA_SPAN).iloc[-1]

        condition = (
            peer_ret > LEADLAG_BLL_PEER_RET
            and peer_ret - own_ret >= LEADLAG_BLL_LAG
            and df["close"].iloc[-1] > e60
        )
        if not condition:
            continue

        key = f"BroadLeadLag:{target}"
        previous = last_signal_times.get(key)
        if previous is not None:
            if (now - previous).total_seconds() < LEADLAG_BLL_COOLDOWN:
                continue

        bars_5m = resample_5m_from_1m(df)
        if len(bars_5m) < 15:
            continue
        atr = atr_sma(bars_5m, LEADLAG_BLL_ATR_PERIOD).iloc[-1]
        if not np.isfinite(atr):
            continue

        signal_close = float(df["close"].iloc[-1])
        entry = signal_close * (1 - LEADLAG_BLL_ENTRY_OFFSET)

        signals.append(
            Signal(
                strategy="BroadLeadLag",
                symbol=target,
                side="BUY",
                signal_time=now.isoformat(),
                limit_price=entry,
                expiry_seconds=LEADLAG_BLL_EXPIRY,
                exit_kind="atr",
                stop_price=entry - LEADLAG_BLL_STOP_ATR * atr,
                take_profit_price=entry + LEADLAG_BLL_TP_ATR * atr,
                note=f"peer1m={peer_ret:.6f}; own1m={own_ret:.6f}",
            )
        )
        last_signal_times[key] = now

    return signals


def mu_to_amd_signal(bars_1m, now):
    mu = bars_1m.get(MU_PAIR)
    amd = bars_1m.get(AMD_PAIR)
    if mu is None or amd is None:
        return []

    r_mu = last_return(mu, 1)
    r_amd = last_return(amd, 1)

    if not (r_mu > LEADLAG_MU2AMD_RET and r_mu - r_amd >= LEADLAG_MU2AMD_LAG):
        return []

    # Frozen v1: signal-close limit (offset 0), no cooldown.
    entry = float(amd["close"].iloc[-1]) * (1 - LEADLAG_MU2AMD_ENTRY_OFFSET)

    return [
        Signal(
            strategy="MU→AMD",
            symbol=AMD_PAIR,
            side="BUY",
            signal_time=now.isoformat(),
            limit_price=entry,
            expiry_seconds=LEADLAG_MU2AMD_EXPIRY,
            exit_kind="time",
            time_exit_seconds=LEADLAG_MU2AMD_TIME_EXIT,
            note=f"MU1m={r_mu:.6f}; AMD1m={r_amd:.6f}",
        )
    ]


def amd_to_mu_signal(bars_1m, now, last_signal_times):
    if not (LEADLAG_AMD2MU_WINDOW_START <= now.hour < LEADLAG_AMD2MU_WINDOW_END):
        return []

    amd = bars_1m.get(AMD_PAIR)
    mu = bars_1m.get(MU_PAIR)
    if amd is None or mu is None:
        return []

    r_amd = last_return(amd, LEADLAG_AMD2MU_RET_BARS)
    r_mu = last_return(mu, LEADLAG_AMD2MU_RET_BARS)

    if not (r_amd > LEADLAG_AMD2MU_RET and r_amd - r_mu >= LEADLAG_AMD2MU_LAG):
        return []

    key = f"AMD→MU:{MU_PAIR}"
    previous = last_signal_times.get(key)
    if previous is not None:
        if (now - previous).total_seconds() < LEADLAG_AMD2MU_COOLDOWN:
            return []

    entry = float(mu["close"].iloc[-1]) * (1 - LEADLAG_AMD2MU_ENTRY_OFFSET)
    last_signal_times[key] = now

    return [
        Signal(
            strategy="AMD→MU",
            symbol=MU_PAIR,
            side="BUY",
            signal_time=now.isoformat(),
            limit_price=entry,
            expiry_seconds=LEADLAG_AMD2MU_EXPIRY,
            exit_kind="time",
            time_exit_seconds=LEADLAG_AMD2MU_TIME_EXIT,
            note=f"AMD3m={r_amd:.6f}; MU3m={r_mu:.6f}",
        )
    ]


def breakout_retest_signals(bars_15m, now):
    if any(len(df) < LEADLAG_BO_MIN_BARS for df in bars_15m.values()):
        return []

    above_now = 0
    above_previous = 0
    ema96 = {}

    for symbol, df in bars_15m.items():
        e = ema(df["close"], LEADLAG_BO_EMA_SPAN)
        ema96[symbol] = e
        above_now += int(df["close"].iloc[-1] > e.iloc[-1])
        above_previous += int(df["close"].iloc[-2] > e.iloc[-2])

    signals = []

    for symbol, df in bars_15m.items():
        close = df["close"]
        high = df["high"]

        r3d_now = close.iloc[-1] / close.iloc[-LEADLAG_BO_R3D_BARS] - 1
        r3d_previous = close.iloc[-2] / close.iloc[-LEADLAG_BO_R3D_BARS - 1] - 1

        # previous 24h high = max high over the prior 96 bars (96 * 15m = 24h)
        prev_24h_high_now = high.iloc[-97:-1].max()
        prev_24h_high_previous = high.iloc[-98:-2].max()

        condition_now = (
            r3d_now > 0
            and above_now >= LEADLAG_BO_BREADTH
            and close.iloc[-1] > prev_24h_high_now * LEADLAG_BO_BREAK_MULT
        )
        condition_previous = (
            r3d_previous > 0
            and above_previous >= LEADLAG_BO_BREADTH
            and close.iloc[-2] > prev_24h_high_previous * LEADLAG_BO_BREAK_MULT
        )

        # Only trigger when entering the full condition.
        if not (condition_now and not condition_previous):
            continue

        atr = atr_wilder_ewm(
            df[["high", "low", "close"]], LEADLAG_BO_ATR_PERIOD
        ).iloc[-1]
        if not np.isfinite(atr):
            continue

        signal_close = float(close.iloc[-1])
        entry = signal_close * (1 - LEADLAG_BO_ENTRY_OFFSET)

        signals.append(
            Signal(
                strategy="Breakout",
                symbol=symbol,
                side="BUY",
                signal_time=now.isoformat(),
                limit_price=entry,
                expiry_seconds=LEADLAG_BO_EXPIRY,
                exit_kind="atr",
                stop_price=entry - LEADLAG_BO_STOP_ATR * atr,
                take_profit_price=entry + LEADLAG_BO_TP_ATR * atr,
                note=f"r3d={r3d_now:.6f}; breadth={above_now}/{len(LEADLAG_PAIRS)}",
            )
        )

    return signals


# ---------------- trading engine ----------------

class LeadLagTrader:
    """Runs the lead-lag pool. Dependencies are injected; NAV comes from the
    shared PortfolioManager (never a fixed per-pool constant)."""

    def __init__(self, roostoo, binance, portfolio_mgr, logger, state, exchange_info):
        self.roostoo = roostoo
        self.binance = binance
        self.portfolio_mgr = portfolio_mgr
        self.logger = logger
        self.state = state  # LeadLagState
        self.exchange_info = exchange_info

        self.bars_1m = {}
        self.bars_15m = {}
        self.last_signal_times = {
            key: pd.Timestamp(value)
            for key, value in self.state.last_signal_times.items()
        }

        # Per-loop context, set in run_once().
        self.tickers = {}
        self.prices = {}
        self.states = {}
        self.equity = 0.0

    # ---- exchange rule helpers ----
    def rules(self, pair):
        return self.exchange_info[pair]

    @staticmethod
    def floor_precision(value, decimals):
        factor = 10 ** int(decimals)
        return math.floor(value * factor) / factor

    def round_price(self, pair, price):
        return self.floor_precision(price, int(self.rules(pair)["PricePrecision"]))

    def round_qty(self, pair, qty):
        return self.floor_precision(qty, int(self.rules(pair)["AmountPrecision"]))

    # ---- portfolio rules ----
    def symbol_busy(self, symbol):
        if symbol in self.state.positions:
            return True
        return any(p.get("symbol") == symbol for p in self.state.pending.values())

    def slots_used(self):
        return len(self.state.positions) + len(self.state.pending)

    # ---- data ----
    def refresh_market_data(self):
        for pair in LEADLAG_PAIRS:
            self.bars_1m[pair] = self.binance.get_klines(pair, "1m", LEADLAG_KLINE_LIMIT)
            self.bars_15m[pair] = self.binance.get_klines(pair, "15m", LEADLAG_KLINE_LIMIT)

    # ---- signal -> entry ----
    def submit_signal(self, signal):
        st = self.states.get(signal.symbol)

        self.logger.log_decision(
            symbol=signal.symbol,
            short_ma=None, long_ma=None,
            prev_short_ma=None, prev_long_ma=None,
            atr=None,
            position=st.position if st is not None else 0.0,
            cash=self.portfolio_mgr.state.cash,
            equity=self.equity,
            signal=f"LL:{signal.strategy}",
            action="SEEN",
            order_id=None,
            extra={"limit_price": signal.limit_price, "note": signal.note},
        )

        if self.symbol_busy(signal.symbol):
            return
        if self.slots_used() >= LEADLAG_MAX_SLOTS:
            return

        roostoo_last = float(self.tickers[signal.symbol]["LastPrice"])
        divergence = abs(roostoo_last / signal.limit_price - 1)
        if divergence > LEADLAG_MAX_PRICE_DIVERGENCE:
            return

        target_value = self.equity * LEADLAG_POSITION_FRACTION
        allowed, reason = self.portfolio_mgr.allowed_entry_notional(
            target_value, self.states, self.prices, self.equity
        )
        if allowed <= 0:
            self.logger.log_decision(
                symbol=signal.symbol,
                short_ma=None, long_ma=None,
                prev_short_ma=None, prev_long_ma=None,
                atr=None, position=0.0,
                cash=self.portfolio_mgr.state.cash, equity=self.equity,
                signal=f"LL:{signal.strategy}", action=f"skip_{reason}",
                order_id=None,
            )
            return

        price = self.round_price(signal.symbol, signal.limit_price)
        quantity = self.round_qty(signal.symbol, min(target_value, allowed) / price)
        if quantity <= 0:
            return

        minimum = float(self.rules(signal.symbol).get("MiniOrder", 0))
        if price * quantity <= minimum:
            return

        now = pd.Timestamp.now(tz="UTC")
        expires = now + pd.Timedelta(seconds=signal.expiry_seconds)

        if DRY_RUN:
            order_id = f"DRY-{int(now.timestamp() * 1000)}-{signal.symbol}"
            self.create_position(signal, signal.symbol, order_id, quantity, price, now)
            return

        response = self.roostoo.place_order(
            pair=signal.symbol,
            side="BUY",
            quantity=quantity,
            order_type="LIMIT",
            price=price,
        )
        if not response.get("Success"):
            self.logger.log_error("leadlag order rejected",
                                  context={"pair": signal.symbol, "resp": response})
            return

        detail = response.get("OrderDetail", {})
        order_id = str(detail.get("OrderID"))

        if str(detail.get("Status", "")).upper() == "FILLED":
            fill_price = float(detail.get("FilledAverPrice") or price)
            filled_qty = float(detail.get("FilledQuantity") or quantity)
            self.create_position(signal, signal.symbol, order_id, filled_qty, fill_price, now)
            return

        self.state.pending[order_id] = {
            "strategy": signal.strategy,
            "symbol": signal.symbol,
            "roostoo_pair": signal.symbol,
            "order_id": order_id,
            "quantity": quantity,
            "limit_price": price,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
            "signal": asdict(signal),
        }
        self.save()

    # ---- filled entry -> position ----
    def create_position(self, signal, pair, order_id, quantity, fill_price, fill_time):
        stop = signal.stop_price
        take_profit = signal.take_profit_price

        if signal.exit_kind == "atr":
            intended = signal.limit_price
            stop = fill_price - (intended - stop)
            take_profit = fill_price + (take_profit - intended)

        exit_deadline = None
        if signal.exit_kind == "time":
            exit_deadline = (
                fill_time + pd.Timedelta(seconds=signal.time_exit_seconds)
            ).isoformat()

        self.state.positions[signal.symbol] = {
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

        # Mirror the held quantity into the shared portfolio view immediately,
        # so equity/exposure reflect it without waiting for next-loop reconcile.
        if pair in self.states:
            self.states[pair].position = quantity

        self.logger.log_trade(
            symbol=pair, side="buy", price=fill_price, quantity=quantity,
            fee=quantity * fill_price * MAKER_FEE, order_id=order_id, signal_reason=signal.strategy,
            equity_before=self.equity, equity_after=self.equity,
            position_after=quantity, note=signal.note,
        )
        self.save()

    # ---- pending orders ----
    def manage_pending(self):
        if not self.state.pending:
            return

        now = pd.Timestamp.now(tz="UTC")

        for order_id, pending in list(self.state.pending.items()):
            if order_id.startswith("DRY-"):
                if now >= pd.Timestamp(pending["expires_at"]):
                    del self.state.pending[order_id]
                continue

            result = self.roostoo.query_order(order_id=order_id)
            matched = result.get("OrderMatched", []) if result else []
            detail = matched[0] if matched else None

            if detail:
                status = str(detail.get("Status", "")).upper()

                if status == "FILLED":
                    signal = Signal(**pending["signal"])
                    fill_price = float(detail.get("FilledAverPrice") or pending["limit_price"])
                    filled_qty = float(detail.get("FilledQuantity") or pending["quantity"])
                    del self.state.pending[order_id]
                    self.create_position(
                        signal, pending["roostoo_pair"], order_id,
                        filled_qty, fill_price, now,
                    )
                    continue

                if status == "CANCELED":
                    del self.state.pending[order_id]
                    continue

            if now >= pd.Timestamp(pending["expires_at"]):
                self.roostoo.cancel_order(order_id=order_id)
                del self.state.pending[order_id]

        self.save()

    # ---- exits ----
    def manage_positions(self):
        now = pd.Timestamp.now(tz="UTC")

        for symbol, position in list(self.state.positions.items()):
            pair = position["roostoo_pair"]
            price = float(self.tickers[pair]["LastPrice"])

            reason = None
            if position.get("stop_price") is not None and price <= float(position["stop_price"]):
                reason = "STOP"
            elif position.get("take_profit_price") is not None and price >= float(position["take_profit_price"]):
                reason = "TAKE_PROFIT"
            elif position.get("exit_deadline") and now >= pd.Timestamp(position["exit_deadline"]):
                reason = "TIME"

            if reason is None:
                continue

            quantity = self.round_qty(symbol, float(position["quantity"]))

            if DRY_RUN:
                self.logger.log_trade(
                    symbol=pair, side="sell", price=price, quantity=quantity,
                    fee=quantity * price * TAKER_FEE, order_id=f"DRY-{int(now.timestamp() * 1000)}",
                    signal_reason=reason, equity_before=self.equity,
                    equity_after=self.equity, position_after=0.0,
                    note=position["strategy"],
                )
            else:
                response = self.roostoo.place_order(
                    pair=pair, side="SELL", quantity=quantity, order_type="MARKET",
                )
                if not response.get("Success"):
                    self.logger.log_error("leadlag exit failed",
                                          context={"pair": pair, "resp": response})
                    continue
                detail = response.get("OrderDetail", {})
                fill_price = float(detail.get("FilledAverPrice") or price)
                self.logger.log_trade(
                    symbol=pair, side="sell", price=fill_price, quantity=quantity,
                    fee=quantity * fill_price * TAKER_FEE, order_id=str(detail.get("OrderID")),
                    signal_reason=reason, equity_before=self.equity,
                    equity_after=self.equity, position_after=0.0,
                    note=position["strategy"],
                )

            if pair in self.states:
                self.states[pair].position = 0.0
            del self.state.positions[symbol]
            self.save()

    # ---- signal evaluation (dedup per bar) ----
    def evaluate_1m(self):
        if not self.bars_1m:
            return []
        latest = min(df.index[-1] for df in self.bars_1m.values())
        key = pd.Timestamp(latest).isoformat()
        if self.state.last_processed_1m == key:
            return []
        self.state.last_processed_1m = key

        now = pd.Timestamp(latest)
        signals = []
        signals += broad_leadlag_signals(self.bars_1m, now, self.last_signal_times)
        signals += mu_to_amd_signal(self.bars_1m, now)
        signals += amd_to_mu_signal(self.bars_1m, now, self.last_signal_times)
        return signals

    def evaluate_15m(self):
        if not self.bars_15m:
            return []
        latest = min(df.index[-1] for df in self.bars_15m.values())
        key = pd.Timestamp(latest).isoformat()
        if self.state.last_processed_15m == key:
            return []
        self.state.last_processed_15m = key
        return breakout_retest_signals(self.bars_15m, pd.Timestamp(latest))

    # ---- state ----
    def save(self):
        self.state.last_signal_times = {
            key: value.isoformat() for key, value in self.last_signal_times.items()
        }
        self.state.save()

    # ---- per-loop entry point ----
    def run_once(self, tickers, prices, states, equity):
        self.tickers = tickers
        self.prices = prices
        self.states = states
        self.equity = equity

        self.manage_pending()
        self.manage_positions()
        self.refresh_market_data()

        signals = self.evaluate_1m() + self.evaluate_15m()

        priority = {"BroadLeadLag": 0, "Breakout": 1, "MU→AMD": 2, "AMD→MU": 3}
        signals.sort(key=lambda s: priority.get(s.strategy, 99))

        for signal in signals:
            self.submit_signal(signal)

        self.save()
