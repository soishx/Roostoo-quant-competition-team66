# strategy.py
import pandas as pd
from typing import Optional, Tuple


def calculate_moving_averages(price: pd.Series,
                              short_hours: int,
                              long_hours: int) -> Tuple[pd.Series, pd.Series]:
    if short_hours <= 0 or long_hours <= 0:
        raise ValueError("window must be > 0")
    if short_hours >= long_hours:
        raise ValueError("short must be < long")
    return (
        price.rolling(window=short_hours).mean(),
        price.rolling(window=long_hours).mean(),
    )


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


class MAStrategy:


    """
    Stateful wrapper around the backtested rule set.

    Each call to `evaluate(df, position, prev_short, prev_long)` returns
    one of:
        "entry"  -> golden cross, we are flat
        "exit"   -> death cross, we are long
        "stop"   -> price crossed the ATR stop (checked separately)
        "none"   -> hold
    """

    def __init__(self,
                 short_hours: int,
                 long_hours: int,
                 atr_period: int,
                 tp1_atr: float,
                 tp1_fraction: float,
                 tp2_atr: float,
                 tp2_fraction: float):
        self.short_hours = short_hours
        self.long_hours = long_hours
        self.atr_period = atr_period
        self.tp1_atr = tp1_atr
        self.tp1_fraction = tp1_fraction
        self.tp2_atr = tp2_atr
        self.tp2_fraction = tp2_fraction

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df with columns short_ma, long_ma, atr appended."""
        out = df.copy()
        out["short_ma"], out["long_ma"] = calculate_moving_averages(
            out["close"], self.short_hours, self.long_hours
        )
        out["atr"] = calculate_atr(out, self.atr_period)
        return out

    def evaluate(self,
                 df: pd.DataFrame,
                 in_position: bool,
                 prev_short_ma: Optional[float],
                 prev_long_ma: Optional[float]) -> dict:
        """
        Returns a dict with the decision + snapshot values for logging.
        Does NOT place orders.
        """
        ind = self.compute_indicators(df)
        if len(ind) < 2:
            return {"signal": "none", "reason": "insufficient_data",
                    "short_ma": None, "long_ma": None, "atr": None}

        cur_short = ind["short_ma"].iloc[-1]
        cur_long  = ind["long_ma"].iloc[-1]
        cur_atr   = ind["atr"].iloc[-1]

        snapshot = {
            "short_ma": cur_short,
            "long_ma":  cur_long,
            "atr":      cur_atr,
            "prev_short_ma": prev_short_ma,
            "prev_long_ma":  prev_long_ma,
        }

        if prev_short_ma is None or prev_long_ma is None:
            return {**snapshot, "signal": "none", "reason": "no_prev_ma"}

        prev_above = prev_short_ma > prev_long_ma
        cur_above  = cur_short > cur_long

        golden_cross = (not prev_above) and cur_above
        death_cross  = prev_above and (not cur_above)

        if in_position and death_cross:
            return {**snapshot, "signal": "exit", "reason": "death_cross"}

        if (not in_position) and golden_cross:
            return {**snapshot, "signal": "entry", "reason": "golden_cross"}

        return {**snapshot, "signal": "none", "reason": "hold"}

    def compute_position_size(self,
                              equity: float,
                              cash: float,
                              price: float,
                              atr: float,
                              risk_per_trade: float,
                              atr_stop_multiplier: float,
                              max_position_pct: float,
                              fee: float) -> Tuple[float, float]:
        """
        Returns (units, stop_price).
        Mirrors the backtest sizing exactly.
        """
        risk_budget = equity * risk_per_trade
        stop_distance = atr * atr_stop_multiplier
        risk_based_value = risk_budget / stop_distance * price
        target_value = min(risk_based_value, equity * max_position_pct)

        units = target_value / price
        max_affordable = cash / (price * (1 + fee))
        units = min(units, max_affordable)

        stop_price = price - stop_distance
        return units, stop_price

    def check_long_exits(self,
                         price: float,
                         atr_now: float,
                         state,   # PairState
                         ) -> dict:
        """
        Called every loop while a long is open (when LONG_TP_ENABLED).
        Decides which take-profit tier to trigger, using ATR frozen at entry.

            - tp1           (frozen at entry ATR)
            - tp2           (frozen at entry ATR)
            - none

        Stop-loss and death-cross exits are handled elsewhere.
        """
        if state.position <= 0:
            return {"action": "none"}

        # ATR used for take-profit targets: frozen at entry.
        atr_entry = state.long_atr_at_entry
        atr_for_tp = atr_entry if (atr_entry is not None and atr_entry > 0) else atr_now

        # 1) Take-profit tier 1 (frozen target)
        if (not state.long_tp1_done
                and state.entry_price is not None
                and atr_for_tp is not None and atr_for_tp > 0):
            tp1 = state.entry_price + atr_for_tp * self.tp1_atr
            if price >= tp1:
                return {"action": "tp1",
                        "price": price,
                        "target": tp1,
                        "close_fraction": self.tp1_fraction}

        # 2) Take-profit tier 2 (frozen target, only after TP1 done)
        if (state.long_tp1_done and not state.long_tp2_done
                and state.entry_price is not None
                and atr_for_tp is not None and atr_for_tp > 0):
            tp2 = state.entry_price + atr_for_tp * self.tp2_atr
            if price >= tp2:
                return {"action": "tp2",
                        "price": price,
                        "target": tp2,
                        "close_fraction": self.tp2_fraction}

        return {"action": "none"}


# =========================================================
# Short-side logic
# =========================================================

class MAShortStrategy:
    """
    Symmetric short signal on the same 24/48 SMA cross,
    but with asymmetric sizing and tiered take-profit.

    Entry : short_ma crosses below long_ma  (death cross)
    Exit  : short_ma crosses above long_ma  (golden cross)
            or stop-loss, or tiered take-profit
    """

    def __init__(self,
                 short_hours: int,
                 long_hours: int,
                 atr_period: int,
                 atr_multiplier: float,
                 risk_scale: float,
                 max_position_pct: float,
                 tp1_atr: float,
                 tp1_fraction: float,
                 tp2_atr: float,
                 tp2_fraction: float,
                 trail_activation_atr: float):
        self.short_hours = short_hours
        self.long_hours = long_hours
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_scale = risk_scale
        self.max_position_pct = max_position_pct
        self.tp1_atr = tp1_atr
        self.tp1_fraction = tp1_fraction
        self.tp2_atr = tp2_atr
        self.tp2_fraction = tp2_fraction
        self.trail_activation_atr = trail_activation_atr

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["short_ma"], out["long_ma"] = calculate_moving_averages(
            out["close"], self.short_hours, self.long_hours
        )
        out["atr"] = calculate_atr(out, self.atr_period)
        return out

    def evaluate(self,
                 df: pd.DataFrame,
                 in_short: bool,
                 prev_short_ma: Optional[float],
                 prev_long_ma: Optional[float]) -> dict:
        """
        Returns dict with signal in {"short_entry", "short_exit", "none"}
        plus indicator snapshot for logging.
        """
        ind = self.compute_indicators(df)
        if len(ind) < 2:
            return {"signal": "none", "reason": "insufficient_data",
                    "short_ma": None, "long_ma": None, "atr": None}

        cur_short = ind["short_ma"].iloc[-1]
        cur_long  = ind["long_ma"].iloc[-1]
        cur_atr   = ind["atr"].iloc[-1]

        snapshot = {
            "short_ma": cur_short,
            "long_ma":  cur_long,
            "atr":      cur_atr,
            "prev_short_ma": prev_short_ma,
            "prev_long_ma":  prev_long_ma,
        }

        if prev_short_ma is None or prev_long_ma is None:
            return {**snapshot, "signal": "none", "reason": "no_prev_ma"}

        prev_above = prev_short_ma > prev_long_ma
        cur_above  = cur_short > cur_long

        golden_cross = (not prev_above) and cur_above
        death_cross  = prev_above and (not cur_above)

        # Mirror of long side
        if in_short and golden_cross:
            return {**snapshot, "signal": "short_exit", "reason": "golden_cross"}

        if (not in_short) and death_cross:
            return {**snapshot, "signal": "short_entry", "reason": "death_cross"}

        return {**snapshot, "signal": "none", "reason": "hold"}

    def compute_short_size(self,
                           equity: float,
                           price: float,
                           atr: float,
                           base_risk_per_trade: float,
                           fee: float) -> dict:
        """
        Returns dict with:
            collateral   : USD to lock on Roostoo
            qty          : short quantity (collateral / price)
            stop_price   : initial stop (entry + ATR * mult)
            tp1_price    : first take-profit trigger
            tp2_price    : second take-profit trigger
        """
        risk_budget = equity * base_risk_per_trade * self.risk_scale
        stop_distance = atr * self.atr_multiplier

        # risk-based notional (same formula as long side, then scaled)
        risk_based_notional = risk_budget / stop_distance * price

        # hard cap
        cap_notional = equity * self.max_position_pct
        notional = min(risk_based_notional, cap_notional)

        # Roostoo caps collateral at available cash too; caller enforces
        qty = notional / price
        stop_price = price + stop_distance
        tp1_price  = price - atr * self.tp1_atr
        tp2_price  = price - atr * self.tp2_atr

        return {
            "collateral": notional,
            "qty": qty,
            "stop_price": stop_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "atr_at_entry": atr,
        }

    def check_short_exits(self,
                          price: float,
                          atr_now: float,
                          state,   # PairState
                          use_stop: bool = True
                          ) -> dict:
        """
        Called every loop while a short is open.
        Decides which (if any) action to take this bar:
            - stop_loss     (uses trailing stop, current ATR semantics)
            - tp1           (frozen at entry ATR)
            - tp2           (frozen at entry ATR)
            - trail_update  (uses current ATR, no order)
            - none
        """
        if state.short_position <= 0:
            return {"action": "none"}

        # ATR used for take-profit targets: frozen at entry.
        # Falls back to atr_now only if the field is missing (legacy state).
        atr_entry = state.short_atr_at_entry
        atr_for_tp = atr_entry if (atr_entry is not None and atr_entry > 0) else atr_now

        # 1) Stop-loss check first (highest priority).
        #    Stop is a trailing stop: it uses whatever price the stop was
        #    last updated to, NOT a recomputed ATR here.
        if use_stop and state.short_stop_armed and state.short_stop_price is not None:
            if price >= state.short_stop_price:
                return {"action": "stop_loss",
                        "price": price,
                        "stop": state.short_stop_price}

        # 2) Take-profit tier 1 (frozen target)
        if (not state.short_tp1_done
                and state.short_entry_price is not None
                and atr_for_tp is not None and atr_for_tp > 0):
            tp1 = state.short_entry_price - atr_for_tp * self.tp1_atr
            if price <= tp1:
                return {"action": "tp1",
                        "price": price,
                        "target": tp1,
                        "close_fraction": self.tp1_fraction}

        # 3) Take-profit tier 2 (frozen target, only after TP1 done)
        if (state.short_tp1_done and not state.short_tp2_done
                and state.short_entry_price is not None
                and atr_for_tp is not None and atr_for_tp > 0):
            tp2 = state.short_entry_price - atr_for_tp * self.tp2_atr
            if price <= tp2:
                return {"action": "tp2",
                        "price": price,
                        "target": tp2,
                        "close_fraction": self.tp2_fraction}

        # 4) Trailing stop update (uses CURRENT ATR, no order).
        #    This is intentionally different from TP targets: the trailing
        #    stop must reflect today's volatility around today's price.
        if (state.short_entry_price is not None
                and atr_now is not None and atr_now > 0):
            profit_atr = (state.short_entry_price - price) / atr_now
            if profit_atr >= self.trail_activation_atr:
                new_stop = price + atr_now * self.atr_multiplier
                # For a short, "tighter" means lower. Only move DOWN.
                if (state.short_stop_price is None
                        or new_stop < state.short_stop_price):
                    return {"action": "trail_update",
                            "new_stop": new_stop,
                            "price": price}

        return {"action": "none"}