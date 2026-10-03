# run.py
import time
import traceback
from datetime import datetime, timezone

from config import (
    PAIRS, SHORT_MA_HOURS, LONG_MA_HOURS, ATR_PERIOD,
    RISK_PER_TRADE, ATR_STOP_MULTIPLIER, MAX_POSITION_PCT,
    TAKER_FEE, LOOP_INTERVAL_SEC, KLINE_INTERVAL, KLINE_LIMIT,
    DRY_RUN, BINANCE_SYMBOL_MAP, USE_ATR_STOP_LONG, USE_ATR_STOP_SHORT, 
    # short
    SHORT_ENABLED, SHORT_RISK_SCALE, SHORT_MAX_POSITION_PCT,
    SHORT_ATR_MULTIPLIER, SHORT_TP1_ATR, SHORT_TP1_FRACTION,
    SHORT_TP2_ATR, SHORT_TP2_FRACTION, SHORT_TRAIL_ACTIVATION_ATR,
)
from exchange_client import RoostooClient
from market_data import BinanceDataClient
from strategy import MAStrategy, MAShortStrategy
from state import PairState
from trade_logger import TradeLogger


def main():
    roostoo = RoostooClient()
    binance = BinanceDataClient()

    st = roostoo.check_server_time()
    print(f"[startup] Roostoo server time: {st}")

    info = roostoo.get_exchange_info()
    if not info or not info.get("IsRunning"):
        raise RuntimeError(f"Roostoo exchange not running: {info}")
    trade_pairs = info["TradePairs"]
    print(f"[startup] Roostoo pairs: {list(trade_pairs.keys())}")

    for p in PAIRS:
        if p not in trade_pairs:
            raise RuntimeError(f"{p} not available on Roostoo")
        if p not in BINANCE_SYMBOL_MAP:
            raise RuntimeError(f"{p} has no Binance mapping")

    long_strats = {
        p: MAStrategy(SHORT_MA_HOURS, LONG_MA_HOURS, ATR_PERIOD) for p in PAIRS
    }
    short_strats = {
        p: MAShortStrategy(
            short_hours=SHORT_MA_HOURS,
            long_hours=LONG_MA_HOURS,
            atr_period=ATR_PERIOD,
            atr_multiplier=SHORT_ATR_MULTIPLIER,
            risk_scale=SHORT_RISK_SCALE,
            max_position_pct=SHORT_MAX_POSITION_PCT,
            tp1_atr=SHORT_TP1_ATR,
            tp1_fraction=SHORT_TP1_FRACTION,
            tp2_atr=SHORT_TP2_ATR,
            tp2_fraction=SHORT_TP2_FRACTION,
            trail_activation_atr=SHORT_TRAIL_ACTIVATION_ATR,
        )
        for p in PAIRS
    }
    states = {p: PairState.load(p) for p in PAIRS}
    logger = TradeLogger(log_dir="logs")

    while True:
        loop_start = time.time()
        for pair in PAIRS:
            try:
                process_pair(
                    pair=pair,
                    roostoo=roostoo,
                    binance=binance,
                    long_strategy=long_strats[pair],
                    short_strategy=short_strats[pair],
                    state=states[pair],
                    pair_info=trade_pairs[pair],
                    logger=logger,
                )
            except Exception as e:
                logger.log_error(str(e), context={
                    "pair": pair, "trace": traceback.format_exc()
                })
                print(f"[{pair}] unhandled error: {e}")

        elapsed = time.time() - loop_start
        time.sleep(max(0, LOOP_INTERVAL_SEC - elapsed))


def process_pair(pair, roostoo, binance, long_strategy, short_strategy,
                 state, pair_info, logger):

    # ---------- 1. Binance klines ----------
    df = binance.get_klines(pair, interval=KLINE_INTERVAL, limit=KLINE_LIMIT)
    min_bars = LONG_MA_HOURS + ATR_PERIOD + 2
    if df is None or len(df) < min_bars:
        logger.log_decision(
            symbol=pair, short_ma=None, long_ma=None,
            prev_short_ma=state.prev_short_ma,
            prev_long_ma=state.prev_long_ma,
            atr=None, position=state.position, cash=state.cash,
            equity=state.cash, signal="none", action="no_data",
            extra={"bars": 0 if df is None else len(df),
                   "min_bars": min_bars},
        )
        return

    # ---------- 2. Roostoo ticker ----------
    ticker_resp = roostoo.get_ticker(pair)
    if not ticker_resp or not ticker_resp.get("Success"):
        logger.log_error("ticker fetch failed",
                         context={"pair": pair, "resp": ticker_resp})
        return
    last_price = float(ticker_resp["Data"][pair]["LastPrice"])

    # ==================================================
    # 3. LONG SIDE (unchanged from before)
    # ==================================================
    long_decision = long_strategy.evaluate(
        df,
        in_position=state.in_position,
        prev_short_ma=state.prev_short_ma,
        prev_long_ma=state.prev_long_ma,
    )
    long_signal = long_decision["signal"]
    cur_short_ma = long_decision["short_ma"]
    cur_long_ma  = long_decision["long_ma"]
    cur_atr      = long_decision["atr"]

    # Long stop check
    if (USE_ATR_STOP_LONG
            and state.in_position
            and state.stop_armed
            and state.stop_price is not None):
        if last_price <= state.stop_price:
            long_signal = "stop"
            long_decision["reason"] = "atr_stop"

    equity = compute_equity(state, last_price)
    long_action = "hold"
    long_order_id = None

    if long_signal in ("exit", "stop") and state.in_position:
        qty = round_to_precision(state.position, pair_info["AmountPrecision"])
        if qty > 0:
            resp = _place_market(roostoo, pair, "SELL", qty, last_price,
                                 logger, DRY_RUN)
            if resp and resp.get("Success"):
                detail = resp["OrderDetail"]
                fill = float(detail["FilledAverPrice"])
                filled = float(detail["FilledQuantity"])
                fee_paid = float(detail.get("CommissionChargeValue", 0) or 0)
                eq_before = equity
                state.cash += filled * fill - fee_paid
                state.position = 0.0
                state.entry_price = None
                state.stop_price = None
                state.stop_armed = False
                eq_after = state.cash
                long_action = "place_order"
                long_order_id = detail["OrderID"]
                logger.log_trade(
                    symbol=pair, side="sell", price=fill, quantity=filled,
                    fee=fee_paid, order_id=long_order_id,
                    signal_reason=long_decision["reason"],
                    equity_before=eq_before, equity_after=eq_after,
                    position_after=0.0,
                )
                state.save()
                equity = compute_equity(state, last_price)

    elif (long_signal == "entry" and not state.in_position
          and cur_atr and state.cash > 0):
        units, stop_price = long_strategy.compute_position_size(
            equity=equity, cash=state.cash, price=last_price, atr=cur_atr,
            risk_per_trade=RISK_PER_TRADE,
            atr_stop_multiplier=ATR_STOP_MULTIPLIER,
            max_position_pct=MAX_POSITION_PCT,
            fee=TAKER_FEE,
        )
        qty = round_to_precision(units, pair_info["AmountPrecision"])
        if qty <= 0 or qty * last_price < pair_info["MiniOrder"]:
            long_action = "skip_min_order"
        else:
            resp = _place_market(roostoo, pair, "BUY", qty, last_price,
                                 logger, DRY_RUN)
            if resp and resp.get("Success"):
                detail = resp["OrderDetail"]
                fill = float(detail["FilledAverPrice"])
                filled = float(detail["FilledQuantity"])
                fee_paid = float(detail.get("CommissionChargeValue", 0) or 0)
                eq_before = equity
                state.cash -= filled * fill + fee_paid
                state.position = filled
                state.entry_price = fill
                state.stop_price = stop_price if USE_ATR_STOP_LONG else None
                state.stop_armed = USE_ATR_STOP_LONG
                state.stop_armed = True
                eq_after = state.cash + filled * fill
                long_action = "place_order"
                long_order_id = detail["OrderID"]
                logger.log_trade(
                    symbol=pair, side="buy", price=fill, quantity=filled,
                    fee=fee_paid, order_id=long_order_id,
                    signal_reason=long_decision["reason"],
                    equity_before=eq_before, equity_after=eq_after,
                    position_after=filled, stop_price=stop_price,
                )
                state.save()
                equity = compute_equity(state, last_price)

    # ==================================================
    # 4. SHORT SIDE
    # ==================================================
    short_action = "hold"
    short_order_id = None
    short_signal = "none"
    short_decision = {"reason": "disabled"}

    if SHORT_ENABLED:
        short_decision = short_strategy.evaluate(
            df,
            in_short=state.in_short,
            prev_short_ma=state.prev_short_ma,
            prev_long_ma=state.prev_long_ma,
        )
        short_signal = short_decision["signal"]

        # ---- 4a. Exits while in short: stop / tp1 / tp2 / trail ----
        if state.in_short:
            ex = short_strategy.check_short_exits(
                price=last_price, atr_now=cur_atr or 0.0, state=state, use_stop=USE_ATR_STOP_SHORT,
            )
            act = ex["action"]

            if act == "stop_loss":
                resp = roostoo.short_close(pair) if not DRY_RUN else _dry_short_close(pair, state, last_price)
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, logger,
                                       reason="atr_stop", last_price=last_price)
                    short_action = "short_stop"
                    short_order_id = resp.get("ID") or resp.get("order_id")
                    state.save()

            elif act == "tp1":
                close_pct = SHORT_TP1_FRACTION * 100.0   # 40.0
                resp = (roostoo.short_close(pair, close_pct=close_pct)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price, close_pct))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, logger,
                                       reason="tp1", last_price=last_price)
                    state.short_tp1_done = True
                    # tighten stop to breakeven after TP1
                    if (state.short_entry_price is not None
                            and state.short_atr_at_entry is not None):
                        state.short_stop_price = (
                            state.short_entry_price
                            - 0.3 * state.short_atr_at_entry
                        )
                    short_action = "short_tp1"
                    short_order_id = resp.get("ID")
                    state.save()
                    equity = compute_equity(state, last_price)

            elif act == "tp2":
                # close 50% of remaining (which = 30% of original)
                remaining_frac = SHORT_TP2_FRACTION / max(
                    1e-9, (1.0 - SHORT_TP1_FRACTION)
                ) * 100.0  # = 50.0
                resp = (roostoo.short_close(pair, close_pct=remaining_frac)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price, remaining_frac))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, logger,
                                       reason="tp2", last_price=last_price)
                    state.short_tp2_done = True
                    short_action = "short_tp2"
                    short_order_id = resp.get("ID")
                    state.save()
                    equity = compute_equity(state, last_price)

            elif act == "trail_update":
                state.short_stop_price = ex["new_stop"]
                short_action = "trail_update"
                state.save()
                equity = compute_equity(state, last_price)

            # Signal exit (golden cross) only if no stop/tp fired this bar
            if short_action == "hold" and short_signal == "short_exit":
                resp = (roostoo.short_close(pair)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, logger,
                                       reason="golden_cross", last_price=last_price)
                    short_action = "short_exit"
                    short_order_id = resp.get("ID")
                    state.save()
                    equity = compute_equity(state, last_price)

        # ---- 4b. Entry ----
        elif (short_signal == "short_entry"
              and not state.in_short
              and cur_atr
              and state.cash > 0):

            sizing = short_strategy.compute_short_size(
                equity=equity, price=last_price, atr=cur_atr,
                base_risk_per_trade=RISK_PER_TRADE, fee=TAKER_FEE,
            )
            collateral = sizing["collateral"]
            # never exceed available cash
            collateral = min(collateral, state.cash)
            # enforce Roostoo minimum collateral ($1)
            if collateral < 1.0:
                short_action = "skip_min_collateral"
            else:
                resp = (roostoo.short_open(pair, collateral=collateral)
                        if not DRY_RUN
                        else _dry_short_open(pair, collateral, last_price))
                if resp and resp.get("Success"):
                    entry = float(resp.get("EntryPrice", last_price))
                    qty   = float(resp.get("ShortQty", collateral / entry))
                    state.cash -= collateral + float(resp.get("OpenFee", 0) or 0)
                    state.short_position = qty
                    state.short_entry_price = entry
                    state.short_collateral = collateral
                    state.short_original_qty = qty
                    state.short_atr_at_entry = cur_atr
                    state.short_stop_price = entry + cur_atr * SHORT_ATR_MULTIPLIER if USE_ATR_STOP_SHORT else None
                    state.short_stop_armed = USE_ATR_STOP_SHORT
                    state.short_tp1_done = False
                    state.short_tp2_done = False
                    short_action = "short_entry"
                    short_order_id = resp.get("ID")
                    logger.log_trade(
                        symbol=pair, side="SHORT_OPEN", price=entry, quantity=qty,
                        fee=float(resp.get("OpenFee", 0) or 0),
                        order_id=str(short_order_id),
                        signal_reason=short_decision["reason"],
                        equity_before=equity,
                        equity_after=state.cash,       # collateral locked
                        position_after=-qty,
                        stop_price=state.short_stop_price,
                    )
                    state.save()

    # ==================================================
    # 5. Persist MA snapshot + heartbeat
    # ==================================================
    state.prev_short_ma = float(cur_short_ma) if cur_short_ma is not None else None
    state.prev_long_ma  = float(cur_long_ma)  if cur_long_ma  is not None else None
    state.save()

    logger.log_decision(
        symbol=pair,
        short_ma=cur_short_ma, long_ma=cur_long_ma,
        prev_short_ma=long_decision["prev_short_ma"],
        prev_long_ma=long_decision["prev_long_ma"],
        atr=cur_atr, position=state.position, cash=state.cash,
        equity=equity,
        signal=f"L:{long_signal}|S:{short_signal}",
        action=f"L:{long_action}|S:{short_action}",
        order_id=long_order_id or short_order_id,
        extra={
            "long_reason": long_decision["reason"],
            "short_reason": short_decision.get("reason"),
            "last_price_roostoo": last_price,
            "last_close_binance": float(df["close"].iloc[-1]),
            "short_position": state.short_position,
            "short_stop_price": state.short_stop_price,
            "short_atr_at_entry": state.short_atr_at_entry,
            "short_entry_price": state.short_entry_price,
        },
    )


# =========================================================
# Helpers
# =========================================================

def compute_equity(state, last_price):
    """Equity = cash + long value + short collateral + short unrealized PnL."""
    long_value = state.position * last_price
    short_pnl = 0.0
    if state.short_position > 0 and state.short_entry_price is not None:
        short_pnl = state.short_position * (state.short_entry_price - last_price)
    return state.cash + long_value + state.short_collateral + short_pnl

def _apply_short_close(pair, resp, state, logger, reason, last_price):
    """Update state after a short_close response."""
    closed_qty  = float(resp.get("ClosedQty", 0))
    close_price = float(resp.get("ClosePrice", last_price))
    close_fee   = float(resp.get("CloseFee", 0) or 0)
    return_amount = float(resp.get("ReturnAmount", 0))
    fully_closed = bool(resp.get("FullyClosed", False))

    eq_before = state.cash + state.short_position * close_price
    state.cash += return_amount
    if fully_closed:
        state.short_position = 0.0
        state.short_entry_price = None
        state.short_stop_price = None
        state.short_stop_armed = False
        state.short_collateral = 0.0
        state.short_original_qty = 0.0
        state.short_tp1_done = False
        state.short_tp2_done = False
    else:
        state.short_position -= closed_qty
        state.short_collateral -= closed_qty * close_price

    eq_after = state.cash + state.short_position * close_price

    logger.log_trade(
        symbol=pair, side="SHORT_CLOSE", price=close_price,
        quantity=closed_qty, fee=close_fee,
        order_id=str(resp.get("ID", "")),
        signal_reason=reason,
        equity_before=eq_before, equity_after=eq_after,
        position_after=-state.short_position,
        stop_price=state.short_stop_price,
    )


def _dry_short_open(pair, collateral, price):
    qty = collateral / price
    return {
        "Success": True,
        "ID": f"dry-{int(time.time())}",
        "EntryPrice": price,
        "ShortQty": qty,
        "Collateral": collateral,
        "OpenFee": collateral * TAKER_FEE,
        "Status": "OPEN",
    }


def _dry_short_close(pair, state, price, close_pct=None):
    qty = state.short_position
    if close_pct is not None:
        qty = state.short_position * close_pct / 100.0
    fully = abs(qty - state.short_position) < 1e-12
    return {
        "Success": True,
        "ID": f"dry-{int(time.time())}",
        "ClosePrice": price,
        "ClosedQty": qty,
        "CloseFee": qty * price * TAKER_FEE,
        "ReturnAmount": qty * price,
        "FullyClosed": fully,
    }


def _place_market(roostoo, pair, side, qty, ref_price, logger, dry_run):
    if dry_run:
        print(f"[DRY_RUN] {side} {qty:.8f} {pair} @ ~{ref_price}")
        return {
            "Success": True,
            "OrderDetail": {
                "OrderID": f"dry-{int(time.time())}",
                "FilledAverPrice": ref_price,
                "FilledQuantity": qty,
                "CommissionChargeValue": qty * ref_price * TAKER_FEE,
            },
        }
    resp = roostoo.place_order(pair, side, qty, order_type="MARKET")
    if not resp or not resp.get("Success"):
        logger.log_error(f"{side} order failed",
                         context={"pair": pair, "qty": qty, "resp": resp})
        return None
    return resp


def round_to_precision(value: float, precision: int) -> float:
    factor = 10 ** precision
    return int(value * factor) / factor


if __name__ == "__main__":
    main()