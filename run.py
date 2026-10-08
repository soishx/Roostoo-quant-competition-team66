# run.py
import time
import traceback
from datetime import datetime, timezone

from config import (
    MA_PAIRS, ALL_PAIRS,
    SHORT_MA_HOURS, LONG_MA_HOURS, ATR_PERIOD,
    RISK_PER_TRADE, ATR_STOP_MULTIPLIER, MAX_POSITION_PCT,
    TAKER_FEE, LOOP_INTERVAL_SEC, LEADLAG_LOOP_SECONDS, LEADLAG_ENABLED,
    KLINE_INTERVAL, KLINE_LIMIT,
    DRY_RUN, BINANCE_SYMBOL_MAP, USE_ATR_STOP_LONG, USE_ATR_STOP_SHORT,
    # long take-profit
    LONG_TP_ENABLED, LONG_TP1_ATR, LONG_TP1_FRACTION,
    LONG_TP2_ATR, LONG_TP2_FRACTION,
    # short
    SHORT_ENABLED, SHORT_RISK_SCALE, SHORT_MAX_POSITION_PCT,
    SHORT_ATR_MULTIPLIER, SHORT_TP1_ATR, SHORT_TP1_FRACTION,
    SHORT_TP2_ATR, SHORT_TP2_FRACTION, SHORT_TRAIL_ACTIVATION_ATR,
)
from exchange_client import RoostooClient
from market_data import BinanceDataClient
from strategy import MAStrategy, MAShortStrategy
from state import PairState, PortfolioState, LeadLagState
from portfolio import PortfolioManager
from lead_lag_strategy import LeadLagTrader
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

    for p in ALL_PAIRS:
        if p not in trade_pairs:
            raise RuntimeError(f"{p} not available on Roostoo")
        if p not in BINANCE_SYMBOL_MAP:
            raise RuntimeError(f"{p} has no Binance mapping")

    long_strats = {
        p: MAStrategy(SHORT_MA_HOURS, LONG_MA_HOURS, ATR_PERIOD,
                      LONG_TP1_ATR, LONG_TP1_FRACTION,
                      LONG_TP2_ATR, LONG_TP2_FRACTION)
        for p in MA_PAIRS
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
        for p in MA_PAIRS
    }
    states = {p: PairState.load(p) for p in ALL_PAIRS}
    portfolio = PortfolioState.load()
    portfolio_mgr = PortfolioManager(portfolio)
    leadlag_state = LeadLagState.load()
    logger = TradeLogger(log_dir="logs")
    leadlag = LeadLagTrader(roostoo, binance, portfolio_mgr, logger, leadlag_state, trade_pairs)

    last_ma_run = 0.0

    while True:
        try:
            loop_start = time.time()

            # ---- Portfolio-level: reconcile cash+positions, equity, breaker ----
            try:
                wallet = roostoo.get_balance()
                short_positions = roostoo.get_short_positions()
                ticker_resp = roostoo.get_ticker()
                if not ticker_resp or not ticker_resp.get("Success"):
                    raise RuntimeError(f"ticker fetch failed: {ticker_resp}")
                ticker_data = ticker_resp["Data"]
                prices = {p: float(ticker_data[p]["LastPrice"]) for p in ALL_PAIRS}

                portfolio_mgr.reconcile(wallet, short_positions, states)
                equity = portfolio_mgr.compute_equity(states, prices)
                drawdown = portfolio_mgr.update_breaker(equity)
                if portfolio.breaker_active:
                    print(f"[portfolio] breaker ACTIVE  equity={equity:.2f} dd={drawdown:.4f}")
            except Exception as e:
                logger.log_error("portfolio reconcile failed",
                                 context={"err": str(e), "trace": traceback.format_exc()})
                print(f"[portfolio] reconcile failed: {e}")
                portfolio.save()
                leadlag_state.save()
                time.sleep(max(0, LEADLAG_LOOP_SECONDS - (time.time() - loop_start)))
                continue

            # ---- Lead-lag pool (every loop, ~5s) ----
            try:
                leadlag.run_once(ticker_data, prices, states, equity, allow_entries=LEADLAG_ENABLED)
            except Exception as e:
                logger.log_error("leadlag pool error",
                                 context={"err": str(e), "trace": traceback.format_exc()})
                print(f"[leadlag] unhandled error: {e}")

            # ---- MA pool (every LOOP_INTERVAL_SEC, ~300s) ----
            if time.time() - last_ma_run >= LOOP_INTERVAL_SEC:
                last_ma_run = time.time()
                for pair in MA_PAIRS:
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
                            portfolio=portfolio,
                            portfolio_mgr=portfolio_mgr,
                            equity=equity,
                            last_price=prices[pair],
                            states=states,
                            prices=prices,
                        )
                    except Exception as e:
                        logger.log_error(str(e), context={
                            "pair": pair, "trace": traceback.format_exc()
                        })
                        print(f"[{pair}] unhandled error: {e}")

            portfolio.save()
            leadlag_state.save()
            elapsed = time.time() - loop_start
            time.sleep(max(0, LEADLAG_LOOP_SECONDS - elapsed))
        except KeyboardInterrupt:
            portfolio.save()
            leadlag_state.save()
            print("[bot] stopped by user")
            break


def process_pair(pair, roostoo, binance, long_strategy, short_strategy,
                 state, pair_info, logger, portfolio, portfolio_mgr,
                 equity, last_price, states, prices):

    # ---------- 1. Binance klines ----------
    df = binance.get_klines(pair, interval=KLINE_INTERVAL, limit=KLINE_LIMIT)
    min_bars = LONG_MA_HOURS + ATR_PERIOD + 2
    if df is None or len(df) < min_bars:
        logger.log_decision(
            symbol=pair, short_ma=None, long_ma=None,
            prev_short_ma=state.prev_short_ma,
            prev_long_ma=state.prev_long_ma,
            atr=None, position=state.position, cash=portfolio.cash,
            equity=equity, signal="none", action="no_data",
            extra={"bars": 0 if df is None else len(df),
                   "min_bars": min_bars},
        )
        return

    # ==================================================
    # 2. LONG SIDE
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

    long_action = "hold"
    long_order_id = None

    # Long take-profit (TP1/TP2), checked before the signal exit (tp-first).
    if LONG_TP_ENABLED and state.in_position:
        ex = long_strategy.check_long_exits(
            price=last_price, atr_now=cur_atr or 0.0, state=state
        )
        act = ex["action"]
        if act in ("tp1", "tp2"):
            # close_fraction is a fraction of the ORIGINAL long qty
            qty = round_to_precision(
                state.long_original_qty * ex["close_fraction"],
                pair_info["AmountPrecision"],
            )
            qty = min(qty, state.position)
            if qty > 0:
                resp = _place_market(roostoo, pair, "SELL", qty, last_price,
                                     logger, DRY_RUN)
                if resp and resp.get("Success"):
                    detail = resp["OrderDetail"]
                    fill = float(detail["FilledAverPrice"])
                    filled = float(detail["FilledQuantity"])
                    fee_paid = float(detail.get("CommissionChargeValue", 0) or 0)
                    eq_before = portfolio_mgr.compute_equity(states, prices)
                    portfolio.cash += filled * fill - fee_paid
                    state.position -= filled
                    if act == "tp1":
                        state.long_tp1_done = True
                    else:
                        state.long_tp2_done = True
                    eq_after = portfolio_mgr.compute_equity(states, prices)
                    long_action = "long_tp1" if act == "tp1" else "long_tp2"
                    long_order_id = detail["OrderID"]
                    logger.log_trade(
                        symbol=pair, side="sell", price=fill, quantity=filled,
                        fee=fee_paid, order_id=long_order_id,
                        signal_reason=act,
                        equity_before=eq_before, equity_after=eq_after,
                        position_after=state.position,
                    )
                    state.save()

    if long_signal in ("exit", "stop") and state.in_position and long_action == "hold":
        qty = round_to_precision(state.position, pair_info["AmountPrecision"])
        if qty > 0:
            resp = _place_market(roostoo, pair, "SELL", qty, last_price,
                                 logger, DRY_RUN)
            if resp and resp.get("Success"):
                detail = resp["OrderDetail"]
                fill = float(detail["FilledAverPrice"])
                filled = float(detail["FilledQuantity"])
                fee_paid = float(detail.get("CommissionChargeValue", 0) or 0)
                eq_before = portfolio_mgr.compute_equity(states, prices)
                portfolio.cash += filled * fill - fee_paid
                state.position = 0.0
                state.entry_price = None
                state.stop_price = None
                state.stop_armed = False
                state.long_atr_at_entry = None
                state.long_original_qty = 0.0
                state.long_tp1_done = False
                state.long_tp2_done = False
                eq_after = portfolio_mgr.compute_equity(states, prices)
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

    # ==================================================
    # 3. SHORT SIDE
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

        # ---- 3a. Exits while in short: stop / tp1 / tp2 / trail ----
        if state.in_short:
            ex = short_strategy.check_short_exits(
                price=last_price, atr_now=cur_atr or 0.0, state=state, use_stop=USE_ATR_STOP_SHORT,
            )
            act = ex["action"]

            if act == "stop_loss":
                resp = roostoo.short_close(pair) if not DRY_RUN else _dry_short_close(pair, state, last_price)
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, portfolio, logger,
                                       reason="atr_stop", last_price=last_price)
                    short_action = "short_stop"
                    short_order_id = resp.get("ID") or resp.get("order_id")
                    state.save()

            elif act == "tp1":
                close_pct = SHORT_TP1_FRACTION * 100.0
                resp = (roostoo.short_close(pair, close_pct=close_pct)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price, close_pct))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, portfolio, logger,
                                       reason="tp1", last_price=last_price)
                    state.short_tp1_done = True
                    if (state.short_entry_price is not None
                            and state.short_atr_at_entry is not None):
                        state.short_stop_price = (
                            state.short_entry_price
                            - 0.3 * state.short_atr_at_entry
                        )
                    short_action = "short_tp1"
                    short_order_id = resp.get("ID")
                    state.save()

            elif act == "tp2":
                remaining_frac = SHORT_TP2_FRACTION / max(
                    1e-9, (1.0 - SHORT_TP1_FRACTION)
                ) * 100.0
                resp = (roostoo.short_close(pair, close_pct=remaining_frac)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price, remaining_frac))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, portfolio, logger,
                                       reason="tp2", last_price=last_price)
                    state.short_tp2_done = True
                    short_action = "short_tp2"
                    short_order_id = resp.get("ID")
                    state.save()

            elif act == "trail_update":
                state.short_stop_price = ex["new_stop"]
                short_action = "trail_update"
                state.save()

            # Signal exit (golden cross) only if no stop/tp fired this bar
            if short_action == "hold" and short_signal == "short_exit":
                resp = (roostoo.short_close(pair)
                        if not DRY_RUN
                        else _dry_short_close(pair, state, last_price))
                if resp and resp.get("Success"):
                    _apply_short_close(pair, resp, state, portfolio, logger,
                                       reason="golden_cross", last_price=last_price)
                    short_action = "short_exit"
                    short_order_id = resp.get("ID")
                    state.save()

        # ---- 3b. Entry ----
        elif (short_signal == "short_entry"
              and not state.in_short
              and cur_atr
              and portfolio.cash > 0):

            sizing = short_strategy.compute_short_size(
                equity=equity, price=last_price, atr=cur_atr,
                base_risk_per_trade=RISK_PER_TRADE, fee=TAKER_FEE,
            )
            collateral = sizing["collateral"]
            allowed_notional, gate_reason = portfolio_mgr.allowed_entry_notional(
                collateral, states, prices, equity
            )
            if allowed_notional <= 0:
                short_action = "skip_" + gate_reason
            else:
                collateral = min(allowed_notional, collateral, portfolio.cash)
                if collateral < 1.0:
                    short_action = "skip_min_collateral"
                else:
                    resp = (roostoo.short_open(pair, collateral=collateral)
                            if not DRY_RUN
                            else _dry_short_open(pair, collateral, last_price))
                    if resp and resp.get("Success"):
                        entry = float(resp.get("EntryPrice", last_price))
                        qty   = float(resp.get("ShortQty", collateral / entry))
                        portfolio.cash -= collateral + float(resp.get("OpenFee", 0) or 0)
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
                        eq_after = portfolio_mgr.compute_equity(states, prices)
                        logger.log_trade(
                            symbol=pair, side="SHORT_OPEN", price=entry, quantity=qty,
                            fee=float(resp.get("OpenFee", 0) or 0),
                            order_id=str(short_order_id),
                            signal_reason=short_decision["reason"],
                            equity_before=portfolio_mgr.compute_equity(states, prices),
                            equity_after=eq_after,
                            position_after=-qty,
                            stop_price=state.short_stop_price,
                        )
                        state.save()

    # ---- Long entry (golden cross) — after exits, so flips are clean ----
    if (long_signal == "entry" and not state.in_position
            and cur_atr and portfolio.cash > 0):
        units, stop_price = long_strategy.compute_position_size(
            equity=equity, cash=portfolio.cash, price=last_price, atr=cur_atr,
            risk_per_trade=RISK_PER_TRADE,
            atr_stop_multiplier=ATR_STOP_MULTIPLIER,
            max_position_pct=MAX_POSITION_PCT,
            fee=TAKER_FEE,
        )
        planned_notional = units * last_price
        allowed_notional, gate_reason = portfolio_mgr.allowed_entry_notional(
            planned_notional, states, prices, equity
        )
        if allowed_notional <= 0:
            long_action = "skip_" + gate_reason
        else:
            units = allowed_notional / last_price
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
                    eq_before = portfolio_mgr.compute_equity(states, prices)
                    portfolio.cash -= filled * fill + fee_paid
                    state.position = filled
                    state.entry_price = fill
                    state.stop_price = stop_price if USE_ATR_STOP_LONG else None
                    state.stop_armed = USE_ATR_STOP_LONG
                    state.long_atr_at_entry = cur_atr
                    state.long_original_qty = filled
                    state.long_tp1_done = False
                    state.long_tp2_done = False
                    eq_after = portfolio_mgr.compute_equity(states, prices)
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

    # ==================================================
    # 4. Persist MA snapshot + heartbeat
    # ==================================================
    state.prev_short_ma = float(cur_short_ma) if cur_short_ma is not None else None
    state.prev_long_ma  = float(cur_long_ma)  if cur_long_ma  is not None else None
    state.save()

    logger.log_decision(
        symbol=pair,
        short_ma=cur_short_ma, long_ma=cur_long_ma,
        prev_short_ma=long_decision["prev_short_ma"],
        prev_long_ma=long_decision["prev_long_ma"],
        atr=cur_atr, position=state.position, cash=portfolio.cash,
        equity=portfolio_mgr.compute_equity(states, prices),
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
            "long_atr_at_entry": state.long_atr_at_entry,
            "long_original_qty": state.long_original_qty,
            "breaker_active": portfolio.breaker_active,
            "peak_equity": portfolio.peak_equity,
        },
    )


# =========================================================
# Helpers
# =========================================================

def _apply_short_close(pair, resp, state, portfolio, logger, reason, last_price):
    """Update state after a short_close response."""
    closed_qty  = float(resp.get("ClosedQty", 0))
    close_price = float(resp.get("ClosePrice", last_price))
    close_fee   = float(resp.get("CloseFee", 0) or 0)
    return_amount = float(resp.get("ReturnAmount", 0))
    fully_closed = bool(resp.get("FullyClosed", False))

    eq_before = portfolio.cash + state.short_position * close_price
    portfolio.cash += return_amount
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
        if "RemainingCollateral" in resp:
            state.short_collateral = float(resp.get("RemainingCollateral", 0) or 0)
        else:
            state.short_collateral -= closed_qty * close_price

    eq_after = portfolio.cash + state.short_position * close_price

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
