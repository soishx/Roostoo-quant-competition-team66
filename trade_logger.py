# trade_logger.py
import csv
import json
import os
from datetime import datetime, timezone, timedelta
from threading import Lock


# Log timestamps are rendered in Hong Kong Time (UTC+8) for readability.
# Trading logic elsewhere continues to use UTC.
HKT = timezone(timedelta(hours=8))


def _now_hkt_iso() -> str:
    """Current time as ISO-8601 string in Hong Kong Time, e.g. 2026-10-03T17:20:16.609057+08:00."""
    return datetime.now(HKT).isoformat()


class TradeLogger:
    """
    Append-only trade and decision logger.

    Two files:
      - decisions.jsonl : one line per loop iteration (heartbeat + signal snapshot)
      - trades.csv      : one row per fill

    Thread-safe via a lock, in case you ever run async.
    """

    def __init__(self, log_dir="logs", run_id=None):
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = run_id or datetime.now(HKT).strftime("%Y%m%dT%H%M%S+08")
        self.decisions_path = os.path.join(log_dir, f"decisions_{self.run_id}.jsonl")
        self.trades_path    = os.path.join(log_dir, f"trades_{self.run_id}.csv")
        self._lock = Lock()
        self._init_trades_file()

    def _init_trades_file(self):
        if not os.path.exists(self.trades_path):
            with open(self.trades_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_hkt", "symbol", "side", "price", "quantity",
                    "notional", "fee", "slippage_bps", "order_id",
                    "signal_reason", "equity_before", "equity_after",
                    "position_after", "stop_price", "note",
                ])

    # --------------------------------------------------
    # Decision heartbeat (every loop iteration)
    # --------------------------------------------------
    def log_decision(
        self,
        symbol,
        short_ma,
        long_ma,
        prev_short_ma,
        prev_long_ma,
        atr,
        position,
        cash,
        equity,
        signal,
        action,
        order_id=None,
        extra=None,
    ):
        row = {
            "ts": _now_hkt_iso(),
            "run_id": self.run_id,
            "symbol": symbol,
            "short_ma": _f(short_ma),
            "long_ma": _f(long_ma),
            "prev_short_ma": _f(prev_short_ma),
            "prev_long_ma": _f(prev_long_ma),
            "atr": _f(atr),
            "position": _f(position),
            "cash": _f(cash),
            "equity": _f(equity),
            "signal": signal,
            "action": action,
            "order_id": order_id,
        }
        if extra:
            row.update(extra)

        with self._lock:
            with open(self.decisions_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    # --------------------------------------------------
    # Fill record (only when an order fills)
    # --------------------------------------------------
    def log_trade(
        self,
        symbol,
        side,
        price,
        quantity,
        fee,
        order_id,
        signal_reason,
        equity_before,
        equity_after,
        position_after,
        stop_price=None,
        slippage_bps=None,
        note=None,
    ):
        notional = price * quantity
        row = [
            _now_hkt_iso(),
            symbol,
            side,
            f"{price:.8f}",
            f"{quantity:.8f}",
            f"{notional:.8f}",
            f"{fee:.8f}",
            f"{slippage_bps:.4f}" if slippage_bps is not None else "",
            order_id,
            signal_reason,
            f"{equity_before:.2f}",
            f"{equity_after:.2f}",
            f"{position_after:.8f}",
            f"{stop_price:.8f}" if stop_price is not None else "",
            note or "",
        ]
        with self._lock:
            with open(self.trades_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    # --------------------------------------------------
    # Generic error/event log
    # --------------------------------------------------
    def log_error(self, message, context=None):
        row = {
            "ts": _now_hkt_iso(),
            "run_id": self.run_id,
            "level": "ERROR",
            "message": message,
            "context": context or {},
        }
        with self._lock:
            with open(self.decisions_path, "a") as f:
                f.write(json.dumps(row) + "\n")


def _f(x):
    """Safe float formatting: None stays None."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
