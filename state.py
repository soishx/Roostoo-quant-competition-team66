# state.py
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

from config import STATE_DIR, INITIAL_CASH


@dataclass
class PairState:
    """Per-pair runtime state. Persisted to disk every loop.

    Cash no longer lives here — it belongs to PortfolioState.
    Positions are reconciled from the exchange wallet each loop.
    """
    pair: str

    # ---- long side ----
    position: float = 0.0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_armed: bool = False
    long_atr_at_entry: Optional[float] = None
    long_original_qty: float = 0.0
    long_tp1_done: bool = False
    long_tp2_done: bool = False

    # ---- short side ----
    short_position: float = 0.0            # qty currently shorted (positive number)
    short_entry_price: Optional[float] = None
    short_stop_price: Optional[float] = None
    short_stop_armed: bool = False
    short_collateral: float = 0.0          # USD locked as collateral on Roostoo
    short_original_qty: float = 0.0        # qty at entry, used for TP fractions
    short_atr_at_entry: Optional[float] = None
    short_tp1_done: bool = False
    short_tp2_done: bool = False

    # ---- shared ----
    prev_short_ma: Optional[float] = None
    prev_long_ma: Optional[float] = None
    last_decision_ts: Optional[str] = None

    @property
    def in_position(self) -> bool:
        return self.position > 0

    @property
    def in_short(self) -> bool:
        return self.short_position > 0

    def save(self):
        path = os.path.join(STATE_DIR, f"state_{self.pair.replace('/', '')}.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, pair: str) -> "PairState":
        path = os.path.join(STATE_DIR, f"state_{pair.replace('/', '')}.json")
        if not os.path.exists(path):
            return cls(pair=pair)
        with open(path) as f:
            data = json.load(f)
        # Backward-compatible: ignore unknown fields (e.g. legacy `cash`)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class PortfolioState:
    """Single shared account across all pairs."""
    cash: float = INITIAL_CASH
    peak_equity: float = INITIAL_CASH
    breaker_active: bool = False

    def save(self):
        path = os.path.join(STATE_DIR, "state_PORTFOLIO.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "PortfolioState":
        path = os.path.join(STATE_DIR, "state_PORTFOLIO.json")
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class LeadLagState:
    """State for the lead-lag (teammate) strategy pool.

    Pending orders and rich position metadata (stop / TP / time-exit) live
    here. The *quantity* held is mirrored into PairState.position via the
    shared wallet reconciliation, so the PortfolioManager always sees it.

    Persisted to logs/state/state_LEADLAG.json.
    """
    pending: dict = field(default_factory=dict)
    positions: dict = field(default_factory=dict)
    last_signal_times: dict = field(default_factory=dict)
    last_processed_1m: Optional[str] = None
    last_processed_15m: Optional[str] = None

    def save(self):
        path = os.path.join(STATE_DIR, "state_LEADLAG.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "LeadLagState":
        path = os.path.join(STATE_DIR, "state_LEADLAG.json")
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
