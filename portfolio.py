# portfolio.py
"""Portfolio-level risk layer.

Only handles what per-asset logic cannot:
  - a single shared cash/equity across all pairs,
  - wallet reconciliation (cash + positions) each loop,
  - a global total-exposure cap, and
  - a drawdown circuit breaker that halts NEW entries.

It never closes existing positions — exits stay with per-asset stop/TP/MA-cross.
"""
from config import MAX_TOTAL_EXPOSURE, DRAWDOWN_TRIGGER, DRAWDOWN_RECOVER
from state import PortfolioState


class PortfolioManager:
    def __init__(self, state: PortfolioState):
        self.state = state

    # ---------------- reconciliation ----------------

    def reconcile(self, wallet, short_positions, states):
        """Overwrite cash + positions from the exchange (source of truth)."""
        # Live API returns "SpotWallet"; older docs show "Wallet" — accept either.
        spot = (wallet.get("SpotWallet") or wallet.get("Wallet") or {}) if wallet else {}
        usd = spot.get("USD", {})

        # Cash = free USD + USD locked as short collateral.
        # (USD "Lock" bucket ignored: we only place market orders.)
        self.state.cash = (
            float(usd.get("Free", 0.0) or 0.0)
            + float(usd.get("ShortCollateral", 0.0) or 0.0)
        )

        # Long positions: SpotWallet[coin]["Free"].
        for pair, st in states.items():
            coin = pair.split("/")[0]
            coin_info = spot.get(coin, {})
            st.position = float(coin_info.get("Free", 0.0) or 0.0)

        # Short positions: /v6/short_positions (authoritative).
        positions = []
        if short_positions and short_positions.get("Success"):
            positions = short_positions.get("Positions", []) or []

        for st in states.values():
            st.short_position = 0.0
            st.short_collateral = 0.0
            st.short_entry_price = None

        for pos in positions:
            pair = pos.get("Pair") or pos.get("pair")
            if pair is None or pair not in states:
                continue
            st = states[pair]
            st.short_position = float(pos.get("ShortQty", 0.0) or 0.0)
            st.short_entry_price = float(pos.get("EntryPrice", 0.0) or 0.0)
            st.short_collateral = float(pos.get("Collateral", 0.0) or 0.0)

    # ---------------- equity ----------------

    def compute_equity(self, states, prices):
        """Equity = cash + long value + short unrealized PnL."""
        equity = self.state.cash
        for pair, st in states.items():
            price = prices.get(pair, 0.0)
            equity += st.position * price
            if st.short_position > 0 and st.short_entry_price is not None:
                equity += st.short_position * (st.short_entry_price - price)
        return equity

    def gross_exposure(self, states, prices):
        """Sum of absolute notional across all positions (long + short)."""
        gross = 0.0
        for pair, st in states.items():
            price = prices.get(pair, 0.0)
            gross += st.position * price
            if st.short_position > 0:
                gross += st.short_position * price
        return gross

    # ---------------- circuit breaker ----------------

    def update_breaker(self, equity):
        """Update peak and (de)activate the breaker via a hysteresis band."""
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        peak = self.state.peak_equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if not self.state.breaker_active and drawdown >= DRAWDOWN_TRIGGER:
            self.state.breaker_active = True
        elif self.state.breaker_active and drawdown <= DRAWDOWN_RECOVER:
            self.state.breaker_active = False
        return drawdown

    # ---------------- entry gating ----------------

    def allowed_entry_notional(self, planned_notional, states, prices, equity):
        """Return (allowed_notional, reason) for a planned new entry.

        `equity` is the loop-start portfolio equity (stable denominator for
        the exposure cap); `states`/`prices` reflect the running gross.
        """
        if self.state.breaker_active:
            return 0.0, "breaker_active"
        gross = self.gross_exposure(states, prices)
        remaining = MAX_TOTAL_EXPOSURE * equity - gross
        if remaining <= 0:
            return 0.0, "exposure_cap"
        return min(planned_notional, remaining), "ok"
