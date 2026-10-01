import numpy as np
import pandas as pd
import vectorbt as vbt


# =========================================================
# Configuration
# =========================================================

DATA_FILE = "ETHUSDT_1h_2023-08-01_to_2023-08-15.csv"

INITIAL_CASH = 100_000.0

# Moving-average parameters
SHORT_MA_HOURS = 48
LONG_MA_HOURS = 72

# ATR / position-sizing parameters
ATR_PERIOD = 48

# Risk budget:
# At each entry, risk approximately this fraction
# of current portfolio equity.
RISK_PER_TRADE = 0.02

# Assumed stop distance = ATR * this multiplier
ATR_STOP_MULTIPLIER = 3.0

# Never allow the position to exceed this fraction
# of current portfolio equity.
MAX_POSITION_PCT = 1.00     # maximum 100% of equity

# Competition taker fee
TAKER_FEE = 0.001


# =========================================================
# Load data
# =========================================================

def load_data(filepath):

    df = pd.read_csv(
        filepath,
        parse_dates=["timestamp"]
    )

    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "MA_6h",
        "MA_24h",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    return df


# =========================================================
# Calculate moving averages
# =========================================================

def calculate_moving_averages(
    price,
    short_hours,
    long_hours
):

    if short_hours <= 0:
        raise ValueError(
            "short_hours must be greater than 0"
        )

    if long_hours <= 0:
        raise ValueError(
            "long_hours must be greater than 0"
        )

    if short_hours >= long_hours:
        raise ValueError(
            "short_hours must be smaller than long_hours"
        )

    short_ma = price.rolling(
        window=short_hours
    ).mean()

    long_ma = price.rolling(
        window=long_hours
    ).mean()

    return short_ma, long_ma


# =========================================================
# Calculate ATR
# =========================================================

def calculate_atr(
    df,
    period
):
    """
    Calculate Average True Range using Wilder's method.

    ATR measures the recent absolute price movement.

    We later shift ATR by one bar before using it for
    an order executed at the current bar's open.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return atr


# =========================================================
# Run backtest
# =========================================================

def run_backtest(
    df,
    short_hours=SHORT_MA_HOURS,
    long_hours=LONG_MA_HOURS,
    atr_period=ATR_PERIOD,
    risk_per_trade=RISK_PER_TRADE,
    atr_stop_multiplier=ATR_STOP_MULTIPLIER,
    max_position_pct=MAX_POSITION_PCT,
    init_cash=100_000.0,
    fee=0.001
):
    """
    Backtest moving-average crossover with
    dynamic ATR-based position sizing.

    Entry:
        Short MA crosses above Long MA.

    Exit:
        Short MA crosses below Long MA.

    Execution:
        Signal generated using candle close.
        Order executed at the next candle's OPEN.

    Position sizing:

        risk_budget =
            current_equity * risk_per_trade

        stop_distance =
            ATR * atr_stop_multiplier

        position_units =
            risk_budget / stop_distance

    The resulting position is capped at
    max_position_pct of current portfolio equity.

    IMPORTANT:
        ATR is shifted by one candle so that an order
        executed at the current candle's OPEN only uses
        information known before that open.
    """

    close = df["close"]

    # -----------------------------------------------------
    # Moving averages
    # -----------------------------------------------------

    short_ma, long_ma = calculate_moving_averages(
        price=close,
        short_hours=short_hours,
        long_hours=long_hours,
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    atr = calculate_atr(
        df=df,
        period=atr_period
    )

    # -----------------------------------------------------
    # Generate crossover signals
    # -----------------------------------------------------

    entries = (
        short_ma.vbt.crossed_above(long_ma)
        .fillna(False)
        .astype(bool)
    )

    exits = (
        short_ma.vbt.crossed_below(long_ma)
        .fillna(False)
        .astype(bool)
    )

    # -----------------------------------------------------
    # Execute signal on next candle's OPEN
    # -----------------------------------------------------

    entries = (
        entries
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    exits = (
        exits
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # The order occurs at current OPEN.
    #
    # Therefore ATR must contain only information
    # available before that OPEN.
    #
    # Example:
    #
    # ATR[i] normally uses candle i.
    # That candle's high/low/close are not known at
    # candle i OPEN.
    #
    # Therefore use ATR[i-1].
    # -----------------------------------------------------

    atr_for_execution = atr.shift(1)

    # -----------------------------------------------------
    # Build dynamic orders sequentially
    # -----------------------------------------------------
    #
    # We calculate the position size ourselves because
    # the desired size depends on CURRENT portfolio equity.
    #
    # This is inherently sequential:
    #
    # trade 1 -> equity changes
    # trade 2 -> size uses new equity
    # trade 3 -> size uses new equity
    #
    # This avoids pretending that the position size can
    # be known in advance.
    # -----------------------------------------------------

    open_price = df["open"].astype(float)
    close_price = df["close"].astype(float)

    order_size = np.zeros(len(df), dtype=float)

    # Current cash
    cash = init_cash

    # Current BTC position
    position = 0.0

    # Track portfolio equity
    equity = init_cash

    for i in range(len(df)):

        current_open = open_price.iloc[i]
        

        # -------------------------------------------------
        # Mark current portfolio to market
        # -------------------------------------------------

        if i == 0:
            previous_close = current_open
            equity = cash
        else:
            previous_close = close_price.iloc[i - 1]

            equity = (
                cash
                + position * previous_close
            )

        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        if exits.iloc[i] and position > 0:

            # Sell the entire position.
            order_size[i] = -position

            # Update our local cash/position model
            #
            # Fees are included here so that the next
            # trade's equity reflects the actual capital.
            proceeds = (
                position
                * current_open
                * (1 - fee)
            )

            cash += proceeds
            position = 0.0

            equity = cash

        # -------------------------------------------------
        # ENTRY
        # -------------------------------------------------

        elif entries.iloc[i] and position == 0:

            current_atr = atr_for_execution.iloc[i]

            # Need valid ATR.
            if (
                pd.notna(current_atr)
                and current_atr > 0
                and current_open > 0
            ):

                # -----------------------------------------
                # Current portfolio equity
                # -----------------------------------------

                current_equity = equity

                # -----------------------------------------
                # Maximum allowed position value
                #
                # Example:
                #
                # equity = $100,000
                # max_position_pct = 1.0
                #
                # maximum position = $100,000
                # -----------------------------------------

                max_position_value = (
                    current_equity
                    * max_position_pct
                )

                # -----------------------------------------
                # Amount we are willing to lose
                # if the assumed ATR stop is hit.
                #
                # Example:
                #
                # equity = $100,000
                # risk = 1%
                #
                # risk budget = $1,000
                # -----------------------------------------

                risk_budget = (
                    current_equity
                    * risk_per_trade
                )

                # -----------------------------------------
                # Assumed stop distance
                # -----------------------------------------

                stop_distance = (
                    current_atr
                    * atr_stop_multiplier
                )

                # -----------------------------------------
                # Position value based on risk
                #
                # If ATR becomes large:
                #
                #     position becomes smaller
                #
                # If ATR becomes small:
                #
                #     position becomes larger
                # -----------------------------------------

                risk_based_position_value = (
                    risk_budget
                    / stop_distance
                    * current_open
                )

                # -----------------------------------------
                # Apply maximum position cap
                # -----------------------------------------

                target_position_value = min(
                    risk_based_position_value,
                    max_position_value
                )

                # -----------------------------------------
                # Convert dollar position value into BTC
                # -----------------------------------------

                units = (
                    target_position_value
                    / current_open
                )

                # Account for entry fee.
                #
                # We need enough cash to buy BTC
                # AND pay the fee.
                max_affordable_units = (
                    cash
                    / (
                        current_open
                        * (1 + fee)
                    )
                )

                units = min(
                    units,
                    max_affordable_units
                )

                if units > 0:

                    order_size[i] = units

                    # Update local portfolio
                    cost = (
                        units
                        * current_open
                    )

                    entry_fee = cost * fee

                    cash -= (
                        cost
                        + entry_fee
                    )

                    position += units

                    equity = (
                        cash
                        + position * current_open
                    )

    # -----------------------------------------------------
    # Convert orders into VectorBT portfolio
    # -----------------------------------------------------

    order_size = pd.Series(
        order_size,
        index=df.index
    )

    pf = vbt.Portfolio.from_orders(
        close=close_price,
        size=order_size,
        size_type="amount",
        direction="both",
        price=open_price,
        init_cash=init_cash,
        fees=fee,
        freq="1h",
    )

    return pf


# =========================================================
# Calculate competition composite score
# =========================================================

def calculate_composite_score(stats):

    sortino = stats["Sortino Ratio"]
    sharpe = stats["Sharpe Ratio"]
    calmar = stats["Calmar Ratio"]

    composite_score = (
        0.4 * sortino
        + 0.3 * sharpe
        + 0.3 * calmar
    )

    return composite_score


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = load_data(DATA_FILE)

    # -----------------------------------------------------
    # Run backtest
    # -----------------------------------------------------

    pf = run_backtest(
        df=df,
        short_hours=SHORT_MA_HOURS,
        long_hours=LONG_MA_HOURS,
        atr_period=ATR_PERIOD,
        risk_per_trade=RISK_PER_TRADE,
        atr_stop_multiplier=ATR_STOP_MULTIPLIER,
        max_position_pct=MAX_POSITION_PCT,
        init_cash=INITIAL_CASH,
        fee=TAKER_FEE,
    )

    # -----------------------------------------------------
    # Get VectorBT statistics
    # -----------------------------------------------------

    stats = pf.stats()

    print(
        "\n>>> print(pf.stats())"
    )

    print(stats)

    # -----------------------------------------------------
    # Competition composite score
    # -----------------------------------------------------

    composite_score = calculate_composite_score(
        stats
    )

    print(
        "\n========== Parameters & Composite Score =========="
    )

    print(
        "\nParameters:"
    )

    print(
        f"Short MA:              "
        f"{SHORT_MA_HOURS} hours"
    )

    print(
        f"Long MA:               "
        f"{LONG_MA_HOURS} hours"
    )

    print(
        f"ATR Period:            "
        f"{ATR_PERIOD} hours"
    )

    print(
        f"Risk Per Trade:        "
        f"{RISK_PER_TRADE:.2%}"
    )

    print(
        f"ATR Stop Multiplier:   "
        f"{ATR_STOP_MULTIPLIER:.2f} × ATR"
    )

    print(
        f"Max Position:          "
        f"{MAX_POSITION_PCT:.2%}"
    )

    print(
        f"Taker Fee:             "
        f"{TAKER_FEE:.3%}"
    )

    print(
        f"\nComposite Score: "
        f"{composite_score:.6f}"
    )


if __name__ == "__main__":
    main()