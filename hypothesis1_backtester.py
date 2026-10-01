import pandas as pd
import vectorbt as vbt


# =========================
# Configuration
# =========================

DATA_FILE = "BTCUSDT_1h_2020-08_to_2023-08.csv"

INITIAL_CASH = 100_000

# Competition fees
TAKER_FEE = 0.001
MAKER_FEE = 0.0005

# Use taker fee for this backtest
FEE = TAKER_FEE


# =========================
# Load data
# =========================

def load_data():
    print("Loading data...")

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["open_time"]
    )

    # Use open_time as the time index
    df = df.set_index("open_time")

    # Make sure data is chronological
    df = df.sort_index()

    # Explicitly convert price and signal columns
    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df["signal_3h"] = pd.to_numeric(
        df["signal_3h"],
        errors="coerce"
    )

    # Remove rows where required data is missing
    df = df.dropna(
        subset=["close", "signal_3h"]
    )

    # Convert signal to boolean
    df["signal_3h"] = df["signal_3h"].astype(bool)

    print(f"Loaded {len(df):,} candles.")
    print(
        f"Period: {df.index[0]} -> {df.index[-1]}"
    )

    return df


# =========================
# Create trading signals
# =========================

def create_signals(df):
    """
    Strategy / hypothesis implementation:

    Hypothesis:
        If the previous 3-hour return was positive,
        BTC is more likely to continue rising.

    Signal:
        signal_3h = 1 -> momentum condition is true
        signal_3h = 0 -> momentum condition is false

    Trading interpretation:
        signal_3h = 1 -> want to be LONG
        signal_3h = 0 -> want to be FLAT

    Entry:
        Change from FLAT -> LONG

    Exit:
        Change from LONG -> FLAT

    We shift the signal by one candle so that
    information from candle t is used to trade
    on candle t+1.
    """

    signal = df["signal_3h"]

    # Desired position based on the PREVIOUS candle.
    #
    # True  = should be long
    # False = should be flat
    desired_long = signal.shift(1)

    # Entry happens when:
    # previous state was flat
    # AND current desired state is long.
    entries = (
        desired_long
        & ~desired_long.shift(1).fillna(False)
    )

    # Exit happens when:
    # previous state was long
    # AND current desired state is flat.
    exits = (
        ~desired_long
        & desired_long.shift(1).fillna(False)
    )

    # Replace NaN with False and make sure the
    # resulting arrays contain actual booleans.
    entries = entries.fillna(False).astype(bool)
    exits = exits.fillna(False).astype(bool)

    print(
        f"Number of entry signals: {entries.sum()}"
    )

    print(
        f"Number of exit signals:  {exits.sum()}"
    )

    return entries, exits


# =========================
# Run VectorBT backtest
# =========================

def run_backtest(df, entries, exits):

    close = df["close"].astype(float)

    # Make absolutely sure VectorBT receives
    # numerical / boolean arrays.
    entries = entries.astype(bool)
    exits = exits.astype(bool)

    print("\nData types passed to VectorBT:")
    print(f"close:   {close.dtype}")
    print(f"entries: {entries.dtype}")
    print(f"exits:   {exits.dtype}")

    portfolio = vbt.Portfolio.from_signals(
        close=close,

        entries=entries,
        exits=exits,

        # Invest 100% of available capital
        size=1.0,
        size_type="percent",

        init_cash=INITIAL_CASH,

        # Trading fee
        fees=FEE,

        # No additional slippage for now
        slippage=0.0,

        # Data frequency
        freq="1h"
    )

    return portfolio


# =========================
# Calculate trade statistics
# =========================

def calculate_trade_statistics(
    portfolio,
    df
):
    """
    Calculate statistics about the trades.

    A trade means one completed LONG position:

        Entry -> Exit

    We count completed trades rather than simply
    counting every True value in the entry array.
    """

    # VectorBT records each actual executed trade.
    trades = portfolio.trades

    # Number of completed trades
    total_trades = trades.count()

    # Number of calendar days in the backtest
    number_of_days = (
        df.index[-1] - df.index[0]
    ).total_seconds() / (24 * 60 * 60)

    # Average completed trades per day
    average_trades_per_day = (
        total_trades / number_of_days
    )

    return {
        "Total Completed Trades": total_trades,
        "Average Trades per Day": average_trades_per_day
    }


# =========================
# Calculate performance
# =========================

def calculate_metrics(portfolio):

    total_return = portfolio.total_return()

    max_drawdown = portfolio.max_drawdown()

    sharpe = portfolio.sharpe_ratio(
        freq="1h"
    )

    sortino = portfolio.sortino_ratio(
        freq="1h"
    )

    calmar = portfolio.calmar_ratio(
        freq="1h"
    )

    # Competition scoring formula
    composite_score = (
        0.4 * sortino
        + 0.3 * sharpe
        + 0.3 * calmar
    )

    return {
        "Total Return (%)": total_return * 100,
        "Max Drawdown (%)": max_drawdown * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Composite Score": composite_score
    }


# =========================
# Main
# =========================

def main():

    # 1. Load CSV
    df = load_data()

    # 2. Convert the hypothesis into
    #    trading entries and exits
    entries, exits = create_signals(df)

    # 3. Run backtest
    portfolio = run_backtest(
        df,
        entries,
        exits
    )

    # 4. Calculate performance
    metrics = calculate_metrics(
        portfolio
    )

    # 5. Calculate trade statistics
    trade_statistics = calculate_trade_statistics(
        portfolio,
        df
    )

    # =========================
    # Print performance
    # =========================

    print("\n=========================")
    print("BACKTEST RESULTS")
    print("=========================")

    for name, value in metrics.items():

        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # =========================
    # Print trade statistics
    # =========================

    print("\n=========================")
    print("TRADE STATISTICS")
    print("=========================")

    for name, value in trade_statistics.items():

        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # =========================
    # Print portfolio statistics
    # =========================

    print("\n=========================")
    print("PORTFOLIO")
    print("=========================")

    print(
        portfolio.stats()
    )


if __name__ == "__main__":
    main()
