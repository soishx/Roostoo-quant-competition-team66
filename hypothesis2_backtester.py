import pandas as pd
import numpy as np
import vectorbt as vbt


# ============================================================
# Configuration
# ============================================================

CSV_FILE = "ETHUSDT_1h_hypothesis2_2023-09_to_2026-08.csv"

INITIAL_CASH = 100_000

# Roostoo competition:
# 0.1% taker fee per executed order
TAKER_FEE = 0.001

# We are using 1-hour candles
FREQUENCY = "1h"

# Holding periods to test
HOLDING_PERIODS = [1, 2, 3]

# Breakout threshold
# 0.001 = price must break previous 24h high by at least 0.1%
BREAKOUT_THRESHOLD = 0.005

# Stop loss
# 0.005 = 0.5% loss from entry price
STOP_LOSS = 0.01

# Take profit
# 0.01 = 1% profit from entry price
TAKE_PROFIT = 0.14


# ============================================================
# Load data
# ============================================================

def load_data():
    df = pd.read_csv(
        CSV_FILE,
        index_col="open_time",
        parse_dates=True
    )

    # Make sure data is chronological
    df = df.sort_index()

    # Make sure numeric columns are actually numeric
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "previous_24h_high",
        "signal",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Remove rows that cannot be used
    df = df.dropna(
        subset=[
            "open",
            "close",
            "previous_24h_high",
            "signal"
        ]
    )
    
    df["signal"] = df["signal"].astype(bool)

    return df


# ============================================================
# Create portfolio for a specific holding period
# ============================================================

def run_backtest(df, holding_period, take_profit):
    """
    Run the breakout strategy with a stop loss
    and take profit.

    Signal:
        Breakout is confirmed when the candle closes.

    Entry:
        Next candle's OPEN.
    """

    breakout_signal = (
        df["close"]
        > df["previous_24h_high"] * (1 + BREAKOUT_THRESHOLD)
    )

    entries = (
        breakout_signal
        .shift(1)
        .fillna(False)
        .to_numpy(dtype=np.bool_)
    )

    exits = None

    price = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    portfolio = vbt.Portfolio.from_signals(
        close=price,
        entries=entries,
        exits=exits,

        high=high,
        low=low,

        size=np.inf,
        direction="longonly",
        fees=TAKER_FEE,
        slippage=0.0,

        sl_stop=STOP_LOSS,
        tp_stop=take_profit,

        init_cash=INITIAL_CASH,
        freq=FREQUENCY,
        accumulate=False,
    )

    return portfolio


# ============================================================
# Calculate composite score
# ============================================================

def calculate_composite_score(
    sharpe,
    sortino,
    calmar
):
    """
    Competition formula:

        0.4 * Sortino
        0.3 * Sharpe
        0.3 * Calmar
    """

    return (
        0.4 * sortino
        + 0.3 * sharpe
        + 0.3 * calmar
    )


# ============================================================
# Print results
# ============================================================

def print_results(
    portfolio,
    holding_period
):

    final_value = portfolio.final_value()

    total_return = portfolio.total_return()

    max_drawdown = portfolio.max_drawdown()

    sharpe = portfolio.sharpe_ratio()

    sortino = portfolio.sortino_ratio()

    calmar = portfolio.calmar_ratio()

    composite_score = calculate_composite_score(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar
    )

    total_trades = portfolio.trades.count()

    total_fees = portfolio.orders.records_readable["Fees"].sum()

    print("\n" + "=" * 65)

    print(
        f"HYPOTHESIS 2 — {holding_period} HOUR EXIT"
    )

    print("=" * 65)

    print(
        f"Initial capital:       ${INITIAL_CASH:,.2f}"
    )

    print(
        f"Final portfolio value: ${final_value:,.2f}"
    )

    print(
        f"Total return:          {total_return:.2%}"
    )

    print(
        f"Maximum drawdown:      {max_drawdown:.2%}"
    )

    print(
        f"Number of trades:      {total_trades}"
    )

    print(
        f"Total fees paid:       ${total_fees:,.2f}"
    )

    print("-" * 65)

    print(
        f"Sharpe ratio:          {sharpe:.4f}"
    )

    print(
        f"Sortino ratio:         {sortino:.4f}"
    )

    print(
        f"Calmar ratio:          {calmar:.4f}"
    )
    
    print(
            f"COMPOSITE SCORE:       {composite_score:.4f}"
        )

    print("-" * 65)

    print("=" * 65)


# ============================================================
# Main
# ============================================================

def main():

    print("Loading Hypothesis 2 data...")

    df = load_data()

    print(
        f"Loaded {len(df):,} hourly candles."
    )

    print(
        f"Period: {df.index.min()} → {df.index.max()}"
    )

    print(
        f"Breakout signals: "
        f"{int(df['signal'].sum()):,}"
    )
    
    # Ask user for take profit
    take_profit = float(
        input("Enter take profit (e.g. 0.15 for 15%): ")
    )

    print(f"\nUsing take profit: {take_profit:.2%}")
    print("\nRunning backtest...")

    portfolio = run_backtest(
        df,
        holding_period=0,
        take_profit=take_profit
    )

    print("\nRunning backtests...")

    results = []
    
    portfolio = run_backtest( df, holding_period=0, take_profit=take_profit)

    # for holding_period in HOLDING_PERIODS:

    #     portfolio = run_backtest(
    #         df,
    #         holding_period
    #     )

    #     print_results(
    #         portfolio,
    #         holding_period
    #     )

        # ----------------------------------------------------
        # Save values for final comparison table
        # ----------------------------------------------------

    sharpe = portfolio.sharpe_ratio()
    sortino = portfolio.sortino_ratio()
    calmar = portfolio.calmar_ratio()

    composite = calculate_composite_score(
        sharpe,
        sortino,
        calmar
    )
    
    final_value = portfolio.final_value() 
    total_return = portfolio.total_return() 
    max_drawdown = portfolio.max_drawdown()
    
    total_trades = portfolio.trades.count() 
    total_fees = portfolio.orders.records_readable["Fees"].sum()

    results.append({
        # "Exit (hours)": holding_period,
        "Final Value": portfolio.final_value(),
        "Total Return": portfolio.total_return(),
        "Max Drawdown": portfolio.max_drawdown(),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Composite Score": composite,
        "Trades": portfolio.trades.count(),
    })
    
        # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "HYPOTHESIS 2 — STOP LOSS / TAKE PROFIT STRATEGY"
    )

    print("=" * 70)

    print(
        f"Initial capital:       ${INITIAL_CASH:,.2f}"
    )

    print(
        f"Breakout threshold:    {BREAKOUT_THRESHOLD:.2%}"
    )

    print(
        f"Stop loss:             -{STOP_LOSS:.2%}"
    )

    print(
        f"Take profit:           +{take_profit:.2%}"
    )

    print("-" * 70)

    print(
        f"Final portfolio value: ${final_value:,.2f}"
    )

    print(
        f"Total return:          {total_return:.2%}"
    )

    print(
        f"Maximum drawdown:      {max_drawdown:.2%}"
    )

    print(
        f"Number of trades:      {total_trades}"
    )

    print(
        f"Total fees paid:       ${total_fees:,.2f}"
    )

    print("-" * 70)

    print(
        f"Sharpe ratio:          {sharpe:.4f}"
    )

    print(
        f"Sortino ratio:         {sortino:.4f}"
    )

    print(
        f"Calmar ratio:          {calmar:.4f}"
    )

    print("-" * 70)


    print("=" * 70)

    # --------------------------------------------------------
    # Detailed VectorBT statistics
    # --------------------------------------------------------

    print("\n===== Detailed Portfolio Statistics =====")
    print(portfolio.stats())
    

    # ========================================================
    # Comparison table
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n\n")
    print("=" * 100)
    print("HYPOTHESIS 2 — EXIT COMPARISON")
    print("=" * 100)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Final Value": lambda x:
                    f"${x:,.2f}",

                "Total Return": lambda x:
                    f"{x:.2%}",

                "Max Drawdown": lambda x:
                    f"{x:.2%}",

                "Sharpe": lambda x:
                    f"{x:.4f}",

                "Sortino": lambda x:
                    f"{x:.4f}",

                "Calmar": lambda x:
                    f"{x:.4f}",

                "Composite Score": lambda x:
                    f"{x:.4f}",
            }
        )
    )

    print("=" * 100)
    
    # for holding_period in HOLDING_PERIODS:
    #     pf = run_backtest(df, holding_period)

    #     print(f"\n===== {holding_period}h Holding Period =====")
    #     print(pf.stats())


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
