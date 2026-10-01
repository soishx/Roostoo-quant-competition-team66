import pandas as pd
import numpy as np
import vectorbt as vbt
from hypothesis3_backtester import calculate_composite_score

LOOKBACK = 24
ENTRY_Z = 2.0
EXIT_Z = 0.5
DATA_FILE = "data/BTCUSDT_1h_2020-08_to_2023-08.csv"
init_cash = 100_000


def backtest_mean_reversion(
    df,
    price_col="close",
    lookback=LOOKBACK,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
    init_cash=100_000,
    fees=0.001,
):
    """
    Mean-reversion strategy:

        z = (Price - Rolling Mean) / Rolling Std

        z < -entry_z  -> LONG
        z > +entry_z  -> SHORT
        |z| < exit_z  -> EXIT

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OHLC data.

    price_col : str
        Price column used for the strategy.

    lookback : int
        Rolling window.
        For 1-hour candles, 24 = 24 hours.

    entry_z : float
        Z-score threshold for entering a trade.

    exit_z : float
        Z-score threshold for exiting a trade.

    fees : float
        Transaction fee per order.
        Example: 0.001 = 0.1%.

    Returns
    -------
    pf : vectorbt.Portfolio
        Backtest portfolio.

    result : pd.DataFrame
        DataFrame containing indicators and signals.
    """

    result = df.copy()

    # --------------------------------------------------
    # 1. Calculate rolling mean
    # --------------------------------------------------

    result["ma"] = (
        result[price_col].rolling(lookback).mean()
    )

    # --------------------------------------------------
    # 2. Calculate rolling volatility
    # --------------------------------------------------

    result["sigma"] = (
        result[price_col].rolling(lookback).std()
    )

    # --------------------------------------------------
    # 3. Calculate z-score
    #
    # z = (P - MA) / sigma
    # --------------------------------------------------

    result["z_score"] = (
        (result[price_col] - result["ma"])
        / result["sigma"]
    )

    # --------------------------------------------------
    # 4. Entry signals
    # --------------------------------------------------

    long_entries = result["z_score"] < -entry_z

    short_entries = result["z_score"] > entry_z

    # --------------------------------------------------
    # 5. Exit signals
    #
    # Exit when price has returned sufficiently close
    # to the rolling mean.
    # --------------------------------------------------

    long_exits = result["z_score"] >= -exit_z

    short_exits = result["z_score"] <= exit_z
    
    

    # --------------------------------------------------
    # 6. Run vectorbt backtest
    # --------------------------------------------------

    pf = vbt.Portfolio.from_signals(
        close=result[price_col],

        entries=long_entries,
        exits=long_exits,

        short_entries=short_entries,
        short_exits=short_exits,
        init_cash = init_cash,

        fees=fees,

        freq="1h",
    )

    return pf, result

def main():
    df = pd.read_csv(DATA_FILE)
    
    pf, result = backtest_mean_reversion(
    df,
    lookback=24,
    entry_z=2.0,
    exit_z=0.5,
    init_cash=100_000,
    fees=0.001,
)
    stats = pf.stats()
    
    print(
        "\n>>> print(pf.stats())"
    )

    print(stats)
    
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
        f"Lookback:          "
        f"{LOOKBACK}"
    )
    
    print(
            f"Entry z score:          "
            f"{ENTRY_Z:.2}"
        )
    
    print(
            f"Exit z score:          "
            f"{EXIT_Z:.2}"
        )
    
    print(
            f"Data File:          "
            f"{DATA_FILE}"
        )

    print(
        f"\nComposite Score: "
        f"{composite_score:.6f}"
    )
    

if __name__ == "__main__":
    main()