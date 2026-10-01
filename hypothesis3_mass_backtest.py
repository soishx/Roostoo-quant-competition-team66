from hypothesis3_backtester import load_data, run_backtest, DATA_FILE

from itertools import product
import pandas as pd



df = load_data(DATA_FILE)

# 3 values for each parameter
atr_periods = [12, 24, 48]
risk_per_trades = [0.005, 0.01, 0.02]
atr_stop_multipliers = [1.5, 2.0, 3.0]
max_position_pcts = [0.25, 0.5, 1.0]

results = []

for atr_period, risk_per_trade, atr_stop_multiplier, max_position_pct in product(
    atr_periods,
    risk_per_trades,
    atr_stop_multipliers,
    max_position_pcts
):

    pf = run_backtest(
        df,
        atr_period=atr_period,
        risk_per_trade=risk_per_trade,
        atr_stop_multiplier=atr_stop_multiplier,
        max_position_pct=max_position_pct
    )

    # VectorBT metrics
    start_value = pf.init_cash
    end_value = pf.final_value()
    total_return = pf.total_return() * 100
    benchmark_return = pf.benchmark_returns().iloc[-1] * 100
    max_drawdown = pf.max_drawdown() * 100
    total_trades = pf.trades.count()
    sharpe = pf.sharpe_ratio()
    calmar = pf.calmar_ratio()
    sortino = pf.sortino_ratio()

    # Your composite score
    composite = (
        0.4 * sortino
        + 0.3 * sharpe
        + 0.3 * calmar
    )

    results.append({
        "ATR Period": atr_period,
        "Risk Per Trade": risk_per_trade,
        "ATR Stop Multiplier": atr_stop_multiplier,
        "Max Position": max_position_pct,

        "Start Value": start_value,
        "End Value": end_value,
        "Total Return [%]": total_return,
        "Benchmark Return [%]": benchmark_return,
        "Max Drawdown [%]": max_drawdown,
        "Total Trades": total_trades,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Sortino Ratio": sortino,
        "Composite Score": composite
    })

results_df = pd.DataFrame(results)

print("\n=== Parameter Search Results ===")
print(results_df.to_string(index=False))

results_df.to_csv(
    "hypothesis3_parameter_search.csv",
    index=False
)