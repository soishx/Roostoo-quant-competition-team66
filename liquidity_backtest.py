# RVOL + Amihud Illiquidity Ratio + Volume-Weighted Momentum

import glob
import os
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
DATA_DIR = "data"
TRADING_FEE_RATE = 0.001  # 0.1% fee per trade execution (typical for Roostoo/Binance spot)
RISK_FREE_RATE = 0.0  # Annualized risk-free rate for Sharpe Ratio calculation
SYMBOLS = ["AMD", "GOOGL", "INTC", "META", "MU", "NVDA", "SKHY", "SNDK"]

# Standard Binance Daily K-line CSV Column Headers (12 columns, no header row in original CSVs)
BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]


def load_and_preprocess_data(symbol, data_dir=DATA_DIR):
  """Loads all monthly CSV files for a given symbol, concatenates them,

  and prepares a clean DataFrame aligned by timestamp.
  """
  file_pattern = os.path.join(data_dir, f"{symbol}BUSDT-1d-2026-*.csv")
  files = sorted(glob.glob(file_pattern))

  if not files:
    print(f"[Warning] No files found for symbol: {symbol}")
    return None

  df_list = []
  for file_path in files:
    temp_df = pd.read_csv(file_path, header=None, names=BINANCE_COLUMNS)
    df_list.append(temp_df)

  df = pd.concat(df_list, ignore_index=True)

  # Convert open_time (milliseconds/microseconds timestamp) to datetime
  # Binance default timestamps in these CSVs use epoch milliseconds or microseconds
  df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
  # Fallback for microsecond timestamps if ms parsing fails or yields futuristic dates
  if df["datetime"].dt.year.max() > 2050:
    df["datetime"] = pd.to_datetime(df["open_time"], unit="us")

  df = df.sort_values("datetime").reset_index(drop=True)

  # Select relevant price and volume columns
  numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
  for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  df = df[["datetime", "open", "high", "low", "close", "volume", "quote_volume"]]
  return df.set_index("datetime")


def calculate_liquidity_signals(df):
  """Calculates 3 liquidity/turnover signals and generates position signals.

  To prevent look-ahead bias, all signal indicators are lagged by 1 day (shift(1)).
  """
  df = df.copy()

  # 1. Asset Base Return
  df["asset_return"] = df["close"].pct_change()

  # -------------------------------------------------------------
  # Signal 1: Relative Volume (RVOL)
  # -------------------------------------------------------------
  # RVOL = Volume / 20-day Moving Average of Volume
  df["vol_ma20"] = df["volume"].rolling(window=20, min_periods=5).mean()
  df["rvol"] = df["volume"] / df["vol_ma20"]
  # Raw Trigger: Volume surge > 1.5x average AND positive price movement
  raw_sig_rvol = ((df["rvol"] > 1.5) & (df["asset_return"] > 0)).astype(int)

  # -------------------------------------------------------------
  # Signal 2: Amihud Illiquidity Ratio
  # -------------------------------------------------------------
  # Illiquidity = |Return| / Dollar Volume
  df["amihud"] = np.abs(df["asset_return"]) / (df["quote_volume"] + 1e-8)
  df["amihud_ma20"] = df["amihud"].rolling(window=20, min_periods=5).mean()
  # Raw Trigger: Illiquidity drops (smooth institutional absorption) AND positive return
  raw_sig_amihud = (
      (df["amihud"] < df["amihud_ma20"] * 0.8) & (df["asset_return"] > 0)
  ).astype(int)

  # -------------------------------------------------------------
  # Signal 3: Volume-Weighted Momentum (VWM)
  # -------------------------------------------------------------
  df["momentum_5d"] = df["close"].pct_change(5)
  df["vwm"] = df["momentum_5d"] * df["rvol"]
  df["vwm_ma20"] = df["vwm"].rolling(window=20, min_periods=5).mean()
  # Raw Trigger: Volume-Weighted Momentum above its 20-day MA
  raw_sig_vwm = (df["vwm"] > df["vwm_ma20"]).astype(int)

  # CRITICAL: Prevent Look-Ahead Bias by shifting raw signals by 1 period.
  # The decision generated at Day T-1 is executed for Return at Day T.
  df["pos_rvol"] = raw_sig_rvol.shift(1).fillna(0)
  df["pos_amihud"] = raw_sig_amihud.shift(1).fillna(0)
  df["pos_vwm"] = raw_sig_vwm.shift(1).fillna(0)

  return df


def backtest_strategy(df, pos_col, fee_rate=TRADING_FEE_RATE):
  """Simulates trading strategy execution, including transaction fee deductions."""
  position = df[pos_col]
  asset_return = df["asset_return"]

  # Identify position changes to calculate trading turnover fees
  pos_change = position.diff().abs().fillna(0)
  fee_deduction = pos_change * fee_rate

  # Net Strategy Daily Return = (Position * Asset Return) - Trading Fees
  net_strategy_return = (position * asset_return) - fee_deduction
  return net_strategy_return


def evaluate_performance(
    returns_series, risk_free_rate=RISK_FREE_RATE, periods_per_year=252
):
  """Calculates core quantitative metrics: Total Return, Annualized Return,

  Annualized Volatility, Sharpe Ratio, Maximum Drawdown, and Win Rate.
  """
  clean_returns = returns_series.dropna()
  if len(clean_returns) == 0:
    return {}

  # Cumulative Returns & Total Return
  cum_returns = (1 + clean_returns).cumprod()
  total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0.0

  # Annualized Return & Volatility
  mean_ret = clean_returns.mean()
  vol_ret = clean_returns.std()
  ann_return = mean_ret * periods_per_year
  ann_vol = vol_ret * np.sqrt(periods_per_year)

  # Sharpe Ratio
  sharpe_ratio = (
      (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-6 else 0.0
  )

  # Maximum Drawdown (Max DD)
  peak = cum_returns.cummax()
  drawdown = (cum_returns - peak) / peak
  max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

  # Win Rate
  trade_days = clean_returns[clean_returns != 0]
  win_rate = (
      (trade_days > 0).sum() / len(trade_days) if len(trade_days) > 0 else 0.0
  )

  return {
      "Total Return (%)": total_return * 100,
      "Annualized Return (%)": ann_return * 100,
      "Annualized Volatility (%)": ann_vol * 100,
      "Sharpe Ratio": sharpe_ratio,
      "Max Drawdown (%)": max_drawdown * 100,
      "Win Rate (%)": win_rate * 100,
  }


def main():
  print("=" * 70)
  print("      LIQUIDITY & TURNOVER STRATEGY BACKTESTING FRAMEWORK      ")
  print("=" * 70)

  all_data = {}
  for sym in SYMBOLS:
    df = load_and_preprocess_data(sym)
    if df is not None:
      all_data[sym] = calculate_liquidity_signals(df)

  if not all_data:
    print("Error: No data loaded. Please check your data/ folder.")
    return

  signals = {
      "RVOL (Relative Volume)": "pos_rvol",
      "Amihud Illiquidity Ratio": "pos_amihud",
      "Volume-Weighted Momentum": "pos_vwm",
  }

  summary_results = []

  for sig_name, pos_col in signals.items():
    print(f"\n--- Evaluating Signal: {sig_name} ---")
    for sym, df in all_data.items():
      strat_returns = backtest_strategy(df, pos_col)
      metrics = evaluate_performance(strat_returns)

      metrics_entry = {"Signal": sig_name, "Symbol": sym}
      metrics_entry.update(metrics)
      summary_results.append(metrics_entry)

  # Convert results into a structured Comparison Table
  results_df = pd.DataFrame(summary_results)

  # Format & Print Results Summary
  print("\n" + "=" * 80)
  print("                        SUMMARY PERFORMANCE TABLE                     ")
  print("=" * 80)
  print(
      results_df[
          [
              "Signal",
              "Symbol",
              "Total Return (%)",
              "Sharpe Ratio",
              "Max Drawdown (%)",
              "Win Rate (%)",
          ]
      ].to_string(index=False)
  )

  # Aggregate metrics across the asset portfolio
  print("\n" + "=" * 80)
  print("                  AVERAGE SIGNAL PERFORMANCE (PORTFOLIO)               ")
  print("=" * 80)
  avg_summary = results_df.groupby("Signal")[
      ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]
  ].mean()
  print(avg_summary.round(2).to_string())


if __name__ == "__main__":
  main()