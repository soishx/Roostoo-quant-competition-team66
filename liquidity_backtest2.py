import glob
import os
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & PARAMETERS
# ==============================================================================
DATA_DIR = "data"
TRADING_FEE_RATE = 0.001  # 0.1% transaction fee per trade execution
RISK_FREE_RATE = 0.0  # Annualized risk-free rate for Sharpe Ratio
SYMBOLS = ["AMD", "GOOGL", "INTC", "META", "MU", "NVDA", "SKHY", "SNDK"]

# Standard 12-column Binance Daily K-Line structure (no CSV headers)
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


# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_and_preprocess_data(symbol, data_dir=DATA_DIR):
  """Loads all monthly Binance CSVs for a given stock symbol, parses timestamps,

  sorts chronologically, and cleans numeric columns.
  """
  file_pattern = os.path.join(data_dir, f"{symbol}BUSDT-1d-2026-*.csv")
  files = sorted(glob.glob(file_pattern))

  if not files:
    print(f"[Warning] No CSV files found for symbol: {symbol}")
    return None

  df_list = []
  for file_path in files:
    temp_df = pd.read_csv(file_path, header=None, names=BINANCE_COLUMNS)
    df_list.append(temp_df)

  df = pd.concat(df_list, ignore_index=True)

  # Parse Binance timestamp (handles microsecond/millisecond timestamps)
  df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
  if df["datetime"].dt.year.max() > 2050:
    df["datetime"] = pd.to_datetime(df["open_time"], unit="us")

  df = df.sort_values("datetime").reset_index(drop=True)

  numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
  for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  df = df[["datetime", "open", "high", "low", "close", "volume", "quote_volume"]]
  return df.set_index("datetime")


# ==============================================================================
# LIQUIDITY SIGNALS COMPUTATION
# ==============================================================================
def calculate_liquidity_signals(df):
  """Computes Amihud Illiquidity Ratio and 5 parallel Liquidity signals.

  Applies shift(1) to all generated positions to strictly avoid look-ahead
  bias.
  """
  df = df.copy()

  # Asset daily percent return
  df["asset_return"] = df["close"].pct_change()

  # --------------------------------------------------------------------------
  # 1. Amihud Illiquidity Ratio (Baseline Winner)
  # --------------------------------------------------------------------------
  df["amihud"] = np.abs(df["asset_return"]) / (df["quote_volume"] + 1e-8)
  df["amihud_ma20"] = df["amihud"].rolling(window=20, min_periods=5).mean()
  # Buy signal: Illiquidity drops below 80% of 20-day MA during upward price move
  raw_sig_amihud = (
      (df["amihud"] < df["amihud_ma20"] * 0.8) & (df["asset_return"] > 0)
  ).astype(int)

  # --------------------------------------------------------------------------
  # 2. Roll Spread Estimator
  # --------------------------------------------------------------------------
  # Estimates effective spread via serial autocovariance of price changes
  price_diff = df["close"].diff()
  cov_series = (
      price_diff.rolling(window=20, min_periods=5).cov(price_diff.shift(1))
  )
  df["roll_spread"] = 2 * np.sqrt(np.maximum(0, -cov_series))
  df["roll_ma20"] = df["roll_spread"].rolling(window=20, min_periods=5).mean()
  # Buy signal: Roll Spread narrows below historical average (low transaction cost)
  raw_sig_roll = (
      (df["roll_spread"] < df["roll_ma20"]) & (df["asset_return"] > 0)
  ).astype(int)

  # --------------------------------------------------------------------------
  # 3. Corwin-Schultz High-Low Spread
  # --------------------------------------------------------------------------
  # Low-frequency proxy for bid-ask spread using High/Low price ratio
  beta = (np.log(df["high"] / df["low"])) ** 2
  gamma = (
      np.log(
          df["high"].rolling(2).max() / df["low"].rolling(2).min() + 1e-8
      )  # Avoid zero division
  ) ** 2
  alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(
      gamma / (3 - 2 * np.sqrt(2))
  )
  df["cs_spread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
  df["cs_ma20"] = df["cs_spread"].rolling(window=20, min_periods=5).mean()
  # Buy signal: CS spread contracts during positive return days
  raw_sig_cs = (
      (df["cs_spread"] < df["cs_ma20"]) & (df["asset_return"] > 0)
  ).astype(int)

  # --------------------------------------------------------------------------
  # 4. Zero-Return Days Proportion (Liquidity Trap Filter)
  # --------------------------------------------------------------------------
  is_zero_ret = (df["asset_return"].abs() < 1e-5).astype(int)
  df["zero_ret_prop"] = (
      is_zero_ret.rolling(window=20, min_periods=5).mean()
  )
  # Buy signal: Zero-return proportion is zero (high liquidity participation) and momentum is up
  raw_sig_zero = (
      (df["zero_ret_prop"] == 0) & (df["asset_return"] > 0)
  ).astype(int)

  # --------------------------------------------------------------------------
  # 5. Share Turnover Rate (Volume Normalized)
  # --------------------------------------------------------------------------
  df["turnover"] = df["volume"]
  df["turnover_ma20"] = (
      df["turnover"].rolling(window=20, min_periods=5).mean()
  )
  # Buy signal: Turnover expands beyond 1.2x of 20-day moving average
  raw_sig_turnover = (
      (df["turnover"] > df["turnover_ma20"] * 1.2) & (df["asset_return"] > 0)
  ).astype(int)

  # --------------------------------------------------------------------------
  # 6. Kyle's Lambda (Price Impact Coefficient)
  # --------------------------------------------------------------------------
  # Lambda = Return / Signed Volume
  signed_vol = df["quote_volume"] * np.sign(df["asset_return"])
  df["kyle_lambda"] = df["asset_return"] / (signed_vol + 1e-8)
  df["lambda_ma20"] = (
      df["kyle_lambda"].abs().rolling(window=20, min_periods=5).mean()
  )
  # Buy signal: Low price impact (higher market depth) during upward trends
  raw_sig_kyle = (
      (df["kyle_lambda"].abs() < df["lambda_ma20"]) & (df["asset_return"] > 0)
  ).astype(int)

  # ==========================================================================
  # CRITICAL: SHIFT SIGNALS BY 1 DAY TO ELIMINATE LOOK-AHEAD BIAS
  # ==========================================================================
  df["pos_amihud"] = raw_sig_amihud.shift(1).fillna(0)
  df["pos_roll"] = raw_sig_roll.shift(1).fillna(0)
  df["pos_cs"] = raw_sig_cs.shift(1).fillna(0)
  df["pos_zero"] = raw_sig_zero.shift(1).fillna(0)
  df["pos_turnover"] = raw_sig_turnover.shift(1).fillna(0)
  df["pos_kyle"] = raw_sig_kyle.shift(1).fillna(0)

  return df


# ==============================================================================
# BACKTEST EXECUTION & DEDUCTION OF TRANSACTION COSTS
# ==============================================================================
def backtest_strategy(df, pos_col, fee_rate=TRADING_FEE_RATE):
  """Simulates strategy trading returns and deducts transaction fees when position changes."""
  position = df[pos_col]
  asset_return = df["asset_return"]

  # Identify turnover/position switching
  pos_change = position.diff().abs().fillna(0)
  fee_deduction = pos_change * fee_rate

  # Daily Strategy Return = (Position * Asset Return) - Transaction Fees
  strategy_return = (position * asset_return) - fee_deduction
  return strategy_return


# ==============================================================================
# PERFORMANCE EVALUATION METRICS
# ==============================================================================
def evaluate_performance(
    returns_series, risk_free_rate=RISK_FREE_RATE, periods_per_year=252
):
  """Calculates Total Return, Sharpe Ratio, Max Drawdown, and Win Rate."""
  clean_returns = returns_series.dropna()
  if len(clean_returns) == 0:
    return {}

  cum_returns = (1 + clean_returns).cumprod()
  total_return = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0.0

  mean_ret = clean_returns.mean()
  vol_ret = clean_returns.std()
  ann_return = mean_ret * periods_per_year
  ann_vol = vol_ret * np.sqrt(periods_per_year)

  sharpe_ratio = (
      (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-6 else 0.0
  )

  peak = cum_returns.cummax()
  drawdown = (cum_returns - peak) / peak
  max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

  trade_days = clean_returns[clean_returns != 0]
  win_rate = (
      (trade_days > 0).sum() / len(trade_days) if len(trade_days) > 0 else 0.0
  )

  return {
      "Total Return (%)": total_return * 100,
      "Sharpe Ratio": sharpe_ratio,
      "Max Drawdown (%)": max_drawdown * 100,
      "Win Rate (%)": win_rate * 100,
  }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
  all_data = {}
  for sym in SYMBOLS:
    df = load_and_preprocess_data(sym)
    if df is not None:
      all_data[sym] = calculate_liquidity_signals(df)

  if not all_data:
    print("Error: No data loaded. Please verify your data directory.")
    return

  signals = {
      "Amihud Illiquidity Ratio": "pos_amihud",
      "Roll Spread": "pos_roll",
      "Corwin-Schultz High-Low": "pos_cs",
      "Zero-Return Days": "pos_zero",
      "Share Turnover Rate": "pos_turnover",
      "Kyle's Lambda": "pos_kyle",
  }

  summary_results = []

  for sig_name, pos_col in signals.items():
    for sym, df in all_data.items():
      strat_returns = backtest_strategy(df, pos_col)
      metrics = evaluate_performance(strat_returns)

      metrics_entry = {"Signal": sig_name, "Symbol": sym}
      metrics_entry.update(metrics)
      summary_results.append(metrics_entry)

  results_df = pd.DataFrame(summary_results)

  # Display individual asset performance breakdown
  print("\n" + "=" * 80)
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

  # Display averaged overall strategy summary table
  print("\n" + "=" * 80)
  print("                  AVERAGE SIGNAL PERFORMANCE (PORTFOLIO)               ")
  print("=" * 80)
  avg_summary = results_df.groupby("Signal")[
      ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]
  ].mean()

  print(
      avg_summary[
          ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"]
      ]
      .round(2)
      .to_string()
  )


if __name__ == "__main__":
  main()