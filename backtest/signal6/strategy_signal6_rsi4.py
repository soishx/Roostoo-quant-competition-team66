#R

import os
import glob
import pandas as pd
import numpy as np

# 1. Load Binance Vision OHLCV Data (Automatically merge all monthly CSV files)
csv_files = sorted(glob.glob('data/BTCUSDT-1h-*.csv'))
if not csv_files:
    csv_files = sorted(glob.glob('../../data/BTCUSDT-1h-*.csv'))

if not csv_files:
    raise FileNotFoundError("data/ 目录下未找到符合条件的 CSV 文件！")

print(f"Loading {len(csv_files)} monthly files: {[os.path.basename(f) for f in csv_files]}")

df_list = [pd.read_csv(f, header=None).iloc[:, :6] for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

# 2. Calculate 14-period RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# 3. Rigorous Event-Driven Simulation with 1.5% Stop-Loss
position = 0
entry_price = 0.0
stop_loss_pct = 0.015
equity = [1.0]
trades_count = 0

for i in range(1, len(df)):
    prev_rsi = df['rsi'].iloc[i-1]
    current_open = df['open'].iloc[i]
    current_high = df['high'].iloc[i]
    current_low = df['low'].iloc[i]
    current_close = df['close'].iloc[i]
    prev_close = df['close'].iloc[i-1]
    current_equity = equity[-1]

    if position == 1:
        sl_price = entry_price * (1 - stop_loss_pct)
        if current_low <= sl_price:
            ret = (sl_price - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - 0.001)  # Deduct 0.1% Fee
            position = 0
            trades_count += 1
        elif prev_rsi > 75:
            ret = (current_open - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - 0.001)  # Deduct 0.1% Fee
            position = 0
            trades_count += 1
        else:
            ret = (current_close - prev_close) / prev_close
            current_equity *= (1 + ret)
    elif position == 0:
        if prev_rsi < 25:
            position = 1
            entry_price = current_open
            trades_count += 1
            current_equity *= (1 - 0.001)  # Deduct 0.1% Fee
            ret = (current_close - current_open) / current_open
            current_equity *= (1 + ret)

    equity.append(current_equity)

df['equity'] = equity

# 4. Standard Performance Metrics (Daily Equity Resampling for True Sharpe)
daily_equity = df['equity'].iloc[::24]
daily_returns = daily_equity.pct_change().dropna()

total_return = (df['equity'].iloc[-1] - 1) * 100
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0.0

print("\n==========================================================================================")
print("  Signal 6 Variant C: Optimized RSI Reversal (25/75 + 1.5% Stop-Loss)                      ")
print("==========================================================================================")
print(f"Cumulative Return: {total_return:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Trades Triggered: {trades_count}")
print("==========================================================================================\n")