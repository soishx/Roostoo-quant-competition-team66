#R

import os
import pandas as pd
import numpy as np

# 1. Load market data
data_path = 'data/BTCUSDT-1h-2026-08.csv'
if not os.path.exists(data_path):
    data_path = '../../data/BTCUSDT-1h-2026-08.csv'

column_names = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
    'taker_buy_quote_volume', 'ignore'
]

df = pd.read_csv(data_path, header=None, names=column_names)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['open_time'] = pd.to_datetime(df['open_time'], unit='us', errors='coerce')
df = df.sort_values('open_time').reset_index(drop=True)

# 2. Strategy parameters (Variant C: Trend-Filtered Mean Reversion)
return_period = 1
z_window = 20
trend_window = 50
z_buy = -2.0
z_exit = 0.0
stop_loss_pct = 0.015
fee_rate = 0.001

# 3. Indicator calculation
df['returns'] = df['close'].pct_change(return_period)
df['ret_mean'] = df['returns'].rolling(window=z_window).mean()
df['ret_std'] = df['returns'].rolling(window=z_window).std()
df['z_score'] = (df['returns'] - df['ret_mean']) / df['ret_std']
df['ma_trend'] = df['close'].rolling(window=trend_window).mean()

# 4. Rigorous event-driven simulation
position = 0
entry_price = 0.0
equity = [1.0]
trades_count = 0

for i in range(1, len(df)):
    prev_z = df['z_score'].iloc[i-1]
    prev_close = df['close'].iloc[i-1]
    prev_ma = df['ma_trend'].iloc[i-1]
    
    current_open = df['open'].iloc[i]
    current_low = df['low'].iloc[i]
    current_close = df['close'].iloc[i]
    current_equity = equity[-1]

    if position == 1:
        sl_price = entry_price * (1 - stop_loss_pct)
        # Check intra-bar Stop Loss
        if current_low <= sl_price:
            ret = (sl_price - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - fee_rate)
            position = 0
            trades_count += 1
        # Check Z-Score Reversion Exit
        elif prev_z >= z_exit:
            ret = (current_open - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - fee_rate)
            position = 0
            trades_count += 1
        else:
            ret = (current_close - prev_close) / prev_close
            current_equity *= (1 + ret)
    elif position == 0:
        # Entry Filter: Previous Z < z_buy AND Previous Close > Previous MA50
        if (prev_z < z_buy) and (prev_close > prev_ma):
            position = 1
            entry_price = current_open
            trades_count += 1
            current_equity *= (1 - fee_rate)
            ret = (current_close - current_open) / current_open
            current_equity *= (1 + ret)

    equity.append(current_equity)

df['equity'] = equity

# 5. Standard performance metrics
daily_equity = df['equity'].iloc[::24]
daily_returns = daily_equity.pct_change().dropna()

total_return = (df['equity'].iloc[-1] - 1) * 100
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0.0

print("\n==================================================")
print("Signal 4 Variant C: Trend-Filtered (MA50 Filter)")
print("==================================================")
print(f"Cumulative Return: {total_return:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Trades Triggered: {trades_count}")
print("==================================================\n")