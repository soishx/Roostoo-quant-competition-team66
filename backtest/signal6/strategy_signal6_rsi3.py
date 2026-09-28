import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Binance Vision OHLCV Data
df = pd.read_csv('data/BTCUSDT-1h-2026-08.csv', header=None)
df = df.iloc[:, :6]
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
        # 1. Check if intra-bar low hit Stop-Loss
        if current_low <= sl_price:
            ret = (sl_price - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - 0.001)  # Deduct 0.1% Fee
            position = 0
            trades_count += 1
        # 2. Check RSI exit signal
        elif prev_rsi > 70:
            ret = (current_open - prev_close) / prev_close
            current_equity *= (1 + ret) * (1 - 0.001)  # Deduct 0.1% Fee
            position = 0
            trades_count += 1
        else:
            ret = (current_close - prev_close) / prev_close
            current_equity *= (1 + ret)
    elif position == 0:
        if prev_rsi < 30:
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

print("\n==================================================================")
print("  Signal 6 Variant B: RSI Reversal (30/70 + 1.5% Stop-Loss)        ")
print("==================================================================")
print(f"Cumulative Return: {total_return:.2f}%")
print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Trades Triggered: {trades_count}")
print("==================================================================\n")