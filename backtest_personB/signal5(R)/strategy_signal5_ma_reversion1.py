#R

import os
import glob
import pandas as pd
import numpy as np

def run_signal5_baseline():
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_base_vol',
        'taker_quote_vol', 'ignore'
    ]
    
    # 自动读取 data 目录下所有的 BTCUSDT 1h CSV 文件
    csv_files = sorted(glob.glob('data/BTCUSDT-1h-*.csv'))
    if not csv_files:
        csv_files = sorted(glob.glob('../../data/BTCUSDT-1h-*.csv'))

    if not csv_files:
        raise FileNotFoundError("data/ 目录下没有找到符合条件的 CSV 文件！")

    print(f"Loading {len(csv_files)} monthly files: {[os.path.basename(f) for f in csv_files]}")

    df_list = [pd.read_csv(f, header=None, names=columns) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['open_time'] = pd.to_datetime(df['open_time'], unit='us', errors='coerce')
    df = df.sort_values('open_time').reset_index(drop=True)

    # Strategy parameters

    # Strategy parameters
    ma_window = 20
    bias_buy = -0.02
    stop_loss_pct = 0.015
    fee_rate = 0.001

    # Indicators
    df['ma'] = df['close'].rolling(window=ma_window).mean()
    df['bias'] = (df['close'] - df['ma']) / df['ma']

    position = 0
    entry_price = 0.0
    equity = [1.0]
    trades_count = 0

    for i in range(1, len(df)):
        prev_bias = df['bias'].iloc[i-1]
        prev_close_price = df['close'].iloc[i-1]
        prev_ma = df['ma'].iloc[i-1]
        
        current_open = df['open'].iloc[i]
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        current_equity = equity[-1]

        if position == 1:
            sl_price = entry_price * (1 - stop_loss_pct)
            # Check Stop Loss
            if current_low <= sl_price:
                ret = (sl_price - prev_close_price) / prev_close_price
                current_equity *= (1 + ret) * (1 - fee_rate)
                position = 0
                trades_count += 1
            # Check MA Reversion Exit (Close >= MA)
            elif prev_close_price >= prev_ma:
                ret = (current_open - prev_close_price) / prev_close_price
                current_equity *= (1 + ret) * (1 - fee_rate)
                position = 0
                trades_count += 1
            else:
                ret = (current_close - prev_close_price) / prev_close_price
                current_equity *= (1 + ret)
        elif position == 0:
            if prev_bias <= bias_buy:
                position = 1
                entry_price = current_open
                trades_count += 1
                current_equity *= (1 - fee_rate)
                ret = (current_close - current_open) / current_open
                current_equity *= (1 + ret)

        equity.append(current_equity)

    df['equity'] = equity

    # Performance calculation
    daily_equity = df['equity'].iloc[::24]
    daily_returns = daily_equity.pct_change().dropna()

    total_return = (df['equity'].iloc[-1] - 1) * 100
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0.0

    print("\n==================================================")
    print("Signal 5: MA Reversion - Baseline")
    print("==================================================")
    print(f"Cumulative Return: {total_return:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Trades Triggered: {trades_count}")
    print("==================================================\n")

if __name__ == "__main__":
    run_signal5_baseline()