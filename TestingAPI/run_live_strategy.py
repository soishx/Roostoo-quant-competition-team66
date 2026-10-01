import time
import hmac
import hashlib
import glob
import os
import requests
import pandas as pd
import numpy as np

# ============================================================
# API Credentials & Settings
# ============================================================
API_KEY = "srPm3Ubjj6ZLS7YyoLuGmwkyPGbB8NNrMziBuP2dwm1LmOX87JF4RyKO4wvjHv6Z"
SECRET_KEY = "pMCXz3IGal6BqlVr7D7qtIWh4u5SKOxho6v8lu7yweLnS8RuDqllEdjmSo9gkfqo"
BASE_URL = "https://mock-api.roostoo.com"

# Selected 4 mainstream trading pairs
TRADING_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]

# Risk Control Parameters
MAX_SINGLE_POSITION_PCT = 0.20  # Max 20% of available cash per position
MAX_CONCURRENT_POSITIONS = 3    # Max 3 concurrent positions


# ============================================================
# Roostoo API Standard Implementation
# ============================================================
def _get_timestamp():
    """Return a 13-digit millisecond timestamp as string."""
    return str(int(time.time() * 1000))


def _get_signed_headers(payload: dict = {}):
    """Generate signed headers and totalParams for RCL_TopLevelCheck endpoints."""
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        total_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        'RST-API-KEY': API_KEY,
        'MSG-SIGNATURE': signature
    }

    return headers, payload, total_params


def get_ticker(pair=None):
    """Get ticker for one or all pairs (RCL_TSCheck)."""
    url = f"{BASE_URL}/v3/ticker"
    params = {'timestamp': _get_timestamp()}
    if pair:
        params['pair'] = pair
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"[API Error] get_ticker: {e}")
        return None


def get_account_balance():
    """Fetch account balance to check available cash (USD/USDT)."""
    url = f"{BASE_URL}/v3/balance"
    headers, _, _ = _get_signed_headers({})
    try:
        res = requests.get(url, headers=headers, params={'timestamp': _get_timestamp()})
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"[API Error] get_account_balance: {e}")
        return None


def place_order(pair_or_coin, side, quantity, price=None, order_type=None):
    """Place a LIMIT or MARKET order strictly following Roostoo spec."""
    url = f"{BASE_URL}/v3/place_order"
    pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"

    payload = {
        'pair': pair,
        'side': side.upper(),
        'type': order_type.upper(),
        'quantity': str(round(quantity, 4))
    }
    if order_type == 'LIMIT':
        payload['price'] = str(price)

    headers, _, total_params = _get_signed_headers(payload)
    headers['Content-Type'] = 'application/x-www-form-urlencoded'

    try:
        res = requests.post(url, headers=headers, data=total_params)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"[API Error] place_order for {pair}: {e}")
        if e.response is not None:
            print(f"[Response Text]: {e.response.text}")
        return None


# ============================================================
# Strategy Initialization & Execution (Multi-Pair + Risk Control)
# ============================================================
def load_historical_df(coin_symbol: str) -> pd.DataFrame:
    """Load local CSV files for a specific coin to initialize RSI indicator."""
    pattern = f"data/{coin_symbol}USDT-1h-*.csv"
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        csv_files = sorted(glob.glob("../" + pattern))
    if not csv_files:
        csv_files = sorted(glob.glob("../../" + pattern))

    if not csv_files:
        print(f"[Warning] Local CSV files for {coin_symbol} not found. Starting with empty container.")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    print(f"Loading {len(csv_files)} historical files for {coin_symbol}...")
    df_list = [pd.read_csv(f, header=None).iloc[:, :6] for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate 14-period RSI."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_live_strategy():
    coin_data = {}
    positions = {} # 0: No position, 1: Holding position

    for pair in TRADING_PAIRS:
        coin_name = pair.split("/")[0]
        coin_data[pair] = load_historical_df(coin_name)
        positions[pair] = 0

    print("\n==================================================")
    print("   Roostoo Multi-Currency Strategy Bot (Signal 6) ")
    print(f"   Monitored Pairs: {TRADING_PAIRS}                ")
    print(f"   Risk Control: Max 20% Capital / Max 3 Positions")
    print("==================================================\n")

    while True:
        try:
            active_positions_count = sum(positions.values())

            for pair in TRADING_PAIRS:
                coin_symbol = pair.split("/")[0]
                
                # 1. Fetch latest market ticker
                ticker_resp = get_ticker(pair)
                if not ticker_resp or not ticker_resp.get("Success"):
                    print(f"[Market Data Error] Fetch ticker failed for {pair}, skipping...")
                    continue

                last_price = float(ticker_resp["Data"][pair]["LastPrice"])
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {pair} Price: {last_price}")

                # 2. Calculate real-time RSI
                new_row = {
                    'timestamp': int(time.time() * 1000),
                    'open': last_price, 'high': last_price,
                    'low': last_price, 'close': last_price, 'volume': 0.0
                }
                df_temp = pd.concat([coin_data[pair], pd.DataFrame([new_row])], ignore_index=True)
                current_rsi = calculate_rsi(df_temp).iloc[-1]

                print(f"   -> RSI: {current_rsi:.2f} | Position: {positions[pair]} | Active Count: {active_positions_count}")

                # 3. Strategy logic & Risk control evaluation
                # Open condition: No position & RSI < 30 & Under max concurrent limit (3)
                if positions[pair] == 0 and current_rsi < 30:
                    if active_positions_count >= MAX_CONCURRENT_POSITIONS:
                        print(f">> [{pair}] Signal Triggered (RSI < 30), but Max Concurrent Positions ({MAX_CONCURRENT_POSITIONS}) reached. Skipping.")
                        continue

                    print(f">> [{pair}] Signal Triggered: RSI < 30 -> Calculating 20% position size...")
                    
                    balance_resp = get_account_balance()
                    available_cash = 10000.0  # Fallback initial capital
                    if balance_resp and balance_resp.get("Success"):
                        wallet = balance_resp.get("Data", {})
                        available_cash = float(wallet.get("USD", wallet.get("USDT", 10000.0)))

                    target_allocation = available_cash * MAX_SINGLE_POSITION_PCT
                    quantity = target_allocation / last_price

                    print(f"   Available Cash: {available_cash:.2f} | 20% Allocation: {target_allocation:.2f} | Order Qty: {quantity:.4f}")
                    
                    order = place_order(coin_symbol, "BUY", quantity=quantity)
                    print(f"[{pair}] Buy Response:", order)
                    if order and order.get("Success"):
                        positions[pair] = 1
                        active_positions_count += 1

                # Close condition: Holding position & RSI > 70 -> Close position
                elif positions[pair] == 1 and current_rsi > 70:
                    print(f">> [{pair}] Signal Triggered: RSI > 70 -> Selling Spot (Close Position)...")
                    
                    order = place_order(coin_symbol, "SELL", quantity=1.0) 
                    print(f"[{pair}] Sell Response:", order)
                    if order and order.get("Success"):
                        positions[pair] = 0
                        active_positions_count = max(0, active_positions_count - 1)

            time.sleep(10)

        except KeyboardInterrupt:
            print("\nBot execution stopped by user.")
            break
        except Exception as e:
            print(f"Unexpected Exception: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_live_strategy()