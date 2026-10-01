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

TRADING_PAIR = "SOL/USD"
BINANCE_CSV_PATTERN = "data/SOLUSDT-1h-*.csv"


# ============================================================
# Roostoo API Standard Implementation (Directly from README Demo)
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
    """Get ticker for one or all pairs."""
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
        'quantity': str(quantity)
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
        print(f"[API Error] place_order: {e}")
        if e.response is not None:
            print(f"[Response Text]: {e.response.text}")
        return None


# ============================================================
# Strategy Initialization & Execution (Signal 6 RSI)
# ============================================================
def load_historical_df() -> pd.DataFrame:
    """Load local CSV files to initialize RSI indicator."""
    csv_files = sorted(glob.glob(BINANCE_CSV_PATTERN))
    if not csv_files:
        csv_files = sorted(glob.glob("../" + BINANCE_CSV_PATTERN))
    if not csv_files:
        csv_files = sorted(glob.glob("../../" + BINANCE_CSV_PATTERN))

    if not csv_files:
        print("[Warning] Local CSV files not found. Starting with empty container.")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    print(f"Loading {len(csv_files)} historical CSV files for RSI warming up...")
    df_list = [pd.read_csv(f, header=None).iloc[:, :6] for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate 14-period RSI identical to strategy_signal6_rsi1.py."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_live_bot():
    df = load_historical_df()

    # 0 -> Flat (No position)
    # 1 -> Long Position (Spot position held)
    position = 0

    print("\n==========================================")
    print("   Roostoo Live Strategy Bot Started      ")
    print("   Strategy: Signal 6 RSI Reversal        ")
    print("==========================================\n")

    while True:
        try:
            # 1. Fetch latest market ticker
            ticker_resp = get_ticker(TRADING_PAIR)
            if not ticker_resp or not ticker_resp.get("Success"):
                print("[Market Data Error] Fetch ticker failed, retrying in 10s...")
                time.sleep(10)
                continue

            last_price = float(ticker_resp["Data"][TRADING_PAIR]["LastPrice"])
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {TRADING_PAIR} Current Price: {last_price}")

            # 2. Append latest price and recalculate RSI
            new_row = {
                'timestamp': int(time.time() * 1000),
                'open': last_price,
                'high': last_price,
                'low': last_price,
                'close': last_price,
                'volume': 0.0
            }
            df_temp = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            rsi_series = calculate_rsi(df_temp)
            current_rsi = rsi_series.iloc[-1]

            print(f"Current Calculated RSI: {current_rsi:.2f} | Current Position State: {position}")

            # 3. Strictly follow Signal 6 strategy decision logic
            # Open condition: position == 0 and RSI < 30 -> Buy Spot (BUY)
            if position == 0 and current_rsi < 30:
                print(">> Signal Triggered: RSI < 30 -> Buying Spot (LONG)...")
                order = place_order(TRADING_PAIR, "BUY", quantity=1.0)
                print("Buy Order Response:", order)
                if order and order.get("Success"):
                    position = 1

            # Close condition: position == 1 and RSI > 70 -> Sell Spot (SELL)
            elif position == 1 and current_rsi > 70:
                print(">> Signal Triggered: RSI > 70 -> Selling Spot (CLOSE LONG)...")
                order = place_order(TRADING_PAIR, "SELL", quantity=1.0)
                print("Sell Order Response:", order)
                if order and order.get("Success"):
                    position = 0

            time.sleep(10)

        except KeyboardInterrupt:
            print("\nBot execution stopped by user.")
            break
        except Exception as e:
            print(f"Unexpected Exception: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_live_bot()