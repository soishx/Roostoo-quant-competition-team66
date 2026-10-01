import time
import hmac
import hashlib
import glob
import os
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlencode

# ============================================================
# API Credentials & Settings
# ============================================================
API_KEY = "srPm3Ubjj6ZLS7YyoLuGmwkyPGbB8NNrMziBuP2dwm1LmOX87JF4RyKO4wvjHv6Z"
API_SECRET = "pMCXz3IGal6BqlVr7D7qtIWh4u5SKOxho6v8lu7yweLnS8RuDqllEdjmSo9gkfqo"
BASE_URL = "https://mock-api.roostoo.com"

TRADING_PAIR = "SOL/USD"
BINANCE_CSV_PATTERN = "data/SOLUSDT-1h-*.csv"


# ============================================================
# Roostoo API Client
# ============================================================
class RoostooClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def _get_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature."""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _send_request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """Helper to sign and execute HTTP requests."""
        if params is None:
            params = {}

        params["timestamp"] = str(int(time.time() * 1000))
        query_string = urlencode(sorted(params.items()))
        signature = self._get_signature(query_string)

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        url = self.base_url + endpoint

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, data=params)

            return response.json()
        except Exception as e:
            print(f"[API Error] Request failed: {e}")
            return {"Success": False, "ErrMsg": str(e)}

    def get_ticker(self, pair: str = None) -> dict:
        """Fetch market ticker."""
        params = {}
        if pair:
            params["pair"] = pair
        return self._send_request("GET", "/v3/ticker", params)

    def get_balance(self) -> dict:
        """Fetch account balance information."""
        return self._send_request("GET", "/v3/balance")

    def place_order(self, pair: str, side: str, quantity: float, order_type: str = "MARKET", price: float = None) -> dict:
        """Place a spot buy or sell order."""
        params = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity)
        }
        if order_type.upper() == "LIMIT" and price:
            params["price"] = str(price)

        return self._send_request("POST", "/v3/place_order", params)

    def open_short(self, pair: str, collateral: float, order_type: str = "MARKET", price: float = None) -> dict:
        """Open short position (Uses /v6 endpoint)."""
        params = {
            "pair": pair,
            "collateral": str(collateral)
        }
        if order_type.upper() == "LIMIT" and price:
            params["order_type"] = "LIMIT"
            params["price"] = str(price)

        return self._send_request("POST", "/v6/short_open", params)

    def close_short(self, pair: str, close_pct: str = "100") -> dict:
        """Close short position (Uses /v6 endpoint)."""
        params = {
            "pair": pair,
            "close_pct": close_pct
        }
        return self._send_request("POST", "/v6/short_close", params)


# ============================================================
# Strategy Execution Core
# ============================================================
def load_historical_df() -> pd.DataFrame:
    """Load local CSV files to initialize technical indicators."""
    csv_files = sorted(glob.glob(BINANCE_CSV_PATTERN))
    if not csv_files:
        csv_files = sorted(glob.glob("../../" + BINANCE_CSV_PATTERN))

    if not csv_files:
        print("[Warning] Local CSV files not found. Creating a blank container.")
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    print(f"Loading {len(csv_files)} historical CSV files for RSI warming up...")
    df_list = [pd.read_csv(f, header=None).iloc[:, :6] for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_live_bot():
    client = RoostooClient(API_KEY, API_SECRET, BASE_URL)
    df = load_historical_df()

    position = 0  # 0: Flat, 1: Long Position Held
    print("\n==========================================")
    print("   Roostoo Live Strategy Bot Started      ")
    print("   Strategy: Signal 6 RSI Reversal (30/70)")
    print("==========================================\n")

    while True:
        try:
            # 1. Fetch latest market price from Roostoo
            ticker_resp = client.get_ticker(TRADING_PAIR)
            if not ticker_resp.get("Success"):
                print(f"[Market Data Error] {ticker_resp.get('ErrMsg')}")
                time.sleep(10)
                continue

            last_price = float(ticker_resp["Data"][TRADING_PAIR]["LastPrice"])
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {TRADING_PAIR} Current Price: {last_price}")

            # 2. Append latest price to DataFrame for real-time RSI update
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

            print(f"Current Calculated RSI: {current_rsi:.2f}")

            # 3. Execution logic based on signals
            if position == 0 and current_rsi < 30:
                print(">> Signal Triggered: BUY (RSI < 30)")
                # Example: Buy 1.0 SOL
                order = client.place_order(TRADING_PAIR, "BUY", quantity=1.0)
                print("Order Response:", order)
                if order.get("Success"):
                    position = 1

            elif position == 1 and current_rsi > 70:
                print(">> Signal Triggered: SELL (RSI > 70)")
                # Example: Sell 1.0 SOL
                order = client.place_order(TRADING_PAIR, "SELL", quantity=1.0)
                print("Order Response:", order)
                if order.get("Success"):
                    position = 0

            # Sleep before next poll (e.g., 60 seconds)
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nBot execution stopped by user.")
            break
        except Exception as e:
            print(f"Unexpected Exception: {e}")
            time.sleep(10)


if __name__ == "__main__":
    run_live_bot()