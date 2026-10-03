# exchange_client.py
import time
import hmac
import hashlib
import requests
from typing import Optional, Dict, Any

from config import BASE_URL, API_KEY, SECRET_KEY


class RoostooClient:
    """
    Thin wrapper around Roostoo REST API.

    Handles:
      - timestamp generation
      - HMAC-SHA256 signing for RCL_TopLevelCheck endpoints
      - consistent error handling (HTTP 200 + Success=false)
      - server time offset correction
    """

    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY,
                 secret_key: str = SECRET_KEY, timeout: int = 10):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.timeout = timeout
        self._server_offset_ms = 0   # serverTime - localTime

    # ---------------- Low-level helpers ----------------

    def _now_ms(self) -> str:
        return str(int(time.time() * 1000) + self._server_offset_ms)

    def _sign(self, params: Dict[str, str]) -> str:
        sorted_keys = sorted(params.keys())
        total = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        return hmac.new(
            self.secret_key.encode("utf-8"),
            total.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest(), total

    def _signed_headers(self, params: Dict[str, str]):
        params = dict(params)
        params["timestamp"] = self._now_ms()
        sig, total = self._sign(params)
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": sig,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        return headers, params, total

    # ---------------- Public endpoints ----------------

    def check_server_time(self) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/v3/serverTime", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            server_ms = int(data["ServerTime"])
            local_ms = int(time.time() * 1000)
            self._server_offset_ms = server_ms - local_ms
            return data
        except Exception as e:
            print(f"[serverTime] error: {e}")
            return None

    def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/v3/exchangeInfo", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[exchangeInfo] error: {e}")
            return None

    def get_ticker(self, pair: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {"timestamp": self._now_ms()}
        if pair:
            params["pair"] = pair
        try:
            r = requests.get(f"{self.base_url}/v3/ticker", params=params,
                             timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[ticker] error: {e}")
            return None

    # ---------------- Signed endpoints ----------------

    def get_balance(self) -> Optional[Dict[str, Any]]:
        headers, params, _ = self._signed_headers({})
        try:
            r = requests.get(f"{self.base_url}/v3/balance",
                             headers=headers, params=params,
                             timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[balance] error: {e}")
            return None

    def place_order(self, pair: str, side: str, quantity: float,
                    order_type: str = "MARKET",
                    price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                print("[place_order] LIMIT order requires price")
                return None
            payload["price"] = str(price)

        headers, params, total = self._signed_headers(payload)
        try:
            r = requests.post(f"{self.base_url}/v3/place_order",
                              headers=headers, data=total,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[place_order] error: {e}")
            return None

    def query_order(self, order_id: Optional[str] = None,
                    pair: Optional[str] = None,
                    pending_only: Optional[bool] = None,
                    limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        payload: Dict[str, str] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if limit is not None:
                payload["limit"] = str(limit)

        headers, params, total = self._signed_headers(payload)
        try:
            r = requests.post(f"{self.base_url}/v3/query_order",
                              headers=headers, data=total,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[query_order] error: {e}")
            return None

    def cancel_order(self, order_id: Optional[str] = None,
                     pair: Optional[str] = None) -> Optional[Dict[str, Any]]:
        payload: Dict[str, str] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair

        headers, params, total = self._signed_headers(payload)
        try:
            r = requests.post(f"{self.base_url}/v3/cancel_order",
                              headers=headers, data=total,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[cancel_order] error: {e}")
            return None
        # ---------------- Short endpoints (/v6) ----------------

    def short_open(self, pair: str, collateral: float,
                   price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Open (or add to) a short. Sized by collateral in USD, not qty.
        If `price` is given, places a LIMIT order; otherwise MARKET.
        """
        payload = {
            "pair": pair,
            "collateral": str(collateral),
        }
        if price is not None:
            payload["order_type"] = "LIMIT"
            payload["price"] = str(price)

        headers, params, total = self._signed_headers(payload)
        try:
            r = requests.post(f"{self.base_url}/v6/short_open",
                              headers=headers, data=total,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[short_open] error: {e}")
            return None

    def short_close(self, pair: str,
                    close_qty: Optional[float] = None,
                    close_pct: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Close all or part of a short.
        `close_qty` takes precedence over `close_pct`.
        Sending neither closes the whole position.
        """
        payload = {"pair": pair}
        if close_qty is not None:
            payload["close_qty"] = str(close_qty)
        elif close_pct is not None:
            payload["close_pct"] = str(close_pct)

        headers, params, total = self._signed_headers(payload)
        try:
            r = requests.post(f"{self.base_url}/v6/short_close",
                              headers=headers, data=total,
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[short_close] error: {e}")
            return None

    def get_short_positions(self) -> Optional[Dict[str, Any]]:
        headers, params, _ = self._signed_headers({})
        try:
            r = requests.get(f"{self.base_url}/v6/short_positions",
                             headers=headers, params=params,
                             timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[short_positions] error: {e}")
            return None