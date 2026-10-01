import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

# ------------------------------------------------------------
# Corrected Testing Credentials
# ------------------------------------------------------------
API_KEY = "srPm3Ubjj6ZLS7YyoLuGmwkyPGbB8NNrMziBuP2dwm1LmOX87JF4RyKO4wvjHv6Z"
API_SECRET = "pMCXz3lGaI6BqIVr7D7qtIWh4u5SKOxho6v8Iu7yweLnS8RuDqllEdjmSo9gkfqo"
BASE_URL = "https://mock-api.roostoo.com"


def get_signature(query_string: str, secret: str) -> str:
    """Generate HMAC SHA256 signature required by Roostoo API."""
    return hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def check_balance():
    """Test API connectivity and fetch balance information."""
    endpoint = "/v3/balance"
    url = BASE_URL + endpoint

    timestamp = str(int(time.time() * 1000))
    params = {
        "timestamp": timestamp
    }

    query_string = urlencode(params)
    signature = get_signature(query_string, API_SECRET)

    headers = {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        print("HTTP Status Code:", response.status_code)
        print("\n=== Roostoo Server Response ===")
        print(response.json())
    except Exception as e:
        print("Connection failed:", e)


if __name__ == "__main__":
    print("Connecting to Roostoo Mock Trading Platform...")
    check_balance()