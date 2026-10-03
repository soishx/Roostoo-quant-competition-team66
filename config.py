# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- Roostoo (execution) ----------------
BASE_URL   = "https://mock-api.roostoo.com"
API_KEY    = os.getenv("ROOSTOO_API_KEY", "srPm3Ubjj6ZLS7YyoLuGmwkyPGbB8NNrMziBuP2dwm1LmOX87JF4RyKO4wvjHv6Z")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "pMCXz3lGaI6BqIVr7D7qtIWh4u5SKOxho6v8Iu7yweLnS8RuDqllEdjmSo9gkfqo")

# ---------------- Binance (market data) ----------------
BINANCE_BASE_URL = "https://api.binance.com"
# Binance symbol mapping: Roostoo pair -> Binance symbol
BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",   # Binance has no BTCUSD spot; USDT is the proxy
    "ETH/USD": "ETHUSDT",
}

# ---------------- Trading ----------------
PAIRS        = ["ETH/USD", "BTC/USD"]
INITIAL_CASH = 100_000.0

# ---------------- Strategy ----------------
SHORT_MA_HOURS      = 24
LONG_MA_HOURS       = 48
ATR_PERIOD          = 24
RISK_PER_TRADE      = 0.005
ATR_STOP_MULTIPLIER = 3.0
MAX_POSITION_PCT    = 0.50

# ---------------- Fees / Slippage ----------------
TAKER_FEE = 0.001
SLIPPAGE  = 0.0005

# ---------------- Loop ----------------
LOOP_INTERVAL_SEC = 65              # API: 1 trade/min max, leave headroom
KLINE_INTERVAL    = "1h"
KLINE_LIMIT       = 200             # enough for 48h MA + 24h ATR + buffer

# ---------------- Files ----------------
LOG_DIR   = "logs"
STATE_DIR = os.path.join(LOG_DIR, "state")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ---------------- Runtime mode ----------------
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# =========================================================
# Short strategy
# =========================================================

SHORT_ENABLED = True

# Short risk budget = equity * RISK_PER_TRADE * SHORT_RISK_SCALE
SHORT_RISK_SCALE = 0.5

# Hard cap: short notional <= equity * SHORT_MAX_POSITION_PCT (25%)
SHORT_MAX_POSITION_PCT = 0.25

# Stop: tighter than long (long uses 1.5)
SHORT_ATR_MULTIPLIER = 2.5

# Tiered take-profit (in ATR multiples of profit)
SHORT_TP1_ATR      = 3.0
SHORT_TP1_FRACTION = 0.40   # close 40% of original short qty

SHORT_TP2_ATR      = 6.0
SHORT_TP2_FRACTION = 0.30   # close another 30% of original short qty

# Remaining 30% exits via trailing stop

# Trailing stop
SHORT_TRAIL_ACTIVATION_ATR = 1.0   # once profit >= 1.0 ATR, move stop to breakeven

# Roostoo short endpoints
SHORT_OPEN_PATH   = "/v6/short_open"
SHORT_CLOSE_PATH  = "/v6/short_close"
SHORT_POSITIONS_PATH = "/v6/short_positions"

# =========================================================
# Exit mode toggles
# =========================================================

# If True, ATR stop-loss is active for the long side.
USE_ATR_STOP_LONG = True

# If True, ATR stop-loss is active for the short side.
# Take-profits (TP1/TP2) are independent of this and always apply
# when a short is open.
USE_ATR_STOP_SHORT = True