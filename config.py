# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- Roostoo (execution) ----------------
BASE_URL   = "https://mock-api.roostoo.com"
API_KEY    = os.getenv("ROOSTOO_API_KEY", "")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "")

# ---------------- Binance (market data) ----------------
BINANCE_BASE_URL = "https://api.binance.com"
# Binance symbol mapping: Roostoo pair -> Binance symbol
BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "BNB/USD": "BNBUSDT",
    "BONK/USD": "BONKUSDT",
    "AMDB/USD": "AMDBUSDT",
    "CRCLB/USD": "CRCLBUSDT",
    "GOOGLB/USD": "GOOGLBUSDT",
    "INTCB/USD": "INTCBUSDT",
    "METAB/USD": "METABUSDT",
    "MUB/USD": "MUBUSDT",
    "NVDAB/USD": "NVDABUSDT",
    "SKHYB/USD": "SKHYBUSDT",
    "SNDKB/USD": "SNDKBUSDT",
}

# ---------------- Trading (asset pools by strategy) ----------------
# MA strategy universe — user may add bStock pairs here manually.
MA_PAIRS      = ["ETH/USD", "BTC/USD", "SOL/USD", "BNB/USD", "AMDB/USD", "CRCLB/USD", "GOOGLB/USD", "INTCB/USD", "METAB/USD"]
# Lead-lag strategy universe — teammate's semiconductor stocks.
LEADLAG_PAIRS = ["AMDB/USD", "INTCB/USD", "MUB/USD", "NVDAB/USD", "SKHYB/USD", "SNDKB/USD"]
# Union of both pools — used for reconciliation, tickers, and data fetching.
ALL_PAIRS     = sorted(set(MA_PAIRS) | set(LEADLAG_PAIRS))
INITIAL_CASH  = 100_000.0

# ---------------- Strategy ----------------
SHORT_MA_HOURS      = 24
LONG_MA_HOURS       = 48
ATR_PERIOD          = 24
RISK_PER_TRADE      = 0.005
ATR_STOP_MULTIPLIER = 3.0
MAX_POSITION_PCT    = 0.02

# ---------------- Portfolio-level risk ----------------
MAX_TOTAL_EXPOSURE  = 0.60   # total gross exposure cap (fraction of portfolio equity)
DRAWDOWN_TRIGGER    = 0.10   # circuit breaker triggers at this drawdown from peak
DRAWDOWN_RECOVER    = 0.05   # circuit breaker recovers below this drawdown

# ---------------- Lead-lag strategy (teammate's bStock pool) ----------------
LEADLAG_POSITION_FRACTION    = 0.05     # each slot = 5% of portfolio equity
LEADLAG_MAX_SLOTS            = 4        # max positions + pending  (=> 20% cap)
LEADLAG_MAX_PRICE_DIVERGENCE = 0.01     # skip if Binance/Roostoo diverge > 1%
LEADLAG_KLINE_LIMIT          = 500      # 15m breakout needs >= 390 bars
LEADLAG_LOOP_SECONDS         = 5        # lead-lag signal cadence (merged loop)

# BroadLeadLag (1m)
LEADLAG_BLL_EMA_SPAN     = 60
LEADLAG_BLL_PEER_RET     = 0.004
LEADLAG_BLL_LAG          = 0.001
LEADLAG_BLL_MIN_PEERS    = 4
LEADLAG_BLL_MIN_BARS     = 100
LEADLAG_BLL_COOLDOWN     = 600
LEADLAG_BLL_ENTRY_OFFSET = 0.0003
LEADLAG_BLL_ATR_PERIOD   = 14
LEADLAG_BLL_STOP_ATR     = 0.5
LEADLAG_BLL_TP_ATR       = 6.0
LEADLAG_BLL_EXPIRY       = 60

# MU->AMD (1m)
LEADLAG_MU2AMD_RET          = 0.005
LEADLAG_MU2AMD_LAG          = 0.001
LEADLAG_MU2AMD_ENTRY_OFFSET = 0.0
LEADLAG_MU2AMD_TIME_EXIT    = 20 * 60
LEADLAG_MU2AMD_EXPIRY       = 60

# AMD->MU (1m, 08:00-14:00 UTC)
LEADLAG_AMD2MU_WINDOW_START = 8
LEADLAG_AMD2MU_WINDOW_END   = 14
LEADLAG_AMD2MU_RET          = 0.0075
LEADLAG_AMD2MU_RET_BARS     = 3
LEADLAG_AMD2MU_LAG          = 0.001
LEADLAG_AMD2MU_COOLDOWN     = 600
LEADLAG_AMD2MU_ENTRY_OFFSET = 0.0003
LEADLAG_AMD2MU_TIME_EXIT    = 20 * 60
LEADLAG_AMD2MU_EXPIRY       = 60

# Breakout (15m)
LEADLAG_BO_EMA_SPAN     = 96
LEADLAG_BO_MIN_BARS     = 390
LEADLAG_BO_BREADTH      = 5
LEADLAG_BO_R3D_BARS     = 289
LEADLAG_BO_BREAK_MULT   = 1.001
LEADLAG_BO_ATR_PERIOD   = 14
LEADLAG_BO_ENTRY_OFFSET = 0.005
LEADLAG_BO_STOP_ATR     = 0.5
LEADLAG_BO_TP_ATR       = 4.0
LEADLAG_BO_EXPIRY       = 30 * 60

# ---------------- Fees / Slippage ----------------
MAKER_FEE = 0.0005   # limit order fee (0.05%)
TAKER_FEE = 0.001    # market order fee (0.1%)
SLIPPAGE  = 0.0005

# ---------------- Loop ----------------
LOOP_INTERVAL_SEC = 300              # MA strategy cadence (1h klines)
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

# Hard cap: short notional <= equity * SHORT_MAX_POSITION_PCT
SHORT_MAX_POSITION_PCT = 0.015

# Stop: tighter than long (long uses 1.5)
SHORT_ATR_MULTIPLIER = 2.5

# Tiered take-profit (in ATR multiples of profit)
SHORT_TP1_ATR      = 2.0
SHORT_TP1_FRACTION = 0.40   # close 40% of original short qty

SHORT_TP2_ATR      = 4.0
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
USE_ATR_STOP_LONG = False

# If True, ATR stop-loss is active for the short side.
# Take-profits (TP1/TP2) are independent of this and always apply
# when a short is open.
USE_ATR_STOP_SHORT = True

# =========================================================
# Long take-profit
# =========================================================

LONG_TP_ENABLED = True

# Same structure as short side: TP1 at 3.0 ATR closes 40%,
# TP2 at 6.0 ATR closes 30% of original qty.
# Remaining 30% exits via death-cross signal.
LONG_TP1_ATR      = 2.0
LONG_TP1_FRACTION = 0.40

LONG_TP2_ATR      = 4.0
LONG_TP2_FRACTION = 0.30
