# Team 66 — Roostoo Quant Trading Competition

An autonomous Python trading bot for the Hong Kong vs Australia vs India Quant Trading Hackathon. The bot derives signals from Binance public candle data and executes trades on Roostoo's mock exchange. It combines a moving-average trend strategy, volatility-based position sizing, partial profit taking, and shared portfolio entry controls.

The default configuration enables the moving-average long and short strategy. An optional semiconductor lead-lag and breakout module is included, but **new entries from that module are disabled** (`LEADLAG_ENABLED = False`). This is a rule-based strategy; it does not require an LLM or trained model.

## Strategy overview

### Active strategy: 24/48-hour moving-average crossover

`strategy.py` computes 24-bar and 48-bar simple moving averages from closed hourly candles. `market_data.py` discards the newest, unfinished candle. ATR is calculated over 24 bars using Wilder-style exponential smoothing.

| Event | Bot action |
| --- | --- |
| 24-hour SMA crosses above 48-hour SMA | Enter a long if no long is held; exit an existing short |
| 24-hour SMA crosses below or equals 48-hour SMA after being above | Exit an existing long; enter a short if no short is held |
| Long price reaches entry + 2 × entry ATR | Sell 40% of the original long quantity |
| Long price reaches entry + 4 × entry ATR, after TP1 | Sell another 30% of the original long quantity |
| Short price reaches entry − 2 × entry ATR | Close 40% of the current short; tighten its stop to entry − 0.3 × entry ATR |
| Short price reaches entry − 4 × entry ATR, after TP1 | Close 50% of the remaining short, targeting another 30% of the original position |

The remaining long quantity exits on a bearish crossover. The remaining short quantity exits through its stop or a bullish crossover. Short stops begin at entry + 2.5 × ATR. When profit reaches at least one current ATR, the strategy can lower the stop to current price + 2.5 × current ATR; it only tightens the stop.

**Long ATR stops are disabled in the committed defaults.** The 3 × ATR distance still determines long position sizing, but it is not an enforced loss limit. Short ATR stops are enabled. These exits are checked during the moving-average processing cycle, approximately every five minutes, and are implemented by the bot rather than resting exchange stop orders. A long take-profit takes precedence over a crossover exit within the same cycle; a short stop takes precedence over short take-profits.

The first evaluation with no saved moving-average snapshot records the indicators and does not enter. Later evaluations compare the latest indicators with the stored previous values, so the bot waits for a detected crossover rather than immediately entering an existing trend.

### Asset universe

The moving-average pool currently includes:

`ETH/USD`, `BTC/USD`, `SOL/USD`, `BNB/USD`, `ZEC/USD`, `ADA/USD`, `XRP/USD`, `AMDB/USD`, `CRCLB/USD`, `GOOGLB/USD`, `INTCB/USD`, and `METAB/USD`.

The optional lead-lag pool includes:

`AMDB/USD`, `INTCB/USD`, `MUB/USD`, `NVDAB/USD`, `SKHYB/USD`, and `SNDKB/USD`.

Roostoo USD pairs are mapped to Binance USDT symbols in `config.py`. Startup requires every pair in the union of both pools to exist on Roostoo, even when new lead-lag entries are disabled. Historical data availability for each mapped Binance symbol must also be checked; the mapping itself does not guarantee availability.

### Optional lead-lag module

`lead_lag_strategy.py` implements four long-only signal families:

- **BroadLeadLag:** buy a lagging semiconductor asset when peer returns are strong and the asset is above its EMA.
- **MU → AMD:** buy AMD when MU's one-minute return leads AMD sufficiently.
- **AMD → MU:** buy MU when AMD's three-minute return leads MU sufficiently, within the configured UTC trading window.
- **Breakout/retest:** use 15-minute candles, sector breadth, and trend filters to identify breakout entries.

Entries use limit buys; exits use market sells with ATR or time-based conditions. Defaults allocate 5% of equity per slot, permit up to four positions plus pending orders, and reject entries when the Roostoo last price differs from the proposed limit price by more than 1%. Pending orders have expiry handling. With `LEADLAG_ENABLED = False`, pending-order and existing-position management still run.

## Position sizing and portfolio controls

For a long, the initial target quantity is:

```text
risk budget = equity × 0.005
stop distance = ATR × 3.0
quantity = min(
    risk budget / stop distance,
    equity × 0.35 / price,
    available cash / (price × (1 + taker fee))
)
```

For a short, the risk budget is halved to 0.25% of equity, the sizing distance is 2.5 × ATR, and notional is capped at 35% of equity. Roostoo short orders are sized in USD collateral. Both sides are further constrained by the shared gross exposure cap.

| Setting | Default | Meaning |
| --- | --- | --- |
| `INITIAL_CASH` | 100,000 | Initial local portfolio and peak-equity state; live balances are reconciled from Roostoo |
| `MAX_POSITION_PCT` | 0.35 | Maximum target long notional as a fraction of equity |
| `SHORT_MAX_POSITION_PCT` | 0.35 | Maximum target short notional as a fraction of equity |
| `MAX_TOTAL_EXPOSURE` | 0.60 | Entry cap on combined long and short gross notional |
| `DRAWDOWN_TRIGGER` | 0.10 | Pause new entries at 10% drawdown from tracked peak equity |
| `DRAWDOWN_RECOVER` | 0.05 | Resume new entries at drawdown of 5% or less |
| `TAKER_FEE` | 0.001 | 0.1% fee assumption for market orders |
| `MAKER_FEE` | 0.0005 | 0.05% fee assumption for limit orders |

The circuit breaker blocks new entries and leaves exits to the strategies. The exposure cap gates entries; it does not automatically rebalance positions after price changes. ATR sizing budgets are estimates, not guarantees of maximum realized loss.

Portfolio equity is calculated from reconciled cash, long market value, and short unrealized P&L. Gross exposure sums long and short absolute market notionals. The client uses Roostoo's collateral-based short endpoints without an explicit leverage parameter.

## Runtime flow

1. Synchronize the Roostoo server-time offset and validate exchange availability and configured pairs.
2. Load saved pair, portfolio, and lead-lag state.
3. Approximately every five seconds, fetch balances, short positions, and Roostoo tickers; reconcile holdings and update the drawdown breaker.
4. Manage optional lead-lag pending orders and positions; evaluate new entries only when enabled.
5. Approximately every 300 seconds, fetch hourly Binance candles for each moving-average asset, evaluate exits and entries, execute applicable orders, and record decisions.
6. Save state and repeat. A portfolio reconciliation exception skips strategy processing for that cycle; individual strategy exceptions are logged.

The loop is sequential, so network requests, retries, and timeouts can extend these intervals. After startup, trades are driven by the strategy logic without manual buy/sell input.

## Repository structure

| File | Purpose |
| --- | --- |
| `run.py` | Startup, main loop, MA order execution, and simulated order helpers |
| `config.py` | Credentials, asset pools, strategy settings, risk limits, and runtime flags |
| `strategy.py` | MA/ATR indicators, crossover signals, sizing, and exit rules |
| `lead_lag_strategy.py` | Optional lead-lag/breakout signals and limit-order lifecycle |
| `market_data.py` | Binance public OHLCV requests, retries, and closed-candle filtering |
| `exchange_client.py` | Roostoo REST requests, HMAC-SHA256 signing, and time correction |
| `portfolio.py` | Wallet reconciliation, equity/exposure calculation, and entry gates |
| `state.py` | JSON persistence for pair, portfolio, and lead-lag state |
| `trade_logger.py` | Decision/error JSONL and fill CSV logs with Hong Kong timestamps |
| `requirements.txt` | Python dependencies |

## Installation and configuration

Use Python 3.10 or newer. On an Ubuntu EC2 instance:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv git
git clone --branch master https://github.com/soishx/Roostoo-quant-competition-team66.git
cd Roostoo-quant-competition-team66
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` in the repository root:

```dotenv
ROOSTOO_API_KEY=your_competition_api_key
ROOSTOO_SECRET_KEY=your_competition_secret_key
DRY_RUN=true
```

Protect the file with `chmod 600 .env`. Keep credentials out of Git commits and shared logs. Add `.env`, `.venv/`, `__pycache__/`, and generated `logs/` to your local Git exclusions or a reviewed `.gitignore` before committing runtime files. Binance market data requests do not require a Binance API key.

Strategy parameters are edited in `config.py`; only the two credentials and `DRY_RUN` are read from environment variables. Restart the process after configuration changes. If `DRY_RUN` is omitted, it defaults to `false`, which enables orders on the Roostoo mock exchange.

## Run the bot

From the repository root, with the virtual environment active:

```bash
python -u run.py
```

For competition execution, set `DRY_RUN=false` in `.env` and restart. The account needs valid Roostoo competition credentials, available pairs, and access to the `/v6/short_*` endpoints. The machine needs outbound HTTPS access to `mock-api.roostoo.com` and `api.binance.com`.

**Dry-run scope:** `DRY_RUN=true` simulates MA order placement and lead-lag entries/exits, while still reading live market data and authenticated account balances. It is not an offline backtest or an isolated paper portfolio: each cycle reconciles state against the real mock-exchange account. Pending-order expiry handling can still call the cancellation endpoint for saved pending orders. Use a separate checkout/state directory and an account without existing pending orders for a dry-run smoke check; do not treat its P&L as backtest evidence.

### Continuous deployment on AWS EC2

Use the competition-provided EC2 environment, install the dependencies above, and configure the credentials on the instance. One simple process-management option is `tmux`:

```bash
sudo apt-get install -y tmux
tmux new -s team66
source .venv/bin/activate
python -u run.py
```

Detach with `Ctrl+B`, then `D`; reconnect with `tmux attach -t team66`. This survives SSH disconnection but does not provide automatic restart after a crash or VM reboot. Run only one bot process per account/state directory. Preserve `logs/state/` across restarts because it holds entry metadata, take-profit flags, previous indicators, and the tracked equity peak.

Stop an interactive run with `Ctrl+C`. Stopping the process saves portfolio and lead-lag state but does not flatten open positions or cancel all pending orders. Bot-managed exits cannot execute while the process is stopped.

## Logs and judging evidence

The bot creates:

- `logs/decisions_<run_id>.jsonl`: MA signal snapshots, chosen actions, portfolio context, and error events.
- `logs/trades_<run_id>.csv`: fills, quantities, fees, order IDs, strategy reasons, and position context.
- `logs/state/state_<PAIR>.json`: per-asset indicators and position-management metadata.
- `logs/state/state_PORTFOLIO.json`: cash, tracked peak equity, and breaker status.
- `logs/state/state_LEADLAG.json`: optional strategy pending orders, positions, and signal timestamps.

Log timestamps use Hong Kong time (UTC+8); market-data timestamps and lead-lag time windows use UTC. Use trade logs together with Roostoo order/account records and the Git commit history to explain autonomous execution and strategy changes. Some short-close log equity fields are local approximations rather than a full portfolio valuation; use exchange records for authoritative performance reporting.

This checkout does not include backtest reports, a test suite, a performance-metric calculator, or verified live return/Sharpe/Sortino/Calmar results. No performance figures are claimed here. Competition active-day requirements must be demonstrated by actual execution records; a crossover strategy does not guarantee a trade every day.

## Operational limitations

- Binance USDT signals and Roostoo USD execution prices can differ. The optional lead-lag module has a divergence check; the MA path does not.
- Missing Binance candles skip MA processing for that asset, including its bot-managed exits. Startup also requires all configured union pairs to be available on Roostoo.
- Reconciliation reads exchange holdings, but reconstructing full entry/ATR/TP metadata depends on preserved local state. API failure handling does not validate every unsuccessful balance or short-position response before reconciliation.
- Both strategy pools share one account and some symbols. Enabling both requires review of ownership and reconciliation behavior for overlapping positions and locked limit-order balances.
- Optional minute-based lead-lag behavior should be reviewed against the competition's restrictions before enabling it. Its presence does not establish organizer approval or rule compliance.
- JSON state writes are ordinary file writes without atomic replacement. Use one process and maintain backups; interrupted writes can require recovery.

## Reference

[Roostoo API documentation](https://github.com/roostoo/Roostoo-API-Documents)

This README describes the `master` checkout reviewed at commit `e05c818badd05c6ad9a4d57426a538d96b4cb346`. Recheck the documented defaults when submitting a later strategy revision.
