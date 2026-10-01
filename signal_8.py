import pandas as pd

# ============================================================
# 1. FILES
# ============================================================

FILES = {
    "BTC": "BTCUSDT_1h_2024-09_to_2026-08.csv",
    "ETH": "ETHUSDT_1h_2024-09_to_2026-08.csv",
    "SOL": "SOLUSDT_1h_2024-09_to_2026-08.csv",
    "BNB": "BNBUSDT_1h_2024-09_to_2026-08.csv"
}

# ============================================================
# 2. FIXED SIGNAL SETTINGS
# ============================================================

SHOCK_THRESHOLDS = [
    0.02,   # -2%
    0.03    # -3%
]

VOLUME_THRESHOLD = 1.0

LOOKBACK_HOURS = 6
FUTURE_HOURS = 6

ROUND_TRIP_FEE = 0.002   # 0.2%

# ============================================================
# 3. TIME PERIODS
# ============================================================

PERIODS = {
    "P1_2024-09_to_2025-02": (
        "2024-09-01",
        "2025-03-01"
    ),

    "P2_2025-03_to_2025-08": (
        "2025-03-01",
        "2025-09-01"
    ),

    "P3_2025-09_to_2026-02": (
        "2025-09-01",
        "2026-03-01"
    ),

    "P4_2026-03_to_2026-08": (
        "2026-03-01",
        "2026-09-01"
    )
}

# ============================================================
# 4. ANALYZE ONE ASSET
# ============================================================

def analyze_asset(asset, file_name):

    df = pd.read_csv(file_name)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # Past 6h return
    df["ret_6h"] = (
        df["close"].pct_change(
            LOOKBACK_HOURS
        )
    )

    # Previous 24h average volume
    df["volume_ma_24"] = (
        df["volume"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_ma_24"]
    )

    # Future 6h return
    df["future_ret_6h"] = (
        df["close"].shift(-FUTURE_HOURS)
        / df["close"]
        - 1
    )

    results = []

    for threshold in SHOCK_THRESHOLDS:

        for period_name, (
            start_date,
            end_date
        ) in PERIODS.items():

            start = pd.Timestamp(
                start_date,
                tz="UTC"
            )

            end = pd.Timestamp(
                end_date,
                tz="UTC"
            )

            period_mask = (
                (df["timestamp"] >= start)
                &
                (df["timestamp"] < end)
            )

            event_mask = (
                period_mask
                &
                (df["ret_6h"] < -threshold)
                &
                (
                    df["volume_ratio"]
                    < VOLUME_THRESHOLD
                )
            )

            sample = (
                df.loc[
                    event_mask,
                    "future_ret_6h"
                ]
                .dropna()
            )

            if len(sample) == 0:

                avg_return = None
                median_return = None
                win_rate = None
                net_return = None

            else:

                avg_return = sample.mean()

                median_return = sample.median()

                win_rate = (
                    sample > 0
                ).mean()

                net_return = (
                    avg_return
                    - ROUND_TRIP_FEE
                )

            results.append({

                "asset":
                    asset,

                "shock_threshold_pct":
                    threshold * 100,

                "period":
                    period_name,

                "events":
                    len(sample),

                "avg_future_6h_return_pct":
                    None
                    if avg_return is None
                    else avg_return * 100,

                "median_future_6h_return_pct":
                    None
                    if median_return is None
                    else median_return * 100,

                "rebound_win_rate_pct":
                    None
                    if win_rate is None
                    else win_rate * 100,

                "net_return_after_0.2pct_fee_pct":
                    None
                    if net_return is None
                    else net_return * 100
            })

    return results

# ============================================================
# 5. RUN ALL ASSETS
# ============================================================

all_results = []

for asset, file_name in FILES.items():

    print("\n")
    print("=" * 70)
    print(f"TESTING {asset}")
    print("=" * 70)

    asset_results = analyze_asset(
        asset,
        file_name
    )

    all_results.extend(
        asset_results
    )

# ============================================================
# 6. RESULT TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "hypothesis_8_time_stability_results.csv",
    index=False
)

# ============================================================
# 7. PRINT FULL TABLE
# ============================================================

print("\n\n")
print("=" * 100)
print("HYPOTHESIS 8 TIME STABILITY TEST")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# 8. SUMMARY BY ASSET + THRESHOLD
# ============================================================

summary = (
    results_df
    .groupby(
        [
            "asset",
            "shock_threshold_pct"
        ]
    )
    .agg(

        periods_tested=(
            "period",
            "count"
        ),

        positive_periods=(
            "avg_future_6h_return_pct",
            lambda x: (x > 0).sum()
        ),

        positive_after_fee_periods=(
            "net_return_after_0.2pct_fee_pct",
            lambda x: (x > 0).sum()
        ),

        avg_return_across_periods_pct=(
            "avg_future_6h_return_pct",
            "mean"
        ),

        total_events=(
            "events",
            "sum"
        )
    )
    .reset_index()
)

summary.to_csv(
    "hypothesis_8_time_stability_summary.csv",
    index=False
)

print("\n\n")
print("=" * 100)
print("SUMMARY BY ASSET")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

# ============================================================
# 9. CROSS-ASSET + CROSS-TIME SUMMARY
# ============================================================

cross_summary = (
    results_df
    .groupby(
        "shock_threshold_pct"
    )
    .agg(

        total_asset_periods=(
            "period",
            "count"
        ),

        positive_asset_periods=(
            "avg_future_6h_return_pct",
            lambda x: (x > 0).sum()
        ),

        positive_after_fee_asset_periods=(
            "net_return_after_0.2pct_fee_pct",
            lambda x: (x > 0).sum()
        ),

        avg_return_all_asset_periods_pct=(
            "avg_future_6h_return_pct",
            "mean"
        ),

        total_events=(
            "events",
            "sum"
        )
    )
    .reset_index()
)

cross_summary.to_csv(
    "hypothesis_8_time_cross_summary.csv",
    index=False
)

print("\n\n")
print("=" * 100)
print("CROSS-ASSET + CROSS-TIME SUMMARY")
print("=" * 100)

print(
    cross_summary.to_string(
        index=False
    )
)

print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Saved:\n"
    "1. hypothesis_8_time_stability_results.csv\n"
    "2. hypothesis_8_time_stability_summary.csv\n"
    "3. hypothesis_8_time_cross_summary.csv"
)
