import os
import logging
import time

import pandas as pd

from binance_sdk_spot.spot import (
    Spot,
    ConfigurationRestAPI,
    SPOT_REST_API_PROD_URL
)
from binance_sdk_spot.rest_api.models import KlinesIntervalEnum


# ============================================================
# 1. CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
)

client = Spot(config_rest_api=configuration_rest_api)


SYMBOL = "BTCUSDT"
INTERVAL = KlinesIntervalEnum["INTERVAL_1h"].value

START_DATE = "2020-08-01"
END_DATE = "2023-09-01"       # exclusive end date

OUTPUT_FILE = "BTCUSDT_1h_2020-08_to_2023-08.csv"


# ============================================================
# 2. DOWNLOAD ONE MONTH OF DATA
# ============================================================

def fetch_month(start_date, end_date):
    """
    Download 1-hour BTCUSDT klines between start_date and end_date.

    Binance returns each kline approximately as:

    [
        open_time,
        open,
        high,
        low,
        close,
        volume,
        close_time,
        quote_asset_volume,
        number_of_trades,
        ...
    ]
    """

    try:
        response = client.rest_api.klines(
            symbol=SYMBOL,
            interval=INTERVAL,
            start_time=int(start_date.timestamp() * 1000),
            end_time=int(end_date.timestamp() * 1000),
            limit=1000,
        )

        data = response.data()

        logging.info(
            f"Downloaded {start_date} -> {end_date}: "
            f"{len(data)} rows"
        )

        return data

    except Exception as e:
        logging.error(
            f"Error downloading {start_date} -> {end_date}: {e}"
        )
        return []


# ============================================================
# 3. CONVERT RAW BINANCE DATA INTO A DATAFRAME
# ============================================================

def klines_to_dataframe(raw_data):
    """
    Convert Binance's raw kline response into a pandas DataFrame.
    """

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore"
    ]

    df = pd.DataFrame(raw_data, columns=columns)

    # Convert timestamps
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # Convert numerical columns from strings to floats
    numerical_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column])

    return df


# ============================================================
# 4. DOWNLOAD THE WHOLE PERIOD MONTH BY MONTH
# ============================================================

def download_data():

    all_data = []

    # Generate monthly boundaries
    month_starts = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="MS",
        tz="UTC"
    )

    for i in range(len(month_starts) - 1):

        month_start = month_starts[i]
        month_end = month_starts[i + 1]

        raw_data = fetch_month(
            month_start,
            month_end
        )

        if raw_data:
            month_df = klines_to_dataframe(raw_data)
            all_data.append(month_df)

        # Avoid sending requests too quickly
        time.sleep(0.2)

    if not all_data:
        raise RuntimeError("No data was downloaded.")

    df = pd.concat(all_data, ignore_index=True)

    # Remove duplicate candles
    df = df.drop_duplicates(
        subset="open_time"
    )

    # Sort chronologically
    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# 5. CLEAN THE DATA
# ============================================================

def clean_data(df):

    # Remove rows with missing prices
    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    # Make sure prices are positive
    df = df[
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0)
    ]

    # Check that high >= low
    df = df[
        df["high"] >= df["low"]
    ]

    # Check chronological order
    df = df.sort_values(
        "open_time"
    ).reset_index(drop=True)

    return df


# ============================================================
# 6. CALCULATE THE MOMENTUM VARIABLES
# ============================================================

def create_momentum_features(df):

    # --------------------------------------------------------
    # Past 3-hour return
    #
    # Example:
    #
    # close[t] = price now
    # close[t-3] = price 3 hours ago
    #
    # past_return_3h =
    #       close[t] / close[t-3] - 1
    # --------------------------------------------------------

    df["past_return_3h"] = (
        df["close"] /
        df["close"].shift(3)
        - 1
    )
    
    df["past_return_1h"] = (
        df["close"] /
        df["close"].shift(1)
        - 1
    )

    df["past_return_2h"] = (
        df["close"] /
        df["close"].shift(2)
        - 1
    )

    df["past_return_3h"] = (
        df["close"] /
        df["close"].shift(3)
        - 1
    )

    # --------------------------------------------------------
    # Past 6-hour return
    # --------------------------------------------------------

    df["past_return_6h"] = (
        df["close"] /
        df["close"].shift(6)
        - 1
    )

    # --------------------------------------------------------
    # FUTURE returns
    #
    # These are NOT available at the time the signal is made.
    #
    # We calculate them now only so that later we can evaluate
    # whether the hypothesis is actually true.
    # --------------------------------------------------------

    df["future_return_1h"] = (
        df["close"].shift(-1) /
        df["close"]
        - 1
    )

    df["future_return_2h"] = (
        df["close"].shift(-2) /
        df["close"]
        - 1
    )

    df["future_return_3h"] = (
        df["close"].shift(-3) /
        df["close"]
        - 1
    )

    return df


# ============================================================
# 7. CREATE THE TRADING SIGNAL
# ============================================================

def create_signal(df):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We first define the hypothesis mathematically.
    #
    # "Past 3h / 6h went up"
    #
    # For now we use 0% as the threshold.
    #
    # Later, we can test different thresholds such as:
    #
    # 0.1%
    # 0.2%
    # 0.5%
    # 1.0%
    # etc.
    # --------------------------------------------------------

    threshold = 0.0

    # 3-hour momentum signal
    df["signal_3h"] = (
        df["past_return_3h"] > threshold
    ).astype(int)

    # 6-hour momentum signal
    df["signal_6h"] = (
        df["past_return_6h"] > threshold
    ).astype(int)

    return df


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    logging.info("Starting data download...")

    # Download
    df = download_data()

    logging.info(
        f"Downloaded {len(df)} total rows."
    )

    # Clean
    df = clean_data(df)

    logging.info(
        f"After cleaning: {len(df)} rows."
    )

    # Create momentum variables
    df = create_momentum_features(df)

    # Create signals
    df = create_signal(df)

    # Remove rows where the calculations are impossible
    #
    # The first 6 rows cannot have a 6-hour past return.
    # The last 3 rows cannot have a 3-hour future return.
    #
    df = df.dropna(
        subset=[
            "past_return_3h",
            "past_return_6h",
            "future_return_1h",
            "future_return_2h",
            "future_return_3h"
        ]
    )

    # Save the processed dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logging.info(
        f"Saved processed data to {OUTPUT_FILE}"
    )

    # Show a small sample
    print("\nFirst 10 rows:")
    print(
        df[
            [
                "open_time",
                "close",
                "past_return_3h",
                "past_return_6h",
                "future_return_1h",
                "future_return_2h",
                "future_return_3h",
                "signal_3h",
                "signal_6h"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()