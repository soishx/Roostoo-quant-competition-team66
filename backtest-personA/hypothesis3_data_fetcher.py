import os
import logging
import time

import pandas as pd

from binance_sdk_spot.spot import (
    Spot,
    ConfigurationRestAPI,
    SPOT_REST_API_PROD_URL,
)
from binance_sdk_spot.rest_api.models import KlinesIntervalEnum


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SYMBOL = "BTCUSDT"
INTERVAL = KlinesIntervalEnum["INTERVAL_1h"].value

START_DATE = "2023-08-01"
END_DATE = "2023-08-15"

OUTPUT_FILE = f"{SYMBOL}_1h_{START_DATE}_to_{END_DATE}.csv"


# ---------------------------------------------------------
# Binance client
# ---------------------------------------------------------

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=os.getenv("BASE_PATH", SPOT_REST_API_PROD_URL),
)

client = Spot(config_rest_api=configuration_rest_api)


# ---------------------------------------------------------
# Moving average calculation
# ---------------------------------------------------------

def calculate_moving_average(df, hours):
    """
    Calculate a simple moving average using hourly closing prices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'close' column.

    hours : int
        Number of hours in the moving-average window.

    Returns
    -------
    pandas.Series
        Moving average of the closing price.
    """

    if hours <= 0:
        raise ValueError("hours must be greater than 0")

    return df["close"].rolling(window=hours).mean()


# ---------------------------------------------------------
# Fetch Binance hourly klines
# ---------------------------------------------------------

def fetch_klines(symbol, start_date, end_date):
    """
    Fetch hourly OHLCV data from Binance.

    Returns a DataFrame containing:
    timestamp, open, high, low, close, volume
    """

    logging.info(
        f"Starting data fetch for {symbol}"
    )
    logging.info(
        f"Period: {start_date} to {end_date}"
    )

    # Convert dates to Binance-compatible timestamps
    start_timestamp = int(
        pd.Timestamp(start_date, tz="UTC").timestamp() * 1000
    )

    # End date is inclusive, so use the beginning of the next day
    end_timestamp = int(
        (
            pd.Timestamp(end_date, tz="UTC")
            + pd.Timedelta(days=1)
        ).timestamp() * 1000
    )

    all_data = []

    current_start = start_timestamp

    while current_start < end_timestamp:

        logging.info(
            f"Fetching candles starting from "
            f"{pd.to_datetime(current_start, unit='ms', utc=True)}"
        )

        try:
            response = client.rest_api.klines(
                symbol=symbol,
                interval=INTERVAL,
                start_time=current_start,
                end_time=end_timestamp,
                limit=1000,
            )

            rate_limits = response.rate_limits
            logging.debug(
                f"Rate limits: {rate_limits}"
            )

            data = response.data()

            if not data:
                logging.info("No more data returned.")
                break

            all_data.extend(data)

            # Binance returns candles in chronological order.
            # The first element is the opening timestamp.
            last_timestamp = int(data[-1][0])

            # Move to the next candle.
            current_start = last_timestamp + 1

            # Avoid hitting Binance too quickly.
            time.sleep(0.1)

            # If fewer than 1000 candles were returned,
            # we have probably reached the end.
            if len(data) < 1000:
                break

        except Exception as e:
            logging.error(
                f"Error while fetching klines: {e}"
            )
            raise

    logging.info(
        f"Fetched {len(all_data)} raw candles."
    )

    # -----------------------------------------------------
    # Convert Binance response into DataFrame
    # -----------------------------------------------------

    # Binance kline format:
    #
    # [
    #   0  Open time
    #   1  Open
    #   2  High
    #   3  Low
    #   4  Close
    #   5  Volume
    #   6  Close time
    #   7  Quote asset volume
    #   8  Number of trades
    #   9  Taker buy base asset volume
    #   10 Taker buy quote asset volume
    #   11 Ignore
    # ]

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
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
            "ignore",
        ],
    )

    # Keep only the columns we need.
    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    # Convert numerical columns from strings to floats.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df[numeric_columns] = df[numeric_columns].astype(float)

    # Convert timestamp to UTC datetime.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    # Remove duplicate candles, if any.
    df = df.drop_duplicates(
        subset="timestamp"
    )

    # Sort chronologically.
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------
# Add moving-average features
# ---------------------------------------------------------

def add_moving_averages(df):
    """
    Add the 6-hour and 24-hour simple moving averages.

    Both averages are calculated from hourly closing prices.
    """

    df = df.copy()

    df["MA_6h"] = calculate_moving_average(
        df,
        hours=6
    )

    df["MA_24h"] = calculate_moving_average(
        df,
        hours=24
    )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    df = fetch_klines(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # Calculate moving averages.
    df = add_moving_averages(df)

    # Save to CSV.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    logging.info(
        f"Saved {len(df)} rows to {OUTPUT_FILE}"
    )

    logging.info(
        f"Columns: {list(df.columns)}"
    )

    print("\nFirst rows:")
    print(df.head(30))

    print("\nLast rows:")
    print(df.tail())


if __name__ == "__main__":
    main()