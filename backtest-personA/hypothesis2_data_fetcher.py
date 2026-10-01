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


# ============================================================
# Configuration
# ============================================================

SYMBOL = input("Enter the trading pair: ")
INTERVAL = KlinesIntervalEnum["INTERVAL_1h"].value

# Same time period as Hypothesis 1
START_DATE = "2023-09-01"
END_DATE = "2026-08-01"

OUTPUT_FILE = f"{SYMBOL}_1h_hypothesis2_2023-09_to_2026-08.csv"

# Binance Spot API returns at most 1000 candles per request.
LIMIT = 1000


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================
# Binance client
# ============================================================

configuration_rest_api = ConfigurationRestAPI(
    api_key=os.getenv("API_KEY", ""),
    api_secret=os.getenv("API_SECRET", ""),
    base_path=os.getenv(
        "BINANCE_BASE_PATH",
        SPOT_REST_API_PROD_URL
    ),
)

client = Spot(config_rest_api=configuration_rest_api)


# ============================================================
# Helper: convert datetime to Binance timestamp
# ============================================================

def to_milliseconds(date_string):
    """
    Convert a YYYY-MM-DD date string to Unix timestamp in milliseconds.
    """

    timestamp = pd.Timestamp(date_string, tz="UTC")

    return int(timestamp.timestamp() * 1000)


# ============================================================
# Fetch historical 1-hour candles
# ============================================================

def fetch_klines(symbol, interval, start_date, end_date):
    """
    Download historical Binance Spot candlestick data.

    Binance limits each request to 1000 candles, so this function
    repeatedly requests data using startTime/endTime.
    """

    start_time = to_milliseconds(start_date)

    # Add one day so that the end date itself is included.
    end_time = to_milliseconds(end_date) + 24 * 60 * 60 * 1000 - 1

    all_rows = []

    current_start = start_time

    while current_start < end_time:

        logging.info(
            "Fetching candles starting from %s",
            pd.to_datetime(current_start, unit="ms", utc=True)
        )

        try:
            response = client.rest_api.klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=LIMIT,
            )

            rows = response.data()

        except Exception as e:
            logging.error("Error fetching data: %s", e)

            # Wait briefly before retrying.
            time.sleep(2)
            continue

        if not rows:
            logging.info("No more data returned by Binance.")
            break

        all_rows.extend(rows)

        # Binance returns candles in chronological order.
        # The first element is the opening timestamp.
        last_open_time = rows[-1][0]

        # Move one millisecond forward so we don't request
        # the same candle again.
        current_start = last_open_time + 1

        logging.info(
            "Received %d candles. Total: %d",
            len(rows),
            len(all_rows)
        )

        # Avoid unnecessarily hitting the API too quickly.
        time.sleep(0.1)

        # If Binance returned fewer than the maximum number,
        # we have probably reached the end.
        if len(rows) < LIMIT:
            break

    return all_rows


# ============================================================
# Convert raw Binance data to DataFrame
# ============================================================

def create_dataframe(rows):
    """
    Convert Binance kline data into a clean DataFrame.
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
        "ignore",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Convert timestamp to datetime.
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    # Convert price/volume columns from strings to numbers.
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert trade count to integer.
    df["number_of_trades"] = pd.to_numeric(
        df["number_of_trades"],
        errors="coerce"
    )

    # Sort chronologically.
    df = df.sort_values("open_time")

    # Remove duplicate candles.
    df = df.drop_duplicates(
        subset="open_time",
        keep="first"
    )

    # Use open_time as the index.
    df = df.set_index("open_time")

    return df


# ============================================================
# Create Hypothesis 2 variables
# ============================================================

def add_hypothesis2_columns(df):
    """
    Create the variables needed for the breakout-continuation
    hypothesis.

    Hypothesis:

        If the current closing price is above the highest
        price during the previous 24 hours, this is a breakout.

    Signal:

        close > previous_24h_high
    """

    # --------------------------------------------------------
    # Previous 24-hour high
    # --------------------------------------------------------
    #
    # We use:
    #
    #     rolling(24).max().shift(1)
    #
    # The rolling window contains the previous 24 completed
    # hourly candles.
    #
    # shift(1) is critical:
    # it prevents the current candle from being included in
    # the 24-hour high.
    #
    # This avoids look-ahead bias.
    # --------------------------------------------------------

    df["previous_24h_high"] = (
        df["high"]
        .rolling(window=24)
        .max()
        .shift(1)
    )

    # --------------------------------------------------------
    # Breakout signal
    # --------------------------------------------------------
    #
    # 1 = breakout occurred
    # 0 = no breakout
    #
    # The signal uses the current candle's closing price.
    # Therefore, in the eventual trading backtest, this signal
    # should be executed at the NEXT candle, not the current
    # candle's close.
    # --------------------------------------------------------

    df["signal"] = (
        df["close"] > df["previous_24h_high"]
    ).astype(int)

    return df


# ============================================================
# Main
# ============================================================

def main():

    logging.info(
        "Starting Hypothesis 2 data fetch for %s",
        SYMBOL
    )

    logging.info(
        "Period: %s to %s",
        START_DATE,
        END_DATE
    )

    # --------------------------------------------------------
    # 1. Download raw Binance data
    # --------------------------------------------------------

    rows = fetch_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if not rows:
        raise RuntimeError(
            "No data was returned from Binance."
        )

    # --------------------------------------------------------
    # 2. Create clean DataFrame
    # --------------------------------------------------------

    df = create_dataframe(rows)

    # --------------------------------------------------------
    # 3. Add Hypothesis 2 variables
    # --------------------------------------------------------

    df = add_hypothesis2_columns(df)

    # --------------------------------------------------------
    # 4. Keep relevant columns
    # --------------------------------------------------------

    output_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "previous_24h_high",
        "signal",
    ]

    df = df[output_columns]

    # --------------------------------------------------------
    # 5. Save CSV
    # --------------------------------------------------------

    df.to_csv(OUTPUT_FILE)

    # --------------------------------------------------------
    # 6. Print summary
    # --------------------------------------------------------

    logging.info(
        "Saved %d rows to %s",
        len(df),
        OUTPUT_FILE
    )

    logging.info(
        "Number of breakout signals: %d",
        df["signal"].sum()
    )

    logging.info(
        "First timestamp: %s",
        df.index.min()
    )

    logging.info(
        "Last timestamp: %s",
        df.index.max()
    )

    print("\nFirst 30 rows:")
    print(df.head(30))

    print("\nRows containing breakout signals:")
    print(
        df[df["signal"] == 1][
            ["close", "previous_24h_high", "signal"]
        ].head(20)
    )


if __name__ == "__main__":
    main()