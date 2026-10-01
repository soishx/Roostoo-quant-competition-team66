import pandas as pd

from scipy.stats import pearsonr, spearmanr, ttest_1samp
import statsmodels.api as sm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "BTCUSDT_1h_2020-08_to_2023-08.csv"

# ============================================================
# RETURN CONFIGURATION
# ============================================================

PAST_RETURNS = [
    "past_return_1h",
    "past_return_2h",
    "past_return_3h"
]

FUTURE_RETURNS = [
    "future_return_1h",
    "future_return_2h",
    "future_return_3h"
]

# ============================================================
# MOMENTUM THRESHOLD
# ============================================================

# Change this value to test different momentum thresholds.
#
# 0.0   = 0%
# 0.001 = 0.1%
# 0.002 = 0.2%
# 0.005 = 0.5%
# 0.01  = 1.0%
#
# For example, 0.005 means:
# "Only consider periods where the past return
# was greater than 0.5%."

MOMENTUM_THRESHOLD = 0.05


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["open_time"]
    )

    df = df.dropna(
        subset=PAST_RETURNS + FUTURE_RETURNS
    )

    return df


# ============================================================
# 2. CORRELATION MATRIX
# ============================================================

def correlation_analysis(df):

    print("\n=========================")
    print("PEARSON CORRELATION")
    print("=========================")

    pearson_matrix = pd.DataFrame(
        index=PAST_RETURNS,
        columns=FUTURE_RETURNS,
        dtype=float
    )

    for past in PAST_RETURNS:

        for future in FUTURE_RETURNS:

            data = df[
                [past, future]
            ].dropna()

            r, p_value = pearsonr(
                data[past],
                data[future]
            )

            pearson_matrix.loc[
                past,
                future
            ] = r

    print(pearson_matrix)

    print("\n=========================")
    print("SPEARMAN CORRELATION")
    print("=========================")

    spearman_matrix = pd.DataFrame(
        index=PAST_RETURNS,
        columns=FUTURE_RETURNS,
        dtype=float
    )

    for past in PAST_RETURNS:

        for future in FUTURE_RETURNS:

            data = df[
                [past, future]
            ].dropna()

            rho, p_value = spearmanr(
                data[past],
                data[future]
            )

            spearman_matrix.loc[
                past,
                future
            ] = rho

    print(spearman_matrix)

    return pearson_matrix, spearman_matrix


# ============================================================
# 4. QUANTILE / BUCKET ANALYSIS
# ============================================================

def bucket_analysis(df):

    print("\n=========================")
    print("MOMENTUM BUCKET ANALYSIS")
    print("=========================")

    for past in PAST_RETURNS:

        print(f"\n--- {past} ---")

        # Divide past return into 5 groups:
        #
        # Q1 = lowest 20%
        # Q2 = 20-40%
        # Q3 = 40-60%
        # Q4 = 60-80%
        # Q5 = highest 20%

        buckets = pd.qcut(
            df[past],
            q=5,
            labels=[
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Q5"
            ]
        )

        for future in FUTURE_RETURNS:

            temp = pd.DataFrame({
                "bucket": buckets,
                "future_return": df[future]
            }).dropna()

            result = (
                temp
                .groupby("bucket", observed=True)
                ["future_return"]
                .agg(
                    count="count",
                    mean="mean",
                    median="median"
                )
            )

            result["mean_percent"] = (
                result["mean"] * 100
            )

            print(
                f"\n{past} -> {future}"
            )

            print(
                result[
                    [
                        "count",
                        "mean_percent",
                        "median"
                    ]
                ]
            )


# ============================================================
# 5. CONDITIONAL WIN RATE
# ============================================================

def conditional_win_rate(df):

    print("\n=========================")
    print("CONDITIONAL WIN RATE")
    print("=========================")

    # Change this threshold at the top of the file.
    threshold = MOMENTUM_THRESHOLD

    for past in PAST_RETURNS:

        condition = (
            df[past] > threshold
        )

        print(
            f"\n--- {past} > "
            f"{threshold:.2%} ---"
        )

        print(
            f"Observations satisfying condition: "
            f"{condition.sum():,}"
        )

        for future in FUTURE_RETURNS:

            future_positive = (
                df[future] > 0
            )

            win_rate = (
                future_positive[condition]
                .mean()
            )

            mean_future_return = (
                df.loc[
                    condition,
                    future
                ]
                .mean()
            )

            print(
                f"{future}: "
                f"win rate = {win_rate:.4%}, "
                f"mean return = "
                f"{mean_future_return:.4%}"
            )


# ============================================================
# 6. T-TEST
# ============================================================

def t_test_analysis(df):

    print("\n=========================")
    print("T-TEST")
    print("=========================")

    for past in PAST_RETURNS:

        condition = (
            df[past] > MOMENTUM_THRESHOLD
        )

        print(
            f"\n--- {past} > "
            f"{MOMENTUM_THRESHOLD:.2%} ---"
        )

        for future in FUTURE_RETURNS:

            future_returns = df.loc[
                condition,
                future
            ].dropna()

            t_stat, p_value = ttest_1samp(
                future_returns,
                popmean=0
            )

            print(
                f"{future}: "
                f"mean = "
                f"{future_returns.mean():.6%}, "
                f"t = {t_stat:.4f}, "
                f"p = {p_value:.6g}"
            )


# ============================================================
# 7. LINEAR REGRESSION
# ============================================================

def regression_analysis(df):

    print("\n=========================")
    print("LINEAR REGRESSION")
    print("=========================")

    for past in PAST_RETURNS:

        for future in FUTURE_RETURNS:

            data = df[
                [past, future]
            ].dropna()

            x = data[past]
            y = data[future]

            X = sm.add_constant(x)

            model = sm.OLS(
                y,
                X
            ).fit()

            coefficient = model.params[past]
            r_squared = model.rsquared
            p_value = model.pvalues[past]

            print(
                f"\n{past} -> {future}"
            )

            print(
                f"Coefficient: "
                f"{coefficient:.6f}"
            )

            print(
                f"R²: "
                f"{r_squared:.6f}"
            )

            print(
                f"p-value: "
                f"{p_value:.6g}"
            )



# ============================================================
# 9. MAIN
# ============================================================

def main():

    df = load_data()

    print(
        f"Number of observations: {len(df):,}"
    )

    correlation_analysis(df)

    bucket_analysis(df)

    conditional_win_rate(df)

    t_test_analysis(df)

    regression_analysis(df)


if __name__ == "__main__":
    main()