from hypothesis2_backtester import run_backtest, load_data
import matplotlib.pyplot as plt

# TP values to test
tp_values = [
    0.01,   # 1%
    0.02,   # 2%
    0.03,   # 3%
    0.04,   # 4%
    0.05,   # 5%
    0.06,   # 6%
    0.07,   # 7%
    0.08,   # 8%
    0.09,   # 9%
    0.10,   # 10%
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20
]


# Load data once
df = load_data()

returns = []
composite_scores = []

for tp in tp_values:

    print(f"Running backtest with TP = {tp:.2%}")

    portfolio = run_backtest(
        df,
        holding_period=0,
        take_profit=tp
    )

    total_return = portfolio.total_return()
    composite_score = portfolio.sharpe_ratio() * 0.3 + portfolio.calmar_ratio() * 0.3 + portfolio.sortino_ratio() * 0.4

    returns.append(total_return)
    composite_scores.append(composite_score)


# Plot results
plt.figure()
plt.plot(tp_values, returns, marker="o")

plt.xlabel("Take Profit")
plt.ylabel("Total Return")
plt.title("Hypothesis 2: Total Return vs Take Profit")
plt.grid(True)

plt.figure()
plt.plot(tp_values, composite_scores, marker="o")

plt.xlabel("Take Profit")
plt.ylabel("Composite Score")
plt.title("Hypothesis 2: Composite Score vs Take Profit")
plt.grid(True)


plt.show()