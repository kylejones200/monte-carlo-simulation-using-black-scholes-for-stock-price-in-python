"""Generated from Jupyter notebook: Summary

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# getting historical data for RDS-A. This code calls the API and transforms the result into a DataFrame.
import numpy as np


def main():
    np.random.seed(4373)
    import datetime

    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import norm

    # %matplotlib inline  # Jupyter-only


    # --- code cell ---

    df = pd.read_csv("data/AOIL data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    ticker = "AOIL"


    # --- code cell ---

    df.head()


    # --- code cell ---

    # Plot of asset historical closing price
    df["adjClose"].plot(
        figsize=(10, 6),
        title=f"Price of {ticker} from {df.index.min()} to {df.index.max()}",
    )


    # --- code cell ---

    pred_end_date = datetime.datetime(2020, 6, 20)
    forecast_dates = [
        d if d.isoweekday() in range(1, 6) else np.nan
        for d in pd.date_range(df.index.max(), pred_end_date)
    ]
    intervals = len(forecast_dates)
    iterations = 1000
    # Preparing log returns from data
    log_returns = np.log(1 + df["adjClose"].pct_change())

    # Setting up drift and random component in relation to asset data
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(intervals, iterations)))

    # Takes last data point as startpoint point for simulation
    S0 = df["adjClose"].iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    # Applies Monte Carlo simulation in asset
    for t in range(1, intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    forecast_df = pd.DataFrame(price_list)


    forecast_df.plot(
        figsize=(10, 6), legend=False, title=f"{iterations} Simulated Future Paths"
    )


    # --- code cell ---

    # Plotting with a histogram

    x = forecast_df.values[-1]
    sigma = np.std(x)
    mu = np.mean(x)

    num_bins = 100

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, density=1, alpha=0.75)

    # add a 'best fit' line
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)
    ax.plot(bins, y, "--")
    ax.axvline(np.mean(x), color="r")
    ax.axvline(mu + sigma * 1.96, color="g", ls="--")
    ax.axvline(mu - sigma * 1.96, color="g", ls="--")
    ax.axvline(S0)
    ax.set_xlabel(f"Predicted Price on {pred_end_date}")
    ax.set_ylabel("Probability density")
    ax.set_title(rf"Histogram of {ticker}: $\mu={mu:.02f}$, $\sigma={sigma:.02f}$")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
