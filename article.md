# Monte Carlo simulation using Black-Scholes for stock price in Python

This project continues another project I did looking at the price of
Telsa stock. Please note, this is not financial advice, just a fun...

::::::::### Monte Carlo simulation using Black-Scholes for stock price in Python 

This project continues another project I did looking at the price of
Telsa stock. Please note, this is not financial advice, just a fun
project.

[**Time Series Forecasting for Stock Prediction in Python**\
*This project introduces common techniques to manipulate time series and
make predictions using an example of a
stock...*python.plainenglish.io](https://python.plainenglish.io/time-series-forecasting-for-stock-prediction-in-python-710a88b7ccbb "https://python.plainenglish.io/time-series-forecasting-for-stock-prediction-in-python-710a88b7ccbb")[](https://python.plainenglish.io/time-series-forecasting-for-stock-prediction-in-python-710a88b7ccbb)
*I updated the code and graphs on 2024--12--23 to fix an error spotted
by* [*Amr Abdeldayem*](https://medium.com/u/69ea4839cf7f)*.
Thanks!*

I wanted to combine this approach with the [Black-Scholes
algo](https://www.investopedia.com/terms/b/blackscholes.asp) and then use Monte Carlo simulations to
predict the value of the Tesla stock in \~90 days.

Like before, I am pulling the data from YFinance because it is easy.

```python
# Import necessary libraries
import numpy as np
np.random.seed(3363)
import pandas as pd
from scipy.stats import norm
import datetime 
import matplotlib.pyplot as plt
%matplotlib inline

# Import yfinance and download data
import yfinance as yf

ticker = "TSLA"  # Tesla Stock
df = yf.download(ticker, period='5y')

# Plot using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label="Close Price")
plt.title(f"Price of {ticker} from {df.index.min().date()} to {df.index.max().date()}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)  # Optional: Add a grid for better readability
plt.show()
```


We pull in the values and plot the data. This is a nice looking time
series! You can see lots of volatility in the stock which makes it
interesting to imagine what will happen next?

#### Monte Carlo Simulation
We need to set two parameters to run the simulation. One is the number
of days we will predict into the future and the other is how many times
we will simulate the future. For this project, I set it future date to
be the beginning of the next financial quarter.

I like to do 1000 simulations. You can do more but there are diminishing
returns as you do more simulations (statistics).

In this code, I use the historic variability of the stock as a
constraint for how much the value of the stock can fluctuate from day to
day. The python goes through and takes the pervious value to predict the
current value. The prediction is based on the Black Scholes method and
includes historic volatility plus some randomness.

### Geometric Brownian Motion
I assume the log of the returns (percent changes) are normally
distributed and that the market is efficient. The formula for the change
in price between periods is the price of the stock in t_0 multiplied by
the expected drift (average change in price) plus an exogenous shock.


<figcaption>Formula for Geometric Brownian Motion. Nerd Alert! You don’t
have to actually use this because the code is below.
You’re welcome!</figcaption>


```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime
import matplotlib.pyplot as plt

def monte_carlo(pred_end_date, df, iterations=1000, plot=True):
    """
    Simulates future stock prices using the Monte Carlo method.

    Parameters:
        pred_end_date (datetime): The end date for the forecast.
        df (pd.DataFrame): Historical stock price data with a 'Close' column.
        iterations (int): Number of Monte Carlo iterations. Default is 1000.

    Returns:
        forecast_df (pd.DataFrame): Simulated future price paths.
        final_prices (np.ndarray): Final prices from each iteration.
    """
    
    # Validate date range
    if pred_end_date <= df.index.max():
        raise ValueError("Prediction end date must be later than the last available date in the data.")

    # Generate business days between the last available date and the prediction end date
    forecast_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), end=pred_end_date, freq='B')
    
    # Validate forecast_dates
    if forecast_dates.empty:
        raise ValueError("No forecast dates generated. Check the input date range.")

    # Number of intervals
    intervals = len(forecast_dates)

    # Prepare log returns from data
    log_returns = np.log(1 + df['Close'].pct_change().dropna())

    # Setting up drift and random component in relation to asset data
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    # Ensure drift and stdev are scalars (use .iloc[0] if they are Series)
    if hasattr(drift, 'iloc'):
        drift = drift.iloc[0]
    if hasattr(stdev, 'iloc'):
        stdev = stdev.iloc[0]
    
    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(intervals, iterations)))

    # Initialize price list for simulation
    price_list = np.zeros((intervals, iterations))
    price_list[0] = df['Close'].iloc[-1]

    # Apply Monte Carlo simulation
    for t in range(1, intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    # Convert results into DataFrame with correct index
    forecast_df = pd.DataFrame(price_list, index=forecast_dates)

    # Plot if needed
    if plot:
        forecast_df.plot(figsize=(10, 6), legend=False, title=f"{iterations} Simulated Future Paths")
        plt.xlabel("Date")
        plt.ylabel("Price")
                plt.show()

    # Extract the final simulated values
    end_values_df = forecast_df.iloc[-1].values.flatten()  # Extract the last row as a 1D array

    return forecast_df, end_values_df
```


#### Histogram of final predicted values
The simulated future paths graph is neat but very hard to read. I'm
interested in what the price will be on July 1, 2024. So I want to look
at the predicted values for that day only and see the distribution of
those values.

I can do this with a histogram. The code is based on another project I
did for histograms.

[**Visualizing the normal distribution with Python and Matplotlib**\
*This is a simple python project to show how to simulate a normal
distribution and plot it using
Matplotlib.*medium.com](https://medium.com/@kylejones_47003/visualizing-the-normal-distribution-with-python-and-matplotlib-c501e3c594f8 "https://medium.com/@kylejones_47003/visualizing-the-normal-distribution-with-python-and-matplotlib-c501e3c594f8")[](https://medium.com/@kylejones_47003/visualizing-the-normal-distribution-with-python-and-matplotlib-c501e3c594f8)
I assume the values are normally distributed because of the randomness
in the Monte Carlo process.

```python
def plot_norm_hist(s, vline=True, title=True):
    """
    Plots a histogram of the given data with a normal distribution overlay.

    Parameters:
        s (array-like): Input data (e.g., simulated final prices) to plot.
        vline (bool): Whether to include vertical lines at ±0.67 standard deviations. Default is True.
        title (bool): Whether to display a title with mean and standard deviation. Default is True.

    Returns:
        None: Displays the histogram with the normal distribution overlay.
    """
    mu, sigma = np.mean(s), np.std(s)  # Mean and standard deviation

    # Plot histogram and normal distribution
    count, bins, ignored = plt.hist(s, bins=30, density=True, alpha=0.75, color='blue')
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp(-(bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='red')

    # Add vertical lines for ±0.67σ (arbitrary choice for spread)
    if vline:
        lline = -0.67 * sigma + mu
        uline = 0.67 * sigma + mu
        plt.axvline(lline, color='green', linestyle='--', label=f"Lower Bound ({lline:.2f})")
        plt.axvline(uline, color='green', linestyle='--', label=f"Upper Bound ({uline:.2f})")

    # Add title and labels
    if title:
        plt.title(f"Final Price Distribution\nMean: ${mu:.2f}, Std Dev: ${sigma:.2f}")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
        plt.show()


# Example usage
ticker = "TSLA"
import yfinance as yf
df = yf.download(ticker, period='5y')

# Ensure prediction end date is valid
pred_end_date = datetime.datetime(2025, 7, 1)

try:
    forecast_df, end_values_df = monte_carlo(pred_end_date, df)
    # Plot the final price distribution
    plot_norm_hist(end_values_df, vline=True, title=True)
except ValueError as e:
    print(e)
```


Now I have a very simple chart and some useful info. The predicted mean
value for the stock is \$576.33 for July 1, 2025.

So this analysis suggests that the future price will very likely be more
than the current price. In fact, this approach predicts the value of
Tesla will be higher than the current price in 62.8% of the simulations.

Based on this, would you buy the stock today?

### Related Stories
- [[Time Series Forecasting for Stock Prediction in
  Python](https://medium.com/python-in-plain-english/time-series-forecasting-for-stock-prediction-in-python-710a88b7ccbb)]
- [[Visualizing the normal distribution with Python and
  Matplotlib](https://medium.com/@kylejones_47003/visualizing-the-normal-distribution-with-python-and-matplotlib-c501e3c594f8)]
- [[Building a Recommendation Engine using Association Rules for item to
  item similarity in
  R](https://medium.com/@kylejones_47003/association-rules-for-item-to-time-personalization-in-r-9d6de7d6db8e)]
::::::::::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[March 29, 2024](https://medium.com/p/808574935473).

[Canonical
link](https://medium.com/@kyle-t-jones/monte-carlo-simulation-using-black-scholes-for-stock-price-in-python-808574935473)

Exported from [Medium](https://medium.com) on November 10, 2025.
