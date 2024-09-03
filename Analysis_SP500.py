#!/usr/bin/env python
# coding: utf-8

# ## Computing the daily and log returns
# 
# The column "adjclose" contains the time series of asset price $S_t$ that we want to work with. The difference between "close" and "adjclose" is that the latter has been properly adjusted for dividends payout.
# 
# Let's create a column of daily simple return and a column of daily log return. Simple return is given by $$R_t:=\frac{S_t-S_{t-1}}{S_{t-1}},$$ which can be computed by the panda function "pct_change()". The log return is given by $$r_t:=\ln S_t - \ln S_{t-1}=\ln(1+R_t).$$ whih we will attempt to use for our analysis since it does seem like there could be added value in profiting from it.
# 
# We store the values in two new pandas columns called "simplereturn" and "logreturn". We also multiply the two columns by 100 so the values are now in percentage terms.

# In[1]:


# Import necessary packages
import yahoo_fin.stock_info as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns      #seaborn which specialises in data visualisation tools
import plotly.express as px    #has gained popularity
from scipy.stats import norm # import the "norm" function from the Scipy.stats library as we will need the normal pdf function
import statsmodels.api as sm
from scipy.optimize import minimize #import minimize function from SciPy
# Style of plotting - not important here
plt.style.use('ggplot')
# Download the data of S&P500 (which Yahoo ticker is "^GSPC")
# Specify the data range in American format
sp500 = yf.get_data("^gspc", start_date="09/21/1988", end_date="08/09/2019")
# Print the first few rows of the dataset to verify the data
print(sp500.head())
# Plot the closing price of S&P 500
plt.figure(figsize=(10, 5))
plt.plot(sp500.index, sp500['close'], label='S&P 500')
plt.title('S&P 500 Closing Price (09/21/1988 - 08/09/2019)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.savefig(r"C:\Users\user\Downloads\SP500fig.pdf", format='pdf', dpi=300)
plt.show()


# In[2]:


sp500['simplereturn'] = sp500['adjclose'].pct_change() #this performs the formula above with an embedded package already installed
sp500['logreturn'] = np.log(1 + sp500['simplereturn']) 

# Multiply the return values by 100 so they represent percentage values
sp500['simplereturn']  = sp500['simplereturn']  * 100
sp500['logreturn'] = sp500['logreturn'] * 100


# In[3]:


import matplotlib.pyplot as plt

# Create a figure with two subplots, one on top of the other
fig, axs = plt.subplots(2, 1, figsize=(20, 10))  # 2 rows, 1 column

# First subplot: Adjusted closing price
axs[0].plot(sp500['adjclose'])
axs[0].set_ylabel(r"$S_t$ (adjusted closing price)")
axs[0].set_title(r"S&P500 (Adjusted Closing Price)")
axs[0].set_facecolor("white")

# Second subplot: Daily log returns
axs[1].plot(sp500['logreturn'])
axs[1].set_ylabel(r"$r_t$ (daily log returns in %)")
axs[1].set_title(r"S&P500 (Daily Log Returns)")
axs[1].set_facecolor("white")

# Adjust layout
plt.tight_layout()

# Save the combined figure as a high-resolution PDF
plt.savefig(r"C:\Users\user\Downloads\sp500logvsdaily_combined.pdf", format='pdf', dpi=300)

# Show the figure
plt.show()


# # Estract this data from specific groups of stocks

# In[5]:


# Import necessary packages
import yahoo_fin.stock_info as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# List of stock tickers from S&P 500
tickers = ['AAPL', 'MSFT']  # Example tickers

# Initialize an empty dictionary to store data
data = {}

# Download the data for these tickers with error handling
for ticker in tickers:
    try:
        data[ticker] = yf.get_data(ticker, start_date="09/21/1988", end_date="08/09/2019")
        print(f"Downloaded data for {ticker}")
    except AssertionError as e:
        print(f"Could not download data for {ticker}: {e}")

# Filter out tickers for which data was successfully downloaded
data = {ticker: df for ticker, df in data.items() if not df.empty}

# Calculate daily returns for each stock
returns = pd.DataFrame()
for ticker in data:
    returns[ticker] = data[ticker]['close'].pct_change()

# Plot the returns
plt.figure(figsize=(12, 6))
for ticker in returns.columns:
    plt.plot(returns.index, returns[ticker], label=ticker)

plt.title('Daily Returns of Selected S&P 500 Stocks')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()



# In[6]:


# Import necessary packages
import yahoo_fin.stock_info as yf
import matplotlib.pyplot as plt
import pandas as pd

# List of stock tickers from S&P 500
tickers = ['AAPL', 'MSFT']  # Example tickers

# Initialize an empty dictionary to store data
data = {}

# Download the data for these tickers with error handling
for ticker in tickers:
    try:
        data[ticker] = yf.get_data(ticker, start_date="09/21/1988", end_date="08/09/2019")
        print(f"Downloaded data for {ticker}")
    except AssertionError as e:
        print(f"Could not download data for {ticker}: {e}")

# Filter out tickers for which data was successfully downloaded
data = {ticker: df for ticker, df in data.items() if not df.empty}

# Create a DataFrame to hold the closing prices
closing_prices = pd.DataFrame()

for ticker in data:
    closing_prices[ticker] = data[ticker]['close']

# Plot the closing prices with a white background
plt.figure(figsize=(12, 6))
for ticker in closing_prices.columns:
    plt.plot(closing_prices.index, closing_prices[ticker], label=ticker)

plt.title('Closing Prices of Selected S&P 500 Stocks')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)  # Add grid lines if desired
plt.show()



# # this analysis can help understanding correlations between stocks over larger periods of time, we see that tech companies do not always behave the same
