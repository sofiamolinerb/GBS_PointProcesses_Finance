#!/usr/bin/env python
# coding: utf-8

# ## Computing the daily and log returns
# 
#  ### DISCLAIMER: Some of this analysis belongs to the Lecture Notes from Lukas Gonon, from Imperial College London
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



# In[7]:


# Understand why log returns should be used
# The below shows the histogram of the log return
sns.histplot(data=sp500['logreturn'], bins=50, stat="density") #density normalises to 1, it is a gaussian


# In[8]:


# Notice that the values are not the same as the previous plot because of the normalisation under stat="density".
sns.histplot(data=sp500['logreturn'], bins=50, stat="density", kde=True)


# Next, we try to plot a $N(\mu,\sigma)$ pdf on top of the fitted kernel density estimator. The $\mu$ and $\sigma$ are estimated from the sample mean and standard deviation of the daily log returns. Essentially a normalised with certain parameters.
# 
# Now we try to fit a kernel estimator below, The fitted kernel estimator has a more acute peak and heavier tails relative to the normal pdf.

# In[9]:


mu = sp500['logreturn'].mean()
sigma = sp500['logreturn'].std()

x = np.linspace(-10,10,1000)

# Add the plot of N(\mu,\sigma) pdf on top of the seaborn histogram
sns_ax = sns.histplot(data=sp500['logreturn'], bins=50, color="b", stat="density", kde=True, label="kernel fit")
sns_ax.plot(x,norm.pdf(x,mu,sigma),label="normal")
plt.legend()


# In[10]:


#Quantile-Quantile (Q-Q) plot, which is a graphical tool to help assess if a set of data plausibly came from some theoretical distribution such as a Normal distribution. 
#straight line corresponds to points following the normal distribution, this plot showcases the tails that come out of this distribution
z = (sp500['logreturn'] - mu)/sigma  # remember to normalise the data against the mean and sd!

sm.qqplot(z, line='45')
plt.show()


# # Now see how you can analyse why certain correlation patterns arise with GBS
# 
# Plotting th eclosing price we see that the correlations in daily returns may not hold constantly for long periods of time. Take as an exapmple two companies which would seem very related, and that depending on the time periods do not always show similarity in behaviour 
# 
# ### This analysis can help understanding correlations between stocks over larger periods of time, we see that tech companies do not always behave the same

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



# # Future areas of exploration: Volatitlity Clustering in ACF: from Imperial College London

# Below is the ACF plot of the daily return $\rho(k)=Cor(r_t,r_{t-k})$ for $k=0,1,...,50$. The correlation values are very small (beyond $k=0$). This suggests forecasting daily returns based on historical values tends to be hard because of the lack of (linear) dependence.
# 
# The graph represents the Autocorrelation Function (ACF) of daily returns for the S&P 500 index.
# 
# Here are the explanations of the axis labels:
# 
# - **Y-axis (Vertical):** This axis represents the autocorrelation values, which range from -1 to 1. Autocorrelation values measure the correlation of a time series with its own past values. An autocorrelation of 1 indicates perfect positive correlation, while -1 indicates perfect negative correlation. A value of 0 indicates no correlation.
# 
# - **X-axis (Horizontal):** This axis represents the number of lags, which are the time intervals by which the series is shifted to compute the autocorrelation. In this graph, lags range from 0 to 50.
# 
# The ACF plot helps in identifying any patterns or dependencies in the time series data. In this plot, the autocorrelations for all lags except the first one (lag 0) appear to be close to zero, which suggests that daily returns are approximately uncorrelated with past returns. This is consistent with the behavior of a random walk, which is often assumed for financial time series in efficient markets.

# In[11]:


sm.graphics.tsa.plot_acf(sp500['logreturn'].dropna(), lags=50, title="ACF of daily return") #this corresponds to the linear time relationship of the whole sandp
plt.show()


# But if we plot the instead the ACF of the absolute daily return $\rho(k)=Cor(|r_t|,|r_{t-k}|)$, the values are now much more significant. This indicates the phenomenon of **volatility clustering**: a large move today tends to result in another large move in the near future (although we are not certain about the direction of move). Here we do not take into account the direction of movement (where things could cancel out) but instead the absolute movement as a whole.
# 
# ## There is potential in using GBS to calculate this volatility clustering
# 
# The effect is also very persistent over time where $\rho(k)$ remains significant for large value of $k$: the impact of a large move today can influence the move many days beyond.

# In[14]:


sm.graphics.tsa.plot_acf(np.abs(sp500['logreturn']).dropna(), lags=50, title="ACF of absolute daily return")
plt.show()


# In[ ]:




