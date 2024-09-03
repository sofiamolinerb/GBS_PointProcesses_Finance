#!/usr/bin/env python
# coding: utf-8

# # Finding the Most Correlated Stocks Using Thermal Sampling
# 
# This code will find the most correlated stocks between two different sets using Thermal Sampli P, as discussed by Jahangiri et al. (2020) in their paper on Point Processes with Gaussian Boson Samplin* [^1 This algorithm can be computed with thermal samples and also provides a Permanental Point Process (it is less accurate but may work for several ccases).].
# 
# ## References
# 
# [^1]: Jahangiri, S., Arrazola, J. M., Quesada, N., & Killoran, N. (2020). Point Processes With Gaussian Boson Sampling. *Physical Review E*, 101, 022134. https://doi.org/10.1103/physreve.101.22134
# 

# In[34]:


import numpy as np
import pandas as pd
import seaborn as sns
import zipfile
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal
import yfinance as yf
import pandas as pd

# Function to extract closing prices for a specific stock
def extract_closing_prices(stock_ticker, df, start_date=None, end_date=None):
    if start_date and end_date:
        return df.loc[start_date:end_date, stock_ticker]
    elif start_date:
        return df.loc[start_date:, stock_ticker]
    elif end_date:
        return df.loc[:end_date, stock_ticker]
    else:
        return df[stock_ticker]

# Function to compute daily returns
def compute_daily_returns(df):
    daily_returns = df.pct_change().dropna()
    return daily_returns

#Function to compute log returns

def compute_log_returns(df):
    daily_returns = df.pct_change().dropna()
    log_returns = np.log(1 + daily_returns)
    return log_returns

# Function to compute correlation matrix jahanguiri
    
def compute_correlation_matrix_J(returns):
    n_days = len(returns)
    correlation_matrix_0 = sum(np.outer(returns.iloc[i], returns.iloc[i]) for i in range(n_days)) / n_days
    rescaling_factor = n_days
    correlation_matrix = correlation_matrix_0 * rescaling_factor
    #correlation_matrix = correlation_matrix + 1j * np.zeros_like(correlation_matrix)
    return correlation_matrix
    
 #Function to compute co-variance matrix from Python function

def compute_covariance_matrix(returns):
    n_days = len(returns)
    covariance_matrix  = np.cov(returns, rowvar=False)
    return covariance_matrix

#Compute Correlation matrix according to Python

def correlation_matrix (returns):
    correlation_matrix = daily_returns.corr()
    return correlation_matrix

# Function to perform Takagi-Autonne decomposition

def takagi_autonne_decomposition(A):
    A = (A + A.T.conj()) / 2
    U, s, _ = np.linalg.svd(A)
    lambdas = s
    return lambdas, U

# Function to solve for the constant c
def solve_for_c(lambdas, target_n_mean):
    def mean_photon_number(c):
        return np.sum((c * lambdas)**2 / (1 - (c * lambdas)**2))
    
    result = root_scalar(lambda c: mean_photon_number(c) - target_n_mean, bracket=[0, 1 / np.max(lambdas) - 1e-6])
    return result.root

# Function to calculate squeezing parameters
def calculate_squeezing_parameters(lambdas, c):
    squeezing_parameters = np.arctanh(c * lambdas)
    return squeezing_parameters

# Function to plot a heatmap
def plot_heatmap(matrix, title, ax=None, vmin=None, vmax=None):
    matrix_magnitude = np.abs(matrix)
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(title)


# # Thermal Sampling

# In[35]:


from strawberryfields.apps import sample
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal
# Function to generate samples

def sample(K, n_mean, n_samples):
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()


# In[44]:


# Load the data
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# Choose subsets of stocks

selected_stocks_diff_sector = ["AAPL", "JPM", "XOM", "JNJ", "DIS", "CAT", "PG", "NKE", "CLX"]
selected_stocks_same_sector = [ "MRO","DVN","COP", "KMI","COG", "PSX", "CVX","HAL", "XOM" ]   # Example energy stocks


# Filter the dataset for the selected stocks

filtered_df_diff_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_diff_sector)]
filtered_df_same_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_same_sector)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df_diff_sector = filtered_df_diff_sector.pivot(index='date', columns='Name', values='close')
pivot_df_same_sector = filtered_df_same_sector.pivot(index='date', columns='Name', values='close')

# Compute log returns
daily_returns_diff_sector = compute_log_returns(pivot_df_diff_sector)
daily_returns_same_sector = compute_log_returns(pivot_df_same_sector)

# Compute correlation matrices
correlation_matrix_diff_sector = compute_correlation_matrix_J(daily_returns_diff_sector)
correlation_matrix_same_sector = compute_correlation_matrix_J(daily_returns_same_sector)

# Perform Takagi-Autonne decomposition, i n this case you need to extract mu's
mu_diff, U_diff = takagi_autonne_decomposition(correlation_matrix_diff_sector)
mu_same, U_same = takagi_autonne_decomposition(correlation_matrix_same_sector)

# Solve for c given a target mean photon number
target_n_mean = 9
c_diff = solve_for_c(lambdas_diff, target_n_mean)
c_same = solve_for_c(lambdas_same, target_n_mean)

#remember these are thermal states and thus we do not need to calculate squeezing parameters

# Generate the kernel matrices: wrong this is only for squeezed
C_diff = U_diff @ np.diag(c_diff * mu_diff) @ U_diff.T.conj()
C_same = U_same @ np.diag(c_same * mu_same) @ U_same.T.conj()

# Generate samples
n_samples = 500  # number of samples to generate
samples_diff = sample(C_diff, target_n_mean, n_samples)
samples_same = sample(C_same, target_n_mean, n_samples)

# Plot the kernel matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot the correlation matrices with stock names on the axes
sns.heatmap(C_diff, annot=False, cmap="inferno", vmin=0, vmax=0.9, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0])
axes[0].set_title("Kernel Matrix - GBS (Different Sectors)")

sns.heatmap(C_same, annot=False, cmap="inferno", vmin=0, vmax=0.7, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[1])
axes[1].set_title("Kernel Matrix - GBS (Same Sector)")

plt.tight_layout()
plt.show()


# Plot the output state distributions
sample_counts_diff = np.sum(samples_diff, axis=0)
sample_counts_same = np.sum(samples_same, axis=0)

# Set common y-axis limits
max_count = max(sample_counts_diff.max(), sample_counts_same.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].bar(range(len(sample_counts_diff)), sample_counts_diff, color='skyblue')
axes[0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[0].set_xlabel('Mode')
axes[0].set_ylabel('Count')
axes[0].set_ylim(0, max_count+200)  # Set the same y-axis scaling for both plots
axes[0].set_title('Output State Distribution from GBS (Different Sectors)')

axes[1].bar(range(len(sample_counts_same)), sample_counts_same, color='skyblue')
axes[1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1].set_xlabel('Mode')
axes[1].set_ylabel('Count')
axes[1].set_ylim(0, max_count + 200)  # Set the same y-axis scaling for both plots
axes[1].set_title('Output State Distribution from GBS (Same Sector)')

plt.tight_layout()

plt.show()

# Additional verification to ensure the unitary matrix is indeed complex
print("Is the unitary matrix U_diff complex? ", np.iscomplexobj(U_diff))
print("Is the unitary matrix U_same complex? ", np.iscomplexobj(U_same))

# Verification to check if the matrices are unitary
identity_approx_diff = np.eye(U_diff.shape[0])
unitary_check_diff = np.allclose(U_diff @ U_diff.conj().T, identity_approx_diff)
print("Is the unitary matrix U_diff unitary? ", unitary_check_diff)

identity_approx_same = np.eye(U_same.shape[0])
unitary_check_same = np.allclose(U_same @ U_same.conj().T, identity_approx_same)
print("Is the unitary matrix U_same unitary? ", unitary_check_same)
plt.savefig(r"C:\Users\user\Downloads\exploredifferentstocks_thermal.pdf", format='pdf', dpi=300)

#It is actually interesting that when you use the same subsets of stocks in different places the correlations change between them, its like it accounts for the whole space and shows the correlations in general, which is MORE ODD since if ALL the stocks are very correlated then the overal matrix is actually less correlate


# # Comparing two sets of stocks from the same sector (tech vs energy) using Thermal Sampling
# 
# ## This code shows that Energy stocks are more correlated than Tech stocks
# 
# To confirm this see SP500 Analysis and compare two stock's closing prices form both sectors

# In[75]:


import numpy as np
import pandas as pd
import seaborn as sns
import zipfile
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal
import yfinance as yf
import pandas as pd

# Function to extract closing prices for a specific stock
def extract_closing_prices(stock_ticker, df, start_date=None, end_date=None):
    if start_date and end_date:
        return df.loc[start_date:end_date, stock_ticker]
    elif start_date:
        return df.loc[start_date:, stock_ticker]
    elif end_date:
        return df.loc[:end_date, stock_ticker]
    else:
        return df[stock_ticker]

# Function to compute daily returns
def compute_daily_returns(df):
    daily_returns = df.pct_change().dropna()
    return daily_returns

# Function to compute log returns
def compute_log_returns(df):
    daily_returns = df.pct_change().dropna()
    log_returns = np.log(1 + daily_returns)
    return log_returns

# Function to compute correlation matrix using the methodology from Jahanguiri et al.
def compute_correlation_matrix_J(returns):
    n_days = len(returns)
    correlation_matrix_0 = sum(np.outer(returns.iloc[i], returns.iloc[i]) for i in range(n_days)) / n_days
    rescaling_factor = n_days
    correlation_matrix = correlation_matrix_0 * rescaling_factor
    return correlation_matrix
    
# Function to compute covariance matrix
def compute_covariance_matrix(returns):
    n_days = len(returns)
    covariance_matrix  = np.cov(returns, rowvar=False)
    return covariance_matrix

# Function to perform Takagi-Autonne decomposition
def takagi_autonne_decomposition(A):
    A = (A + A.T.conj()) / 2
    U, s, _ = np.linalg.svd(A)
    lambdas = s
    return lambdas, U

# Function to solve for the constant c
def solve_for_c(lambdas, target_n_mean):
    def mean_photon_number(c):
        return np.sum((c * lambdas)**2 / (1 - (c * lambdas)**2))
    
    result = root_scalar(lambda c: mean_photon_number(c) - target_n_mean, bracket=[0, 1 / np.max(lambdas) - 1e-6])
    return result.root

# Function to calculate squeezing parameters
def calculate_squeezing_parameters(lambdas, c):
    squeezing_parameters = np.arctanh(c * lambdas)
    return squeezing_parameters

# Function to generate samples
def sample(K, n_mean, n_samples):
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()

# Load the data
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# Choose subsets of stocks
selected_stocks_energy_sector = ["ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU"]
selected_stocks_tech_sector = ["AAPL", "MSFT", "GOOGL", "AMZN", "PYPL", "CSCO", "NVDA", "ORCL", "ADBE"]

# Filter the dataset for the selected stocks
filtered_df_tech_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_tech_sector)]
filtered_df_energy_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_energy_sector)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df_tech_sector = filtered_df_tech_sector.pivot(index='date', columns='Name', values='close') #pivot automatically orders the stocks in alphabetical order
pivot_df_energy_sector = filtered_df_energy_sector.pivot(index='date', columns='Name', values='close')

# Compute log returns
daily_returns_tech_sector = compute_log_returns(pivot_df_tech_sector)
daily_returns_energy_sector = compute_log_returns(pivot_df_energy_sector)

# Compute correlation matrices
correlation_matrix_tech_sector = compute_correlation_matrix_J(daily_returns_tech_sector)
correlation_matrix_energy_sector = compute_correlation_matrix_J(daily_returns_energy_sector)


# Perform Takagi-Autonne decomposition
mu_tech, U_tech = takagi_autonne_decomposition(correlation_matrix_tech_sector)
mu_energy, U_energy = takagi_autonne_decomposition(correlation_matrix_energy_sector)

# Solve for c given a target mean photon number
target_n_mean = 9
c_tech = solve_for_c(mu_tech, target_n_mean)
c_energy = solve_for_c(mu_energy, target_n_mean)

# Remember these are thermal states adn thus there are no squeezing parameters

# Generate the kernel matrices
C_tech = U_tech @ np.diag(c_tech * mu_tech) @ U_tech.T.conj()
C_energy = U_energy @ np.diag(c_energy * mu_energy) @ U_energy.T.conj()

# Generate samples: you want to sample from the scaled mtrix according to the thermal state
n_samples = 400  # number of samples to generate
samples_tech = sample(C_tech, target_n_mean, n_samples)
samples_energy = sample(C_energy, target_n_mean, n_samples)

# Create the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Plot the kernel matrices
sns.heatmap(correlation_matrix_tech_sector, annot=False, cmap="inferno", vmin=0, vmax=1, 
            xticklabels=selected_stocks_tech_sector, yticklabels=selected_stocks_tech_sector, ax=axes[0, 0])
axes[0, 0].set_title("Kernel Matrix - GBS (Technology Sectors)")

sns.heatmap(correlation_matrix_energy_sector, annot=False, cmap="inferno", vmin=0, vmax=1, 
            xticklabels=selected_stocks_energy_sector, yticklabels=selected_stocks_energy_sector, ax=axes[0, 1])
axes[0, 1].set_title("Kernel Matrix - GBS (Energy Sector)")

# Plot the output state distributions
sample_counts_tech = np.sum(samples_tech, axis=0)
sample_counts_energy = np.sum(samples_energy, axis=0)

# Set common y-axis limits
max_count = max(sample_counts_tech.max(), sample_counts_energy.max())

axes[1, 0].bar(range(len(sample_counts_tech)), sample_counts_tech, color='skyblue')
axes[1, 0].set_xticks(range(len(selected_stocks_tech_sector)))
axes[1, 0].set_xticklabels(selected_stocks_tech_sector, rotation=90)
axes[1, 0].set_xlabel('Mode')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_ylim(0, max_count + 200)  # Set the same y-axis scaling for both plots
axes[1, 0].set_title('Output State Distribution from GBS (Technology Sectors)')

axes[1, 1].bar(range(len(sample_counts_energy)), sample_counts_energy, color='skyblue')
axes[1, 1].set_xticks(range(len(selected_stocks_energy_sector)))
axes[1, 1].set_xticklabels(selected_stocks_energy_sector, rotation=90)
axes[1, 1].set_xlabel('Mode')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_ylim(0, max_count + 200)  # Set the same y-axis scaling for both plots
axes[1, 1].set_title('Output State Distribution from GBS (Energy Sector)')

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\exploredifferentstocksthermal_evst.pdf", format='pdf', dpi=300)
plt.show()


# In[76]:


# Normalize the sample counts
normalized_sample_counts_tech = sample_counts_tech / np.sum(sample_counts_tech)
normalized_sample_counts_energy = sample_counts_energy / np.sum(sample_counts_energy)

# Calculate the area under the curves using the trapezoidal rule
auc_tech = np.trapz(normalized_sample_counts_tech, dx=1)
auc_energy = np.trapz(normalized_sample_counts_energy, dx=1)

# Print the AUC values
print(f"Area under the curve (Technology Sectors): {auc_tech}")
print(f"Area under the curve (Energy Sectors): {auc_energy}")

# Plot the normalized distributions
plt.figure(figsize=(12, 6))

# Plot for Technology Sectors
plt.plot(range(len(normalized_sample_counts_tech)), normalized_sample_counts_tech, marker='o', linestyle='-', color='b', label=f'Technology Sectors (AUC = {auc_tech:.4f})')
plt.fill_between(range(len(normalized_sample_counts_tech)), normalized_sample_counts_tech, alpha=0.2, color='b')

# Plot for Energy Sectors
plt.plot(range(len(normalized_sample_counts_energy)), normalized_sample_counts_energy, marker='o', linestyle='-', color='g', label=f'Energy Sectors (AUC = {auc_energy:.4f})')
plt.fill_between(range(len(normalized_sample_counts_energy)), normalized_sample_counts_energy, alpha=0.2, color='g')

# Add title, labels, grid, and legend
plt.title('Normalized Output State Distributions from GBS')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\normalized_distribution_auc.pdf", format='pdf', dpi=300)
plt.show()


# # Checking using closing prices that indeed the Energy Sector is more correlated

# In[77]:


import yahoo_fin.stock_info as yf
import matplotlib.pyplot as plt
import pandas as pd

# List of stock tickers from S&P 500 for two sectors
tech_tickers = ['AAPL', 'MSFT']  # Tech stocks
energy_tickers = ["COP", "DVN"]  # Energy stocks

# Initialize an empty dictionary to store data
data = {}

# Download the data for tech stocks with error handling
for ticker in tech_tickers:
    try:
        data[ticker] = yf.get_data(ticker, start_date="09/21/1988", end_date="08/09/2019")
        print(f"Downloaded data for {ticker}")
    except AssertionError as e:
        print(f"Could not download data for {ticker}: {e}")

# Download the data for energy stocks with error handling
for ticker in energy_tickers:
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

# Plot the closing prices side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot tech stocks on the first subplot
for ticker in tech_tickers:
    if ticker in closing_prices.columns:
        axes[0].plot(closing_prices.index, closing_prices[ticker], label=ticker)

axes[0].set_title('Tech Stocks Closing Prices')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Closing Price')
axes[0].legend()
axes[0].grid(True)

# Plot energy stocks on the second subplot
for ticker in energy_tickers:
    if ticker in closing_prices.columns:
        axes[1].plot(closing_prices.index, closing_prices[ticker], label=ticker)

axes[1].set_title('Energy Stocks Closing Prices')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Closing Price')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# In[23]:


import yahoo_fin.stock_info as yf
import matplotlib.pyplot as plt
import pandas as pd

# List of stock tickers from S&P 500 for two sectors
tech_tickers = ['AAPL', 'MSFT']  # Tech stocks
energy_tickers = ["COP", "DVN"]  # Energy stocks

# Initialize an empty dictionary to store data
data = {}

# Download the data for tech stocks with error handling
for ticker in tech_tickers:
    try:
        data[ticker] = yf.get_data(ticker, start_date="09/21/1988", end_date="08/09/2019")
        print(f"Downloaded data for {ticker}")
    except AssertionError as e:
        print(f"Could not download data for {ticker}: {e}")

# Download the data for energy stocks with error handling
for ticker in energy_tickers:
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
plt.figure(figsize=(14, 8))

# Plot tech stocks
for ticker in tech_tickers:
    if ticker in closing_prices.columns:
        plt.plot(closing_prices.index, closing_prices[ticker], label=f'{ticker} (Tech)', linestyle='-', color='blue')

# Plot energy stocks
for ticker in energy_tickers:
    if ticker in closing_prices.columns:
        plt.plot(closing_prices.index, closing_prices[ticker], label=f'{ticker} (Energy)', linestyle='--', color='orange')

# Add titles and labels
plt.title('Closing Prices of Selected Tech and Energy Stocks')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)  # Add grid lines if desired
plt.show()


# # This code shows that for several cases we can use Thermal Sampling for stock correlation analysis
