#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import random

# Helper functions
def decompose_kernel(L):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return {'V': eigenvectors, 'D': eigenvalues}

def sample_dpp(L_decomposed):
    D = L_decomposed['D'] / (1 + L_decomposed['D'])
    selected_indices = np.where(np.random.rand(len(D)) < D)[0]
    V = L_decomposed['V'][:, selected_indices]
    Y = []

    while V.shape[1] > 0:
        P = np.sum(V**2, axis=1)
        P = P / np.sum(P)
        i = np.random.choice(len(P), p=P)
        Y.append(i)
        j = np.argmax(np.abs(V[i, :]))
        Vj = V[:, j]
        V = V - np.outer(Vj, V[i, :] / Vj[i])
        Q, R = np.linalg.qr(V)
        V = Q[:, np.abs(np.diag(R)) > 1e-10]
    
    return np.array(Y)

# Path to the zip file
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# List of selected stock names
selected_stocks = ["CA", "CHK", "SIG", "LH", "AVB", "BMY", "EFX", "RRC", "TEL", "INTU", "PRGO", "NVDA", "SCG", "SLG", "DVN", "ISRG", "ARE", 
                   "NFLX", "ANDV", "BEN", "AMT", "COL", "DTE", "AMAT", "JBHT", "HON", "GGP", "HCN", "TGT", "DLR", "AJG", "CBG", "AMG", "EXC", 
                   "ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU", "KMI", "LRCX", "LUK", "MRO", "MU", "PCLN", "SNA", "VRTX"]

# Filter the dataset for the selected stocks
filtered_df = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')

# Calculate the daily returns and drop rows with NaN values
daily_returns = pivot_df.pct_change(fill_method=None).dropna()

# Compute the log returns
log_returns = np.log(1 + daily_returns)

# Number of stocks
n_stocks = len(selected_stocks)
n_days = len(daily_returns)
print(f"The number of stocks is {n_stocks} and the number of days used is {n_days}")

# Compute the correlation matrix
correlation_matrix_0 = sum(np.outer(log_returns.iloc[i], log_returns.iloc[i]) for i in range(n_days)) / n_days
rescaling_factor = n_days
correlation_matrix = correlation_matrix_0 * rescaling_factor

# Decompose the kernel matrix
L_decomposed = decompose_kernel(correlation_matrix)

# Sample from the DPP
dpp_sample_indices = sample_dpp(L_decomposed)
dpp_sample_stocks = [selected_stocks[i] for i in dpp_sample_indices]

# Sample independently
ind_sample_indices = random.sample(range(n_stocks), len(dpp_sample_indices))
ind_sample_stocks = [selected_stocks[i] for i in ind_sample_indices]

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(dpp_sample_stocks, [1]*len(dpp_sample_stocks), color='blue')
plt.title('DPP Sampled Stocks')
plt.xticks(rotation=90)
plt.ylabel('Selection (1 = Selected)')

plt.subplot(1, 2, 2)
plt.bar(ind_sample_stocks, [1]*len(ind_sample_stocks), color='red')
plt.title('Independent Sampled Stocks')
plt.xticks(rotation=90)
plt.ylabel('Selection (1 = Selected)')

plt.tight_layout()
plt.show()

print("DPP Sampled stocks:", dpp_sample_stocks)
print("Independent Sampled stocks:", ind_sample_stocks)

# Define the selected stocks from the image for DPP and PPP
dpp_selected_stocks_image = [
    "CA", "CHK", "SIG", "LH", "AVB", "BMY", "EFX", "RRC", "TEL", 
    "INTU", "PRGO", "NVDA", "SCG", "SLG", "DVN", "ISRG", "ARE"
]

ppp_selected_stocks_image = [
    "NFLX", "ANDV", "BEN", "AMT", "COL", "DTE", "AMAT", "JBHT", 
    "HON", "GGP", "HCN", "TGT", "DLR", "AJG", "CBG", "AMG", "EXC"
]

# Compare the DPP selected stocks with the ones from Jahanguiri et. al. (no need for this)
common_dpp_stocks = set(dpp_sample_stocks).intersection(dpp_selected_stocks_image)
print("\nCommon DPP Stocks between script and image:")
print(common_dpp_stocks)

# Compare the PPP selected stocks
common_ppp_stocks = set(ind_sample_stocks).intersection(ppp_selected_stocks_image)
print("\nCommon PPP Stocks between script and image:")
print(common_ppp_stocks)


# In[3]:


import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Helper functions
def decompose_kernel(L):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    return {'V': eigenvectors, 'D': eigenvalues}

def sample_dpp(L_decomposed, k):
    D = L_decomposed['D'] / (1 + L_decomposed['D'])
    selected_indices = np.where(np.random.rand(len(D)) < D)[0]
    V = L_decomposed['V'][:, selected_indices]
    Y = []

    while V.shape[1] > 0 and len(Y) < k:
        P = np.sum(V**2, axis=1)
        P = P / np.sum(P)
        i = np.random.choice(len(P), p=P)
        Y.append(i)
        j = np.argmax(np.abs(V[i, :]))
        Vj = V[:, j]
        V = V - np.outer(Vj, V[i, :] / Vj[i])
        Q, R = np.linalg.qr(V)
        V = Q[:, np.abs(np.diag(R)) > 1e-10]
    
    return np.array(Y)

# Path to the zip file
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# Consider all stocks in the CSV file
all_stocks = all_stocks_df['Name'].unique()

# Filter the dataset for all stocks
filtered_df = all_stocks_df[all_stocks_df['Name'].isin(all_stocks)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')

# Calculate the daily returns and drop rows with NaN values
daily_returns = pivot_df.pct_change(fill_method=None).dropna()

# Compute the log returns
log_returns = np.log(1 + daily_returns)

# Number of stocks
n_stocks = len(all_stocks)
n_days = len(daily_returns)
print(f"The number of stocks is {n_stocks} and the number of days used is {n_days}")

# Compute the correlation matrix
correlation_matrix_0 = sum(np.outer(log_returns.iloc[i], log_returns.iloc[i]) for i in range(n_days)) / n_days
rescaling_factor = n_days
correlation_matrix_J = correlation_matrix_0 * rescaling_factor

# Plot the overall correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_J, annot=False, cmap='coolwarm')
plt.title('Overall Stock Correlation Matrix')
plt.show()

correlation_matrix_python = log_returns.corr()
# Decompose the kernel matrix
L_decomposed = decompose_kernel(correlation_matrix_python)

# Number of stocks to sample (as in the image)
n_sample = 17  # This can be adjusted to match the number of stocks in the image

# Sample from the DPP
dpp_sample_indices = sample_dpp(L_decomposed, n_sample)
dpp_sample_stocks = [all_stocks[i] for i in dpp_sample_indices]

# Sample independently
ppp_sample_indices = random.sample(range(n_stocks), n_sample)
ppp_sample_stocks = [all_stocks[i] for i in ppp_sample_indices]

# Define the selected stocks from the image for DPP and PPP
dpp_selected_stocks_image = ['PRGO', 'SCG', 'DVN', 'AJG', 'LUK', 'DVN', 'AMAT', 'SLG', 'MRO', 'INTU', 'MU', 'AMT']
ppp_selected_stocks_image = ['EXC', 'DTE', 'NFLX', 'TGT', 'LUK', 'AMT', 'AVB', 'DLR', 'TEL', 'CHK', 'LH', 'DVN']

# Compare the DPP selected stocks
common_dpp_stocks = set(dpp_sample_stocks).intersection(dpp_selected_stocks_image)
print("\nCommon DPP Stocks between script and image:")
print(common_dpp_stocks)

# Compare the PPP selected stocks
common_ppp_stocks = set(ppp_sample_stocks).intersection(ppp_selected_stocks_image)
print("\nCommon PPP Stocks between script and image:")
print(common_ppp_stocks)

# Extract the stock prices for the selected stocks
dpp_prices = pivot_df[dpp_sample_stocks]
ppp_prices = pivot_df[ppp_sample_stocks]

# Compute the log returns for the selected stocks
dpp_log_returns = np.log(1 + dpp_prices.pct_change(fill_method=None)).dropna()
ppp_log_returns = np.log(1 + ppp_prices.pct_change(fill_method=None)).dropna()


# In[4]:


def compute_correlation_matrix_J(returns):
    n_days = len(returns)
    if n_days == 0:
        return np.zeros((returns.shape[1], returns.shape[1]))
    correlation_matrix_0 = sum(np.outer(returns.iloc[i], returns.iloc[i]) for i in range(n_days)) / n_days
    rescaling_factor = n_days
    correlation_matrix = correlation_matrix_0 * rescaling_factor
    return correlation_matrix


# In[5]:


# Calculate the correlation matrices for the selected stocks

correlation_matrix_dpp = compute_correlation_matrix_J(dpp_log_returns)
correlation_matrix_ppp = compute_correlation_matrix_J(ppp_log_returns)


# Plot the correlation matrices for the selected stocks
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix_dpp, annot=False, fmt=".2f", cmap='inferno', vmax=0.5)
plt.title('DPP Selected Stocks Correlation Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix_ppp, annot=False, fmt=".2f", cmap='inferno',vmax=0.5)
plt.title('PPP Selected Stocks Correlation Matrix')

plt.tight_layout()
plt.show()

