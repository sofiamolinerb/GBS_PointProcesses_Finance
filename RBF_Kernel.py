#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import zipfile
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from strawberryfields.apps import sample

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


# In[6]:


# Load the data
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# Choose subsets of stocks
selected_stocks_diff_sector = ["AAPL", "JPM", "XOM", "JNJ", "DIS", "CAT", "PG", "NKE", "CLX"]
selected_stocks_same_sector = ["MRO", "DVN", "COP", "KMI", "COG", "PSX", "CVX", "OXY", "PXD"] 
#These correspond to Energy


# In[9]:

# Define the RBF kernel function
def rbf_kernel(log_returns, sigma):
    n_stocks = log_returns.shape[1]
    K = np.zeros((n_stocks, n_stocks))
    
    # Compute the RBF kernel matrix
    for i in range(n_stocks):
        for j in range(n_stocks):
            diff = log_returns.iloc[:, i] - log_returns.iloc[:, j]
            K[i, j] = np.exp(-np.sum(diff**2) / (2 * sigma ** 2))
    
    return K

# Define sigma for the RBF kernel
sigma = 15  # Adjust this value to control the width of the Gaussian

# Compute the RBF kernel matrices for different and same sector stocks
K_diff_sector = rbf_kernel(log_returns_diff_sector, sigma)
K_same_sector = rbf_kernel(log_returns_same_sector, sigma)

# Plot the heat maps for both sectors
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(K_diff_sector, xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, cmap='viridis')
plt.title('RBF Kernel Matrix (Different Sectors)')

plt.subplot(1, 2, 2)
sns.heatmap(K_same_sector, xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, cmap='viridis')
plt.title('RBF Kernel Matrix (Same Sector)')

plt.tight_layout()
plt.show()


# In[10]:


# Generate samples
n_mean = 9
samples = 300

s_thresh_diff = sample.sample(K_diff_sector, n_mean, samples, threshold=True)
s_thresh_same = sample.sample(K_same_sector , n_mean, samples, threshold=True)
s_pnr_diff = sample.sample(K_diff_sector, n_mean, samples, threshold=False)
s_pnr_same = sample.sample(K_same_sector, n_mean, samples, threshold=False)

# Convert the lists to NumPy arrays
s_thresh_diff = np.array(s_thresh_diff)
s_thresh_same = np.array(s_thresh_same)
s_pnr_diff = np.array(s_pnr_diff)
s_pnr_same = np.array(s_pnr_same)


# In[ ]:




