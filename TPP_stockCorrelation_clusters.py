#!/usr/bin/env python
# coding: utf-8

# In[11]:
import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import random
import strawberryfields as sf
from strawberryfields.apps import points, plot, sample
from strawberryfields.ops import *
from sklearn.datasets import make_blobs
from thewalrus import hafnian

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from itertools import product
from scipy.special import poch, factorial
from thewalrus.quantum import density_matrix_element, reduced_gaussian, Qmat, Xmat, Amat
from thewalrus.random import random_covariance
from thewalrus import (
    tor,
    ltor,
    threshold_detection_prob,
    numba_tor,
    numba_ltor,
    rec_torontonian,
    rec_ltorontonian,
)
from thewalrus.symplectic import two_mode_squeezing
from thewalrus._torontonian import numba_vac_prob


# In[13]:


def tor(A, fsum=False):
    """Returns the Torontonian of a matrix.

    For more direct control, you may wish to call :func:`tor_real` or
    :func:`tor_complex` directly.

    The input matrix is cast to quadruple precision
    internally for a quadruple precision torontonian computation.

    Args:
        A (array): a np.complex128, square, symmetric array of even dimensions.
        fsum (bool): if ``True``, the `Shewchuck algorithm <https://github.com/achan001/fsum>`_
            for more accurate summation is performed. This can significantly increase
            the `accuracy of the computation <https://link.springer.com/article/10.1007%2FPL00009321>`_,
            but no casting to quadruple precision takes place, as the Shewchuck algorithm
            only supports double precision.

    Returns:
        np.float64 or np.complex128: the torontonian of matrix A.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input matrix must be a NumPy array.")

    matshape = A.shape

    if matshape[0] != matshape[1]:
        raise ValueError("Input matrix must be square.")

    if A.dtype == np.complex:
        if np.any(np.iscomplex(A)):
            return tor_complex(A, fsum=fsum)
        return tor_real(np.float64(A.real), fsum=fsum)

    return tor_real(A, fsum=fsum)


# In[14]:


def is_positive_semidefinite(matrix, tol=1e-10):
    # Check if the input is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        return False, "The matrix is not square."
    
    # Compute the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    # Check if all eigenvalues are non-negative
    if np.all(eigenvalues >= -tol):  # Allowing a small tolerance for numerical stability
        return True, "The matrix is positive semidefinite."
    else:
        return False, "The matrix is not positive semidefinite."

def regularize_matrix(matrix, tol=1e-10):
    # Ensure the matrix is positive semidefinite
    matrix = matrix.astype(float)  # Convert the matrix to float type
    min_eig = np.min(np.linalg.eigvalsh(matrix))
    if min_eig < 0:
        matrix -= np.eye(matrix.shape[0]) * min_eig
    matrix += np.eye(matrix.shape[0]) * tol
    return matrix

def sample_torontonian_point_process(K, num_samples, tol=1e-10):
    """
    Sample points using a Torontonian point process from a given kernel matrix K.

    Parameters:
    K (numpy.ndarray): The kernel matrix.
    num_samples (int): The number of points to sample.

    Returns:
    list: Indices of the sampled points.
    """
    # Regularize the kernel matrix to ensure it is positive semidefinite
    K = regularize_matrix(K, tol)
    
    n = K.shape[0]
    sampled_indices = []

    for _ in range(num_samples):
        # Compute the torontonian for each subset of indices
        probabilities = []
        for i in range(n):
            subset_indices = sampled_indices + [i]
            subset_matrix = K[np.ix_(subset_indices, subset_indices)]
            try:
                tor_val = tor(subset_matrix)
                probabilities.append(tor_val)
            except np.linalg.LinAlgError:
                probabilities.append(0)  # Handle non-positive definite submatrices
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        if np.sum(probabilities) == 0:
            probabilities = np.ones_like(probabilities) / len(probabilities)  # Avoid division by zero
        else:
            probabilities /= np.sum(probabilities)
        
        # Sample a new index based on the probabilities
        new_index = np.random.choice(n, p=probabilities)
        sampled_indices.append(new_index)
    
    return sampled_indices


# In[37]:


# Function to calculate correlation matrix for selected stocks using all available dates
def calculate_correlation_matrix(selected_stocks):
    filtered_df = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks)]
    pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')
    pivot_df.replace(0, np.nan, inplace=True)
    daily_returns = pivot_df.pct_change().dropna()
    log_returns = np.log(1 + daily_returns)
    n_stocks = len(selected_stocks)
    n_days = len(daily_returns)
    print("The number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))
    correlation_matrix = log_returns.corr().values
    return correlation_matrix

def calculate_correlation_matrix_log(selected_stocks):
    # Filter the dataset for the selected stocks
    filtered_df = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks)]
    
    # Pivot the data to have stock names as columns and dates as rows
    pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')
    
    # Replace zeros with NaNs to handle missing data
    pivot_df.replace(0, np.nan, inplace=True)
    
    # Calculate the daily returns and drop rows with NaN values
    daily_returns = pivot_df.pct_change().dropna()
    
    # Compute the log returns
    log_returns = np.log(1 + daily_returns)
    
    # Number of stocks and days
    n_stocks = len(selected_stocks)
    n_days = len(daily_returns)
    print("The number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))
    
    # Compute the correlation matrix: Need to revise why we need that scaling factor
    correlation_matrix_0 = sum(np.outer(log_returns.iloc[i], log_returns.iloc[i]) for i in range(n_days)) /n)days
    rescaling_factor = n_days
    correlation_matrix = correlation_matrix_0 * rescaling_factor #this is included because in Jahangiri's paper there is no rescaling factor but itt seems to be needed here
    
    return correlation_matrix  # Corrected return statement



def calculate_correlation_matrix_daily(selected_stocks):
    # Filter the dataset for the selected stocks
    filtered_df = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks)]
    
    # Pivot the data to have stock names as columns and dates as rows
    pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')
    
    # Replace zeros with NaNs to handle missing data
    pivot_df.replace(0, np.nan, inplace=True)
    
    # Calculate the daily returns and drop rows with NaN values
    daily_returns = pivot_df.pct_change().dropna()
    
    # Number of stocks and days
    n_stocks = len(selected_stocks)
    n_days = len(daily_returns)
    print("The number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))
    
    # Compute the correlation matrix
    correlation_matrix_0 = sum(np.outer(daily_returns.iloc[i], daily_returns.iloc[i]) for i in range(n_days)) / n_days
    rescaling_factor = 1
    correlation_matrix_daily = correlation_matrix_0 * rescaling_factor
    
    return correlation_matrix_daily


# In[40]:


# Use log returns
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)
# List of selected stock names
selected_stocks = ["ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU", "KMI", "LRCX", "LUK", "MRO", "MU", "PCLN", "SNA", "VRTX"]

# Compute the correlation matrix
correlation_matrix_log = calculate_correlation_matrix_log(selected_stocks)
correlation_matrix_daily = calculate_correlation_matrix_daily(selected_stocks)
# Create a figure with one subplot
fig, axes = plt.subplots(1, 1, figsize=(10, 8))

# Plotting the original correlation matrix as a heatmap
sns.heatmap(correlation_matrix_log, annot=False, cmap="inferno", 
            xticklabels=selected_stocks, yticklabels=selected_stocks, 
            ax=axes, vmin=0, vmax=0.6)

axes.set_title("Correlation Matrix of S&P 500 Stocks")
plt.savefig(r"C:\Users\user\Downloads\jahanguiri17s.pdf", format='pdf', dpi=300) 
plt.show()


# In[41]:


# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Adjust the figsize for better sizing

# Plot the log return correlation matrix as a heatmap
sns.heatmap(correlation_matrix_log, annot=False, cmap="inferno", 
            xticklabels=selected_stocks, yticklabels=selected_stocks, 
            ax=axes[0], vmin=0, vmax=0.6)
axes[0].set_title("Correlation Matrix of S&P 500 Stocks (Log Returns)")

# Plot the daily return correlation matrix as a heatmap
sns.heatmap(correlation_matrix_daily, annot=False, cmap="inferno", 
            xticklabels=selected_stocks, yticklabels=selected_stocks, 
            ax=axes[1], vmin=0, vmax=0.6)
axes[1].set_title("Correlation Matrix of S&P 500 Stocks (Daily Returns)")

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\jahanguiri17s_logvsdailycorrmatrix.pdf", format='pdf', dpi=300) 
plt.show()


# # The code below samples from this matrix reproducing th eresults from Jahangiri et. al. which show that log returns can be used instead of daily returns.

# In[44]:


modes = 17
n_mean = 6 #mean number of output photons
samples = 300

s_thresh = sample.sample(correlation_matrix_log, n_mean, samples, threshold=True)
s_pnr = sample.sample(correlation_matrix_log, n_mean, samples, threshold=False)

#print(s_thresh)
#print(s_pnr)


# In[45]:


# Count the frequency of each stock in the threshold samples
stock_counts_thresh = np.sum(s_thresh, axis=0)

# Count the frequency of each stock in the PNR samples
stock_counts_pnr = np.sum(s_pnr, axis=0)

# Select the most frequently sampled stocks (Top 17)
most_sampled_indices_thresh = np.argsort(stock_counts_thresh)[-17:]
most_sampled_indices_thresh = most_sampled_indices_thresh[::-1]  # Reverse to get most sampled first

most_sampled_indices_pnr = np.argsort(stock_counts_pnr)[-17:]
most_sampled_indices_pnr = most_sampled_indices_pnr[::-1]  # Reverse to get most sampled first

most_sampled_stocks_thresh = [selected_stocks[i] for i in most_sampled_indices_thresh]
most_sampled_stocks_pnr = [selected_stocks[i] for i in most_sampled_indices_pnr]
centre_thres = most_sampled_stocks_thresh[:9]
centre_pnr = most_sampled_stocks_pnr[:9]

print("Most Sampled Stocks (Threshold):", most_sampled_stocks_thresh)
print("Most Sampled Stocks (PNR):", most_sampled_stocks_pnr)
print("2 most Sampled Stocks (thres):", centre_thres)


# ### Built with the correlation matrices from the two clusters developped before

# In[47]:


# Define the clusters
cluster_0 = ['MRO', 'DVN', 'COP', 'VRTX', 'HAL', 'MU', 'KMI', 'COG', 'LRCX', "ALXN"]
cluster_1 = ['ALXN', 'BA', 'CMI', 'HON', 'INTU', 'LUK', 'PCLN', 'SNA']

# Calculate the correlation matrices for each cluster
correlation_matrix_0 = calculate_correlation_matrix_log(cluster_0)
correlation_matrix_1 = calculate_correlation_matrix_log(cluster_1)

print("Correlation Matrix for Cluster 0:")
print(correlation_matrix_0)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Adjusted figure size

# Plot the correlation matrix for Cluster 0
sns.heatmap(correlation_matrix_0, annot=False, cmap="inferno", 
            xticklabels=cluster_0, yticklabels=cluster_0, ax=axes[0], vmin=0, vmax=0.7)
axes[0].set_title("Correlation Matrix of Cluster 0 Stocks", fontsize=20)  # Larger title font

# Plot the correlation matrix for Cluster 1
sns.heatmap(correlation_matrix_1, annot=False, cmap="inferno", 
            xticklabels=cluster_1, yticklabels=cluster_1, ax=axes[1], vmin=0, vmax=0.7)
axes[1].set_title("Correlation Matrix of Cluster 1 Stocks", fontsize=20)  # Larger title font

plt.tight_layout()
plt.savefig(r"C:\Users\user\Downloads\clusteringnicer.pdf", format='pdf', dpi=300) 
plt.show()



# In[ ]:





# In[ ]:




