#!/usr/bin/env python
# coding: utf-8

# In[23]:


import yahoo_fin.stock_info as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
import random
import strawberryfields as sf
from strawberryfields.apps import points, plot, sample
from strawberryfields.ops import *
from sklearn.datasets import make_blobs
from thewalrus import hafnian, rec_torontonian as tor
from thewalrus.quantum import density_matrix_element, reduced_gaussian, Qmat, Xmat, Amat
from thewalrus.random import random_covariance
from thewalrus import (
    ltor,
    threshold_detection_prob,
    numba_tor,
    numba_ltor,
    rec_ltorontonian,
)
from thewalrus.symplectic import two_mode_squeezing
from thewalrus._torontonian import numba_vac_prob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from itertools import product
from scipy.special import poch, factorial


# ## Motivation: Simulate Hawkes Process
# 
# The `simulate_hawkes` function generates timestamps of events using given parameters `mu` (baseline intensity), `alpha` (excitation parameter), and `beta` (decay parameter) over a time horizon `T`. The function uses an iterative approach to simulate the process where the intensity of events depends on past events.
# 
# ```python
# def simulate_hawkes(mu, alpha, beta, T):
#     events = []
#     s = 0.0  # current time
#     while s < T:
#         intensity = mu + alpha * np.sum(np.exp(-beta * (s - np.array(events))))
#         u = np.random.uniform(0, 1)
#         s -= np.log(u) / intensity
#         d = np.random.uniform(0, 1)
#         if d < intensity / (mu + alpha):
#             events.append(s)
#     return events

# In[24]:
def simulate_hawkes(mu, alpha, beta, T):
    events = []
    s = 0.0  # current time
    while s < T:
        intensity = mu + alpha * np.sum(np.exp(-beta * (s - np.array(events))))
        u = np.random.uniform(0, 1)
        s -= np.log(u) / intensity
        d = np.random.uniform(0, 1)
        if d < intensity / (mu + alpha):
            events.append(s)
    return events

def log_likelihood(params, events, T):
    mu, alpha, beta = params
    n = len(events)
    comp1 = n * np.log(mu)
    comp2 = -mu * T
    comp3 = 0
    comp4 = 0
    for i in range(n):
        comp3 += np.log(mu + alpha * np.sum(np.exp(-beta * (events[i] - np.array(events[:i])))))
    for i in range(n):
        if i > 0:
            comp4 += (1 - np.exp(-beta * (T - events[i-1]))) / beta
    return -(comp1 + comp2 + comp3 - alpha * comp4)

def estimate_params(events, T, initial_params):
    result = minimize(log_likelihood, initial_params, args=(events, T), method='L-BFGS-B', bounds=[(0, 2), (0, 2), (0, 2)])
    return result


# In[25]:


# Step 1: Download and preprocess the data
sp500 = yf.get_data("^gspc", start_date="01/01/2008", end_date="01/01/2015")
sp500['daily_return'] = sp500['close'].pct_change()
sp500['log_return'] = np.log(1 + sp500['daily_return'])
sp500.dropna(inplace=True)

# Step 2: Identify significant events
threshold = 3 * sp500['log_return'].std()
significant_events = sp500[sp500['log_return'].abs() > threshold].index

# Convert timestamps to float format representing the number of days since the start date
start_date = datetime.strptime("01/01/2008", "%m/%d/%Y")
timestamps = [(event - start_date).days for event in significant_events]


# In[26]:


# Simulate a Hawkes process
T = max(timestamps)
events = timestamps

# Estimate parameters
initial_params = [0.1, 0.1, 0.1]
result = estimate_params(events, T, initial_params)
estimated_mu, estimated_alpha, estimated_beta = result.x

# Print estimated parameters
print(f"Estimated mu: {estimated_mu}")
print(f"Estimated alpha: {estimated_alpha}")
print(f"Estimated beta: {estimated_beta}")

plt.figure(figsize=(10, 6))
plt.eventplot(events, orientation='horizontal', color='black')
plt.xlabel("Time (years)")
plt.title("Significant Events in S&P 500")

# Format the x-axis to show the years
years = [int(start_date.year + y) for y in np.linspace(0, T, num=10)]
plt.xticks(np.linspace(0, T, num=10), years)

plt.show()


# # Mark significant events using GBS and a temporal kernel. Step by Step
# 
# Based on this motivation we see that significant events are those with greater clustering, let us use GBS to build a tempral kernel and see if we sample the same 'relevant' dates. 
# 
# ## This is an RBF Homogeneous Kernel defined as:
# 
# $$
# K_{i,j} = e^{-\frac{\|r_i - r_j\|^2}{\sigma^2}}
# $$

# In[ ]:

# Step 1: Download and preprocess the data
sp500 = yf.get_data("^gspc", start_date="01/01/2008", end_date="01/01/2015")
sp500['daily_return'] = sp500['close'].pct_change()
sp500['log_return'] = np.log(1 + sp500['daily_return'])
sp500.dropna(inplace=True)

# Step 2: Identify significant events
threshold = 3 * sp500['log_return'].std()
significant_events = sp500[sp500['log_return'].abs() > threshold].index

# Convert timestamps to float format representing the number of days since the start date
start_date = datetime.strptime("01/01/2008", "%m/%d/%Y")
timestamps = [(event - start_date).days for event in significant_events]

# Step 3: Define the RBF kernel function
def rbf_kernel(events, sigma=1.0):
    events = np.array(events).reshape(-1, 1)
    sq_dists = -2 * np.dot(events, events.T) + np.sum(events**2, axis=1) + np.sum(events**2, axis=1)[:, np.newaxis]
    kernel = np.exp(-sq_dists / (2 * sigma**2))
    return kernel

# Calculate the kernel matrix
sigma_value = 10 # Using sigma = 10 for the RBF kernel this culd be modified
kernel_matrix = rbf_kernel(timestamps, sigma=sigma_value)

# Map indices back to dates
event_dates = [start_date + pd.Timedelta(days=int(t)) for t in timestamps]

# Function to create samples matrix
def create_samples_matrix(sampled_points, num_events):
    samples_matrix = np.zeros((num_events, num_events), dtype=int)
    for point in sampled_points:
        samples_matrix[point, point] += 1
    return samples_matrix

# Function to plot samples and count the frequency of sampled dates
def plot_samples(samples, title, event_dates):
    plt.figure(figsize=(10, 6))
    sns.heatmap(samples, annot=False, cmap="viridis", cbar=False, linewidths=0.5)
    plt.title(title)
    plt.xlabel("Event Date")
    plt.ylabel("Event Date")
    
    # Set custom tick labels
    num_ticks = 10
    tick_indices = np.linspace(0, samples.shape[0] - 1, num_ticks).astype(int)
    plt.xticks(tick_indices, [event_dates[i].strftime('%Y-%m-%d') for i in tick_indices], rotation=45)
    plt.yticks(tick_indices, [event_dates[i].strftime('%Y-%m-%d') for i in tick_indices])
    
    plt.show()
    
    # Count the frequency of each date in the samples
    date_counts = np.sum(samples, axis=0)
    
    # Select the most frequently sampled dates (Top 17)
    most_sampled_indices = np.argsort(date_counts)[-17:]
    most_sampled_indices = most_sampled_indices[::-1]  # Reverse to get most sampled first
    most_sampled_dates = [event_dates[i] for i in most_sampled_indices]
    most_sampled_counts = date_counts[most_sampled_indices]
    
    # Print the most frequently sampled dates
    print(f"Most Sampled Dates ({title}):", most_sampled_dates)
    print(f"Sampling Counts ({title}):", most_sampled_counts)
    
    # Plot the most frequently sampled dates
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[date.strftime('%Y-%m-%d') for date in most_sampled_dates], y=most_sampled_counts, palette="viridis")
    plt.title(f"Most Sampled Dates ({title})")
    plt.xlabel("Date")
    plt.ylabel("Sampling Count")
    plt.xticks(rotation=90)
    plt.show()
    
    return most_sampled_dates


# In[ ]:

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

def tor(matrix):
    # Placeholder implementation for the Torontonian calculation
    # Replace this with the actual Torontonian function
    return np.linalg.det(matrix + np.eye(matrix.shape[0]) * 1e-10)

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


# Example usage
n_mean = 9  # Number of samples

# Generate samples
sampled_points = sample_torontonian_point_process(kernel_matrix, n_mean)

# Create samples matrix
samples_matrix = create_samples_matrix(sampled_points, kernel_matrix.shape[0])

# Plot the results and print the most sampled dates
most_sampled_dates = plot_samples(samples_matrix, "Kernel Matrix Heatmap", event_dates)


# In[32]:


# Plot S&P 500 with significant dates highlighted
plt.figure(figsize=(12, 6))
plt.plot(sp500.index, sp500['close'], label='S&P 500 Close Price')
for event in significant_events:
    plt.axvline(x=event, color='r', linestyle='--', alpha=0.7, label='Significant Event' if event == significant_events[0] else "")
for date in most_sampled_dates:
    plt.axvline(x=date, color='b', linestyle='-', alpha=0.7, label='Sampled Date' if date == most_sampled_dates[0] else "")
plt.title('S&P 500 Close Price with Significant Events and Sampled Dates Highlighted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig(r"C:\Users\user\Downloads\significantdates.pdf", format='pdf', dpi=300)  # Save with high resolution (300 dpi)
plt.show()


# In[ ]:




