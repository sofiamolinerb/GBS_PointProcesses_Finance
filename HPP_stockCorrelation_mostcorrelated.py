#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, Rgate, Interferometer
import scipy.linalg
from scipy.optimize import root_scalar
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import pandas as pd
import zipfile
from strawberryfields.apps import sample
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal #in case we need to compare to thermal samples


# In[35]:


#Define functions
def calculate_correlation_matrix_log(selected_stocks): #compute correlation matrix from Jahanguiri et all from a list of selected stocks
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
    correlation_matrix_0 = sum(np.outer(log_returns.iloc[i], log_returns.iloc[i]) for i in range(n_days)) /n_days
    rescaling_factor = n_days
    correlation_matrix = correlation_matrix_0 * rescaling_factor #this is included because in Jahangiri's paper (would be good to investigate)
    return correlation_matrix 
    
def plot_heatmap(matrix, title, ax=None, vmin=None, vmax=None):
    # Calculate the magnitude of the complex matrix
    matrix_magnitude = np.abs(matrix)
    
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=False, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=False, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(title)

# Function to plot heatmap
def plot_heatmap_selected(matrix, title, ax=None, vmin=None, vmax=None, xticklabels=None, yticklabels=None):
    # Calculate the magnitude of the complex matrix
    matrix_magnitude = np.abs(matrix)
    
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=False, cmap="inferno", vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=False, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
        ax.set_title(title)
        
# Function to perform Takagi-Autonne decomposition and return necessary matrices
def takagi_autonne_decomposition(A):
    # Ensure the matrix A is symmetric
    A = (A + A.T) / 2
    # Perform SVD for the Takagi-Autonne decomposition
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

def is_unitary(matrix, tol=1e-10):
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix @ matrix.conj().T, identity, atol=tol)

#Define RBF kernel 
def rbf_kernel(R: np.ndarray, sigma: float) -> np.ndarray: 
    return np.exp(-((scipy.spatial.distance.cdist(R, R)) ** 2) / 2 / sigma**2)

# Define the thermal sampling function
def sample_thermal(K: np.ndarray, n_mean: float, n_samples: int) -> list:
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()
    


# In[36]:


# Path to the zip file
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# List of selected stock names (from the DPP image)
selected_stocks = ["CA", "CHK", "SIG", "LH", "AVB", "BMY", "EFX", "RRC", "TEL", "INTU", "PRGO", "NVDA", "SCG", "SLG", "DVN", 
                   "NFLX", "ANDV", "BEN", "ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU", "KMI", "LRCX", "LUK", "MRO", "MU", "PCLN", "SNA", "VRTX"]

correlation_matrix = calculate_correlation_matrix_log(selected_stocks)


# In[37]:


# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="inferno", 
            xticklabels=selected_stocks, yticklabels=selected_stocks, vmin=0, vmax=0.4)
plt.title("Correlation Matrix of Selected Stocks")
plt.show()


# In[28]:


# n_modes = 35 not needed
from strawberryfields.apps import sample
# Generate samples with PNR detectors
n_mean = 15
n_samples = 30 # number of samples to generate, small to check but increase to increase precision
samples = sample.sample(correlation_matrix, n_mean, n_samples, threshold=True)
        
# Plot the kernel matrix
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_heatmap_selected(correlation_matrix, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.25, xticklabels=selected_stocks, yticklabels=selected_stocks)

# Plot the output state
sample_counts = np.sum(samples, axis=0)

#plot the counts
total_samples = np.sum(sample_counts)  # Total number of samples

# Normalize the sample counts
normalized_counts = sample_counts / total_samples

plt.figure(figsize=(8, 6))
plt.bar(range(len(normalized_counts)), normalized_counts, color='skyblue')
plt.xlabel('Mode')
plt.ylabel('Normalized Count')
plt.title('Normalized Output State Distribution from GBS')

# Adjust the xticks to match the number of modes
num_modes = min(len(normalized_counts), len(selected_stocks))
xticks_positions = np.linspace(0, len(normalized_counts) - 1, num=num_modes).astype(int)

# Set the xticks and corresponding labels
plt.xticks(xticks_positions, np.array(selected_stocks)[xticks_positions], rotation=45, ha='right')

plt.show()


# In[29]:


# Identify the 17 most sampled stocks
most_sampled_indices = np.argsort(sample_counts)[-17:][::-1]
most_sampled_stocks = [selected_stocks[i] for i in most_sampled_indices]
most_sampled_counts = sample_counts[most_sampled_indices]

# Plot the most sampled stocks
plt.figure(figsize=(10, 6))
plt.bar(range(len(most_sampled_counts)), most_sampled_counts, color='skyblue')
plt.xlabel('Stock')
plt.ylabel('Sample Count')
plt.title('17 Most Sampled Stocks')
plt.xticks(range(len(most_sampled_counts)), most_sampled_stocks, rotation=45, ha='right')
plt.show()


# In[39]:


# Filter the dataset for the most sampled stocks
most_sampled_stocks_set = set(most_sampled_stocks)

correlation_matrix_TPP = calculate_correlation_matrix_log(most_sampled_stocks) #correlation matrix of most sampled stocks

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_TPP, annot=False, cmap="inferno", 
            xticklabels=most_sampled_stocks, yticklabels=most_sampled_stocks, vmin=0, vmax=0.9)
plt.title("Correlation Matrix of 17 Most Sampled Stocks")
plt.show()


# In[31]:


# We see that we recover the most sampled stock using a Hafnian Proint Process


# In[ ]:





# In[ ]:




