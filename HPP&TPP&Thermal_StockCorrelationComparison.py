#!/usr/bin/env python
# coding: utf-8

# # Analyse the difference between doing PNR Sampling (HPP) and Threshold Sampling (TPP) and Thermal Sampling (ThermPP) to distinguish the correlation between different sets of stocks
# 
# Apply PP to stocks from the same and different sectors: see if GBS can identify which group of stocks belongs to the same sector.

# In[47]:


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


# In[48]:


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

# Filter the dataset for the selected stocks
filtered_df_diff_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_diff_sector)]
filtered_df_same_sector = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks_same_sector)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df_diff_sector = filtered_df_diff_sector.pivot(index='date', columns='Name', values='close')
pivot_df_same_sector = filtered_df_same_sector.pivot(index='date', columns='Name', values='close')

# Compute log returns
log_returns_diff_sector = compute_log_returns(pivot_df_diff_sector)
log_returns_same_sector = compute_log_returns(pivot_df_same_sector)

# Compute correlation matrices
correlation_matrix_diff_sector = compute_correlation_matrix_J(log_returns_diff_sector)
correlation_matrix_same_sector = compute_correlation_matrix_J(log_returns_same_sector)

# Perform Takagi-Autonne decomposition
lambdas_diff, U_diff = takagi_autonne_decomposition(correlation_matrix_diff_sector)
lambdas_same, U_same = takagi_autonne_decomposition(correlation_matrix_same_sector)

# Solve for c given a target mean photon number
target_n_mean = 5
c_diff = solve_for_c(lambdas_diff, target_n_mean)
c_same = solve_for_c(lambdas_same, target_n_mean)

# Calculate the squeezing parameters
squeezing_parameters_diff = calculate_squeezing_parameters(lambdas_diff, c_diff)
squeezing_parameters_same = calculate_squeezing_parameters(lambdas_same, c_same)

# Generate the kernel matrices, this is for a non-real world data case to satisfy th ephysical realisability of the quantum state of GBS: 
#SKIP if real world data
B_diff = U_diff @ np.diag(c_diff * lambdas_diff) @ U_diff.T.conj()
B_same = U_same @ np.diag(c_same * lambdas_same) @ U_same.T.conj()

# Generate samples
n_mean = 9
samples = 300

s_thresh_diff = sample.sample(correlation_matrix_diff_sector, n_mean, samples, threshold=True)
s_thresh_same = sample.sample(correlation_matrix_same_sector , n_mean, samples, threshold=True)
s_pnr_diff = sample.sample(correlation_matrix_diff_sector, n_mean, samples, threshold=False)
s_pnr_same = sample.sample(correlation_matrix_same_sector, n_mean, samples, threshold=False)

# Convert the lists to NumPy arrays
s_thresh_diff = np.array(s_thresh_diff)
s_thresh_same = np.array(s_thresh_same)
s_pnr_diff = np.array(s_pnr_diff)
s_pnr_same = np.array(s_pnr_same)


# # Compare PNR (HPP) and Threshold (TPP) detection probability distributions 

# In[80]:


# Create the figure with subplots (3 rows and 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(20, 24))

# Plot the kernel matrices
sns.heatmap(correlation_matrix_diff_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0, 0])
axes[0, 0].set_title("Kernel Matrix - GBS (Different Sectors)")

sns.heatmap(correlation_matrix_same_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[0, 1])
axes[0, 1].set_title("Kernel Matrix - GBS (Same Sector)")

# Calculate the summed and normalized probabilities
sum_pnr_diff = np.sum(s_pnr_diff, axis=0)
sum_pnr_same = np.sum(s_pnr_same, axis=0)

# Find the maximum count for normalization
max_count_pnr = max(sum_pnr_diff.max(), sum_pnr_same.max())

# Normalize the summed probabilities
s_pnr_diff_norm = sum_pnr_diff / max_count_pnr
s_pnr_same_norm = sum_pnr_same / max_count_pnr

# Plot the normalized probability distribution for the different sectors
axes[1, 0].bar(range(len(s_pnr_diff_norm)), s_pnr_diff_norm, color='skyblue')
axes[1, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[1, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[1, 0].set_xlabel('Mode')
axes[1, 0].set_ylabel('Normalized Probability')
axes[1, 0].set_ylim(0, 1.1)  # Normalized probabilities should be between 0 and 1
axes[1, 0].set_title('PNR detector Normalized Probability Distribution from GBS (Tech Sectors)')

# Plot the normalized probability distribution for the same sector
axes[1, 1].bar(range(len(s_pnr_same_norm)), s_pnr_same_norm, color='skyblue')
axes[1, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1, 1].set_xlabel('Mode')
axes[1, 1].set_ylabel('Normalized Probability')
axes[1, 1].set_ylim(0, 1.1)  # Normalized probabilities should be between 0 and 1
axes[1, 1].set_title('PNR detector Normalized Probability Distribution from GBS (Energy Sector)')

# Plot the unnormalized summed probabilities for the different sectors
axes[2, 0].bar(range(len(sum_pnr_diff)), sum_pnr_diff, color='#90EE90')
axes[2, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[2, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[2, 0].set_xlabel('Mode')
axes[2, 0].set_ylabel('Summed Probability')
axes[2, 0].set_ylim(0, max_count_pnr +20) 
axes[2, 0].set_title('PNR detector Unnormalized Summed Probability Distribution from GBS (Tech Sectors)')

# Plot the unnormalized summed probabilities for the same sector
axes[2, 1].bar(range(len(sum_pnr_same)), sum_pnr_same, color='#90EE90')
axes[2, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[2, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[2, 1].set_xlabel('Mode')
axes[2, 1].set_ylabel('Summed Probability')
axes[2, 1].set_ylim(0, max_count_pnr +20) 
axes[2, 1].set_title('PNR detector Unnormalized Summed Probability Distribution from GBS (Energy Sector)')

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\norm_probability_distribution_pnr.pdf", format='pdf', dpi=300)

# Now show the plot
plt.show()


# ## 1 - PNR Detectors (HPP)

# In[81]:


# Set common y-axis limits
max_count_pnr = max(probability_distribution_pnr_diff.max(), probability_distribution_pnr_same.max())

# Create a single figure for both curves
plt.figure(figsize=(12, 6))

# Plot the probability distribution curve for Threshold Detector (Different Sectors)
plt.plot(range(len(probability_distribution_pnr_diff)), probability_distribution_pnr_diff, marker='o', linestyle='-', color='skyblue', label='PNR Detector (Different Sectors)')
plt.fill_between(range(len(probability_distribution_pnr_diff)), probability_distribution_pnr_diff, alpha=0.2, color='skyblue')

# Plot the probability distribution curve for Threshold Detector (Same Sectors)
plt.plot(range(len(probability_distribution_pnr_same)), probability_distribution_pnr_same, marker='o', linestyle='-', color='b', label='PNR Detector (Same Sectors)')
plt.fill_between(range(len(probability_distribution_pnr_same)), probability_distribution_pnr_same, alpha=0.2, color='b')

# Add title, labels, grid, and legend
plt.title('Probability Distribution Curve (PNR Detector)')
plt.xlabel('Mode')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\pnr_distribution_combined.pdf", format='pdf', dpi=300)
plt.show()


# ## Threshold detectors (TPP)

# In[82]:


# Set common y-axis limits
max_count_thres = max(probability_distribution_thresh_diff.max(), probability_distribution_thresh_same.max())

# Create a single figure for both curves
plt.figure(figsize=(12, 6))

# Plot the probability distribution curve for Threshold Detector (Different Sectors)
plt.plot(range(len(probability_distribution_thresh_diff)), probability_distribution_thresh_diff, marker='o', linestyle='-', color='#90EE90', label='Threshold Detector (Different Sectors)')
plt.fill_between(range(len(probability_distribution_thresh_diff)), probability_distribution_thresh_diff, alpha=0.2, color='#90EE90')

# Plot the probability distribution curve for Threshold Detector (Same Sectors)
plt.plot(range(len(probability_distribution_thresh_same)), probability_distribution_thresh_same, marker='o', linestyle='-', color='g', label='Threshold Detector (Same Sectors)')
plt.fill_between(range(len(probability_distribution_thresh_same)), probability_distribution_thresh_same, alpha=0.2, color='g')

# Calculate the area under the curves using the trapezoidal rule
auc_thresh_diff = np.trapz(probability_distribution_thresh_diff, dx=1)
auc_thresh_same = np.trapz(probability_distribution_thresh_same, dx=1)

# Print the areas
print(f"Area under the curve (Different Sectors): {auc_thresh_diff}")
print(f"Area under the curve (Same Sectors): {auc_thresh_same}")

# Add title, labels, grid, and legend
plt.title('Probability Distribution Curve (Threshold Detector)')
plt.xlabel('Mode')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\threshold_distribution_combined.pdf", format='pdf', dpi=300)
plt.show()


# # 3- Thermal Sampling (ThermPP)

# In[64]:


from strawberryfields.apps import sample
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal
# Function to generate samples

def sample(K, n_mean, n_samples):
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()


# In[84]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize the sample counts for different and same sectors
normalized_sample_counts_diff = sample_counts_diff / sample_counts_diff.max()
normalized_sample_counts_same = sample_counts_same / sample_counts_same.max()

# Set common y-axis limits for the normalized plots
max_count_norm = 1.0  # Since they are normalized, the maximum should be 1.0

# Plot the kernel matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(correlation_matrix_diff_sector, annot=False, cmap="inferno", vmin=0, vmax=0.9, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0])
axes[0].set_title("Kernel Matrix - GBS (Different Sectors)")

sns.heatmap(correlation_matrix_same_sector, annot=False, cmap="inferno", vmin=0, vmax=0.7, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[1])
axes[1].set_title("Kernel Matrix - GBS (Same Sector)")

plt.tight_layout()
plt.show()

# Plot the normalized output state distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Normalized plot for different sectors
axes[0].bar(range(len(normalized_sample_counts_diff)), normalized_sample_counts_diff, color='orange')
axes[0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[0].set_xlabel('Mode')
axes[0].set_ylabel('Normalized Probability')
axes[0].set_ylim(0, max_count_norm+ 0.2)  # Normalized probabilities should be between 0 and 1
axes[0].set_title('ThermalPP Normalized Output State Distribution from GBS (Different Sectors)')

# Normalized plot for same sectors
axes[1].bar(range(len(normalized_sample_counts_same)), normalized_sample_counts_same, color='orange')
axes[1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1].set_xlabel('Mode')
axes[1].set_ylabel('Normalized Probability')
axes[1].set_ylim(0, max_count_norm +0.2)  # Normalized probabilities should be between 0 and 1
axes[1].set_title('ThermalPP Normalized Output State Distribution from GBS (Same Sector)')

plt.tight_layout()

# Save the figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\normalized_probability_distribution_GBS.pdf", format='pdf', dpi=300)

# Now show the plot
plt.show()


# # All together

# In[79]:


# Create the figure with subplots (4 rows and 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(20, 32))

# Plot the kernel matrices for different and same sectors
sns.heatmap(correlation_matrix_diff_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0, 0])
axes[0, 0].set_title("Kernel Matrix - GBS (Different Sectors)")

sns.heatmap(correlation_matrix_same_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[0, 1])
axes[0, 1].set_title("Kernel Matrix - GBS (Same Sector)")

# Normalized plot for different sectors (Thermal - Technology)
axes[1, 0].bar(range(len(normalized_sample_counts_diff)), normalized_sample_counts_diff, color='orange')
axes[1, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[1, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[1, 0].set_xlabel('Mode')
axes[1, 0].set_ylabel('Normalized Probability')
axes[1, 0].set_ylim(0, max_count_norm + 0.2)  # Normalized probabilities should be between 0 and 1
axes[1, 0].set_title('Thermal Normalized Output State Distribution (Technology - Different Sectors)')

# Normalized plot for same sectors (Thermal - Energy)
axes[1, 1].bar(range(len(normalized_sample_counts_same)), normalized_sample_counts_same, color='orange')  # Lighter orange for energy
axes[1, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1, 1].set_xlabel('Mode')
axes[1, 1].set_ylabel('Normalized Probability')
axes[1, 1].set_ylim(0, max_count_norm + 0.2)  # Normalized probabilities should be between 0 and 1
axes[1, 1].set_title('Thermal Normalized Output State Distribution (Energy - Same Sectors)')

# Plot the normalized probability distribution for the different sectors using PNR Detector
axes[2, 0].bar(range(len(s_pnr_diff_norm)), s_pnr_diff_norm, color='skyblue')
axes[2, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[2, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[2, 0].set_xlabel('Mode')
axes[2, 0].set_ylabel('Normalized Probability')
axes[2, 0].set_ylim(0, 1.1)
axes[2, 0].set_title('PNR Detector Normalized Probability Distribution (Technology - Different Sectors)')

# Plot the normalized probability distribution for the same sector using PNR Detector
axes[2, 1].bar(range(len(s_pnr_same_norm)), s_pnr_same_norm, color='skyblue')  # Lighter blue for energy
axes[2, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[2, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[2, 1].set_xlabel('Mode')
axes[2, 1].set_ylabel('Normalized Probability')
axes[2, 1].set_ylim(0, 1.1)
axes[2, 1].set_title('PNR Detector Normalized Probability Distribution (Energy - Same Sectors)')

# Plot the normalized probability distribution for the different sectors using Threshold Detector
axes[3, 0].bar(range(len(s_thresh_diff_norm)), s_thresh_diff_norm, color='#90EE90')
axes[3, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[3, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[3, 0].set_xlabel('Mode')
axes[3, 0].set_ylabel('Normalized Probability')
axes[3, 0].set_ylim(0, 1.1)
axes[3, 0].set_title('Threshold Detector Normalized Probability Distribution (Technology - Different Sectors)')

# Plot the normalized probability distribution for the same sector using Threshold Detector
axes[3, 1].bar(range(len(s_thresh_same_norm)), s_thresh_same_norm, color='#90EE90')  # Lighter green for energy
axes[3, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[3, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[3, 1].set_xlabel('Mode')
axes[3, 1].set_ylabel('Normalized Probability')
axes[3, 1].set_ylim(0, 1.1)
axes[3, 1].set_title('Threshold Detector Normalized Probability Distribution (Energy - Same Sectors)')

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\norm_probability_distribution_comparison.pdf", format='pdf', dpi=300)

# Now show the plot
plt.show()




# ## We see that what really makes a difference is the 'clustering' in correlation not the overal correlation itself. 
# 
# Since the point is that regardless of how the stocks are more or less correlated overall this point process is good at distinguishing correlations CLUSTERS. We see that PNR is more accurate n finding a larger area under the cure of the prob distribution with less samples compared the Threshold detectors.
# 
# ### Therefore what matters is where these are located within the correlation matrix.
# 
# Since both are sampled with the same number of mean photons, the normalised probability distribution will only be significant the closer the most sampled stocks are from one another in the feature space.
