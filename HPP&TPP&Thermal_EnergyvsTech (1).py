#!/usr/bin/env python
# coding: utf-8

# # Analyse the difference between doing PNR Sampling (HPP) and Threshold Sampling (TPP) nd Thermal Sampling (ThermPP) to distinguish the correlation between different sets of stocks
# 
# Choose two sets of stocks from the same sector. Energy and Technology and show which sector shows a higher degree of correlation using GBS point processes (HPP and TPP) and classical quantum-inspired point processes with thermal states (ThermPP)  [^1].
# 
# ## References
# 
# [^1]: Jahangiri, S., Arrazola, J. M., Quesada, N., & Killoran, N. (2020). Point Processes With Gaussian Boson Sampling. *Physical Review E*, 101, 022134. https://doi.org/10.1103/physreve.101.022134
# 
# 

# In[61]:


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


# In[63]:


# Load the data
zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# Choose subsets of stocks 
selected_stocks_same_sector = ["ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU"] #energy
selected_stocks_diff_sector = ["AAPL", "MSFT", "GOOGL", "AMZN", "PYPL", "CSCO", "NVDA", "ORCL", "ADBE"] #technology

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


# ## 1 - PNR Detectors

# In[64]:


# Create the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Plot the kernel matrices
sns.heatmap(correlation_matrix_diff_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0, 0])
axes[0, 0].set_title("Kernel Matrix - GBS (Technology Sector)")

sns.heatmap(correlation_matrix_same_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[0, 1])
axes[0, 1].set_title("Kernel Matrix - GBS (Energy Sector)")

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
axes[1, 0].set_title('PNR detector Normalized Probability Distribution from GBS (Technology Sector)')

# Plot the normalized probability distribution for the same sector
axes[1, 1].bar(range(len(s_pnr_same_norm)), s_pnr_same_norm, color='skyblue')
axes[1, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1, 1].set_xlabel('Mode')
axes[1, 1].set_ylabel('Normalized Probability')
axes[1, 1].set_ylim(0, 1.1)  # Normalized probabilities should be between 0 and 1
axes[1, 1].set_title('PNR detector Normalized Probability Distribution from GBS (Energy Sector)')

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\norm_probability_distribution_PNR_EvsT.pdf", format='pdf', dpi=300)

# Now show the plot
plt.show()


# In[81]:


# Normalize the probability distributions
sum_pnr_diff_norm = sum_pnr_diff / sum_pnr_diff.max()
sum_pnr_same_norm = sum_pnr_same / sum_pnr_same.max()

# Set common y-axis limits
max_count_pnr = max(sum_pnr_diff_norm.max(), sum_pnr_same_norm.max())

# Create a single figure for both curves
plt.figure(figsize=(12, 6))

# Plot the normalized probability distribution curve for PNR Detector (Technology Sector)
plt.plot(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, marker='o', linestyle='-', color='b', label='PNR Detector (Technology Sector)')
plt.fill_between(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, alpha=0.2, color='b')

# Plot the normalized probability distribution curve for PNR Detector (Energy Sector)
plt.plot(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, marker='o', linestyle='-', color='g', label='PNR Detector (Energy Sector)')
plt.fill_between(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, alpha=0.2, color='g')

# Calculate the area under the curves using the trapezoidal rule
auc_pnr_diff_norm = np.trapz(sum_pnr_diff_norm, dx=1)
auc_pnr_same_norm = np.trapz(sum_pnr_same_norm, dx=1)

# Print the areas
print(f"Area under the normalized curve (Technology Sector): {auc_pnr_diff_norm}")
print(f"Area under the normalized curve (Energy Sector): {auc_pnr_same_norm}")

# Add title, labels, grid, and legend
plt.title('Normalized Probability Distribution Curve (PNR Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Add annotations for the area under the curve (AUC)
plt.text(len(sum_pnr_diff_norm) - 2, sum_pnr_diff_norm[-1], f'AUC Tech: {auc_pnr_diff_norm:.2f}', color='b', fontsize=12, verticalalignment='bottom')
plt.text(len(sum_pnr_same_norm) - 2, sum_pnr_same_norm[-1], f'AUC Energy: {auc_pnr_same_norm:.2f}', color='g', fontsize=12, verticalalignment='bottom')

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\PNR_distribution_combined_EvsT_normalized.pdf", format='pdf', dpi=300)
plt.show()



# ## 2- Threshold detectors

# In[67]:


# Create the figure with subplots (3 rows and 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(20, 24))

# Plot the kernel matrices
sns.heatmap(correlation_matrix_diff_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_diff_sector, yticklabels=selected_stocks_diff_sector, ax=axes[0, 0])
axes[0, 0].set_title("Kernel Matrix - GBS (Technology Sectors)")

sns.heatmap(correlation_matrix_same_sector, annot=False, cmap="inferno", vmin=0, vmax=0.8, 
            xticklabels=selected_stocks_same_sector, yticklabels=selected_stocks_same_sector, ax=axes[0, 1])
axes[0, 1].set_title("Kernel Matrix - GBS (Energy Sector)")

# Calculate the summed and normalized probabilities
sum_thresh_diff = np.sum(s_thresh_diff, axis=0)
sum_thresh_same = np.sum(s_thresh_same, axis=0)

# Normalize the summed probabilities
s_thresh_diff_norm = sum_thresh_diff / sum_thresh_diff.max()
s_thresh_same_norm = sum_thresh_same / sum_thresh_same.max()

# Find the maximum count for normalization
max_count_thres = max(sum_thresh_diff.max(), sum_thresh_same.max())

# Plot the normalized probability distribution for the different sectors
axes[1, 0].bar(range(len(s_thresh_diff_norm)), s_thresh_diff_norm, color='skyblue')
axes[1, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[1, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[1, 0].set_xlabel('Mode')
axes[1, 0].set_ylabel('Normalized Probability')
axes[1, 0].set_ylim(0, 1.1)  # Normalized probabilities should be between 0 and 1
axes[1, 0].set_title('Threshold detector Normalized Probability Distribution from GBS (Technology Sector)')

# Plot the normalized probability distribution for the same sector
axes[1, 1].bar(range(len(s_thresh_same_norm)), s_thresh_same_norm, color='skyblue')
axes[1, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[1, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[1, 1].set_xlabel('Mode')
axes[1, 1].set_ylabel('Normalized Probability')
axes[1, 1].set_ylim(0, 1.1)  # Normalized probabilities should be between 0 and 1
axes[1, 1].set_title('Threshold detector Normalized Probability Distribution from GBS (Energy Sector)')

# Plot the unnormalized summed probabilities for the different sectors
axes[2, 0].bar(range(len(sum_thresh_diff)), sum_thresh_diff, color='orange')
axes[2, 0].set_xticks(range(len(selected_stocks_diff_sector)))
axes[2, 0].set_xticklabels(selected_stocks_diff_sector, rotation=90)
axes[2, 0].set_xlabel('Mode')
axes[2, 0].set_ylabel('Summed Probability')
axes[2, 0].set_ylim(0, max_count_thres) 
axes[2, 0].set_title('Threshold detector Unnormalized Summed Probability Distribution from GBS (Technology Sector)')

# Plot the unnormalized summed probabilities for the same sector
axes[2, 1].bar(range(len(sum_thresh_same)), sum_thresh_same, color='orange')
axes[2, 1].set_xticks(range(len(selected_stocks_same_sector)))
axes[2, 1].set_xticklabels(selected_stocks_same_sector, rotation=90)
axes[2, 1].set_xlabel('Mode')
axes[2, 1].set_ylabel('Summed Probability')
axes[2, 1].set_ylim(0, max_count_thres) 
axes[2, 1].set_title('Threshold detector Unnormalized Summed Probability Distribution from GBS (Energy Sector)')

plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\norm&unnorm_probability_distribution_Thresh_EvsT.pdf", format='pdf', dpi=300)

# Now show the plot
plt.show()


# In[85]:


# Normalize the probability distributions by dividing by the maximum value
sum_thresh_diff_norm = sum_thresh_diff / sum_thresh_diff.max()
sum_thresh_same_norm = sum_thresh_same / sum_thresh_same.max()

# Set common y-axis limits
max_count_thres = max(sum_thresh_diff_norm.max(), sum_thresh_same_norm.max())

# Calculate the area under the curves using the trapezoidal rule
auc_thresh_diff_norm = np.trapz(sum_thresh_diff_norm, dx=1)
auc_thresh_same_norm = np.trapz(sum_thresh_same_norm, dx=1)

# Create a single figure for both curves
plt.figure(figsize=(12, 6))

# Plot the normalized probability distribution curve for Threshold Detector (Energy Sectors)
plt.plot(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, marker='o', linestyle='-', color='b', 
         label=f'Threshold Detector (Energy Sectors), AUC: {auc_thresh_same_norm:.2f}')
plt.fill_between(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, alpha=0.2, color='b')

# Plot the normalized probability distribution curve for Threshold Detector (Technology Sectors)
plt.plot(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, marker='o', linestyle='-', color='g', 
         label=f'Threshold Detector (Technology Sectors), AUC: {auc_thresh_diff_norm:.2f}')
plt.fill_between(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, alpha=0.2, color='g')

# Print the areas
print(f"Area under the normalized curve (Energy Sectors): {auc_thresh_same_norm}")
print(f"Area under the normalized curve (Technology Sectors): {auc_thresh_diff_norm}")

# Add title, labels, grid, and legend
plt.title('Normalized Probability Distribution Curve (Threshold Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\Threshold_distribution_combined_normalized.pdf", format='pdf', dpi=300)
plt.show()


# # Compare both GBS Point Processes
# 
# HPP and TPP

# In[69]:


# Normalize the probability distributions for PNR Detector
sum_pnr_diff_norm = sum_pnr_diff / sum_pnr_diff.max()
sum_pnr_same_norm = sum_pnr_same / sum_pnr_same.max()

# Normalize the probability distributions for Threshold Detector
sum_thresh_diff_norm = sum_thresh_diff / sum_thresh_diff.max()
sum_thresh_same_norm = sum_thresh_same / sum_thresh_same.max()

# Create a figure with two subplots side by side
plt.figure(figsize=(16, 6))

# Plot for PNR Detector
plt.subplot(1, 2, 1)
plt.plot(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, marker='o', linestyle='-', color='b', label='PNR Detector (Technology Sectors)')
plt.fill_between(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, alpha=0.2, color='b')
plt.plot(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, marker='o', linestyle='-', color='g', label='PNR Detector (Energy Sectors)')
plt.fill_between(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, alpha=0.2, color='g')
plt.title('Normalized Probability Distribution (PNR Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Plot for Threshold Detector
plt.subplot(1, 2, 2)
plt.plot(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, marker='o', linestyle='-', color='r', label='Threshold Detector (Technology Sectors)')
plt.fill_between(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, alpha=0.2, color='r')
plt.plot(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, marker='o', linestyle='-', color='orange', label='Threshold Detector (Energy Sectors)')
plt.fill_between(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, alpha=0.2, color='orange')
plt.title('Normalized Probability Distribution (Threshold Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\PNR_and_Threshold_Distribution_Normalized_Side_by_Side.pdf", format='pdf', dpi=300)
plt.show()



# # 3-Thermal Sampling

# In[70]:


from strawberryfields.apps import sample
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal
# Function to generate samples

def sample(K, n_mean, n_samples):
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()


# In[71]:


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


# In[72]:


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
sample_counts_tech = np.sum(samples_diff, axis=0)
sample_counts_energy = np.sum(samples_same, axis=0)

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


# In[73]:


# Normalize the sample counts for thermal
normalized_sample_counts_tech = sample_counts_tech / sample_counts_tech.max()
normalized_sample_counts_energy = sample_counts_energy / sample_counts_energy.max()

# Calculate the area under the curves using the trapezoidal rule
auc_tech = np.trapz(normalized_sample_counts_tech, dx=1)
auc_energy = np.trapz(normalized_sample_counts_energy, dx=1)

# Print the AUC values
print(f"Area under the normalized curve (Technology Sectors): {auc_tech}")
print(f"Area under the normalized curve (Energy Sectors): {auc_energy}")
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


# # Calculate the area under the probability distribution curve: which group of stocks is most correlated?
# 
# Compare all methods: ThermalPP, TPP and HPP

# In[80]:


# Normalize the sample counts for Thermal Detector
normalized_sample_counts_tech = sample_counts_tech / sample_counts_tech.max()
normalized_sample_counts_energy = sample_counts_energy / sample_counts_energy.max()

# Calculate the area under the curves using the trapezoidal rule for Thermal Detector
auc_tech = np.trapz(normalized_sample_counts_tech, dx=1)
auc_energy = np.trapz(normalized_sample_counts_energy, dx=1)

# Normalize the probability distributions for PNR Detector
sum_pnr_diff_norm = sum_pnr_diff / sum_pnr_diff.max()
sum_pnr_same_norm = sum_pnr_same / sum_pnr_same.max()

# Normalize the probability distributions for Threshold Detector
sum_thresh_diff_norm = sum_thresh_diff / sum_thresh_diff.max()
sum_thresh_same_norm = sum_thresh_same / sum_thresh_same.max()

# Create a figure with three subplots, one for each detector type
plt.figure(figsize=(18, 12))

# Plot for Thermal Detector (yellow for tech, orange for energy)
plt.subplot(3, 1, 1)
plt.plot(range(len(normalized_sample_counts_tech)), normalized_sample_counts_tech, marker='o', linestyle='-', color='yellow', label=f'Technology Sector (AUC = {auc_tech:.4f})')
plt.fill_between(range(len(normalized_sample_counts_tech)), normalized_sample_counts_tech, alpha=0.2, color='yellow')
plt.plot(range(len(normalized_sample_counts_energy)), normalized_sample_counts_energy, marker='o', linestyle='-', color='orange', label=f'Energy Sector (AUC = {auc_energy:.4f})')
plt.fill_between(range(len(normalized_sample_counts_energy)), normalized_sample_counts_energy, alpha=0.2, color='orange')
plt.title('Normalized Output State Distributions (Thermal Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Plot for PNR Detector (blue for tech, brighter blue for energy)
plt.subplot(3, 1, 2)
plt.plot(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, marker='o', linestyle='-', color='blue', label=f'Technology Sector (AUC = {auc_pnr_diff_norm:.4f})')
plt.fill_between(range(len(sum_pnr_diff_norm)), sum_pnr_diff_norm, alpha=0.2, color='blue')
plt.plot(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, marker='o', linestyle='-', color='#ADD8E6', label=f'Energy Sector (AUC = {auc_pnr_same_norm:.4f})')  # Brighter blue
plt.fill_between(range(len(sum_pnr_same_norm)), sum_pnr_same_norm, alpha=0.2, color='#ADD8E6')
plt.title('Normalized Probability Distribution (PNR Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Plot for Threshold Detector (green for tech, brighter green for energy)
plt.subplot(3, 1, 3)
plt.plot(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, marker='o', linestyle='-', color='green', label=f'Technology Sector (AUC = {auc_thresh_diff_norm:.4f})')
plt.fill_between(range(len(sum_thresh_diff_norm)), sum_thresh_diff_norm, alpha=0.2, color='green')
plt.plot(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, marker='o', linestyle='-', color='#90EE90', label=f'Energy Sector (AUC = {auc_thresh_same_norm:.4f})')  # Brighter green
plt.fill_between(range(len(sum_thresh_same_norm)), sum_thresh_same_norm, alpha=0.2, color='#90EE90')
plt.title('Normalized Probability Distribution (Threshold Detector)')
plt.xlabel('Mode')
plt.ylabel('Normalized Probability')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig(r"C:\Users\user\Downloads\Normalized_Distribution_Thermal_PNR_Threshold.pdf", format='pdf', dpi=300)
plt.show()

plt.show()


# # According to these findings Energy is more correlated than Tech
# 
# Let us plot the closing price of the two most correlated stock sin each sector to compare

# In[75]:


import yahoo_fin.stock_info as yf
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
plt.savefig(r"C:\Users\user\Downloads\energyvstechClosingprices.pdf", format='pdf', dpi=300)
plt.show()


# In[ ]:


# This exploration yields satisfactory results

