#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import random
import strawberryfields as sf
from strawberryfields.apps import points, plot
from strawberryfields.ops import *
from sklearn.datasets import make_blobs
from thewalrus import hafnian
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from strawberryfields.apps import sample
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
import numpy as np
from thewalrus import rec_torontonian as tor
from thewalrus.quantum import Qmat, Xmat, Amat


# # Simulating and Comparing Hawkes and Poisson Processes for Market Buy Events
# 
# This Python code simulates a Hawkes process and compares it with a Poisson process in the context of modeling market buy events within a specific time window. The code also visualizes the actual market buy events against the simulated processes.

# In[19]:


import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Define the path to your CSV file
file_path = r"C:\Users\user\Downloads\AMZN_2012-06-21_34200000_57600000_message_1.csv"

# Load the message file (CSV)
messages = pd.read_csv(file_path, header=None, names=['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction'])

# Convert Time from seconds after midnight to a datetime object
start_time = datetime.datetime(2012, 6, 21)  # Date of the data
messages['DateTime'] = messages['Time'].apply(lambda x: start_time + datetime.timedelta(seconds=x))

# Filter to include only the "market buy" orders (assuming Direction 1 indicates a buy order)
market_buys = messages[(messages['Type'] == 4) & (messages['Direction'] == 1)]

# Further filter to include only events between 11:00 AM and 12:00 PM
start_time_filter = datetime.datetime(2012, 6, 21, 11, 0)
end_time_filter = datetime.datetime(2012, 6, 21, 12, 0)
market_buys = market_buys[(market_buys['DateTime'] >= start_time_filter) & (market_buys['DateTime'] < end_time_filter)]

# Extract the time of the events in seconds
event_times = (market_buys['DateTime'] - start_time_filter).dt.total_seconds().values

# Define a simple exponential kernel for the Hawkes process
def hawkes_intensity(t, event_times, mu, alpha, beta):
    """Compute the intensity of the Hawkes process at time t."""
    intensity = mu + alpha * np.sum(np.exp(-beta * (t - event_times[event_times < t])))
    return intensity

def simulate_hawkes_process(mu, alpha, beta, T, event_times):
    """Simulate a Hawkes process up to time T."""
    simulated_events = []
    current_time = 0
    
    while current_time < T:
        intensity = hawkes_intensity(current_time, np.array(simulated_events + event_times.tolist()), mu, alpha, beta)
        current_time += np.random.exponential(1.0 / intensity)
        if current_time < T:
            if np.random.rand() < intensity / hawkes_intensity(current_time, np.array(simulated_events + event_times.tolist()), mu, alpha, beta):
                simulated_events.append(current_time)
                
    return np.array(simulated_events)

# Parameters for the Hawkes process: optimised in another section, change if needed
mu = 0.1  # Baseline intensity
alpha = 0.8  # Excitation parameter
beta = 1.3 # Decay parameter

# Simulate the Hawkes process using the fitted parameters
T = 3600  # Time period of 1 hour (in seconds)
simulated_event_times = simulate_hawkes_process(mu, alpha, beta, T, event_times)

# Convert simulated event times back to DateTime format for plotting
simulated_datetimes = [start_time_filter + datetime.timedelta(seconds=t) for t in simulated_event_times]

# Group by minute to count the number of market buy orders per minute
market_buys_per_minute = market_buys.set_index('DateTime').resample('T').size()

# Calculate the mean rate (λ) for the Poisson process
lambda_poisson = market_buys_per_minute.mean()

# Generate a Poisson process using the calculated λ
poisson_simulated = np.random.poisson(lambda_poisson, len(market_buys_per_minute))

# Convert the Poisson simulated values to a time series
poisson_simulated_series = pd.Series(poisson_simulated, index=market_buys_per_minute.index)

# Group the simulated events by minute for the Hawkes process
simulated_events_per_minute = pd.Series(1, index=pd.to_datetime(simulated_datetimes)).resample('T').size()

# Plotting the actual, Hawkes simulated, and Poisson simulated number of events per minute
plt.figure(figsize=(10, 6))
plt.step(market_buys_per_minute.index, market_buys_per_minute.values, color='blue', where='mid', label="Actual Market Buy Orders")
plt.step(simulated_events_per_minute.index, simulated_events_per_minute.values, color='green', linestyle='--', where='mid', label="Simulated Hawkes Process")
plt.step(poisson_simulated_series.index, poisson_simulated_series.values, color='red', linestyle=':', where='mid', label="Simulated Poisson Process")
plt.xlabel('Time')
plt.ylabel('Number of Events per Minute')
plt.title('Actual vs. Simulated Hawkes and Poisson Processes for Market Buy Events (11:00 AM to 12:00 PM)')
plt.legend()
plt.grid(True)
# Save the plot as a high-resolution PDF in your Downloads folder
plt.savefig(r"C:\Users\user\Downloads\hawkesvsPoisson.pdf", format='pdf', dpi=300)  # Save with high resolution (300 dpi)
plt.show()



# # Kernel Matrix: Inhomogeneours RBF kernel with a density vector to add control over the location of the clusters
# 
# $$
# K_{i,j} = \lambda_i \lambda_j e^{-\frac{\|r_i - r_j\|^2}{\sigma^2}}
# $$
# 
# - The location of the clusters is extracted from the event counts in each time step
# 
# - $\sigma$ denotes how qiuckly the excitaton decays and cna therefroe be modified below to increase the brightness of the kernel at ht eexpense of a longer computation time.

# In[11]:


# Select a subset of evenly spaced timestamps for building the kernel matrix
num_selected_points = 60  # Number of points to select
selected_indices = np.linspace(0, len(market_buys_per_minute) - 1, num=num_selected_points).astype(int)
selected_event_counts = market_buys_per_minute.values[selected_indices]

# Calculate the pairwise time differences and build the kernel matrix using the selected timestamps
sigma_value = 30  # Adjust sigma as necessary
kernel_matrix = np.zeros((num_selected_points, num_selected_points)) #compute kernel matrix as 

for i in range(num_selected_points):
    for j in range(num_selected_points):
        time_difference = np.abs(selected_indices[i] - selected_indices[j])
        kernel_matrix[i, j] = selected_event_counts[i] * selected_event_counts[j] * np.exp(-time_difference**2 / (sigma_value**2))

# Normalize the kernel matrix
kernel_matrix /= np.max(kernel_matrix)

# Plot the normalized kernel matrix
plt.figure(figsize=(12, 10))
plt.imshow(kernel_matrix, cmap='hot', interpolation='nearest', vmax=0.5)
plt.colorbar(label='Normalized Kernel Value')
plt.title('Normalized Temporal Kernel Matrix Using Event Counts')

# Set custom tick labels
time_labels = market_buys_per_minute.index[selected_indices]
# Only show every 5th label
tick_indices = np.arange(0, len(time_labels), 5)
plt.xticks(tick_indices, [time_labels[i].strftime('%H:%M') for i in tick_indices], rotation=45)
plt.yticks(tick_indices, [time_labels[i].strftime('%H:%M') for i in tick_indices])

plt.xlabel('Event Time')
plt.ylabel('Event Time')

# Save the kernel matrix plot as a high-resolution PDF in your Downloads folder
plt.savefig(r"C:\Users\user\Downloads\kernel_matrix_plot_sigma30.pdf", format='pdf', dpi=300)  # Save with high resolution (300 dpi)
plt.show()


# In[12]:


modes = 60 #this is done automatically by the sampling function
n_mean = 20
samples = 800 #change as needed to increase accuracy at the expense of computation time

#StrawberryFields in-built Sampling functions

s_thresh_800 = sample.sample(kernel_matrix, n_mean, samples, threshold=True) #function
s_pnr_800 = sample.sample(kernel_matrix, n_mean, samples, threshold=False)

#print(s_thresh)
#print(s_pnr)


# In[13]:


# Convert the lists of samples to numpy arrays for easier manipulation
s_thresh_np = np.array(s_thresh_800)
s_pnr_np = np.array(s_pnr_800)

# Calculate the probability of detection for each mode (position in the array)
probability_distribution_thresh_800 = np.sum(s_thresh_np > 0, axis=0) / s_thresh_np.shape[0]
probability_distribution_pnr_800 = np.sum(s_pnr_np > 0, axis=0) / s_pnr_np.shape[0]

# Plot the probability distribution curve for Threshold Detector
plt.figure(figsize=(12, 6))
plt.plot(range(len(probability_distribution_thresh_800)), probability_distribution_thresh_800, marker='o', linestyle='-', color='b', label='Threshold Detector')
plt.fill_between(range(len(probability_distribution_thresh_800)), probability_distribution_thresh_800, alpha=0.2, color='b')
plt.title('Probability Distribution Curve (Threshold Detector) 800 Samples')
plt.xlabel('Mode')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()

# Plot the probability distribution curve for PNR Detector
plt.figure(figsize=(12, 6))
plt.plot(range(len(probability_distribution_pnr_800)), probability_distribution_pnr_800, marker='o', linestyle='-', color='r', label='PNR Detector')
plt.fill_between(range(len(probability_distribution_pnr_800)), probability_distribution_pnr_800, alpha=0.2, color='r')
plt.title('Probability Distribution Curve (PNR Detector) 800 Samples')
plt.xlabel('Mode')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.savefig(r"C:\Users\user\Downloads\temporal_kernel_distribution_curves.pdf", format='pdf', dpi=300)  # Save with high resolution (300 dpi)
plt.show()


# In[17]:


plt.figure(figsize=(14, 8))

# Plot the actual market buy events
plt.step(market_buys_per_minute.index, market_buys_per_minute.values, color='blue', where='mid', label="Actual Market Buy Orders")

# Plot the simulated market buy events
plt.step(simulated_events_per_minute.index, simulated_events_per_minute.values, color='green', linestyle='--', where='mid', label="Simulated Hawkes Process")

# Plot the normalized quantum probability distributions, correctly aligned with time
plt.plot(time_labels, normalized_probability_distribution_thresh_800, marker='o', linestyle='-', color='b', alpha=0.6, label='Normalized Threshold Detector Probability')
plt.plot(time_labels, normalized_probability_distribution_pnr_800, marker='o', linestyle='-', color='r', alpha=0.6, label='Normalized PNR Detector Probability')

# Corrected title with sigma and samples values
plt.title(f'GBS vs Hawkes Processes in time-series modelling of LOB, with σ = {sigma_value} and {samples} samples', fontsize=16)

plt.xlabel('Time (11:00 AM to 12:00 PM)', fontsize=14)
plt.ylabel('Number of Events / Normalized Probability', fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Save the plot as a high-resolution PDF in your Downloads folder
plt.savefig(r"C:\Users\user\Downloads\GBSvsHawkesLOB.pdf", format='pdf', dpi=300)  # Save with high resolution (300 dpi)
plt.show()


# In[ ]:




