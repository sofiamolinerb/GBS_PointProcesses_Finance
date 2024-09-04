#!/usr/bin/env python
# coding: utf-8

# In[6]:
import numpy as np
import plotly
from sklearn.datasets import make_blobs#
from strawberryfields.apps import*
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import yfinance as yf
import scipy
from scipy.integrate import simps # Path to the zip file
from scipy.optimize import root_scalar
import scipy.linalg
from PIL import Image
from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal

#The idea is to create a GBS device capable of sampling form a covariance/ correlation matrix from different stock prices

#First we extract the information from the stocks

zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# List of selected stock names
selected_stocks = ["ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU"]

# Filter the dataset for the selected stocks
filtered_df = all_stocks_df[all_stocks_df['Name'].isin(selected_stocks)]

# Pivot the data to have stock names as columns and dates as rows
pivot_df = filtered_df.pivot(index='date', columns='Name', values='close')

# Calculate the daily returns and drop rows with NaN values
daily_returns = pivot_df.pct_change().dropna()

# Compute the log returns
log_returns = np.log(1 + daily_returns)

# Number of stocks
n_stocks = len(selected_stocks)
n_days = len(daily_returns)
print("the number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))

correlation_matrix_0 = sum(np.outer(log_returns.iloc[i], log_returns.iloc[i]) for i in range(n_days)) /n_days
rescaling_factor = n_days
correlation_matrix = correlation_matrix_0 * rescaling_factor

# Display the correlation matrix
#print("\nCorrelation Matrix:")
#print(correlation_matrix)


# Print the vector of returns for each day
#for j in range(n_days):
#    R_j = log_returns.iloc[j].values
#    print(f"Day {j+1} returns: {R_j}")

n_stocks = len(selected_stocks)
n_days = len(daily_returns)
print("the number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))

# Load the image to compare
image_path = r'C:\Users\user\Downloads\TPP_reduced.png'  # Path to your image
image = Image.open(image_path)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=False, cmap="inferno", xticklabels=selected_stocks, yticklabels=selected_stocks, ax=axes[0], vmin=0, vmax=0.40)
axes[0].set_title("Correlation Matrix")

# Display the image shown in the paper of Jahanguiri et. al. ho used Hafnian point processes to select subsets of stocks with a greater clustering 
axes[1].imshow(image)
axes[1].axis('off')
axes[1].set_title("Comparison Image")

# Ensure both subplots have the same size
axes[0].set_aspect('auto')
axes[1].set_aspect('auto')

# Show the plot
plt.tight_layout()
plt.show()


# In[7]:

def plot_heatmap(matrix, title, ax=None, vmin=None, vmax=None):
    # Calculate the magnitude of the complex matrix
    matrix_magnitude = np.abs(matrix)
    
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax)
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

# Perform Takagi-Autonne decomposition
lambdas, U_corr = takagi_autonne_decomposition(correlation_matrix)

# Solve for c given a target mean photon number
target_n_mean = 5  # Example target mean photon number
c = solve_for_c(lambdas, target_n_mean)

# Calculate the squeezing parameters
squeezing_parameters = calculate_squeezing_parameters(lambdas, c)

# Generate the kernel matrix
B = U_corr @ np.diag(c * lambdas) @ U_corr.T

# Create a figure with one subplot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the kernel matrix
plot_heatmap(B, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.25)

# Print out the unitary matrix and squeezing parameters
print("Unitary matrix U:\n", U_corr)
print("Squeezing parameters:\n", squeezing_parameters)

# Additional verification to ensure the unitary matrix is indeed complex
print("Is the unitary matrix U complex? ", np.iscomplexobj(U_corr))
# Verification to check if the matrix is unitary
identity_approx = np.eye(U_corr.shape[0])
unitary_check = np.allclose(U_corr @ U_corr.conj().T, identity_approx)
print("Is the unitary matrix U unitary? ", unitary_check)

#NOTE: Please let me know if these squeezing parameters are not too small, I do think the last ones are maybe too small, if you think the decomposition is wrong let me know
#NOTE 2: I am also a bit consfused since I do not know how to make the Unitary matrix complex.


# In[12]:
def is_unitary(matrix, tol=1e-10):
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix @ matrix.conj().T, identity, atol=tol)

# Assuming U_corr is defined
#U_corr = np.array([[0.70710678 + 0.j, 0.5 + 0.5j],[0.5 - 0.5j, 0.70710678 + 0.j]])  # Example unitary matrix

if not is_unitary(U_corr):
    raise ValueError("The input matrix is not unitary")

# Continue with the Strawberry Fields program
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, Rgate, Interferometer

# Initialize the Strawberry Fields program
prog = sf.Program(U_corr.shape[0])

with prog.context as q:
    # Apply the squeezing gates
    for i in range(len(squeezing_parameters)):
        Sgate(squeezing_parameters[i]) | q[i]

    # Create the interferometer instance
    interferometer = Interferometer(U_corr, mesh='rectangular', drop_identity=True)
    # Decompose the interferometer
    cmds = interferometer.decompose(q)

# Print the decomposed operations
for cmd in cmds:
    print(cmd)

# Optionally, draw the quantum circuit if desired
def draw_quantum_circuit(gate_sequence, N, squeezing_parameters):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('off')

    x_pos = [0.1 * i for i in range(len(gate_sequence) + 1)]
    y_pos = [0.9 - i * 0.2 for i in range(N)]

    for y in y_pos:
        ax.plot([0, max(x_pos) + 0.1], [y, y], 'k')
    
    # Plot squeezing gates
    for i, r in enumerate(squeezing_parameters):
        ax.text(-0.1, y_pos[i], f"S({r:.2f})", ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    for i, cmd in enumerate(gate_sequence):
        x = x_pos[i+1]

        if isinstance(cmd.op, Rgate):
            reg = cmd.reg[0].ind
            y = y_pos[reg]
            param = cmd.op.p[0]
            ax.text(x, y, f'R({param:.2f})', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

        elif isinstance(cmd.op, BSgate):
            reg1 = cmd.reg[0].ind
            reg2 = cmd.reg[1].ind
            y1 = y_pos[reg1]
            y2 = y_pos[reg2]
            param = cmd.op.p[0]
            ax.plot([x, x], [y1, y2], 'k')
            ax.text(x, (y1 + y2) / 2, f'BS({param:.2f})', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

    plt.savefig(r"C:\Users\user\Downloads\gbsdecompositionpython.pdf", format='pdf', dpi=300) 
    plt.show()
    
# Draw the quantum circuit
draw_quantum_circuit(cmds, U_corr.shape[0], squeezing_parameters)


# # Sanity Check of Decomposition

# In[9]:

# Define the RBF kernel function
def rbf_kernel(R: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-((scipy.spatial.distance.cdist(R, R)) ** 2) / 2 / sigma**2)

# Define the sampling function
def sample(K: np.ndarray, n_mean: float, n_samples: int) -> list:
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()

print(target_n_mean)

print(B)


# Generate samples
n_samples = 300 # number of samples to generate
samples = sample(B, target_n_mean, n_samples)

# Function to plot heatmap
def plot_heatmap(matrix, title, ax=None, vmin=None, vmax=None, xticklabels=None, yticklabels=None):
    # Calculate the magnitude of the complex matrix
    matrix_magnitude = np.abs(matrix)
    
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
        plt.title(title)
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels)
        ax.set_title(title)
        
# Plot the kernel matrix
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_heatmap(B, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.25, xticklabels=selected_stocks, yticklabels=selected_stocks)

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
plt.xticks(range(len(normalized_counts)), selected_stocks, rotation=45, ha='right')
plt.show()


# In[10]:

def plot_samples(samples, title, xticklabels=None):
    """
    Plots the given samples in a heatmap-like fashion.
    
    Args:
        samples (array-like): A 2D array where each row represents a sample and each column a mode.
        title (str): Title of the plot.
        xticklabels (list): Labels for the x-axis (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(samples, aspect='auto', cmap='inferno', interpolation='nearest', vmax = 10)
    
    plt.colorbar(label='Photon Count')
    plt.title(title)
    plt.ylabel('Sample')
    plt.xlabel('Mode')

    if xticklabels is not None:
        plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=90)
    
    plt.show()

# Example usage:
# Assuming s_thresh is your sample data and DPP_selected_stocks are the x-axis labels
plot_samples(samples, "Threshold Samples", selected_stocks)

# If you want to plot the PNR samples:
# plot_samples(s_pnr, "PNR Samples", selected_stocks)


# # It would be good to understand what is going on here and ALSO if there is anything else I could do to check this is correct

# In[ ]:




