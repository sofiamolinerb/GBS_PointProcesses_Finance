#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import plotly
from sklearn.datasets import make_blobs#
from strawberryfields.apps import*
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import yfinance as yf
from scipy.integrate import simps # Path to the zip file
import pandas as pd
import zipfile
from PIL import Image
import scipy.linalg
from scipy.optimize import root_scalar
# Continue with the Strawberry Fields program
import strawberryfields as sf
from strawberryfields.ops import Sgate, BSgate, Rgate, Interferometer

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

#The idea is to create a GBS device capable of sampling from correlation matrix from different stock prices

#First we extract the information from the stocks

zip_file_path = r'C:\Users\user\Downloads\archive (1).zip'
csv_file_name = 'all_stocks_5yr.csv'

# Extract the specific CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open(csv_file_name) as f:
        all_stocks_df = pd.read_csv(f)

# List of selected stock names
selected_stocks = ["ALXN", "BA", "CMI", "COG", "COP", "DVN", "HAL", "HON", "INTU"]

# Number of stocks
n_stocks = len(selected_stocks)
n_days = len(daily_returns)
print("the number of stocks is " + str(n_stocks) + " and the number of days used is " + str(n_days))

correlation_matrix = calculate_correlation_matrix_log(selected_stocks)


# In[26]:


#Compare with jahanguiri results
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


# In[27]:


from itertools import groupby
from collections import defaultdict
from scipy.linalg import block_diag, sqrtm, polar, schur
from thewalrus.quantum import adj_scaling
from thewalrus.symplectic import sympmat, xpxp_to_xxpp

#Import StrawberryFields source code for Takagi Decomposition
def takagi(N, tol=1e-13, rounding=13): 
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.

    Note that singular values of N are considered equal if they are equal after np.round(values, tol).

    See :cite:`cariolaro2016` and references therein for a derivation.

    Args:
        N (array[complex]): square, symmetric matrix N
        rounding (int): the number of decimal places to use when rounding the singular values of N
        tol (float): the tolerance used when checking if the input matrix is symmetric: :math:`|N-N^T| <` tol

    Returns:
        tuple[array, array]: (rl, U), where rl are the (rounded) singular values,
            and U is the Takagi unitary, such that :math:`N = U \diag(rl) U^T`.
    """
    (n, m) = N.shape
    if n != m:
        raise ValueError("The input matrix must be square")
    if np.linalg.norm(N - np.transpose(N)) >= tol:
        raise ValueError("The input matrix is not symmetric")

    N = np.real_if_close(N)

    if np.allclose(N, 0):
        return np.zeros(n), np.eye(n)

    if np.isrealobj(N):
        # If the matrix N is real one can be more clever and use its eigendecomposition
        l, U = np.linalg.eigh(N)
        vals = np.abs(l)  # These are the Takagi eigenvalues
        phases = np.sqrt(np.complex128([1 if i > 0 else -1 for i in l]))
        Uc = U @ np.diag(phases)  # One needs to readjust the phases
        list_vals = [(vals[i], i) for i in range(len(vals))]
        list_vals.sort(reverse=True)
        sorted_l, permutation = zip(*list_vals)
        permutation = np.array(permutation)
        Uc = Uc[:, permutation]
        # And also rearrange the unitary and values so that they are decreasingly ordered
        return np.array(sorted_l), Uc

    v, l, ws = np.linalg.svd(N)
    w = np.transpose(np.conjugate(ws))
    rl = np.round(l, rounding)

    # Generate list with degenerancies
    result = []
    for k, g in groupby(rl):
        result.append(list(g))

    # Generate lists containing the columns that correspond to degenerancies
    kk = 0
    for k in result:
        for ind, j in enumerate(k):  # pylint: disable=unused-variable
            k[ind] = kk
            kk = kk + 1

    # Generate the lists with the degenerate column subspaces
    vas = []
    was = []
    for i in result:
        vas.append(v[:, i])
        was.append(w[:, i])

    # Generate the matrices qs of the degenerate subspaces
    qs = []
    for i in range(len(result)):
        qs.append(sqrtm(np.transpose(vas[i]) @ was[i]))

    # Construct the Takagi unitary
    qb = block_diag(*qs)

    U = v @ np.conj(qb)
    return rl, U


# In[28]:


# Define functions

def plot_heatmap(matrix, title, ax=None, vmin=None, vmax=None, save_path=None):
    # Calculate the magnitude of the complex matrix
    matrix_magnitude = np.abs(matrix)
    
    if ax is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, format="png", dpi=300)  # Save the heatmap
        plt.show()
    else:
        sns.heatmap(matrix_magnitude, annot=True, cmap="inferno", ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path, format="png", dpi=300)  # Save the heatmap if the path is provided


# Function to perform Takagi-Autonne decomposition and return necessary matrices (manually performed to see if it works for real matrices) 
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

# Plot the kernel matrix and save the heatmap
plot_heatmap(correlation_matrix, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.4, save_path="kernel_matrix_heatmap_9stocks.pdf")


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


# # Using StrawberryFields Takagi function to compare

# In[29]:


# Attempt to use the StrawberryFields function and compare

# Perform Takagi-Autonne decomposition
lambdas, U_corr = takagi(correlation_matrix)

# Solve for c given a target mean photon number
target_n_mean = 5  # Example target mean photon number
c = solve_for_c(lambdas, target_n_mean)

# Calculate the squeezing parameters
squeezing_parameters = calculate_squeezing_parameters(lambdas, c)
# Create a figure with one subplot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the kernel matrix
plot_heatmap(correlation_matrix, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.4)

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
#NOTE 2: BOTH MATRICES ARE THE SAME so the manual decomposition is correct for real matrices


# # Both Takagi Decompositions yield the same unitary and squeezing parameters so we can keep the simple one for real-world data

# In[30]:


def is_unitary(matrix, tol=1e-10):
    identity = np.eye(matrix.shape[0])
    return np.allclose(matrix @ matrix.conj().T, identity, atol=tol)

# Assuming U_corr is defined
#U_corr = np.array([[0.70710678 + 0.j, 0.5 + 0.5j],[0.5 - 0.5j, 0.70710678 + 0.j]])  # Example unitary matrix

if not is_unitary(U_corr):
    raise ValueError("The input matrix is not unitary")

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


# # Thermal sampling from this matrix: it is enough to show the success of the protocol (used in SF source code)

# In[33]:


from thewalrus.csamples import generate_thermal_samples, rescale_adjacency_matrix_thermal

# Define the sampling function
def sample(K: np.ndarray, n_mean: float, n_samples: int) -> list:
    ls, O = rescale_adjacency_matrix_thermal(K, n_mean)
    return np.array(generate_thermal_samples(ls, O, num_samples=n_samples)).tolist()

# Generate samples
n_samples = 300 # number of samples to generate
samples = sample(correlation_matrix, target_n_mean, n_samples)

# Function to plot heatmap, I used a different one, this one does not save the figure and includes the stocks in x and y axis
def plot_heatmap_selected(matrix, title, ax=None, vmin=None, vmax=None, xticklabels=None, yticklabels=None):
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
plot_heatmap_selected(correlation_matrix, "Kernel Matrix - GBS", ax=ax, vmin=0, vmax=0.4, xticklabels=selected_stocks, yticklabels=selected_stocks)

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
plt.savefig(r"C:\Users\user\Downloads\kernelandsamplinggbsdecomp.pdf", format='pdf', dpi=300)
plt.show()


# In[34]:


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
    plt.savefig(r"C:\Users\user\Downloads\samplinggbssanitycheck.pdf", format='pdf', dpi=300)
    plt.show()

# Example usage:
# Assuming s_thresh is your sample data and DPP_selected_stocks are the x-axis labels
plot_samples(samples, "Threshold Samples", selected_stocks)

# If you want to plot the PNR samples:
# plot_samples(s_pnr, "PNR Samples", selected_stocks)


# In[35]:


#Now use StrawberryFields to calculate Hafnian and confirm the samples agree.


# In[36]:


# Create the GBS program
# Define a function to create the GBS program
def create_gbs_program(prog, squeezing_parameters, unitary_matrix):
    with prog.context as q:
        # Apply squeezing gates
        for i in range(len(squeezing_parameters)):
            Sgate(squeezing_parameters[i]) | q[i]

        # Apply the unitary transformation (interferometer)
        Interferometer(unitary_matrix) | q

# Apply to different and same sectors
create_gbs_program(prog, squeezing_parameters , U_corr)

# Run the programs on the Fock backend with 500 samples
eng = sf.Engine(backend="gaussian", backend_options={"cutoff_dim": 6})
samples = eng.run(prog)


# In[37]:


# Higher counts in the sides: not very likely if we see the sample image 
state_high_ends = [3, 0, 0, 0, 0, 0, 0, 0, 3]

# Higher counts in the center: very likely if we see the sample image
state_high_center= [0, 0, 0, 3, 0, 3, 0, 0, 0]

# Other examples of possible states
state_example = [2, 0, 0, 0, 3, 0, 0, 0, 1]

# Calculate probabilities for each state
measure_states = [state_high_ends, state_high_center, state_example]

for i in measure_states:
    prob = samples.state.fock_prob(i)
    print("|{}>: {}".format("".join(str(j) for j in i), prob))


# # This agrees with the image showing the succes of the decomposition protocol
