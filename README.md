# GBS_PointProcesses_Finance
Exploratory Study of Simulating Financial Markets with Gaussian Boson Sampling

This code provides tools to perform statistical analysis of financial data using Gaussian Boson Sampling (GBS) simulation software. It is built on StrawberryFields, an open-source Python package developed by Xanadu.

The link to the project with all the required information can be found here: https://www.overleaf.com/read/kvjyvsrphqxm#e2b352

The files contain the following:

- Analysis_SP500_Returns: Extracts SP500 data and analyzes returns, log returns (and normalized probability density), and close prices. The analysis explains why log returns are used in this exploration due to their statistical properties. It includes future research areas, such as the Autocorrelation Function (ACF) of daily returns, an indicator of volatility and volatility clustering, which is suitable for GBS modeling. This analysis is derived from Lukas Gonon's lecture notes at Imperial College London.

- DPP_Kulesza_adapted: Extracts MATLAB code from Kulesza's Determinantal Point Process (DPP) analysis and adapts it to Python. Applies these point processes to stock data, reproducing the results from Jahangiri et al.

- GBS_Decomposition_Stocks: This notebook includes the protocol to decompose a kernel matrix derived from stock data (log returns) into a GBS device, specifying squeezing parameters, beam splitters, and phase shifters according to the Clements decomposition. It tests the success of this process using Strawberry Fields sampling functions.

- HPP&TPP&Thermal_EnergyvsTech: Provides an analysis of Hafnian Point Processes (HPP) using Photon Number Resolving (PNR) detectors, Torontonian Point Processes (TPP) using Threshold detectors, and a classical quantum-inspired Thermal PP (which uses thermal states). It explores the effect of these processes on stocks from the technology and energy sectors, assessing which group is more correlated by calculating the area under the curve (AUC) of the probability distribution.

- HPP&TPP&Thermal_StockCorrelation: Similar to the above, but uses stocks from the same sector versus stocks from different sectors, identifying which stocks belong to the same sector. The selected stocks can be modified.

- RBFKernel_GBSPointProcesses: Provides an overview of the StrawberryFields module for Sampling Point Processes using a homogeneous RBF Kernel function (based on 2D Euclidean distance between points). It attempts to adapt this analysis to higher-dimensional data such as stock log returns, providing samples and a starting point for further exploration.

- Significant_dates_GBS: Constructs a temporal kernel matrix from the 2008 financial crisis and identifies significant events in the time series using GBS Point Process Sampling.

- Modelling_LOB: Uses a similar temporal kernel matrix to outperform a Hawkes Process and Poisson Process in modeling the time series of the Limit Order Book (LOB).

- HPP_stockCorrelation: Focuses on Hafnian Point Processes (using PNR detectors) to sample from a larger set of stocks and extract the most correlated ones, attempting to reproduce the results from Jahangiri et al. It also outlines the GBS experimental setup.

- TPP_stockCorrelation: Focuses on Torontonian Point Processes (using Threshold detectors) to sample from 17 stocks from Jahangiri et al., identifying two different clusters from two sectors. It hints at a method to find the centers for k-means clustering algorithms.
