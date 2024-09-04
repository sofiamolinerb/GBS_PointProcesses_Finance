# GBS_PointProcesses_Finance
Exploratory study of simulating financial  markets with Gaussian boson sampling

This code will provide the code to perform statistical analysis on financial data using Gaussian Boson Sampling simulation software. It is built from the software from StrawberryFields, an open source Python Package developped by Xanadu.

The link to the project with all the required information can be found here: https://www.overleaf.com/read/kvjyvsrphqxm#e2b352

The files contain the following:

  - Analysis_SP500_Returns: extraction of SP500 data and analysis of returns, log returns (and normalised probability density) and close price. The analysis shows why log returns are used in this exploration due to their statistical properties. Includes analysis which could not be integrated in the project (areas of future research) such as the use of the Autocorrelation Function of daily returns (ACF) which is an indicator of volatility and shows volatility clustering (very suitable for GBS modelling), and . This analysis is extracted from Lukas Gonon's lectures notes from Imperial College London.
    
  - DPP_Kulesza_adapted: This code extracts the MATLAB code from Kulesza's Determinantal Point Process analysis and adapts it to Python. It applies these point processes to stocks and reproduces the results from Jahangiri et. al.
    
  - GBS_Deomposition_Stocks: This notebook includes the protocol to decompose a kernel matrix derived from stock data (log returns) into a GBS device (squeezing parameters, beam splitters and phase shifters) according to the Clements decomposition. It tests its success using Strawberry Fields Sampling functions.
    
  - HPP&TPP&Thermal_EnergyvsTech: Provides an analysis of Hafnian Point Processes (HPP) sampling from GBS with PNR detectors, Torontonian PP using Threshold detectors, and a classical quantum inspired Thermal PP (uses thermal states) exploring their effect in stocks from two sectors: technology and energy, in order to assess which group is most correlated by calculating the area under the curve of the probability distribution.
    
  - HPP&TPP&Thermal_StockCorrelation: Provides a similar analysis but using stocks form the same sector versus stocks from different sectors, identifying which ones belongs to the same sector. The selected stocks can be modified.
    
  - RBFKernel_GBSPointProcesses: Provides an overview on the StrawberryFieds module for Sampling for Point Processes using an homogeneous RBF Kernel function (based on 2D euclidean distance between points). It attempts to adapt this analysis to higher dimensional data such as stock log returns, providing samples and a starting point for further exploration.
    
  - Significant_dates_GBS: Constructs a temporal kernel matrix from the 2008 crisis and identifies significant events in the time series using GBS PP Sampling.
    
  - Modelling_LOB: Uses the same temporal kernel to outperform a Hawkes Process and a Poisson Process in the modelling og the time series of the Limit Order Book (LOB).
    
  - HPP_stockCorrelation: Focuses on Hafnian PP (using PNR detectors) to sample from a larger number of stocks and extract the most correlated ones, attempting to reproduce the results from Jahanguiri. It derives the GBS experimental setup.
    
  - TPP_stockCorrelation: Focuses on Torontonian PP (using Threshold detectors) to sample from the 17 stocks from Jahangiri, finding two different clusters from two sectors. It hints at a method to find the centres of k-means clustering algorithms.
