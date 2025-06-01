import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# Import functions from the main script
from run_multistock_abm import load_real_market_data, load_simulation_data, calculate_correlation_matrix, calculate_pcmi, create_pcmi_plots, perform_pcmi_analysis

# Test the PCMI analysis directly
base_output_dir = "multistock_abm_outputs"
sp500_csv_path = r"C:\Oepnfilelearningpath\Thesis\PFNN\sp500_csv_data"

# Perform the PCMI analysis
if os.path.exists(base_output_dir):
    perform_pcmi_analysis(base_output_dir, sp500_csv_path, max_stocks=20)
else:
    print("No simulation data found. Please run the full simulation first.")
