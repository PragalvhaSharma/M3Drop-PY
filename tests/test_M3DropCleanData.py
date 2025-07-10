import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
from m3Drop.Normalization import M3DropCleanData
import numpy as np

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data
# Using raw counts as the function expects counts
raw_counts = adata.to_df().T


# Step 3: Run M3DropCleanData Analysis
print("Running M3DropCleanData...")
cleaned_data_dict = M3DropCleanData(raw_counts, is_counts=True)
cleaned_data = cleaned_data_dict['data']

# Step 4: Print the cleaned data info
print("Data after cleaning:")
print(cleaned_data)
print(f"Original shape: {raw_counts.shape}, Cleaned shape: {cleaned_data.shape}")

# Basic check to ensure the output is a numpy array and that some filtering happened
assert isinstance(cleaned_data, np.ndarray)
assert cleaned_data.shape[0] < raw_counts.shape[0] 
assert cleaned_data.shape[1] <= raw_counts.shape[1]

print("Test passed: M3DropCleanData ran successfully.") 