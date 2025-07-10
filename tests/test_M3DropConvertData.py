import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
import numpy as np
from m3Drop.basics import M3DropConvertData

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data
# Use a pandas DataFrame of raw counts
raw_counts_df = adata.to_df().T

# Step 3: Test case 1: Convert raw counts
print("Running M3DropConvertData with raw counts...")
converted_data_counts = M3DropConvertData(raw_counts_df, is_counts=True)
print("Converted data from counts:")
print(converted_data_counts)
assert isinstance(converted_data_counts, pd.DataFrame)
sums = converted_data_counts.sum(axis=0)
assert (np.isclose(sums, 1e6) | np.isclose(sums, 0)).all() # Should be CPM or 0 for empty cells
print("Test 1 passed.")

# Step 4: Test case 2: Convert log-transformed data
print("\nRunning M3DropConvertData with log-transformed data...")
log_transformed_data = np.log1p(raw_counts_df)
converted_data_log = M3DropConvertData(log_transformed_data, is_log=True)
print("Converted data from log:")
print(converted_data_log)
assert isinstance(converted_data_log, pd.DataFrame)
# The output should be un-logged, so values should not be small log values
assert converted_data_log.max().max() > 10 
print("Test 2 passed.")

# Step 5: Test case 3: Convert AnnData object directly
print("\nRunning M3DropConvertData with AnnData object...")
# The function expects genes x cells, so we transpose the AnnData object.
adata_transposed = adata.T
converted_data_adata = M3DropConvertData(adata_transposed, is_counts=True)
print("Converted data from AnnData:")
print(converted_data_adata)
assert isinstance(converted_data_adata, pd.DataFrame)
# The data is returned as a dataframe, so we can check the sum of columns.
sums_adata = converted_data_adata.sum(axis=0)
assert (np.isclose(sums_adata, 1e6) | np.isclose(sums_adata, 0)).all()
print("Test 3 passed.")

print("\nAll tests for M3DropConvertData passed.") 