import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
from m3Drop.Extremes import M3DropTestShift

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data
# Handle sparse matrices by converting to dense format first
if hasattr(adata.X, 'toarray'):
    # If it's a sparse matrix, convert to dense
    raw_counts = adata.X.toarray().T
else:
    # If it's already dense, just transpose
    raw_counts = adata.X.T

if not isinstance(raw_counts, pd.DataFrame):
    raw_counts = pd.DataFrame(raw_counts, index=adata.var_names, columns=adata.obs_names)

# Step 3: Select a subset of genes to test
genes_to_test = raw_counts.index[:10].tolist()
print(f"Genes to test: {genes_to_test}")

# Step 4: Run M3DropTestShift Analysis
print("Running M3DropTestShift...")
shift_results = M3DropTestShift(raw_counts, genes_to_test=genes_to_test, name="First10")

# Step 5: Print the results
print("Shift test results:")
print(shift_results)

# Basic check to ensure the output is a DataFrame
assert isinstance(shift_results, pd.DataFrame)
assert not shift_results.empty
print("Test passed: M3DropTestShift ran successfully.") 