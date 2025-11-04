import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import numpy as np
from m3Drop.basics import M3DropConvertData, SparseMat3Drop
from m3Drop import ann_data_to_sparse_gene_matrix

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = " "
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare a manageable subset to avoid densifying the full matrix
max_cells = min(adata.n_obs, 500)
max_genes = min(adata.n_vars, 500)
adata_subset = adata[:max_cells, :max_genes].copy()
raw_sparse = ann_data_to_sparse_gene_matrix(adata_subset)

# Step 3: Test case 1: Convert raw counts
print("Running M3DropConvertData with raw counts...")
converted_data_counts = M3DropConvertData(raw_sparse, is_counts=True, preserve_sparse=True)
print("Converted data from counts:")
print(converted_data_counts.shape)
assert isinstance(converted_data_counts, SparseMat3Drop)
count_sums = np.array(converted_data_counts.matrix.sum(axis=0)).flatten()
non_zero_cols = count_sums > 0
assert np.allclose(count_sums[non_zero_cols], 1e6, rtol=1e-3)
print("Test 1 passed.")

# Step 4: Test case 2: Convert log-transformed data
print("\nRunning M3DropConvertData with log-transformed data...")
log_sparse = raw_sparse.matrix.copy().astype(np.float32)
log_sparse.data = np.log2(log_sparse.data + 1)
converted_data_log = M3DropConvertData(log_sparse, is_log=True, preserve_sparse=True)
print("Converted data from log:")
print(converted_data_log.shape)
assert isinstance(converted_data_log, SparseMat3Drop)
# The output should be un-logged, so values should not be tiny log-scale numbers
max_value = converted_data_log.matrix.data.max() if converted_data_log.matrix.nnz > 0 else 0
assert max_value > 10 
print("Test 2 passed.")

# Step 5: Test case 3: Convert AnnData object directly
print("\nRunning M3DropConvertData with AnnData object...")
converted_data_adata = M3DropConvertData(adata_subset, is_counts=True, preserve_sparse=True)
print("Converted data from AnnData:")
print(converted_data_adata.shape)
assert isinstance(converted_data_adata, SparseMat3Drop)
count_sums_adata = np.array(converted_data_adata.matrix.sum(axis=0)).flatten()
non_zero_cols_adata = count_sums_adata > 0
assert np.allclose(count_sums_adata[non_zero_cols_adata], 1e6, rtol=1e-3)
print("Test 3 passed.")

print("\nAll tests for M3DropConvertData passed.") 
