import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
from m3Drop.Brennecke_implementation import BrenneckeGetVariableGenes

# Step 1: Load your AnnData (.h5ad) file
# Replace with the actual path to your file if different.
h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data for M3Drop analysis
# M3Drop requires a normalized, non-log-transformed expression matrix.
# We use scanpy's normalize_total for this.
sc.pp.normalize_total(adata, target_sum=1e4)
print("Normalized data for M3Drop.")

# M3Drop expects a pandas DataFrame with genes as rows and cells as columns.
# anndata's to_df() provides this.
# Note: The anndata object stores genes as columns (obs) and cells as rows (vars) by default.
# We need to transpose the DataFrame for M3Drop.
normalized_matrix = adata.to_df().T


# Step 3: Run BrenneckeGetVariableGenes Analysis
print("Running BrenneckeGetVariableGenes...")
hvg_genes = BrenneckeGetVariableGenes(normalized_matrix, fdr=0.1)

# Step 4: Print the highly variable genes
print("Found these highly variable genes:")
print(hvg_genes)

# Basic check to ensure the output is a DataFrame and not empty
assert isinstance(hvg_genes, pd.DataFrame)
assert not hvg_genes.empty
print("Test passed: BrenneckeGetVariableGenes ran successfully and returned a non-empty DataFrame.") 