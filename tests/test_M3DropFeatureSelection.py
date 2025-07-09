import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
from m3Drop.feature_selection import M3DropFeatureSelection

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/newM3dUpdate/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data for M3Drop analysis
sc.pp.normalize_total(adata, target_sum=1e4)
print("Normalized data for M3Drop.")

normalized_matrix = adata.to_df().T


# Step 3: Run M3DropFeatureSelection Analysis
print("Running M3DropFeatureSelection with Brennecke method...")
highly_variable_genes = M3DropFeatureSelection(normalized_matrix, mt_method="fdr_bh", mt_threshold=0.01, method="brennecke")

# Step 4: Print the highly variable genes
print("Found these highly variable genes:")
print(highly_variable_genes)

# Basic check to ensure the output is a DataFrame and not empty
assert isinstance(highly_variable_genes, pd.DataFrame)
assert not highly_variable_genes.empty
print("Test passed: M3DropFeatureSelection with Brennecke ran successfully and returned a non-empty DataFrame.") 