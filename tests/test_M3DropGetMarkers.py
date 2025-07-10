import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
import numpy as np
from m3Drop.basics import M3DropGetMarkers

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data for M3Drop analysis
sc.pp.normalize_total(adata, target_sum=1e4)
print("Normalized data for M3Drop.")

normalized_matrix = adata.to_df().T # Transpose to have genes as rows, cells as columns for M3DropGetMarkers


# Step 3: Create dummy labels for the cells
num_cells = adata.shape[0]
labels = np.array(['GroupA'] * (num_cells // 2) + ['GroupB'] * (num_cells - num_cells // 2))
print(f"Created {len(labels)} labels for {num_cells} cells.")


# Step 4: Run M3DropGetMarkers Analysis
print("Running M3DropGetMarkers...")
marker_genes = M3DropGetMarkers(normalized_matrix, labels)

# Step 5: Print the marker genes
print("Found these marker genes:")
print(marker_genes)

# Basic check to ensure the output is a DataFrame and not empty
assert isinstance(marker_genes, pd.DataFrame)
assert not marker_genes.empty
print("Test passed: M3DropGetMarkers ran successfully and returned a non-empty DataFrame.") 