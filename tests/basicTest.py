import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
from m3Drop.Extremes import M3DropGetExtremes

# Step 1: Load the required libraries are imported above.

# Step 2: Load your AnnData (.h5ad) file
# Replace with the actual path to your file if different.
h5ad_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/newM3dUpdate/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 3: Prepare the data for M3Drop analysis
# M3Drop requires a normalized, non-log-transformed expression matrix.
# We use scanpy's normalize_total for this.
sc.pp.normalize_total(adata, target_sum=1e4)
print("Normalized data for M3Drop.")

# M3Drop expects a pandas DataFrame with genes as rows and cells as columns.
# anndata's to_df() provides this.
# Note: The anndata object stores genes as columns (obs) and cells as rows (vars) by default.
# We need to transpose the DataFrame for M3Drop.
normalized_matrix = adata.to_df().T


# Step 4: Run M3Drop Analysis using M3DropGetExtremes
print("Running M3DropGetExtremes...")
m3drop_features = M3DropGetExtremes(normalized_matrix)

# Step 5: Print the differentially expressed genes
print("Found these differentially expressed genes (right tail):")
print(m3drop_features['right'])
