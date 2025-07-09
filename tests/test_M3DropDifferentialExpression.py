import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
from m3Drop.differential_expression import M3DropDifferentialExpression

# Step 1: Load your AnnData (.h5ad) file
h5ad_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/newM3dUpdate/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
adata = sc.read_h5ad(h5ad_file)
print("AnnData object loaded successfully:")
print(adata)


# Step 2: Prepare the data for M3Drop analysis
# M3Drop functions use raw counts, so we extract the count matrix
# and transpose it to have genes as rows and cells as columns.
raw_counts = adata.to_df().T

if not isinstance(raw_counts, pd.DataFrame):
    raw_counts = pd.DataFrame(raw_counts, index=adata.var_names, columns=adata.obs_names)


# Step 3: Run M3DropDifferentialExpression Analysis
print("Running M3DropDifferentialExpression...")
de_genes = M3DropDifferentialExpression(raw_counts, mt_threshold=0.05)

# Step 4: Print the differentially expressed genes
print("Found these differentially expressed genes:")
print(de_genes)

# Basic check to ensure the output is a DataFrame and not empty
assert isinstance(de_genes, pd.DataFrame)
# It's possible for no genes to be significant, so we don't check for empty
# assert not de_genes.empty 
print("Test passed: M3DropDifferentialExpression ran successfully.") 