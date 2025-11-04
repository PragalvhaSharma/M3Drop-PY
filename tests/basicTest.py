import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import scipy.sparse
from m3Drop.Extremes import M3DropGetExtremes
from m3Drop.basics import M3DropConvertData, compute_gene_statistics_h5ad

# Step 1: Load the required libraries are imported above.

# Step 2: Load your AnnData (.h5ad) file
# Replace with the actual path to your file if different.
h5ad_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/CompSci/myProjects/M3Drop/M3Drop-PY/m3Drop/Human_Heart.h5ad"
print(f"Loading data from: {h5ad_file}")

file_size_bytes = os.path.getsize(h5ad_file)
file_size_gb = file_size_bytes / (1024 ** 3)
streaming_threshold_gb = float(os.environ.get("M3DROP_STREAMING_THRESHOLD_GB", 5))
chunk_size = int(os.environ.get("M3DROP_STREAMING_CHUNK", 5000))
use_streaming = file_size_gb >= streaming_threshold_gb

gene_info = None
normalized_matrix = None

print(f"Detected file size: {file_size_gb:.2f} GB")
if use_streaming:
    print(
        f"Dataset exceeds {streaming_threshold_gb:.0f} GB; "
        f"streaming gene statistics with chunk size {chunk_size}."
    )
    gene_info, n_cells = compute_gene_statistics_h5ad(h5ad_file, chunk_size=chunk_size)
    n_genes = len(gene_info['p'])
    print(f"Computed gene statistics for {n_genes} genes across {n_cells} cells (CPM normalisation).")
else:
    print("Dataset is within in-memory limits; loading into AnnData...")
    adata = sc.read_h5ad(h5ad_file)
    print("AnnData object loaded successfully:")
    print(adata)
    print(f"Data shape: {adata.shape} (cells x genes)")
    print(f"Is sparse: {scipy.sparse.issparse(adata.X)}")

    # Step 3: Prepare the data for M3Drop analysis
    # M3Drop requires a normalized, non-log-transformed expression matrix.
    # Use M3DropConvertData which handles sparse matrices efficiently
    print("Converting data for M3Drop (memory-efficient sparse mode)...")
    preserve_sparse = scipy.sparse.issparse(adata.X)
    if preserve_sparse:
        print("Input matrix is sparse; keeping sparse representation to avoid densifying large datasets.")
    normalized_matrix = M3DropConvertData(
        adata,
        is_counts=True,  # Indicate we have raw counts
        preserve_sparse=preserve_sparse
    )
    print(f"Normalized data shape: {normalized_matrix.shape} (genes x cells)")

# Step 4: Run M3Drop Analysis using M3DropGetExtremes
print("Running M3DropGetExtremes...")
if use_streaming:
    m3drop_features = M3DropGetExtremes(expr_mat=None, gene_info=gene_info, suppress_plot=True)
else:
    m3drop_features = M3DropGetExtremes(normalized_matrix, suppress_plot=True)

# Step 5: Print the differentially expressed genes
print("\nFound these differentially expressed genes (right tail):")
print(f"Number of right-tail genes: {len(m3drop_features['right'])}")
print(f"Number of left-tail genes: {len(m3drop_features['left'])}")
if m3drop_features['right']:
    print("\nTop 10 right-tail genes:")
    print(m3drop_features['right'][:10])
