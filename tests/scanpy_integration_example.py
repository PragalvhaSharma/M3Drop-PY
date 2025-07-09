"""
M3Drop + Scanpy Integration Example
===================================

This example demonstrates how to integrate M3Drop normalization and feature selection
methods into a standard Scanpy single-cell RNA-seq analysis workflow.

Based on the Scanpy clustering tutorial:
https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html
"""

import scanpy as sc
import numpy as np

# Import M3Drop
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import m3Drop as m3d

# Scanpy settings
sc.settings.verbosity = 3  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

def standard_scanpy_workflow(adata):
    """
    Standard Scanpy clustering workflow for comparison.
    """
    print("=== STANDARD SCANPY WORKFLOW ===")
    
    # Make a copy to avoid modifying original data
    adata_standard = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata_standard, min_genes=200)
    sc.pp.filter_genes(adata_standard, min_cells=3)
    
    # Calculate QC metrics
    adata_standard.var['mt'] = adata_standard.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_standard, percent_top=None, log1p=False, inplace=True)
    
    # Standard normalization
    sc.pp.normalize_total(adata_standard, target_sum=1e4)
    sc.pp.log1p(adata_standard)
    
    # Standard feature selection
    sc.pp.highly_variable_genes(adata_standard, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_standard.raw = adata_standard
    adata_standard = adata_standard[:, adata_standard.var.highly_variable]
    
    # Scale and perform PCA
    sc.pp.scale(adata_standard, max_value=10)
    sc.tl.pca(adata_standard, svd_solver='arpack')
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata_standard, n_neighbors=10, n_pcs=40)
    
    # Perform clustering
    sc.tl.leiden(adata_standard)
    
    # Run UMAP
    sc.tl.umap(adata_standard)
    
    print(f"Standard workflow completed:")
    print(f"  - {adata_standard.n_obs} cells")
    print(f"  - {adata_standard.n_vars} highly variable genes")
    print(f"  - {len(adata_standard.obs['leiden'].unique())} clusters identified")
    
    return adata_standard

def m3drop_scanpy_workflow(adata):
    """
    Scanpy workflow enhanced with M3Drop normalization and feature selection.
    """
    print("\n=== M3DROP + SCANPY WORKFLOW ===")
    
    # Make a copy to avoid modifying original data
    adata_m3drop = adata.copy()
    
    # Basic filtering
    sc.pp.filter_cells(adata_m3drop, min_genes=200)
    sc.pp.filter_genes(adata_m3drop, min_cells=3)
    
    # Calculate QC metrics
    adata_m3drop.var['mt'] = adata_m3drop.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_m3drop, percent_top=None, log1p=False, inplace=True)
    
    # Store raw counts for M3Drop methods
    adata_m3drop.layers['counts'] = adata_m3drop.X.copy()
    
    # M3Drop normalization (replaces sc.pp.normalize_total)
    print("Applying M3Drop NBumi normalization...")
    m3d.scanpy.nbumi_normalize(adata_m3drop)
    
    # Optional: log transform after M3Drop normalization
    sc.pp.log1p(adata_m3drop)
    
    # M3Drop feature selection (replaces sc.pp.highly_variable_genes)
    print("Applying M3Drop consensus feature selection...")
    m3d.scanpy.m3drop_highly_variable_genes(adata_m3drop, ntop=2000)
    
    # Subset to highly variable genes
    adata_m3drop.raw = adata_m3drop
    adata_m3drop = adata_m3drop[:, adata_m3drop.var.highly_variable]
    
    # Scale and perform PCA
    sc.pp.scale(adata_m3drop, max_value=10)
    sc.tl.pca(adata_m3drop, svd_solver='arpack')
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata_m3drop, n_neighbors=10, n_pcs=40)
    
    # Perform clustering
    sc.tl.leiden(adata_m3drop)
    
    # Run UMAP
    sc.tl.umap(adata_m3drop)
    
    print(f"M3Drop workflow completed:")
    print(f"  - {adata_m3drop.n_obs} cells")
    print(f"  - {adata_m3drop.n_vars} highly variable genes")
    print(f"  - {len(adata_m3drop.obs['leiden'].unique())} clusters identified")
    
    # Show M3Drop-specific results
    print(f"  - M3Drop consensus ranking available in adata.var['m3drop_consensus_rank']")
    print(f"  - Individual method rankings: {[col for col in adata_m3drop.var.columns if col.startswith('m3drop_') and col.endswith('_rank')]}")
    
    return adata_m3drop

def compare_methods(adata_standard, adata_m3drop):
    """
    Compare results from standard and M3Drop workflows.
    """
    print("\n=== COMPARISON ===")
    
    # Compare number of clusters
    n_clusters_standard = len(adata_standard.obs['leiden'].unique())
    n_clusters_m3drop = len(adata_m3drop.obs['leiden'].unique())
    
    print(f"Number of clusters:")
    print(f"  - Standard Scanpy: {n_clusters_standard}")
    print(f"  - M3Drop + Scanpy: {n_clusters_m3drop}")
    
    # Compare highly variable genes if both have the same number of cells
    if adata_standard.n_obs == adata_m3drop.n_obs:
        # Get original gene names for comparison
        hvg_standard = set(adata_standard.var_names)
        hvg_m3drop = set(adata_m3drop.var_names)
        
        overlap = len(hvg_standard.intersection(hvg_m3drop))
        total_hvg = len(hvg_standard.union(hvg_m3drop))
        
        print(f"Highly variable genes overlap:")
        print(f"  - Standard method: {len(hvg_standard)} genes")
        print(f"  - M3Drop method: {len(hvg_m3drop)} genes")
        print(f"  - Overlap: {overlap} genes ({overlap/min(len(hvg_standard), len(hvg_m3drop))*100:.1f}%)")

def main():
    """
    Main function demonstrating the workflows.
    Replace this section with your own data loading.
    """
    print("M3Drop + Scanpy Integration Example")
    print("===================================")
    
    # Example 1: Load your own data
    # Uncomment and modify one of these options:
    
    # Option 1: 10X data
    # adata = sc.read_10x_mtx(
    #     'path/to/matrix.mtx',  # Path to the count matrix
    #     var_names='gene_symbols',  # use gene symbols for gene names
    #     cache=True  # write a cache file for faster subsequent reading
    # )
    # adata.var_names_unique()
    
    # Option 2: H5 file
    # adata = sc.read_h5ad('path/to/data.h5ad')
    
    # Option 3: CSV file
    # adata = sc.read_csv('path/to/data.csv').T  # Transpose if genes are columns
    
    # Example 2: Generate synthetic data for demonstration
    print("Generating synthetic data for demonstration...")
    np.random.seed(42)
    n_obs, n_vars = 1000, 2000
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
    
    adata = sc.AnnData(X)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    
    print(f"Created synthetic dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Run both workflows
    adata_standard = standard_scanpy_workflow(adata)
    adata_m3drop = m3drop_scanpy_workflow(adata)
    
    # Compare results
    compare_methods(adata_standard, adata_m3drop)
    
    # Plotting (optional)
    print("\n=== PLOTTING ===")
    print("To visualize results, you can use:")
    print("sc.pl.umap(adata_standard, color=['leiden'], title='Standard Scanpy')")
    print("sc.pl.umap(adata_m3drop, color=['leiden'], title='M3Drop + Scanpy')")
    
    return adata_standard, adata_m3drop

if __name__ == "__main__":
    # Run the example
    adata_standard, adata_m3drop = main()
    
    # Optional: Create plots
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sc.pl.umap(adata_standard, color='leiden', ax=ax1, title='Standard Scanpy', show=False)
        sc.pl.umap(adata_m3drop, color='leiden', ax=ax2, title='M3Drop + Scanpy', show=False)
        
        plt.tight_layout()
        plt.savefig('m3drop_scanpy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nComparison plot saved as 'm3drop_scanpy_comparison.png'")
        
    except ImportError:
        print("Matplotlib not available for plotting") 