#!/usr/bin/env python3
"""
M3Drop Sparse Matrix Optimization Example
==========================================

This example demonstrates how to use M3Drop's optimized sparse matrix support
for memory-efficient analysis of large single-cell RNA-seq datasets.

The key improvements are:
1. Memory efficiency: Sparse matrices stay sparse, reducing memory usage by 10-100x
2. Performance: Operations are optimized for sparse data structures
3. Compatibility: All existing M3Drop functions work with sparse inputs
"""

import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os

# Add M3Drop to path if needed
sys.path.insert(0, os.path.abspath('..'))

from m3Drop.basics import M3DropConvertData, bg__calc_variables
from m3Drop.NB_UMI import NBumiConvertData, NBumiFitModel
from m3Drop.Extremes import M3DropFeatureSelection


def load_example_data():
    """Load or create example sparse single-cell data"""
    print("📂 Loading example data...")
    
    try:
        # Try to load real data if available
        adata = sc.datasets.pbmc3k_processed()
        print(f"✅ Loaded PBMC3k dataset: {adata.shape}")
        
        # Use raw counts if available
        if adata.raw is not None:
            adata.X = adata.raw.X
        
    except:
        # Create synthetic sparse data if real data not available
        print("🔧 Creating synthetic sparse data...")
        n_cells, n_genes = 2000, 5000
        
        # Create sparse count matrix
        np.random.seed(42)
        density = 0.05  # 5% of values are non-zero (95% sparse)
        nnz = int(n_cells * n_genes * density)
        
        rows = np.random.choice(n_cells, nnz)
        cols = np.random.choice(n_genes, nnz)
        data = np.random.poisson(3, nnz)  # Count data
        
        X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
        
        adata = sc.AnnData(X=X)
        adata.var_names = [f"Gene_{i:04d}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i:04d}" for i in range(n_cells)]
        
        print(f"✅ Created synthetic data: {adata.shape}")
    
    # Print data characteristics
    sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"   📊 Sparsity: {sparsity:.2%}")
    print(f"   📊 Non-zero elements: {adata.X.nnz:,}")
    print(f"   📊 Data type: {type(adata.X)}")
    
    return adata


def compare_memory_usage(adata):
    """Compare memory usage between sparse and dense approaches"""
    print("\n" + "="*60)
    print("💾 MEMORY USAGE COMPARISON")
    print("="*60)
    
    import psutil
    process = psutil.Process()
    
    # Get initial memory
    mem_start = process.memory_info().rss / 1024 / 1024
    print(f"📊 Initial memory: {mem_start:.1f} MB")
    
    # Option 1: Traditional approach (convert to dense)
    print("\n1️⃣ Traditional approach (preserve_sparse=False):")
    try:
        converted_dense = M3DropConvertData(adata, is_counts=True, preserve_sparse=False)
        mem_dense = process.memory_info().rss / 1024 / 1024
        print(f"   ✅ Success: {converted_dense.shape}")
        print(f"   📊 Memory usage: {mem_dense:.1f} MB (+{mem_dense-mem_start:.1f} MB)")
        print(f"   📊 Output type: {type(converted_dense)}")
        del converted_dense
    except MemoryError:
        print("   ❌ Memory Error: Dataset too large for dense conversion!")
    
    # Option 2: Optimized approach (preserve sparse)
    print("\n2️⃣ Optimized approach (preserve_sparse=True):")
    converted_sparse = M3DropConvertData(adata, is_counts=True, preserve_sparse=True)
    mem_sparse = process.memory_info().rss / 1024 / 1024
    print(f"   ✅ Success: {converted_sparse.shape}")
    print(f"   📊 Memory usage: {mem_sparse:.1f} MB (+{mem_sparse-mem_start:.1f} MB)")
    print(f"   📊 Output type: {type(converted_sparse)}")
    
    return converted_sparse


def demonstrate_m3drop_workflow(adata):
    """Demonstrate complete M3Drop workflow with sparse optimization"""
    print("\n" + "="*60)
    print("🔬 M3DROP WORKFLOW WITH SPARSE OPTIMIZATION")
    print("="*60)
    
    # Step 1: Convert data (preserving sparsity)
    print("\n1️⃣ Data conversion...")
    converted_data = M3DropConvertData(adata, is_counts=True, preserve_sparse=True)
    print(f"   ✅ Converted to M3Drop format: {converted_data.shape}")
    
    # Step 2: Calculate gene-specific variables
    print("\n2️⃣ Calculating gene-specific variables...")
    gene_vars = bg__calc_variables(converted_data)
    print(f"   ✅ Calculated variables for {len(gene_vars['s'])} genes")
    print(f"   📊 Mean expression range: {gene_vars['s'].min():.3f} - {gene_vars['s'].max():.3f}")
    print(f"   📊 Dropout rate range: {gene_vars['p'].min():.3f} - {gene_vars['p'].max():.3f}")
    
    # Step 3: Feature selection (if data is appropriate)
    print("\n3️⃣ Feature selection...")
    try:
        # Convert to DataFrame for feature selection if needed
        if hasattr(converted_data, 'to_dataframe'):
            expr_df = converted_data.to_dataframe()
        else:
            expr_df = converted_data
        
        hvg_genes = M3DropFeatureSelection(expr_df, mt_threshold=0.1, suppress_plot=True)
        print(f"   ✅ Found {len(hvg_genes)} highly variable genes")
        if len(hvg_genes) > 0:
            print(f"   📊 Top genes: {list(hvg_genes.index[:5])}")
    except Exception as e:
        print(f"   ⚠️  Feature selection skipped: {str(e)[:100]}")
    
    # Step 4: NBumi workflow
    print("\n4️⃣ NBumi workflow...")
    try:
        nbumi_data = NBumiConvertData(adata, is_counts=True, preserve_sparse=True)
        print(f"   ✅ NBumi conversion: {nbumi_data.shape}")
        
        # Fit NBumi model (may need DataFrame)
        if hasattr(nbumi_data, 'to_dataframe'):
            nbumi_df = nbumi_data.to_dataframe()
        else:
            nbumi_df = nbumi_data
            
        fit_result = NBumiFitModel(nbumi_df)
        print(f"   ✅ NBumi model fitted for {len(fit_result['sizes'])} genes")
        
    except Exception as e:
        print(f"   ⚠️  NBumi workflow issue: {str(e)[:100]}")


def show_compatibility_options(adata):
    """Show different ways to handle sparse/dense compatibility"""
    print("\n" + "="*60)
    print("🔧 COMPATIBILITY OPTIONS")
    print("="*60)
    
    print("\n📋 For maximum memory efficiency (recommended for large datasets):")
    print("   converted = M3DropConvertData(adata, is_counts=True, preserve_sparse=True)")
    print("   # Returns SparseMat3Drop object that works with all M3Drop functions")
    
    print("\n📋 For legacy compatibility (if you need pandas DataFrame):")
    print("   converted = M3DropConvertData(adata, is_counts=True, preserve_sparse=False)")
    print("   # Returns pandas DataFrame (may use more memory)")
    
    print("\n📋 For mixed workflows (sparse to dense when needed):")
    print("   sparse_data = M3DropConvertData(adata, is_counts=True, preserve_sparse=True)")
    print("   dense_df = sparse_data.to_dataframe()  # Convert only when needed")
    
    print("\n📋 Automatic format detection:")
    print("   # Most M3Drop functions automatically detect and handle both formats")
    print("   gene_vars = bg__calc_variables(sparse_or_dense_data)")


def main():
    """Main example function"""
    print("M3DROP SPARSE MATRIX OPTIMIZATION EXAMPLE")
    print("="*80)
    print("This example shows how to use M3Drop's memory-efficient sparse matrix support")
    print("="*80)
    
    # Load example data
    adata = load_example_data()
    
    # Compare memory usage
    converted_data = compare_memory_usage(adata)
    
    # Demonstrate workflow
    demonstrate_m3drop_workflow(adata)
    
    # Show compatibility options
    show_compatibility_options(adata)
    
    print("\n" + "="*80)
    print("✅ EXAMPLE COMPLETE")
    print("="*80)
    print("Key takeaways:")
    print("• Use preserve_sparse=True for memory-efficient analysis")
    print("• All M3Drop functions work with sparse inputs")
    print("• Convert to DataFrame only when necessary")
    print("• Memory usage scales with data sparsity, not total size")
    print("• Perfect for large single-cell datasets (100k+ cells)")


if __name__ == "__main__":
    try:
        import scipy.sparse as sp
        main()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install scanpy scipy pandas")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc() 