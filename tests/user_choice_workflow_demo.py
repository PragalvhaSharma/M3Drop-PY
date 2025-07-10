"""
M3Drop User Choice Workflow Demo
===============================

This file demonstrates the key requirement from the Instructions.md:
"Note that the user should be able to choose to perform just feature selection 
or just normalization or both."

This example shows three different workflows:
1. Normalization ONLY - Apply M3Drop normalization without feature selection
2. Feature Selection ONLY - Apply M3Drop feature selection without normalization
3. Both - Apply both M3Drop normalization AND feature selection

Each workflow can seamlessly integrate into the standard Scanpy pipeline.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import m3Drop as m3d

# Configure scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

def demo_normalization_only(adata, method='nbumi'):
    """
    OPTION 1: Apply M3Drop NORMALIZATION ONLY
    
    This replaces the standard scanpy normalization step but keeps
    standard scanpy feature selection.
    """
    print("\n" + "="*60)
    print("🔬 OPTION 1: M3Drop NORMALIZATION ONLY")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    adata_norm_only = adata.copy()
    
    # Basic filtering (standard scanpy)
    sc.pp.filter_cells(adata_norm_only, min_genes=200)
    sc.pp.filter_genes(adata_norm_only, min_cells=3)
    
    # Calculate QC metrics (standard scanpy)
    adata_norm_only.var['mt'] = adata_norm_only.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_norm_only, percent_top=None, log1p=False, inplace=True)
    
    # Store raw counts
    adata_norm_only.raw = adata_norm_only
    
    print(f"📊 Applying M3Drop {method} normalization...")
    
    if method == 'nbumi':
        # M3Drop NBumi normalization (REPLACES sc.pp.normalize_total)
        m3d.scanpy.nbumi_normalize(adata_norm_only)
        print("✅ Applied NBumi normalization with NB-UMI model")
        
    elif method == 'pearson_residuals':
        # M3Drop Pearson residuals normalization
        m3d.scanpy.nbumi_normalize(adata_norm_only, use_pearson_residuals=True)
        print("✅ Applied Pearson residuals normalization")
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Log transform (optional, can be skipped if using Pearson residuals)
    if method != 'pearson_residuals':
        sc.pp.log1p(adata_norm_only)
        print("✅ Applied log1p transformation")
    
    # STANDARD SCANPY FEATURE SELECTION (not M3Drop)
    print("📋 Using standard Scanpy feature selection...")
    sc.pp.highly_variable_genes(adata_norm_only, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Continue with standard pipeline
    adata_norm_only = adata_norm_only[:, adata_norm_only.var.highly_variable]
    sc.pp.scale(adata_norm_only, max_value=10)
    sc.tl.pca(adata_norm_only, svd_solver='arpack')
    sc.pp.neighbors(adata_norm_only, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata_norm_only)
    sc.tl.umap(adata_norm_only)
    
    print(f"\n📈 Results with M3Drop NORMALIZATION ONLY:")
    print(f"   • {adata_norm_only.n_obs} cells")
    print(f"   • {adata_norm_only.n_vars} highly variable genes (Scanpy method)")
    print(f"   • {len(adata_norm_only.obs['leiden'].unique())} clusters identified")
    print(f"   • Normalization: M3Drop {method}")
    print(f"   • Feature selection: Standard Scanpy")
    
    return adata_norm_only

def demo_feature_selection_only(adata, method='consensus'):
    """
    OPTION 2: Apply M3Drop FEATURE SELECTION ONLY
    
    This uses standard scanpy normalization but replaces the
    feature selection step with M3Drop methods.
    """
    print("\n" + "="*60)
    print("🧬 OPTION 2: M3Drop FEATURE SELECTION ONLY")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    adata_fs_only = adata.copy()
    
    # Basic filtering (standard scanpy)
    sc.pp.filter_cells(adata_fs_only, min_genes=200)
    sc.pp.filter_genes(adata_fs_only, min_cells=3)
    
    # Calculate QC metrics (standard scanpy)
    adata_fs_only.var['mt'] = adata_fs_only.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_fs_only, percent_top=None, log1p=False, inplace=True)
    
    # Store raw counts in a layer BEFORE normalization (M3Drop needs raw counts for feature selection)
    adata_fs_only.layers['counts'] = adata_fs_only.X.copy()
    
    # Store raw counts
    adata_fs_only.raw = adata_fs_only
    
    # STANDARD SCANPY NORMALIZATION (not M3Drop)
    print("📊 Using standard Scanpy normalization...")
    sc.pp.normalize_total(adata_fs_only, target_sum=1e4)
    sc.pp.log1p(adata_fs_only)
    print("✅ Applied standard total count normalization + log1p")
    
    # M3Drop FEATURE SELECTION (REPLACES sc.pp.highly_variable_genes)
    # Use the raw counts layer for M3Drop feature selection
    print(f"📋 Applying M3Drop {method} feature selection...")
    
    if method == 'consensus':
        # Consensus feature selection combining multiple M3Drop methods
        m3d.scanpy.m3drop_highly_variable_genes(adata_fs_only, method='consensus', ntop=2000, layer='counts')
        print("✅ Applied consensus feature selection (M3Drop + Brennecke + PCA + Gini)")
        
    elif method == 'm3drop':
        # Traditional M3Drop feature selection
        m3d.scanpy.m3drop_highly_variable_genes(adata_fs_only, method='m3drop', fdr_thresh=0.05, layer='counts')
        print("✅ Applied traditional M3Drop feature selection")
        
    elif method == 'danb':
        # DANB method for highly variable genes
        m3d.scanpy.m3drop_highly_variable_genes(adata_fs_only, method='danb', ntop=2000, layer='counts')
        print("✅ Applied DANB highly variable gene selection")
        
    elif method == 'combined_drop':
        # Combined dropout analysis
        m3d.scanpy.m3drop_highly_variable_genes(adata_fs_only, method='combined_drop', ntop=2000, layer='counts')
        print("✅ Applied combined dropout feature selection")
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Continue with standard pipeline
    adata_fs_only = adata_fs_only[:, adata_fs_only.var.highly_variable]
    sc.pp.scale(adata_fs_only, max_value=10)
    sc.tl.pca(adata_fs_only, svd_solver='arpack')
    sc.pp.neighbors(adata_fs_only, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata_fs_only)
    sc.tl.umap(adata_fs_only)
    
    print(f"\n📈 Results with M3Drop FEATURE SELECTION ONLY:")
    print(f"   • {adata_fs_only.n_obs} cells")
    print(f"   • {adata_fs_only.n_vars} highly variable genes (M3Drop {method})")
    print(f"   • {len(adata_fs_only.obs['leiden'].unique())} clusters identified")
    print(f"   • Normalization: Standard Scanpy")
    print(f"   • Feature selection: M3Drop {method}")
    
    return adata_fs_only

def demo_both_normalization_and_feature_selection(adata, norm_method='nbumi', fs_method='consensus'):
    """
    OPTION 3: Apply BOTH M3Drop NORMALIZATION AND FEATURE SELECTION
    
    This replaces both the normalization and feature selection steps
    with M3Drop methods for a fully M3Drop-powered pipeline.
    """
    print("\n" + "="*60)
    print("🚀 OPTION 3: M3Drop NORMALIZATION + FEATURE SELECTION")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    adata_both = adata.copy()
    
    # Basic filtering (standard scanpy)
    sc.pp.filter_cells(adata_both, min_genes=200)
    sc.pp.filter_genes(adata_both, min_cells=3)
    
    # Calculate QC metrics (standard scanpy)
    adata_both.var['mt'] = adata_both.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_both, percent_top=None, log1p=False, inplace=True)
    
    # Store raw counts in a layer BEFORE normalization (for feature selection)
    adata_both.layers['counts'] = adata_both.X.copy()
    
    # Store raw counts
    adata_both.raw = adata_both
    
    # M3Drop NORMALIZATION
    print(f"📊 Applying M3Drop {norm_method} normalization...")
    
    if norm_method == 'nbumi':
        m3d.scanpy.nbumi_normalize(adata_both)
        print("✅ Applied NBumi normalization with NB-UMI model")
        
    elif norm_method == 'pearson_residuals':
        m3d.scanpy.nbumi_normalize(adata_both, use_pearson_residuals=True)
        print("✅ Applied Pearson residuals normalization")
        
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")
    
    # Log transform (optional, skip if using Pearson residuals)
    if norm_method != 'pearson_residuals':
        sc.pp.log1p(adata_both)
        print("✅ Applied log1p transformation")
    
    # M3Drop FEATURE SELECTION
    # Use the raw counts layer for M3Drop feature selection
    print(f"📋 Applying M3Drop {fs_method} feature selection...")
    
    if fs_method == 'consensus':
        m3d.scanpy.m3drop_highly_variable_genes(adata_both, method='consensus', ntop=2000, layer='counts')
        print("✅ Applied consensus feature selection")
        
    elif fs_method == 'm3drop':
        m3d.scanpy.m3drop_highly_variable_genes(adata_both, method='m3drop', fdr_thresh=0.05, layer='counts')
        print("✅ Applied traditional M3Drop feature selection")
        
    elif fs_method == 'danb':
        m3d.scanpy.m3drop_highly_variable_genes(adata_both, method='danb', ntop=2000, layer='counts')
        print("✅ Applied DANB feature selection")
        
    elif fs_method == 'combined_drop':
        m3d.scanpy.m3drop_highly_variable_genes(adata_both, method='combined_drop', ntop=2000, layer='counts')
        print("✅ Applied combined dropout feature selection")
        
    else:
        raise ValueError(f"Unknown feature selection method: {fs_method}")
    
    # Continue with standard pipeline
    adata_both = adata_both[:, adata_both.var.highly_variable]
    sc.pp.scale(adata_both, max_value=10)
    sc.tl.pca(adata_both, svd_solver='arpack')
    sc.pp.neighbors(adata_both, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata_both)
    sc.tl.umap(adata_both)
    
    print(f"\n📈 Results with BOTH M3Drop NORMALIZATION + FEATURE SELECTION:")
    print(f"   • {adata_both.n_obs} cells")
    print(f"   • {adata_both.n_vars} highly variable genes (M3Drop {fs_method})")
    print(f"   • {len(adata_both.obs['leiden'].unique())} clusters identified")
    print(f"   • Normalization: M3Drop {norm_method}")
    print(f"   • Feature selection: M3Drop {fs_method}")
    
    return adata_both

def demo_user_choice_workflow(data_path=None, user_choice='both', use_subset=True, subset_size=(2000, 5000)):
    """
    Main demonstration function showing user choice workflow.
    
    Parameters
    ----------
    data_path : str, optional
        Path to h5ad file. If None, uses the GSM8267529_G-P28_raw_matrix.h5ad file.
    user_choice : str, default='both'
        User's choice: 'normalization', 'feature_selection', or 'both'
    use_subset : bool, default=True
        Whether to use a subset of the data for faster demonstration.
    subset_size : tuple, default=(2000, 5000)
        Size of subset as (n_cells, n_genes) if use_subset=True.
    """
    print("🔬 M3Drop User Choice Workflow Demonstration")
    print("=" * 60)
    print("This demo shows how users can choose to apply:")
    print("1. Just normalization (keeping standard Scanpy feature selection)")
    print("2. Just feature selection (keeping standard Scanpy normalization)")  
    print("3. Both normalization AND feature selection (full M3Drop pipeline)")
    print("=" * 60)
    
    # Load data
    if data_path is None:
        # Use the specific GSM8267529 dataset
        data_path = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/newM3dUpdate/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
        print(f"📁 Loading GSM8267529_G-P28_raw_matrix dataset...")
        adata = sc.read_h5ad(data_path)
        print(f"✅ Loaded GSM8267529 dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    else:
        print(f"📁 Loading data from: {data_path}")
        adata = sc.read_h5ad(data_path)
        print(f"✅ Loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Use subset for faster demonstration if requested
    if use_subset:
        n_cells, n_genes = subset_size
        n_cells = min(n_cells, adata.n_obs)
        n_genes = min(n_genes, adata.n_vars)
        adata = adata[:n_cells, :n_genes].copy()
        print(f"🔬 Using subset for demo: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Execute based on user choice
    if user_choice == 'normalization':
        print(f"\n🎯 User chose: NORMALIZATION ONLY")
        result = demo_normalization_only(adata, method='nbumi')
        
    elif user_choice == 'feature_selection':
        print(f"\n🎯 User chose: FEATURE SELECTION ONLY")
        result = demo_feature_selection_only(adata, method='consensus')
        
    elif user_choice == 'both':
        print(f"\n🎯 User chose: BOTH NORMALIZATION AND FEATURE SELECTION")
        result = demo_both_normalization_and_feature_selection(adata, 
                                                             norm_method='nbumi', 
                                                             fs_method='consensus')
    else:
        raise ValueError(f"Unknown user choice: {user_choice}. Must be 'normalization', 'feature_selection', or 'both'")
    
    print(f"\n✨ Workflow completed successfully!")
    return result

def compare_all_approaches(data_path=None, use_subset=True, subset_size=(2000, 5000)):
    """
    Compare all three approaches side-by-side to show the differences.
    
    Parameters
    ----------
    data_path : str, optional
        Path to h5ad file. If None, uses the GSM8267529_G-P28_raw_matrix.h5ad file.
    use_subset : bool, default=True
        Whether to use a subset of the data for faster demonstration.
    subset_size : tuple, default=(2000, 5000)
        Size of subset as (n_cells, n_genes) if use_subset=True.
    """
    print("\n" + "🔍 COMPARISON OF ALL THREE APPROACHES" + "\n")
    print("=" * 80)
    
    # Load data
    if data_path is None:
        # Use the specific GSM8267529 dataset
        data_path = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/newM3dUpdate/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
        adata = sc.read_h5ad(data_path)
    else:
        adata = sc.read_h5ad(data_path)
    
    # Use subset for faster demonstration if requested
    if use_subset:
        n_cells, n_genes = subset_size
        n_cells = min(n_cells, adata.n_obs)
        n_genes = min(n_genes, adata.n_vars)
        adata = adata[:n_cells, :n_genes].copy()
        print(f"🔬 Using subset for comparison: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Run all three approaches
    print("🏃‍♂️ Running all three approaches for comparison...")
    
    result_norm_only = demo_normalization_only(adata.copy(), method='nbumi')
    result_fs_only = demo_feature_selection_only(adata.copy(), method='consensus')  
    result_both = demo_both_normalization_and_feature_selection(adata.copy(), 
                                                               norm_method='nbumi',
                                                               fs_method='consensus')
    
    # Summary comparison
    print("\n" + "📊 SUMMARY COMPARISON" + "\n")
    print("=" * 80)
    print(f"{'Approach':<25} {'Cells':<8} {'HVGs':<8} {'Clusters':<10} {'Normalization':<15} {'Feature Selection'}")
    print("-" * 80)
    print(f"{'Normalization Only':<25} {result_norm_only.n_obs:<8} {result_norm_only.n_vars:<8} "
          f"{len(result_norm_only.obs['leiden'].unique()):<10} {'M3Drop NBumi':<15} {'Scanpy'}")
    print(f"{'Feature Selection Only':<25} {result_fs_only.n_obs:<8} {result_fs_only.n_vars:<8} "
          f"{len(result_fs_only.obs['leiden'].unique()):<10} {'Scanpy':<15} {'M3Drop Consensus'}")
    print(f"{'Both (Full M3Drop)':<25} {result_both.n_obs:<8} {result_both.n_vars:<8} "
          f"{len(result_both.obs['leiden'].unique()):<10} {'M3Drop NBumi':<15} {'M3Drop Consensus'}")
    
    return {
        'normalization_only': result_norm_only,
        'feature_selection_only': result_fs_only,
        'both': result_both
    }

if __name__ == "__main__":
    """
    Example usage of the user choice workflow.
    """
    
    # Example 1: User wants normalization only
    print("🔬 EXAMPLE 1: User Choice = Normalization Only")
    result1 = demo_user_choice_workflow(user_choice='normalization')
    
    # Example 2: User wants feature selection only  
    print("\n\n🧬 EXAMPLE 2: User Choice = Feature Selection Only")
    result2 = demo_user_choice_workflow(user_choice='feature_selection')
    
    # Example 3: User wants both
    print("\n\n🚀 EXAMPLE 3: User Choice = Both")
    result3 = demo_user_choice_workflow(user_choice='both')
    
    # Example 4: Compare all approaches
    print("\n\n🔍 EXAMPLE 4: Comparison of All Approaches")
    comparison_results = compare_all_approaches()
    
    print(f"\n✅ All examples completed successfully!")
    print(f"📝 This demonstrates the key requirement:")
    print(f"   'User should be able to choose to perform just feature selection")
    print(f"    or just normalization or both.'") 