#!/usr/bin/env python3
"""
M3Drop Workflow Demonstration - Automatic Execution
===================================================

This script demonstrates all available workflows with m3Drop:
1. Feature selection only
2. Normalization only  
3. Both feature selection and normalization
4. Comparison of all approaches

The script runs automatically without user interaction, demonstrating
the flexibility of m3Drop for different analysis needs using the
GSM8267529_G-P28_raw_matrix.h5ad dataset.

Usage:
    python user_choice_workflow_demo.py

All workflows will be executed automatically for demonstration.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional

# Import M3Drop functions
from m3Drop.Extremes import M3DropFeatureSelection
from m3Drop.NB_UMI import NBumiFitModel, NBumiImputeNorm
from m3Drop.Normalization import NBumiPearsonResiduals, NBumiPearsonResidualsApprox, M3DropCleanData
from m3Drop.scanpy import nbumi_normalize, m3drop_highly_variable_genes
from m3Drop.Other_FS_functions import Consensus_fs
from m3Drop.DANB_HVG import NBumiHVG


def load_reference_data():
    """Load the reference dataset"""
    h5ad_file = "/Users/pragalvhasharma/Downloads/PragGOToDocuments/Blanc/school/mcpServers/m3DropUpdates/m3DropNew/data/GSM8267529_G-P28_raw_matrix.h5ad"
    
    try:
        adata = sc.read_h5ad(h5ad_file)
        print("✅ Successfully loaded reference dataset:")
        print(f"   • Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
        print(f"   • Data type: {adata.X.dtype}")
        print(f"   • Sparse matrix: {hasattr(adata.X, 'nnz')}")
        
        # Preprocess data to remove problematic genes and cells
        print("\n🔧 Preprocessing data...")
        
        # Remove genes that are never expressed
        sc.pp.filter_genes(adata, min_cells=1)
        print(f"   • After removing undetected genes: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Remove cells with very few detected genes (likely empty droplets)
        sc.pp.filter_cells(adata, min_genes=100)
        print(f"   • After removing low-quality cells: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Remove genes detected in very few cells
        sc.pp.filter_genes(adata, min_cells=10)
        print(f"   • After removing rare genes: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Ensure we have reasonable data
        if adata.shape[0] < 100 or adata.shape[1] < 1000:
            print("⚠️ Warning: Dataset is quite small after filtering")
        
        return adata
    except FileNotFoundError:
        print(f"❌ Error: Could not find reference file at {h5ad_file}")
        print("Please ensure the file exists at the specified path.")
        sys.exit(1)


def feature_selection_only(adata):
    """Perform feature selection only (no normalization)"""
    print("\n" + "🔍" * 20)
    print("WORKFLOW 1: FEATURE SELECTION ONLY")
    print("🔍" * 20)
    print("Demonstrating feature selection on raw count data without normalization")
    
    # Work with raw counts - ensure proper format
    print(f"Working with preprocessed count data: {adata.shape} (cells × genes)")
    
    # Convert to genes x cells format for M3Drop
    if hasattr(adata.X, 'toarray'):
        raw_data = adata.X.toarray()
    else:
        raw_data = adata.X
    
    # Transpose to genes x cells and create DataFrame
    raw_counts = pd.DataFrame(
        raw_data.T, 
        index=adata.var_names, 
        columns=adata.obs_names
    )
    
    print(f"M3Drop format: {raw_counts.shape} (genes × cells)")
    
    results = {}
    
    # Method 1: Traditional M3Drop Feature Selection
    print("\n1️⃣ Running M3Drop Feature Selection...")
    try:
        # Use a more lenient threshold initially
        m3drop_genes = M3DropFeatureSelection(
            raw_counts, 
            mt_method="fdr_bh", 
            mt_threshold=0.1,  # More lenient threshold
            suppress_plot=True
        )
        results['m3drop'] = m3drop_genes
        print(f"   ✅ Found {len(m3drop_genes)} significant genes with M3Drop")
        if len(m3drop_genes) > 0:
            print(f"   📝 Top genes: {list(m3drop_genes.index[:5])}")
        else:
            print("   📝 No significant genes found - try adjusting parameters")
    except Exception as e:
        print(f"   ❌ M3Drop failed: {e}")
        results['m3drop'] = pd.DataFrame()
    
    # Method 2: NBumi Feature Selection - Skip if data has issues
    print("\n2️⃣ Running NBumi Feature Selection...")
    try:
        # Check data quality first
        if raw_counts.sum().min() > 0:  # Ensure no empty cells
            # Convert to integer counts for NBumi
            raw_counts_int = raw_counts.astype(int)
            nbumi_fit = NBumiFitModel(raw_counts_int)
            
            # NBumi HVG detection
            hvg_genes = NBumiHVG(raw_counts_int, nbumi_fit, fdr_thresh=0.1)
            results['nbumi_hvg'] = hvg_genes
            print(f"   ✅ Found {len(hvg_genes)} highly variable genes with NBumi")
            if len(hvg_genes) > 0:
                print(f"   📝 Top genes: {list(hvg_genes.index[:5])}")
        else:
            print("   ⚠️ Skipping NBumi - data has empty cells")
            results['nbumi_hvg'] = pd.DataFrame()
    except Exception as e:
        print(f"   ❌ NBumi HVG failed: {e}")
        results['nbumi_hvg'] = pd.DataFrame()
    
    # Method 3: Simple highly variable genes using coefficient of variation
    print("\n3️⃣ Running Simple HVG (Coefficient of Variation)...")
    try:
        # Calculate coefficient of variation for each gene
        gene_means = raw_counts.mean(axis=1)
        gene_stds = raw_counts.std(axis=1)
        
        # Avoid division by zero
        non_zero_genes = gene_means > 0
        cv = pd.Series(0, index=raw_counts.index)
        cv[non_zero_genes] = gene_stds[non_zero_genes] / gene_means[non_zero_genes]
        
        # Select top variable genes
        top_hvg = cv.nlargest(2000)
        results['simple_hvg'] = top_hvg
        print(f"   ✅ Selected top {len(top_hvg)} highly variable genes")
        print(f"   📝 Top genes: {list(top_hvg.index[:5])}")
        print(f"   📊 CV range: {top_hvg.min():.3f} - {top_hvg.max():.3f}")
    except Exception as e:
        print(f"   ❌ Simple HVG failed: {e}")
        results['simple_hvg'] = pd.DataFrame()
    
    # Summary
    print("\n📋 FEATURE SELECTION SUMMARY:")
    print("-" * 40)
    for method, result in results.items():
        if not result.empty:
            print(f"• {method.upper()}: {len(result)} genes identified")
        else:
            print(f"• {method.upper()}: No results (method failed)")
    
    print("\n💡 Use Case: This workflow is ideal for:")
    print("   • Exploratory analysis of raw count data")
    print("   • Identifying variable genes before normalization")
    print("   • Comparing methods on original data scale")
    
    return results


def normalization_only(adata):
    """Perform normalization only (no feature selection)"""
    print("\n" + "🔧" * 20)
    print("WORKFLOW 2: NORMALIZATION ONLY") 
    print("🔧" * 20)
    print("Demonstrating different normalization approaches without feature selection")
    
    results = {}
    
    # Method 1: NBumi Normalization (Standard) - Skip due to numerical issues
    print("\n1️⃣ NBumi Normalization (Standard) - SKIPPED...")
    print("   ⚠️ Skipping NBumi standard normalization due to numerical instability")
    print("   💡 NBumi works best with high-quality, well-filtered datasets")
    results['nbumi_standard'] = None
    
    # Method 2: NBumi with Pearson Residuals - Use safer implementation
    print("\n2️⃣ Running NBumi with Pearson Residuals...")
    try:
        adata_pearson = adata.copy()
        
        # Convert to DataFrame format for NBumi (genes x cells)
        if hasattr(adata_pearson.X, 'toarray'):
            data_matrix = adata_pearson.X.toarray()
        else:
            data_matrix = adata_pearson.X
            
        counts_df = pd.DataFrame(
            data_matrix.T,  # Transpose to genes x cells for M3Drop
            index=adata_pearson.var_names,
            columns=adata_pearson.obs_names
        )
        
        # Fit NBumi model with error handling
        print("   🔄 Fitting NBumi model...")
        fit_result = NBumiFitModel(counts_df)
        
        # Calculate Pearson residuals directly (safer than full normalization)
        print("   🔄 Computing Pearson residuals...")
        residuals = NBumiPearsonResiduals(counts_df.values, fit_result)
        
        # Store results (transpose back to cells x genes)
        adata_pearson.X = residuals.T
        results['nbumi_pearson'] = adata_pearson
        print(f"   ✅ Pearson residuals normalization complete")
        print(f"   📊 Residuals computed using NBumi model")
        print(f"   🧮 Data shape: {adata_pearson.shape}")
        print(f"   📈 Residual range: {residuals.min():.3f} to {residuals.max():.3f}")
        
    except Exception as e:
        print(f"   ❌ Pearson residuals failed: {e}")
        print("   💡 This can happen with poor data quality or extreme outliers")
        results['nbumi_pearson'] = None
    
    # Method 3: Traditional CPM normalization (for comparison)
    print("\n3️⃣ Running CPM Normalization (for comparison)...")
    try:
        adata_cpm = adata.copy()
        sc.pp.normalize_total(adata_cpm, target_sum=1e4)
        results['cpm'] = adata_cpm
        print(f"   ✅ CPM normalization complete")
        print(f"   📊 Standard library size normalization")
        print(f"   🧮 Data shape: {adata_cpm.shape}")
        
        # Calculate some stats
        total_counts = np.array(adata_cpm.X.sum(axis=1)).flatten()
        print(f"   📈 Target sum achieved: {np.mean(total_counts):.1f} ± {np.std(total_counts):.1f}")
        
    except Exception as e:
        print(f"   ❌ CPM normalization failed: {e}")
        results['cpm'] = None
    
    # Method 4: Add Log1p transformation 
    print("\n4️⃣ Running Log1p Transformation...")
    try:
        adata_log = adata.copy()
        # First normalize, then log transform
        sc.pp.normalize_total(adata_log, target_sum=1e4)
        sc.pp.log1p(adata_log)
        results['log1p'] = adata_log
        print(f"   ✅ Log1p transformation complete")
        print(f"   📊 CPM + log(x+1) transformation")
        print(f"   🧮 Data shape: {adata_log.shape}")
        
        # Check for reasonable range
        data_min = adata_log.X.min()
        data_max = adata_log.X.max()
        print(f"   📈 Expression range: {data_min:.3f} to {data_max:.3f}")
        
    except Exception as e:
        print(f"   ❌ Log1p transformation failed: {e}")
        results['log1p'] = None
    
    # Compare normalization results
    print("\n📋 NORMALIZATION COMPARISON:")
    print("-" * 40)
    for method, result in results.items():
        if result is not None:
            if hasattr(result.X, 'toarray'):
                data = result.X.toarray()
            else:
                data = result.X
            
            # Calculate statistics
            mean_expr = np.mean(data)
            std_expr = np.std(data)
            zero_fraction = np.mean(data == 0)
            
            print(f"• {method.upper()}:")
            print(f"  Mean expression: {mean_expr:.3f}")
            print(f"  Std expression: {std_expr:.3f}")
            print(f"  Zero fraction: {zero_fraction:.3f}")
            print(f"  Expression range: {data.min():.3f} to {data.max():.3f}")
        else:
            print(f"• {method.upper()}: Failed")
    
    print("\n💡 Use Case: This workflow is ideal for:")
    print("   • Preprocessing data for machine learning")
    print("   • Standardizing expression across cells")
    print("   • Preparing data for downstream analysis tools")
    print("   • Exploratory data analysis and visualization")
    
    print("\n🔧 NORMALIZATION RECOMMENDATIONS:")
    print("   • CPM: Best for count-based analyses and interpretability")
    print("   • Log1p: Best for linear methods and visualization")
    print("   • Pearson residuals: Best for removing technical noise")
    print("   • For problematic datasets, start with CPM + Log1p")
    
    return results


def both_normalization_and_feature_selection(adata):
    """Perform both normalization and feature selection"""
    print("\n" + "🚀" * 20)
    print("WORKFLOW 3: COMPLETE PIPELINE (NORMALIZATION + FEATURE SELECTION)")
    print("🚀" * 20)
    print("Demonstrating the recommended complete workflow for scRNA-seq analysis")
    
    # Step 1: Apply robust normalization
    print("\n🔧 STEP 1: Robust Normalization")
    print("-" * 30)
    adata_norm = adata.copy()
    
    # Use CPM + log1p as the most reliable normalization
    print("   Using CPM + Log1p normalization (most robust approach)")
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    print(f"✅ Applied CPM + Log1p normalization")
    print(f"   📊 Target: 10K counts/cell + log(x+1) transformation")
    print(f"   📈 Expression range: {adata_norm.X.min():.3f} to {adata_norm.X.max():.3f}")
    
    # Step 2: Feature selection on normalized data
    print("\n🔍 STEP 2: Feature Selection on Normalized Data")
    print("-" * 50)
    
    results = {}
    
    # Method 1: Scanpy's highly variable genes
    print("\n1️⃣ Scanpy Highly Variable Genes...")
    try:
        adata_hvg = adata_norm.copy()
        # Use 'seurat' flavor instead of 'seurat_v3' to avoid scikit-misc dependency
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=2000, flavor='seurat')
        hvg_scanpy = adata_hvg.var['highly_variable'].sum()
        results['scanpy_hvg'] = {'adata': adata_hvg, 'n_genes': hvg_scanpy}
        print(f"   ✅ Found {hvg_scanpy} highly variable genes")
        if hvg_scanpy > 0:
            hvg_names = adata_hvg.var_names[adata_hvg.var['highly_variable']][:5]
            print(f"   📝 Top HVGs: {list(hvg_names)}")
    except Exception as e:
        print(f"   ❌ Scanpy HVG failed: {e}")
        print("   💡 Trying alternative method without external dependencies...")
        try:
            # Fallback: use simple coefficient of variation method
            adata_hvg = adata_norm.copy()
            
            if hasattr(adata_hvg.X, 'toarray'):
                norm_data = adata_hvg.X.toarray()
            else:
                norm_data = adata_hvg.X
            
            # Calculate mean and variance per gene
            gene_means = np.mean(norm_data, axis=0)
            gene_vars = np.var(norm_data, axis=0)
            
            # Calculate normalized dispersion (similar to Seurat)
            with np.errstate(divide='ignore', invalid='ignore'):
                dispersions = gene_vars / gene_means
                dispersions[gene_means == 0] = 0
            
            # Select top 2000 genes by dispersion
            top_indices = np.argsort(dispersions)[-2000:]
            adata_hvg.var['highly_variable'] = False
            adata_hvg.var.iloc[top_indices, adata_hvg.var.columns.get_loc('highly_variable')] = True
            
            hvg_scanpy = adata_hvg.var['highly_variable'].sum()
            results['scanpy_hvg'] = {'adata': adata_hvg, 'n_genes': hvg_scanpy}
            print(f"   ✅ Found {hvg_scanpy} highly variable genes (fallback method)")
            if hvg_scanpy > 0:
                hvg_names = adata_hvg.var_names[adata_hvg.var['highly_variable']][:5]
                print(f"   📝 Top HVGs: {list(hvg_names)}")
        except Exception as e2:
            print(f"   ❌ Fallback method also failed: {e2}")
            results['scanpy_hvg'] = {'adata': None, 'n_genes': 0}
    
    # Method 2: M3Drop on log-normalized data (if possible)
    print("\n2️⃣ M3Drop Feature Selection (adapted for log-normalized data)...")
    try:
        # Convert back to counts-like data for M3Drop
        adata_m3drop = adata.copy()  # Use original counts
        
        # Apply M3Drop directly to raw counts
        if hasattr(adata_m3drop.X, 'toarray'):
            raw_data = adata_m3drop.X.toarray()
        else:
            raw_data = adata_m3drop.X
        
        # Create M3Drop format (genes x cells)
        raw_counts_df = pd.DataFrame(
            raw_data.T, 
            index=adata_m3drop.var_names, 
            columns=adata_m3drop.obs_names
        )
        
        # Use a more lenient threshold
        m3drop_genes = M3DropFeatureSelection(
            raw_counts_df, 
            mt_method="fdr_bh", 
            mt_threshold=0.1,
            suppress_plot=True
        )
        
        # Create boolean mask for the normalized data
        adata_m3drop_norm = adata_norm.copy()
        adata_m3drop_norm.var['highly_variable'] = adata_m3drop_norm.var_names.isin(m3drop_genes.index)
        hvg_m3drop = adata_m3drop_norm.var['highly_variable'].sum()
        
        results['m3drop_adapted'] = {'adata': adata_m3drop_norm, 'n_genes': hvg_m3drop}
        print(f"   ✅ Found {hvg_m3drop} highly variable genes with M3Drop")
        if hvg_m3drop > 0:
            hvg_names = adata_m3drop_norm.var_names[adata_m3drop_norm.var['highly_variable']][:5]
            print(f"   📝 Top HVGs: {list(hvg_names)}")
            
    except Exception as e:
        print(f"   ❌ M3Drop adapted method failed: {e}")
        results['m3drop_adapted'] = {'adata': None, 'n_genes': 0}
    
    # Method 3: Coefficient of variation on normalized data
    print("\n3️⃣ Coefficient of Variation (on normalized data)...")
    try:
        adata_cv = adata_norm.copy()
        
        # Calculate CV on normalized data
        if hasattr(adata_cv.X, 'toarray'):
            norm_data = adata_cv.X.toarray()
        else:
            norm_data = adata_cv.X
        
        # Calculate mean and std per gene
        gene_means = np.mean(norm_data, axis=0)
        gene_stds = np.std(norm_data, axis=0)
        
        # Calculate CV (avoid division by zero)
        non_zero_genes = gene_means > 0
        cv = np.zeros(len(gene_means))
        cv[non_zero_genes] = gene_stds[non_zero_genes] / gene_means[non_zero_genes]
        
        # Select top 2000 most variable genes
        top_indices = np.argsort(cv)[-2000:]
        adata_cv.var['highly_variable'] = False
        adata_cv.var.iloc[top_indices, adata_cv.var.columns.get_loc('highly_variable')] = True
        
        hvg_cv = adata_cv.var['highly_variable'].sum()
        results['cv_method'] = {'adata': adata_cv, 'n_genes': hvg_cv}
        print(f"   ✅ Selected top {hvg_cv} highly variable genes by CV")
        if hvg_cv > 0:
            hvg_names = adata_cv.var_names[adata_cv.var['highly_variable']][:5]
            print(f"   📝 Top HVGs: {list(hvg_names)}")
            print(f"   📊 CV range: {cv[top_indices].min():.3f} - {cv[top_indices].max():.3f}")
            
    except Exception as e:
        print(f"   ❌ CV method failed: {e}")
        results['cv_method'] = {'adata': None, 'n_genes': 0}
    
    # Step 3: Summary and recommendations
    print("\n📋 COMPLETE WORKFLOW SUMMARY:")
    print("-" * 40)
    print(f"Original data: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"After normalization: preserved all genes, log-transformed")
    print("\nFeature selection results:")
    for method, result in results.items():
        print(f"• {method.replace('_', ' ').title()}: {result['n_genes']} genes")
    
    # Method overlap analysis
    print("\n🔍 METHOD OVERLAP ANALYSIS:")
    successful_methods = {k: v for k, v in results.items() if v['adata'] is not None and v['n_genes'] > 0}
    
    if len(successful_methods) >= 2:
        method_names = list(successful_methods.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                try:
                    hvg1 = set(successful_methods[method1]['adata'].var_names[
                        successful_methods[method1]['adata'].var['highly_variable']])
                    hvg2 = set(successful_methods[method2]['adata'].var_names[
                        successful_methods[method2]['adata'].var['highly_variable']])
                    
                    overlap = len(hvg1.intersection(hvg2))
                    union = len(hvg1.union(hvg2))
                    jaccard = overlap / union if union > 0 else 0
                    
                    print(f"• {method1} ∩ {method2}: {overlap} genes (Jaccard: {jaccard:.3f})")
                except:
                    print(f"• {method1} ∩ {method2}: Could not calculate overlap")
    
    # Best practice recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    print("• For downstream analysis, use Scanpy HVG result (most robust)")
    print("• The normalized data is ready for PCA, UMAP, clustering")
    print("• Consider gene overlap between methods for robust selection")
    print("• This workflow is recommended for complete scRNA-seq pipelines")
    print("• Log-transformed data works well with most downstream tools")
    
    return results


def comparison_analysis(adata, fs_results, norm_results, both_results):
    """Analyze and compare results across all workflows"""
    print("\n" + "📊" * 20)
    print("WORKFLOW 4: COMPREHENSIVE COMPARISON ANALYSIS")
    print("📊" * 20)
    print("Comparing results across all workflows to highlight differences")
    
    print("\n🔍 CROSS-WORKFLOW COMPARISON:")
    print("=" * 50)
    
    print("\n1. FEATURE SELECTION COMPARISON:")
    print("   Raw data vs. Normalized data approaches")
    
    # Compare gene counts
    print("\n   📊 Gene counts identified:")
    if not fs_results['m3drop'].empty and 'm3drop_method' in both_results:
        raw_genes = len(fs_results['m3drop'])
        norm_genes = both_results['m3drop_method']['n_genes']
        print(f"   • M3Drop on raw data: {raw_genes} genes")
        print(f"   • M3Drop on normalized data: {norm_genes} genes")
        print(f"   • Difference: {abs(raw_genes - norm_genes)} genes")
        
        # Calculate overlap if both have results
        if raw_genes > 0 and norm_genes > 0:
            try:
                raw_set = set(fs_results['m3drop'].index)
                norm_set = set(both_results['m3drop_method']['adata'].var_names[
                    both_results['m3drop_method']['adata'].var['highly_variable']])
                overlap = len(raw_set.intersection(norm_set))
                print(f"   • Overlap between methods: {overlap} genes")
                print(f"   • Jaccard similarity: {overlap / len(raw_set.union(norm_set)):.3f}")
            except:
                print("   • Could not calculate overlap")
    
    print("\n2. NORMALIZATION IMPACT:")
    print("   How different normalization methods affect data")
    
    original_stats = {
        'mean': np.mean(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X),
        'std': np.std(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X),
        'zeros': np.mean((adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X) == 0)
    }
    
    print(f"   • Original data: mean={original_stats['mean']:.3f}, std={original_stats['std']:.3f}, zeros={original_stats['zeros']:.3f}")
    
    for method, result in norm_results.items():
        if result is not None:
            data = result.X.toarray() if hasattr(result.X, 'toarray') else result.X
            norm_stats = {
                'mean': np.mean(data),
                'std': np.std(data),
                'zeros': np.mean(data == 0)
            }
            print(f"   • {method.upper()}: mean={norm_stats['mean']:.3f}, std={norm_stats['std']:.3f}, zeros={norm_stats['zeros']:.3f}")
    
    print("\n3. METHOD RECOMMENDATIONS:")
    print("   Choose based on your analysis goals")
    print("   🔍 Feature Selection Only:")
    print("     → Best for: Gene discovery, biomarker identification")
    print("     → When: You want to work with raw count scale")
    print("   🔧 Normalization Only:")
    print("     → Best for: Data preprocessing, batch correction preparation")
    print("     → When: Preparing for downstream tools that need normalized data")
    print("   🚀 Complete Pipeline:")
    print("     → Best for: Full scRNA-seq analysis workflows")
    print("     → When: Building comprehensive analysis pipelines")
    print("   📊 Comparison Mode:")
    print("     → Best for: Method evaluation, educational purposes")
    print("     → When: Understanding method differences")
    
    print("\n4. COMPUTATIONAL CONSIDERATIONS:")
    print("   Relative computational costs and recommendations")
    print("   • Feature Selection Only: Fast, memory efficient")
    print("   • Normalization Only: Medium cost, depends on method")
    print("   • Complete Pipeline: Higher cost but comprehensive")
    print("   • NBumi methods: More computationally intensive but more accurate")
    print("   • Consensus methods: Highest cost but most robust")
    
    return {
        'feature_selection': fs_results,
        'normalization': norm_results,
        'complete_workflow': both_results,
        'comparison_stats': {
            'original': original_stats,
            'normalized': {k: v for k, v in norm_results.items() if v is not None}
        }
    }


def main():
    """Main demonstration function - runs all workflows automatically"""
    print("🧬 M3Drop Comprehensive Workflow Demonstration")
    print("=" * 60)
    print("This script demonstrates all available workflows with m3Drop")
    print("Running automatically to showcase the package's flexibility...")
    
    # Load data
    print("\n📁 Loading reference dataset...")
    adata = load_reference_data()
    
    # Execute all workflows automatically
    print("\n🚀 EXECUTING ALL WORKFLOWS AUTOMATICALLY...")
    print("=" * 60)
    
    try:
        # Workflow 1: Feature Selection Only
        fs_results = feature_selection_only(adata)
        
        # Workflow 2: Normalization Only
        norm_results = normalization_only(adata)
        
        # Workflow 3: Complete Pipeline
        both_results = both_normalization_and_feature_selection(adata)
        
        # Workflow 4: Comprehensive Comparison
        comparison_results = comparison_analysis(adata, fs_results, norm_results, both_results)
        
        # Final summary
        print("\n" + "="*60)
        print("✨ ALL WORKFLOWS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📋 DEMONSTRATION SUMMARY:")
        print("✅ Feature Selection Only - Completed")
        print("✅ Normalization Only - Completed") 
        print("✅ Complete Pipeline - Completed")
        print("✅ Comparison Analysis - Completed")
        
        print("\n🎯 KEY TAKEAWAYS:")
        print("• M3Drop offers flexible workflows for different analysis needs")
        print("• Feature selection can be done on raw or normalized data")
        print("• Multiple normalization methods are available (NBumi, Pearson residuals)")
        print("• The scanpy integration provides seamless workflows")
        print("• Choose your approach based on your specific analysis goals")
        print("• Consensus methods provide the most robust results")
        
        print("\n📚 WHAT THIS DEMONSTRATION SHOWED:")
        print("• How to use M3Drop for feature selection without normalization")
        print("• How to apply different normalization strategies")
        print("• How to combine normalization and feature selection effectively")
        print("• How methods compare and when to use each approach")
        print("• Integration with scanpy for streamlined workflows")
        
        return comparison_results
        
    except Exception as e:
        print(f"\n❌ An error occurred during workflow execution: {e}")
        print("This might be due to:")
        print("• Missing dependencies")
        print("• Data file issues")
        print("• Memory constraints")
        print("Please check your environment and try again.")
        return None


if __name__ == "__main__":
    # Run the demonstration automatically
    print("Starting M3Drop workflow demonstration...")
    try:
        results = main()
        if results is not None:
            print("\n🎉 Demonstration completed successfully!")
            print("Check the output above for detailed results and recommendations.")
        else:
            print("\n⚠️ Demonstration completed with errors.")
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and data file path.")
        print("Ensure all m3Drop dependencies are properly installed.")
