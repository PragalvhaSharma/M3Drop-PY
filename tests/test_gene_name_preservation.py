#!/usr/bin/env python3
"""
Test script to demonstrate gene name preservation in M3Drop pipeline.

This script tests that gene names (feature_name in adata.var) are preserved
throughout the M3Drop data conversion and processing pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc

# Add the M3Drop package to the path
sys.path.insert(0, os.path.abspath('.'))

from m3Drop.basics import M3DropConvertData
from m3Drop.NB_UMI import NBumiConvertData, NBumiFitModel, NBumiImputeNorm
from m3Drop.Normalization import NBumiPearsonResiduals, NBumiPearsonResidualsApprox

def test_gene_name_preservation():
    """Test that gene names are preserved throughout the M3Drop pipeline."""
    
    print("=" * 60)
    print("TESTING GENE NAME PRESERVATION IN M3DROP PIPELINE")
    print("=" * 60)
    
    # Create test data with known gene names
    n_genes = 100
    n_cells = 50
    gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    cell_names = [f"Cell_{i:03d}" for i in range(n_cells)]
    
    # Generate synthetic count data
    np.random.seed(42)
    counts = np.random.poisson(5, size=(n_genes, n_cells))
    
    # Create DataFrame with gene names
    counts_df = pd.DataFrame(counts, index=gene_names, columns=cell_names)
    print(f"✓ Created test data: {counts_df.shape}")
    print(f"  Gene names: {gene_names[:5]}...")
    print(f"  Cell names: {cell_names[:5]}...")
    
    # Test 1: M3DropConvertData preserves names
    print("\n1. Testing M3DropConvertData...")
    converted_m3drop = M3DropConvertData(counts_df, is_counts=True)
    print(f"  Input shape: {counts_df.shape}")
    print(f"  Output shape: {converted_m3drop.shape}")
    print(f"  Output type: {type(converted_m3drop)}")
    if isinstance(converted_m3drop, pd.DataFrame):
        print(f"  ✓ Gene names preserved: {list(converted_m3drop.index[:5])}")
        print(f"  ✓ Cell names preserved: {list(converted_m3drop.columns[:5])}")
    else:
        print("  ✗ Gene names lost - output is not DataFrame")
    
    # Test 2: NBumiConvertData preserves names
    print("\n2. Testing NBumiConvertData...")
    converted_nbumi = NBumiConvertData(counts_df, is_counts=True)
    print(f"  Input shape: {counts_df.shape}")
    print(f"  Output shape: {converted_nbumi.shape}")
    print(f"  Output type: {type(converted_nbumi)}")
    if isinstance(converted_nbumi, pd.DataFrame):
        print(f"  ✓ Gene names preserved: {list(converted_nbumi.index[:5])}")
        print(f"  ✓ Cell names preserved: {list(converted_nbumi.columns[:5])}")
    else:
        print("  ✗ Gene names lost - output is not DataFrame")
    
    # Test 3: NBumiFitModel with DataFrame input
    print("\n3. Testing NBumiFitModel...")
    if isinstance(converted_nbumi, pd.DataFrame):
        fit_result = NBumiFitModel(converted_nbumi)
        print(f"  ✓ Fit successful with DataFrame input")
        print(f"  ✓ Fit contains {len(fit_result['sizes'])} genes")
    else:
        print("  ⚠ Using numpy array for fit (gene names not available)")
        fit_result = NBumiFitModel(converted_nbumi)
    
    # Test 4: NBumiPearsonResiduals preserves names
    print("\n4. Testing NBumiPearsonResiduals...")
    if isinstance(converted_nbumi, pd.DataFrame):
        residuals = NBumiPearsonResiduals(converted_nbumi, fit_result)
        print(f"  Input shape: {converted_nbumi.shape}")
        print(f"  Output shape: {residuals.shape}")
        print(f"  Output type: {type(residuals)}")
        if isinstance(residuals, pd.DataFrame):
            print(f"  ✓ Gene names preserved: {list(residuals.index[:5])}")
            print(f"  ✓ Cell names preserved: {list(residuals.columns[:5])}")
            print(f"  ✓ All original genes present: {set(gene_names).issubset(set(residuals.index))}")
        else:
            print("  ✗ Gene names lost - output is not DataFrame")
    else:
        print("  ⚠ Skipping - input is not DataFrame")
    
    # Test 5: NBumiImputeNorm preserves names
    print("\n5. Testing NBumiImputeNorm...")
    if isinstance(converted_nbumi, pd.DataFrame):
        imputed = NBumiImputeNorm(converted_nbumi, fit_result)
        print(f"  Input shape: {converted_nbumi.shape}")
        print(f"  Output shape: {imputed.shape}")
        print(f"  Output type: {type(imputed)}")
        if isinstance(imputed, pd.DataFrame):
            print(f"  ✓ Gene names preserved: {list(imputed.index[:5])}")
            print(f"  ✓ Cell names preserved: {list(imputed.columns[:5])}")
            print(f"  ✓ All original genes present: {set(gene_names).issubset(set(imputed.index))}")
        else:
            print("  ✗ Gene names lost - output is not DataFrame")
    else:
        print("  ⚠ Skipping - input is not DataFrame")
    
    # Test 6: Test with AnnData object
    print("\n6. Testing with AnnData object...")
    # Create AnnData object
    adata = sc.AnnData(X=counts.T)  # AnnData expects cells x genes
    adata.var_names = gene_names
    adata.obs_names = cell_names
    print(f"  Created AnnData: {adata.shape}")
    print(f"  Gene names in var: {list(adata.var_names[:5])}")
    
    # Test conversion from AnnData
    converted_from_adata = NBumiConvertData(adata, is_counts=True)
    print(f"  Converted shape: {converted_from_adata.shape}")
    print(f"  Converted type: {type(converted_from_adata)}")
    if isinstance(converted_from_adata, pd.DataFrame):
        print(f"  ✓ Gene names preserved: {list(converted_from_adata.index[:5])}")
        print(f"  ✓ Cell names preserved: {list(converted_from_adata.columns[:5])}")
    else:
        print("  ✗ Gene names lost - output is not DataFrame")
    
    print("\n" + "=" * 60)
    print("GENE NAME PRESERVATION TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    if (isinstance(converted_m3drop, pd.DataFrame) and 
        isinstance(converted_nbumi, pd.DataFrame) and
        isinstance(residuals, pd.DataFrame) and 
        isinstance(imputed, pd.DataFrame)):
        print("✅ SUCCESS: All functions preserve gene names!")
        return True
    else:
        print("❌ FAILURE: Some functions still lose gene names")
        return False

if __name__ == "__main__":
    success = test_gene_name_preservation()
    sys.exit(0 if success else 1) 