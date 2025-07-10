import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scanpy as sc
import pandas as pd
from m3Drop.Extremes import M3DropFeatureSelection


def test_m3drop_feature_selection_comprehensive():
    """
    Comprehensive test of M3DropFeatureSelection with both raw and normalized data.
    This demonstrates the key differences between using raw counts vs normalized data.
    """
    
    # Step 1: Load your AnnData (.h5ad) file
    h5ad_file = "data/GSM8267529_G-P28_raw_matrix.h5ad"
    adata = sc.read_h5ad(h5ad_file)
    print("=" * 80)
    print("AnnData object loaded successfully:")
    print(adata)
    print("=" * 80)
    
    # ==========================================
    # TEST 1: M3DropFeatureSelection with RAW COUNTS
    # ==========================================
    print("\n🧬 TEST 1: M3DropFeatureSelection with RAW COUNTS")
    print("-" * 50)
    
    # M3Drop functions traditionally use raw counts
    # Transpose to have genes as rows and cells as columns
    raw_counts = adata.to_df().T
    
    if not isinstance(raw_counts, pd.DataFrame):
        raw_counts = pd.DataFrame(raw_counts, index=adata.var_names, columns=adata.obs_names)
    
    print("Running M3DropFeatureSelection with raw counts...")
    print(f"Data shape: {raw_counts.shape} (genes x cells)")
    print(f"Parameters: mt_threshold=0.05 (default method)")
    
    # Run with raw counts and more lenient threshold
    raw_results = M3DropFeatureSelection(raw_counts, mt_threshold=0.05)
    
    print(f"\nResults with RAW COUNTS:")
    print(f"Number of significant genes: {len(raw_results)}")
    print(f"Columns in results: {list(raw_results.columns)}")
    print(f"Result DataFrame shape: {raw_results.shape}")
    
    if not raw_results.empty:
        print(f"First few rows of results:")
        print(raw_results.head())
        
        # Use the actual column name for sorting (likely 'p.value' or similar)
        if 'p.value' in raw_results.columns:
            top_raw = raw_results.nsmallest(5, 'p.value')  # smallest p-values
            print(f"\nTop 5 genes by significance (smallest p-values):")
            for idx, row in top_raw.iterrows():
                print(f"  {idx}: p-value = {row['p.value']:.2e}")
        else:
            # Just show first 5 rows if we don't know the column structure
            print(f"\nFirst 5 genes in results:")
            for idx, row in raw_results.head().iterrows():
                print(f"  {idx}: {dict(row)}")
    
    # ==========================================
    # TEST 2: M3DropFeatureSelection with NORMALIZED DATA
    # ==========================================
    print("\n🔬 TEST 2: M3DropFeatureSelection with NORMALIZED DATA")
    print("-" * 50)
    
    # Create a copy for normalization to avoid modifying original
    adata_norm = adata.copy()
    
    # Normalize the data (standard scRNA-seq preprocessing)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    print("Applied normalization: total counts per cell = 10,000")
    
    normalized_matrix = adata_norm.to_df().T
    
    print(f"Data shape: {normalized_matrix.shape} (genes x cells)")
    print(f"Parameters: mt_method='fdr_bh', mt_threshold=0.01")
    
    # Run with normalized data and stricter threshold + FDR correction
    norm_results = M3DropFeatureSelection(
        normalized_matrix, 
        mt_method="fdr_bh", 
        mt_threshold=0.01
    )
    
    print(f"\nResults with NORMALIZED DATA:")
    print(f"Number of significant genes: {len(norm_results)}")
    print(f"Columns in results: {list(norm_results.columns)}")
    print(f"Result DataFrame shape: {norm_results.shape}")
    
    if not norm_results.empty:
        print(f"First few rows of results:")
        print(norm_results.head())
        
        # Use the actual column name for sorting
        if 'p.value' in norm_results.columns:
            top_norm = norm_results.nsmallest(5, 'p.value')  # smallest p-values
            print(f"\nTop 5 genes by significance (smallest p-values):")
            for idx, row in top_norm.iterrows():
                print(f"  {idx}: p-value = {row['p.value']:.2e}")
        else:
            # Just show first 5 rows if we don't know the column structure
            print(f"\nFirst 5 genes in results:")
            for idx, row in norm_results.head().iterrows():
                print(f"  {idx}: {dict(row)}")
    
    # ==========================================
    # COMPARISON AND EXPLANATION
    # ==========================================
    print("\n📊 COMPARISON AND KEY DIFFERENCES")
    print("=" * 80)
    
    print(f"Raw counts approach:")
    print(f"  • Genes found: {len(raw_results)}")
    print(f"  • Uses original count data")
    print(f"  • Threshold: 0.05 (more lenient)")
    print(f"  • Multiple testing: default method")
    
    print(f"\nNormalized approach:")
    print(f"  • Genes found: {len(norm_results)}")
    print(f"  • Uses normalized data (10K reads per cell)")
    print(f"  • Threshold: 0.01 (more stringent)")
    print(f"  • Multiple testing: FDR (Benjamini-Hochberg)")
    
    # Find overlapping genes
    if not raw_results.empty and not norm_results.empty:
        raw_genes = set(raw_results.index)
        norm_genes = set(norm_results.index)
        overlap = raw_genes & norm_genes
        
        print(f"\nGene overlap:")
        print(f"  • Genes in both methods: {len(overlap)}")
        print(f"  • Raw-only genes: {len(raw_genes - norm_genes)}")
        print(f"  • Normalized-only genes: {len(norm_genes - raw_genes)}")
        print(f"  • Overlap percentage: {len(overlap)/max(len(raw_genes), len(norm_genes))*100:.1f}%")
        
        # Show some examples of overlapping vs unique genes
        if len(overlap) > 0:
            print(f"\nSample overlapping genes: {list(overlap)[:5]}")
        if len(raw_genes - norm_genes) > 0:
            print(f"Sample raw-only genes: {list(raw_genes - norm_genes)[:5]}")
        if len(norm_genes - raw_genes) > 0:
            print(f"Sample norm-only genes: {list(norm_genes - raw_genes)[:5]}")
    
    print(f"\n💡 BIOLOGICAL INTERPRETATION:")
    print(f"• Raw counts preserve absolute expression differences")
    print(f"• Normalization removes cell-size effects but may alter gene rankings")
    print(f"• Different significant genes suggest normalization affects variability patterns")
    print(f"• FDR correction (normalized) is more conservative than default method")
    print(f"• The number of genes found differs significantly between approaches")
    
    # ==========================================
    # ASSERTIONS FOR TESTING
    # ==========================================
    print(f"\n✅ RUNNING ASSERTIONS...")
    
    # Basic checks for both results
    assert isinstance(raw_results, pd.DataFrame), "Raw results should be DataFrame"
    assert isinstance(norm_results, pd.DataFrame), "Normalized results should be DataFrame"
    
    # The raw counts test allows empty results (as in original)
    print("✓ Raw counts test: DataFrame type check passed")
    
    # The normalized test requires non-empty results (as in original)
    assert not norm_results.empty, "Normalized results should not be empty"
    print("✓ Normalized test: Non-empty DataFrame check passed")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Both M3DropFeatureSelection approaches completed successfully.")
    
    return raw_results, norm_results


if __name__ == "__main__":
    # Run the comprehensive test
    raw_genes, norm_genes = test_m3drop_feature_selection_comprehensive() 