import sys
import os

# Add the parent directory to the path to import m3Drop
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Disable ALL matplotlib plotting before importing anything else
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

import scanpy as sc
import pandas as pd
import numpy as np
from m3Drop import *

def test_nbumi_functions_no_plots():
    """Test all NBumi functions with NO plotting whatsoever"""
    
    print("="*60)
    print("NBUMI FUNCTIONS TEST - NO PLOTS")
    print("="*60)
    
    # Test results dictionary
    results = {}
    
    # Step 1: Load the data using relative path
    data_dir = os.path.join(parent_dir, "data")
    h5ad_file = os.path.join(data_dir, "GSM8267529_G-P28_raw_matrix.h5ad")
    
    # Check if data file exists
    if not os.path.exists(h5ad_file):
        print(f"✗ Data file not found: {h5ad_file}")
        print("Please ensure the data file exists in the correct location.")
        results["Data Loading"] = "FAILED - FILE NOT FOUND"
        return results
    
    try:
        adata = sc.read_h5ad(h5ad_file)
        print(f"✓ Data loaded: {adata.shape}")
        results["Data Loading"] = "PASSED"
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        results["Data Loading"] = "FAILED"
        return results
    
    # Step 2: Prepare small subset for fast testing
    try:
        # Take a very small subset for speed
        adata_small = adata[:100, :100].copy()  # 100 cells, 100 genes
        
        # Basic filtering
        sc.pp.filter_cells(adata_small, min_genes=1)
        sc.pp.filter_genes(adata_small, min_cells=1)
        
        # Get counts matrix
        counts = adata_small.to_df().T
        print(f"✓ Test data prepared: {counts.shape}")
        results["Data Preparation"] = "PASSED"
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        results["Data Preparation"] = "FAILED"
        return results
    
    # Test each function quickly
    test_functions = [
        ("NBumiConvertData", lambda: NBumiConvertData(counts, is_counts=True)),
        ("NBumiConvertToInteger", lambda: NBumiConvertToInteger(counts)),
        ("NBumiFitModel", lambda: NBumiFitModel(counts.astype(int))),
        ("NBumiFitBasicModel", lambda: NBumiFitBasicModel(counts.astype(int))),
    ]
    
    # Test basic functions first
    converted_data = counts
    fit_model = None
    
    for func_name, func_call in test_functions:
        try:
            print(f"Testing {func_name}...")
            result = func_call()
            print(f"✓ {func_name} PASSED")
            results[func_name] = "PASSED"
            
            if func_name == "NBumiConvertData":
                converted_data = result
            elif func_name == "NBumiFitModel" and result is not None:
                fit_model = result
        except Exception as e:
            print(f"✗ {func_name} FAILED: {str(e)[:100]}...")
            results[func_name] = "FAILED"
    
    # Test functions that need fit model
    if fit_model is not None:
        fit_dependent_tests = [
            ("NBumiFitDispVsMean", lambda: NBumiFitDispVsMean(fit_model, suppress_plot=True)),
            ("NBumiCheckFit", lambda: NBumiCheckFit(converted_data.astype(int), fit_model, suppress_plot=True)),
            ("NBumiCheckFitFS", lambda: NBumiCheckFitFS(converted_data.astype(int), fit_model, suppress_plot=True)),
            ("NBumiFeatureSelectionHighVar", lambda: NBumiFeatureSelectionHighVar(fit_model)),
            ("NBumiFeatureSelectionCombinedDrop", lambda: NBumiFeatureSelectionCombinedDrop(fit_model, suppress_plot=True)),
        ]
        
        for func_name, func_call in fit_dependent_tests:
            try:
                print(f"Testing {func_name}...")
                result = func_call()
                print(f"✓ {func_name} PASSED")
                results[func_name] = "PASSED"
            except Exception as e:
                print(f"✗ {func_name} FAILED: {str(e)[:100]}...")
                results[func_name] = "FAILED"
        
        # Test more complex functions with simplified parameters
        try:
            print("Testing NBumiPearsonResidualsApprox...")
            residuals = NBumiPearsonResidualsApprox(converted_data.astype(int), fit_model)
            print(f"✓ NBumiPearsonResidualsApprox PASSED")
            results["NBumiPearsonResidualsApprox"] = "PASSED"
        except Exception as e:
            print(f"✗ NBumiPearsonResidualsApprox FAILED: {str(e)[:100]}...")
            results["NBumiPearsonResidualsApprox"] = "FAILED"
        
        try:
            print("Testing NBumiHVG...")
            # Convert to int but keep as DataFrame for NBumiHVG
            if isinstance(converted_data, pd.DataFrame):
                hvg_data = converted_data.astype(int)
            else:
                hvg_data = pd.DataFrame(converted_data.astype(int))
            hvg = NBumiHVG(hvg_data, fit_model, suppress_plot=True)
            print(f"✓ NBumiHVG PASSED")
            results["NBumiHVG"] = "PASSED"
        except Exception as e:
            print(f"✗ NBumiHVG FAILED: {str(e)[:100]}...")
            results["NBumiHVG"] = "FAILED"
    
    else:
        print("⚠️  No fit model available - skipping fit-dependent tests")
        for func_name in ["NBumiFitDispVsMean", "NBumiCheckFit", "NBumiCheckFitFS", 
                         "NBumiFeatureSelectionHighVar", "NBumiFeatureSelectionCombinedDrop",
                         "NBumiPearsonResidualsApprox", "NBumiHVG"]:
            results[func_name] = "SKIPPED"
    
    # Test standalone functions
    try:
        print("Testing NBumiCompareModels...")
        compare = NBumiCompareModels(converted_data.astype(int))
        print(f"✓ NBumiCompareModels PASSED")
        results["NBumiCompareModels"] = "PASSED"
    except Exception as e:
        print(f"✗ NBumiCompareModels FAILED: {str(e)[:100]}...")
        results["NBumiCompareModels"] = "FAILED"
    
    # Functions that are known to have issues - test but expect potential failures
    problematic_functions = [
        ("NBumiImputeNorm", lambda: NBumiImputeNorm(converted_data.astype(int), fit_model) if fit_model else None),
        ("NBumiPearsonResiduals", lambda: NBumiPearsonResiduals(converted_data.astype(int), fit_model) if fit_model else None),
    ]
    
    for func_name, func_call in problematic_functions:
        if fit_model is not None:
            try:
                print(f"Testing {func_name}...")
                result = func_call()
                if result is not None:
                    print(f"✓ {func_name} PASSED")
                    results[func_name] = "PASSED"
                else:
                    print(f"⚠️ {func_name} RETURNED NONE")
                    results[func_name] = "WARNING"
            except Exception as e:
                print(f"✗ {func_name} FAILED: {str(e)[:100]}...")
                results[func_name] = "FAILED"
        else:
            results[func_name] = "SKIPPED"
    
    # Make sure to close any plots that might have been created
    plt.close('all')
    
    return results

def print_summary(results):
    """Print a summary of all test results"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = failed = skipped = warning = 0
    
    for test_name, result in results.items():
        if result == "PASSED":
            symbol = "✓"
            passed += 1
        elif result == "FAILED":
            symbol = "✗"
            failed += 1
        elif result == "WARNING":
            symbol = "⚠️"
            warning += 1
        else:
            symbol = "⚠️"
            skipped += 1
        
        print(f"{symbol} {test_name}: {result}")
    
    print("\n" + "-"*40)
    print(f"TOTAL TESTS: {len(results)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"WARNINGS: {warning}")
    print(f"SKIPPED: {skipped}")
    print("-"*40)
    
    if failed == 0:
        print("🎉 All critical tests passed!")
    else:
        print(f"❌ {failed} tests failed - check function implementations")

if __name__ == "__main__":
    test_results = test_nbumi_functions_no_plots()
    print_summary(test_results) 