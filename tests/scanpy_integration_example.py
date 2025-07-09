import sys
import os
import unittest
import warnings

# Add the parent directory to the path to import m3Drop
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Disable matplotlib plotting for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData

# Import M3Drop scanpy functions
from m3Drop.scanpy import (
    fit_nbumi_model,
    nbumi_normalize,
    m3drop_highly_variable_genes,
    nbumi_impute,
    _ensure_raw_counts
)

class TestM3DropScanpyIntegration(unittest.TestCase):
    """Test suite for M3Drop scanpy integration functions"""
    
    def setUp(self):
        """Set up test data for each test"""
        # Suppress warnings during testing
        warnings.filterwarnings('ignore')
        
        # Create synthetic count data that resembles real scRNA-seq data
        np.random.seed(42)
        n_cells = 100
        n_genes = 200
        
        # Generate count matrix with realistic properties
        # Some genes should be highly expressed, others lowly expressed
        base_expression = np.random.exponential(scale=0.5, size=(n_genes, n_cells))
        
        # Add some structure - some genes correlated, some independent
        for i in range(0, n_genes, 10):
            # Make some genes correlated
            if i + 5 < n_genes:
                base_expression[i:i+5] *= np.random.exponential(scale=2, size=(1, n_cells))
        
        # Add technical dropout (zeros)
        dropout_prob = np.random.beta(2, 8, size=(n_genes, n_cells))
        mask = np.random.random((n_genes, n_cells)) < dropout_prob
        base_expression[mask] = 0
        
        # Convert to integer counts and ensure we have some variation
        counts = np.random.poisson(base_expression * 10).astype(int)
        
        # Ensure at least some counts in each gene and cell
        counts = np.maximum(counts, np.random.poisson(0.1, size=counts.shape))
        
        # Create gene and cell names
        gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]
        cell_names = [f"Cell_{i:03d}" for i in range(n_cells)]
        
        # Create AnnData object
        self.adata = AnnData(
            X=counts.T,  # AnnData expects cells x genes
            obs=pd.DataFrame(index=cell_names),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Also create a sparse version for testing
        self.adata_sparse = self.adata.copy()
        self.adata_sparse.X = sp.csr_matrix(self.adata_sparse.X)
        
        # Create normalized data for negative testing
        self.adata_normalized = self.adata.copy()
        sc.pp.normalize_total(self.adata_normalized, target_sum=1e4)
        sc.pp.log1p(self.adata_normalized)
        
        # Create data with layers
        self.adata_layers = self.adata.copy()
        self.adata_layers.layers['counts'] = self.adata_layers.X.copy()
        self.adata_layers.layers['normalized'] = self.adata_normalized.X.copy()
    
    def test_ensure_raw_counts_basic(self):
        """Test _ensure_raw_counts function with basic input"""
        # Test with dense data
        counts = _ensure_raw_counts(self.adata)
        self.assertEqual(counts.shape, (self.adata.n_vars, self.adata.n_obs))
        self.assertTrue(np.all(counts >= 0))
        self.assertTrue(np.allclose(counts, np.round(counts)))
        
        # Test with sparse data
        counts_sparse = _ensure_raw_counts(self.adata_sparse)
        self.assertEqual(counts_sparse.shape, (self.adata_sparse.n_vars, self.adata_sparse.n_obs))
        self.assertTrue(np.all(counts_sparse >= 0))
    
    def test_ensure_raw_counts_with_layers(self):
        """Test _ensure_raw_counts with different layers"""
        # Test with counts layer
        counts = _ensure_raw_counts(self.adata_layers, layer='counts')
        self.assertEqual(counts.shape, (self.adata_layers.n_vars, self.adata_layers.n_obs))
        
        # Test with non-existent layer
        with self.assertRaises(ValueError):
            _ensure_raw_counts(self.adata_layers, layer='nonexistent')
    
    def test_ensure_raw_counts_normalized_data(self):
        """Test _ensure_raw_counts rejects normalized data"""
        with self.assertRaises(ValueError):
            _ensure_raw_counts(self.adata_normalized)
    
    def test_ensure_raw_counts_negative_values(self):
        """Test _ensure_raw_counts rejects negative values"""
        adata_negative = self.adata.copy()
        adata_negative.X[0, 0] = -1
        
        with self.assertRaises(ValueError):
            _ensure_raw_counts(adata_negative)
    
    def test_fit_nbumi_model_basic(self):
        """Test fit_nbumi_model basic functionality"""
        # Test without copy
        result = fit_nbumi_model(self.adata, key_added='test_fit')
        self.assertIsNone(result)
        self.assertIn('test_fit', self.adata.uns)
        self.assertIn('test_fit_size', self.adata.var.columns)
        self.assertIn('test_fit_mean', self.adata.var.columns)
        
        # Check that fit results are reasonable
        fit_result = self.adata.uns['test_fit']
        self.assertIn('sizes', fit_result)
        self.assertIn('vals', fit_result)
        self.assertEqual(len(fit_result['sizes']), self.adata.n_vars)
    
    def test_fit_nbumi_model_copy(self):
        """Test fit_nbumi_model with copy=True"""
        adata_copy = fit_nbumi_model(self.adata, copy=True)
        self.assertIsNotNone(adata_copy)
        self.assertIn('nbumi_fit', adata_copy.uns)
        self.assertNotIn('nbumi_fit', self.adata.uns)  # Original should be unchanged
    
    def test_fit_nbumi_model_with_layer(self):
        """Test fit_nbumi_model with specific layer"""
        result = fit_nbumi_model(self.adata_layers, layer='counts')
        self.assertIsNone(result)
        self.assertIn('nbumi_fit', self.adata_layers.uns)
    
    def test_nbumi_normalize_basic(self):
        """Test nbumi_normalize basic functionality"""
        original_X = self.adata.X.copy()
        
        result = nbumi_normalize(self.adata)
        self.assertIsNone(result)
        
        # Check that data was normalized (should be different from original)
        self.assertFalse(np.allclose(self.adata.X, original_X))
        
        # Check that raw data was stored
        self.assertIsNotNone(self.adata.raw)
        
        # Check that fit results were stored
        self.assertIn('nbumi_fit', self.adata.uns)
        self.assertIn('nbumi_size', self.adata.var.columns)
        self.assertIn('nbumi_mean', self.adata.var.columns)
    
    def test_nbumi_normalize_copy(self):
        """Test nbumi_normalize with copy=True"""
        original_X = self.adata.X.copy()
        
        adata_copy = nbumi_normalize(self.adata, copy=True)
        self.assertIsNotNone(adata_copy)
        
        # Original should be unchanged
        self.assertTrue(np.allclose(self.adata.X, original_X))
        
        # Copy should be normalized
        self.assertFalse(np.allclose(adata_copy.X, original_X))
        self.assertIn('nbumi_fit', adata_copy.uns)
    
    def test_nbumi_normalize_pearson_residuals(self):
        """Test nbumi_normalize with Pearson residuals"""
        result = nbumi_normalize(self.adata, use_pearson_residuals=True)
        self.assertIsNone(result)
        
        # Check that we get residuals (can be negative)
        self.assertTrue(np.any(self.adata.X < 0))
        self.assertIn('nbumi_fit', self.adata.uns)
    
    def test_nbumi_normalize_target_sum(self):
        """Test nbumi_normalize with specific target sum"""
        target_sum = 5000
        result = nbumi_normalize(self.adata, target_sum=target_sum)
        self.assertIsNone(result)
        
        # Check that the normalization used the target sum appropriately
        # (exact check depends on implementation details)
        self.assertIn('nbumi_fit', self.adata.uns)
    
    def test_m3drop_highly_variable_genes_consensus(self):
        """Test m3drop_highly_variable_genes with consensus method"""
        result = m3drop_highly_variable_genes(self.adata, method='consensus', ntop=50)
        self.assertIsNone(result)
        
        # Check that HVG annotation was added
        self.assertIn('highly_variable', self.adata.var.columns)
        self.assertIn('m3drop_consensus_rank', self.adata.var.columns)
        
        # Check that we got the right number of HVGs
        n_hvg = np.sum(self.adata.var['highly_variable'])
        self.assertLessEqual(n_hvg, 50)  # Should be <= ntop
        self.assertGreater(n_hvg, 0)     # Should find some HVGs
    
    def test_m3drop_highly_variable_genes_danb(self):
        """Test m3drop_highly_variable_genes with DANB method"""
        result = m3drop_highly_variable_genes(self.adata, method='danb', ntop=30)
        self.assertIsNone(result)
        
        # Check that HVG annotation and DANB-specific columns were added
        self.assertIn('highly_variable', self.adata.var.columns)
        self.assertIn('m3drop_danb_effect_size', self.adata.var.columns)
        self.assertIn('m3drop_danb_pvalue', self.adata.var.columns)
        self.assertIn('m3drop_danb_qvalue', self.adata.var.columns)
        self.assertIn('nbumi_fit', self.adata.uns)
    
    def test_m3drop_highly_variable_genes_combined_drop(self):
        """Test m3drop_highly_variable_genes with combined_drop method"""
        result = m3drop_highly_variable_genes(self.adata, method='combined_drop', ntop=40)
        self.assertIsNone(result)
        
        # Check that HVG annotation and combined_drop-specific columns were added
        self.assertIn('highly_variable', self.adata.var.columns)
        self.assertIn('m3drop_combined_effect_size', self.adata.var.columns)
        self.assertIn('m3drop_combined_pvalue', self.adata.var.columns)
        self.assertIn('m3drop_combined_qvalue', self.adata.var.columns)
        self.assertIn('nbumi_fit', self.adata.uns)
    
    def test_m3drop_highly_variable_genes_m3drop(self):
        """Test m3drop_highly_variable_genes with traditional M3Drop method"""
        result = m3drop_highly_variable_genes(self.adata, method='m3drop', fdr_thresh=0.1)
        self.assertIsNone(result)
        
        # Check that HVG annotation and M3Drop-specific columns were added
        self.assertIn('highly_variable', self.adata.var.columns)
        self.assertIn('m3drop_effect_size', self.adata.var.columns)
        self.assertIn('m3drop_pvalue', self.adata.var.columns)
        self.assertIn('m3drop_qvalue', self.adata.var.columns)
    
    def test_m3drop_highly_variable_genes_invalid_method(self):
        """Test m3drop_highly_variable_genes with invalid method"""
        with self.assertRaises(ValueError):
            m3drop_highly_variable_genes(self.adata, method='invalid_method')
    
    def test_m3drop_highly_variable_genes_copy(self):
        """Test m3drop_highly_variable_genes with copy=True"""
        original_vars = self.adata.var.columns.tolist()
        
        adata_copy = m3drop_highly_variable_genes(self.adata, method='consensus', copy=True)
        self.assertIsNotNone(adata_copy)
        
        # Original should be unchanged
        self.assertEqual(self.adata.var.columns.tolist(), original_vars)
        
        # Copy should have HVG annotations
        self.assertIn('highly_variable', adata_copy.var.columns)
    
    def test_nbumi_impute_basic(self):
        """Test nbumi_impute basic functionality"""
        original_X = self.adata.X.copy()
        
        result = nbumi_impute(self.adata)
        self.assertIsNone(result)
        
        # Check that data was imputed (should be different from original)
        self.assertFalse(np.allclose(self.adata.X, original_X))
        
        # Check that raw data was stored
        self.assertIsNotNone(self.adata.raw)
        
        # Check that fit results were stored
        self.assertIn('nbumi_fit', self.adata.uns)
        self.assertIn('nbumi_size', self.adata.var.columns)
        self.assertIn('nbumi_mean', self.adata.var.columns)
    
    def test_nbumi_impute_copy(self):
        """Test nbumi_impute with copy=True"""
        original_X = self.adata.X.copy()
        
        adata_copy = nbumi_impute(self.adata, copy=True)
        self.assertIsNotNone(adata_copy)
        
        # Original should be unchanged
        self.assertTrue(np.allclose(self.adata.X, original_X))
        
        # Copy should be imputed
        self.assertFalse(np.allclose(adata_copy.X, original_X))
        self.assertIn('nbumi_fit', adata_copy.uns)
    
    def test_nbumi_impute_target_sum(self):
        """Test nbumi_impute with specific target sum"""
        target_sum = 8000
        result = nbumi_impute(self.adata, target_sum=target_sum)
        self.assertIsNone(result)
        
        # Check that the imputation used the target sum appropriately
        self.assertIn('nbumi_fit', self.adata.uns)
    
    def test_nbumi_impute_with_layer(self):
        """Test nbumi_impute with specific layer"""
        result = nbumi_impute(self.adata_layers, layer='counts')
        self.assertIsNone(result)
        self.assertIn('nbumi_fit', self.adata_layers.uns)
    
    def test_sparse_data_handling(self):
        """Test that all functions handle sparse data correctly"""
        # Test with sparse data
        adata_sparse = self.adata_sparse.copy()
        
        # These functions should work with sparse data
        fit_nbumi_model(adata_sparse)
        self.assertIn('nbumi_fit', adata_sparse.uns)
        
        nbumi_normalize(adata_sparse)
        self.assertIsNotNone(adata_sparse.raw)
        
        m3drop_highly_variable_genes(adata_sparse, method='consensus', ntop=20)
        self.assertIn('highly_variable', adata_sparse.var.columns)
        
        # Reset for imputation test
        adata_sparse = self.adata_sparse.copy()
        nbumi_impute(adata_sparse)
        self.assertIn('nbumi_fit', adata_sparse.uns)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with very small dataset
        adata_small = self.adata[:10, :20].copy()
        
        # Should still work but might have warnings
        fit_nbumi_model(adata_small)
        self.assertIn('nbumi_fit', adata_small.uns)
        
        # Test with data that has many zeros
        adata_sparse_counts = self.adata.copy()
        adata_sparse_counts.X[adata_sparse_counts.X < 2] = 0
        
        fit_nbumi_model(adata_sparse_counts)
        self.assertIn('nbumi_fit', adata_sparse_counts.uns)
    
    def test_integration_workflow(self):
        """Test a complete integration workflow"""
        # Start with fresh data
        adata = self.adata.copy()
        
        # Step 1: Fit model
        fit_nbumi_model(adata)
        self.assertIn('nbumi_fit', adata.uns)
        
        # Step 2: Find highly variable genes
        m3drop_highly_variable_genes(adata, method='consensus', ntop=50)
        self.assertIn('highly_variable', adata.var.columns)
        
        # Step 3: Normalize
        nbumi_normalize(adata)
        self.assertIsNotNone(adata.raw)
        
        # Check that all steps completed successfully
        self.assertTrue(np.any(adata.var['highly_variable']))
        self.assertIn('nbumi_fit', adata.uns)
        self.assertIn('m3drop_consensus_rank', adata.var.columns)
    
    def tearDown(self):
        """Clean up after tests"""
        plt.close('all')


def run_comprehensive_tests():
    """Run all tests and provide detailed output"""
    print("="*70)
    print("M3DROP SCANPY INTEGRATION COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestM3DropScanpyIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the comprehensive tests
    success = run_comprehensive_tests()
    
    print("\n" + "="*70)
    if success:
        print("🎉 ALL TESTS PASSED! M3Drop scanpy integration is working correctly.")
    else:
        print("❌ SOME TESTS FAILED. Please check the output above for details.")
    print("="*70)
    