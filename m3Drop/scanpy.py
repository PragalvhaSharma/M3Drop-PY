"""
M3Drop + Scanpy Integration
===========================

This module provides seamless integration between M3Drop and Scanpy for single-cell RNA-seq analysis.
It wraps M3Drop's NBumi normalization and feature selection methods to work with AnnData objects
and follow scanpy conventions.

Key Functions:
- nbumi_normalize: Apply M3Drop's NBumi normalization
- m3drop_highly_variable_genes: Find highly variable genes using M3Drop methods
- nbumi_impute: Impute and normalize using NBumi methods
- fit_nbumi_model: Fit negative binomial UMI model

Usage:
    import scanpy as sc
    import m3Drop as m3d
    
    # Standard scanpy workflow with M3Drop
    adata = sc.read_10x_mtx('data.mtx')
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Replace standard normalization with M3Drop
    m3d.scanpy.nbumi_normalize(adata)
    
    # Replace standard HVG selection with M3Drop
    m3d.scanpy.m3drop_highly_variable_genes(adata, ntop=2000)
    
    # Continue with standard scanpy workflow
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    # ... etc
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Optional, Union, Sequence, Literal

# Import the core M3Drop functions
from .NB_UMI import (
    NBumiFitModel, NBumiFitBasicModel, NBumiConvertData, NBumiImputeNorm,
    NBumiFeatureSelectionHighVar, NBumiFeatureSelectionCombinedDrop,
    NBumiConvertToInteger, hidden_calc_vals
)
from .DANB_HVG import NBumiHVG
from .Other_FS_functions import Consensus_fs
from .Extremes import M3DropFeatureSelection
from .Normalization import NBumiPearsonResiduals, NBumiPearsonResidualsApprox
from .basics import M3DropConvertData

def _ensure_raw_counts(adata, layer=None):
    """
    Ensure we have raw count data for M3Drop analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
        
    Returns
    -------
    np.ndarray
        Raw count matrix (genes × cells).
    """
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Convert to dense if sparse
    if sp.issparse(X):
        X = X.toarray()
    
    # Transpose to genes × cells (M3Drop convention)
    X = X.T
    
    # Check if data looks like raw counts
    if not np.allclose(X, np.round(X)):
        raise ValueError(
            "Data does not appear to be raw counts. M3Drop requires integer count data. "
            "If you have raw counts in a layer, specify it with the 'layer' parameter."
        )
    
    if np.any(X < 0):
        raise ValueError("Count data contains negative values")
    
    return X.astype(int)

def fit_nbumi_model(
    adata,
    layer: Optional[str] = None,
    copy: bool = False,
    key_added: str = 'nbumi_fit'
):
    """
    Fit NBumi negative binomial model to count data.
    
    This function fits M3Drop's negative binomial UMI model to the data,
    which accounts for technical dropout and sequencing depth effects.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
    copy : bool, default False
        Return a copy instead of writing to adata.
    key_added : str, default 'nbumi_fit'
        Key under which to add the fit results to adata.uns.
        
    Returns
    -------
    adata : AnnData, optional
        If copy=True, returns AnnData object with fit results.
    """
    adata = adata.copy() if copy else adata
    
    # Get raw counts
    counts = _ensure_raw_counts(adata, layer)
    
    # Convert to DataFrame with gene and cell names
    counts_df = pd.DataFrame(
        counts,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Fit the model
    fit_result = NBumiFitModel(counts_df)
    
    # Store results
    adata.uns[key_added] = fit_result
    
    # Also store some key metrics in var for easy access
    adata.var[f'{key_added}_size'] = fit_result['sizes']
    adata.var[f'{key_added}_mean'] = fit_result['vals']['tjs'] / fit_result['vals']['nc']
    
    return adata if copy else None

def nbumi_normalize(
    adata,
    target_sum: Optional[float] = None,
    layer: Optional[str] = None,
    copy: bool = False,
    use_pearson_residuals: bool = False
):
    """
    Normalize counts using M3Drop's NBumi method.
    
    This method uses the negative binomial UMI model to perform depth-adjusted
    normalization that accounts for technical dropout effects.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_sum : float, optional
        Target sum for normalization. If None, uses median total counts.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
    copy : bool, default False
        Return a copy instead of writing to adata.
    use_pearson_residuals : bool, default False
        If True, compute Pearson residuals instead of normalized counts.
        
    Returns
    -------
    adata : AnnData, optional
        If copy=True, returns AnnData object with normalized data.
    """
    adata = adata.copy() if copy else adata
    
    # Get raw counts
    counts = _ensure_raw_counts(adata, layer)
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(
        counts,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Store raw counts if not already stored
    if adata.raw is None:
        adata.raw = adata
    
    if use_pearson_residuals:
        # Fit model first
        fit_result = NBumiFitModel(counts_df)
        
        # Compute Pearson residuals - keep as DataFrame to preserve gene names
        normalized = NBumiPearsonResiduals(counts_df, fit_result)
        
        # Store fit results
        adata.uns['nbumi_fit'] = fit_result
        adata.var['nbumi_size'] = fit_result['sizes']
        adata.var['nbumi_mean'] = fit_result['vals']['tjs'] / fit_result['vals']['nc']
        
    else:
        # Fit model
        fit_result = NBumiFitModel(counts_df)
        
        # Normalize using NBumi method - keep as DataFrame to preserve gene names
        normalized = NBumiImputeNorm(counts_df, fit_result, target_sum)
        
        # Store fit results
        adata.uns['nbumi_fit'] = fit_result
        adata.var['nbumi_size'] = fit_result['sizes']
        adata.var['nbumi_mean'] = fit_result['vals']['tjs'] / fit_result['vals']['nc']
    
    # Update the expression matrix (transpose back to cells × genes)
    # Convert to numpy array for adata.X but gene order is preserved
    if isinstance(normalized, pd.DataFrame):
        adata.X = normalized.T.values
    else:
        adata.X = normalized.T
    
    return adata if copy else None

def m3drop_highly_variable_genes(
    adata,
    layer: Optional[str] = None,
    ntop: Optional[int] = None,
    method: Literal['consensus', 'danb', 'combined_drop', 'm3drop'] = 'consensus',
    fdr_thresh: float = 0.05,
    copy: bool = False,
    **kwargs
):
    """
    Find highly variable genes using M3Drop methods.
    
    This function provides access to M3Drop's various feature selection methods,
    including the consensus method that combines multiple approaches.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
    ntop : int, optional
        Number of top genes to select. If None, uses fdr_thresh.
    method : {'consensus', 'danb', 'combined_drop', 'm3drop'}, default 'consensus'
        Method to use for feature selection:
        - 'consensus': Combines multiple methods (recommended)
        - 'danb': DANB method for highly variable genes
        - 'combined_drop': Combined dropout analysis
        - 'm3drop': Traditional M3Drop feature selection
    fdr_thresh : float, default 0.05
        FDR threshold for significance testing.
    copy : bool, default False
        Return a copy instead of writing to adata.
    **kwargs
        Additional parameters passed to the specific method.
        
    Returns
    -------
    adata : AnnData, optional
        If copy=True, returns AnnData object with HVG annotations.
    """
    adata = adata.copy() if copy else adata
    
    # Get raw counts
    counts = _ensure_raw_counts(adata, layer)
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(
        counts,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    if method == 'consensus':
        # Use consensus feature selection
        result = Consensus_fs(counts_df, **kwargs)
        
        # Add rankings to var
        for col in result.columns:
            adata.var[f'm3drop_{col.lower()}_rank'] = result[col]
        
        # Overall consensus ranking
        adata.var['m3drop_consensus_rank'] = result['Cons']
        
        # Select highly variable genes
        if ntop is not None:
            highly_variable = adata.var['m3drop_consensus_rank'] <= ntop
        else:
            # Use top genes based on consensus
            highly_variable = adata.var['m3drop_consensus_rank'] <= len(adata.var) * 0.1
        
    elif method == 'danb':
        # Fit NBumi model first
        fit_result = NBumiFitModel(counts_df)
        adata.uns['nbumi_fit'] = fit_result
        
        # DANB highly variable genes
        hvg_result = NBumiHVG(counts_df, fit_result, fdr_thresh=fdr_thresh, **kwargs)
        
        # Mark highly variable genes
        highly_variable = adata.var_names.isin(hvg_result['Gene'])
        
        # Add effect sizes and p-values
        hvg_dict = hvg_result.set_index('Gene').to_dict()
        adata.var['m3drop_danb_effect_size'] = [
            hvg_dict['effect.size'].get(gene, 0) for gene in adata.var_names
        ]
        adata.var['m3drop_danb_pvalue'] = [
            hvg_dict['p.value'].get(gene, 1) for gene in adata.var_names
        ]
        adata.var['m3drop_danb_qvalue'] = [
            hvg_dict['q.value'].get(gene, 1) for gene in adata.var_names
        ]
        
    elif method == 'combined_drop':
        # Fit NBumi model first
        fit_result = NBumiFitModel(counts_df)
        adata.uns['nbumi_fit'] = fit_result
        
        # Combined dropout analysis
        hvg_result = NBumiFeatureSelectionCombinedDrop(
            fit_result, ntop=ntop, qval_thresh=fdr_thresh, **kwargs
        )
        
        # Mark highly variable genes
        highly_variable = adata.var_names.isin(hvg_result['Gene'])
        
        # Add results to var
        hvg_dict = hvg_result.set_index('Gene').to_dict()
        adata.var['m3drop_combined_effect_size'] = [
            hvg_dict['effect_size'].get(gene, 0) for gene in adata.var_names
        ]
        adata.var['m3drop_combined_pvalue'] = [
            hvg_dict['p_value'].get(gene, 1) for gene in adata.var_names
        ]
        adata.var['m3drop_combined_qvalue'] = [
            hvg_dict['q_value'].get(gene, 1) for gene in adata.var_names
        ]
        
    elif method == 'm3drop':
        # Traditional M3Drop feature selection
        # First normalize the data for M3Drop
        sf = np.array(counts_df.sum(axis=0))
        sf = sf / np.median(sf)
        norm_df = counts_df / sf
        
        hvg_result = M3DropFeatureSelection(
            norm_df, mt_threshold=fdr_thresh, suppress_plot=True, **kwargs
        )
        
        # Mark highly variable genes
        highly_variable = adata.var_names.isin(hvg_result['Gene'])
        
        # Add results to var
        hvg_dict = hvg_result.set_index('Gene').to_dict()
        adata.var['m3drop_effect_size'] = [
            hvg_dict['effect.size'].get(gene, 0) for gene in adata.var_names
        ]
        adata.var['m3drop_pvalue'] = [
            hvg_dict['p.value'].get(gene, 1) for gene in adata.var_names
        ]
        adata.var['m3drop_qvalue'] = [
            hvg_dict['q.value'].get(gene, 1) for gene in adata.var_names
        ]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Set highly variable genes
    adata.var['highly_variable'] = highly_variable
    
    # Add summary statistics
    n_hvg = np.sum(highly_variable)
    print(f"Found {n_hvg} highly variable genes using M3Drop {method} method")
    
    return adata if copy else None

def nbumi_impute(
    adata,
    target_sum: Optional[float] = None,
    layer: Optional[str] = None,
    copy: bool = False
):
    """
    Impute and normalize counts using NBumi method.
    
    This method performs imputation of dropout events followed by normalization
    using the negative binomial UMI model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_sum : float, optional
        Target sum for normalization. If None, uses median total counts.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
    copy : bool, default False
        Return a copy instead of writing to adata.
        
    Returns
    -------
    adata : AnnData, optional
        If copy=True, returns AnnData object with imputed data.
    """
    adata = adata.copy() if copy else adata
    
    # Get raw counts
    counts = _ensure_raw_counts(adata, layer)
    
    # Convert to DataFrame
    counts_df = pd.DataFrame(
        counts,
        index=adata.var_names,
        columns=adata.obs_names
    )
    
    # Store raw counts if not already stored
    if adata.raw is None:
        adata.raw = adata
    
    # Fit NBumi model
    fit_result = NBumiFitModel(counts_df)
    
    # Impute and normalize - keep as DataFrame to preserve gene names
    imputed = NBumiImputeNorm(counts_df, fit_result, target_sum)
    
    # Update the expression matrix (transpose back to cells × genes)
    # Convert to numpy array for adata.X but gene order is preserved
    if isinstance(imputed, pd.DataFrame):
        adata.X = imputed.T.values
    else:
        adata.X = imputed.T
    
    # Store fit results
    adata.uns['nbumi_fit'] = fit_result
    adata.var['nbumi_size'] = fit_result['sizes']
    adata.var['nbumi_mean'] = fit_result['vals']['tjs'] / fit_result['vals']['nc']
    
    return adata if copy else None

# Add all functions to a scanpy-like namespace for easy access
def _register_functions():
    """Register functions in the scanpy namespace style"""
    # This would be called when the module is imported
    pass

# For backwards compatibility with the example
def nbumi_normalize_legacy(adata):
    """Legacy function for backwards compatibility"""
    return nbumi_normalize(adata, use_pearson_residuals=False)

def m3drop_highly_variable_genes_legacy(adata, ntop=2000):
    """Legacy function for backwards compatibility"""
    return m3drop_highly_variable_genes(adata, ntop=ntop, method='consensus')
