import numpy as np
import pandas as pd
import cupy as cp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import h5py
import os
import time
import psutil
import gc
from scipy import sparse
from scipy import stats
import anndata

# [GOVERNOR INTEGRATION] 
from .coreGPU import hidden_calc_valsGPU, NBumiFitModelGPU, NBumiFitDispVsMeanGPU, get_optimal_chunk_size
from cupy.sparse import csr_matrix as cp_csr_matrix
import scipy.sparse as sp
from scipy.sparse import csr_matrix as sp_csr_matrix
import statsmodels.api as sm

def NBumiFitBasicModelGPU(
    cleaned_filename: str,
    stats: dict,
    is_logged=False,
    chunk_size: int = None 
) -> dict:
    """
    Fits a simpler, unadjusted NB model out-of-core.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModel() | FILE: {cleaned_filename}")

    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=3.0, is_dense=True)

    # --- Phase 1: Initialization ---
    tjs = stats['tjs'].values
    nc, ng = stats['nc'], stats['ng']

    tjs_gpu = cp.asarray(tjs, dtype=cp.float64)
    sum_x_sq_gpu = cp.zeros(ng, dtype=cp.float64)

    # --- Phase 2: Calculate Variance from Data Chunks ---
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue
            
            # Original processing
            data_slice = h5_data[start_idx:end_idx]
            indices_slice = h5_indices[start_idx:end_idx]

            data_gpu = cp.asarray(data_slice, dtype=cp.float64)
            indices_gpu = cp.asarray(indices_slice)

            cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            del data_gpu, indices_gpu
            cp.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/2]: COMPLETE{' '*50}")

    mean_x_sq_gpu = sum_x_sq_gpu / nc
    mean_mu_gpu = tjs_gpu / nc
    my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
    size_gpu = mean_mu_gpu**2 / (my_rowvar_gpu - mean_mu_gpu)
    
    max_size_val = cp.nanmax(size_gpu) * 10
    if cp.isnan(max_size_val): max_size_val = 1000
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    my_rowvar_cpu = my_rowvar_gpu.get()
    sizes_cpu = size_gpu.get()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'var_obs': pd.Series(my_rowvar_cpu, index=stats['tjs'].index),
        'sizes': pd.Series(sizes_cpu, index=stats['tjs'].index),
        'vals': stats
    }

def NBumiCheckFitFSGPU(
    cleaned_filename: str,
    fit: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    Calculates fit errors. [FIXED] Added clamps to prevent >1.0 probability errors.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFS() | FILE: {cleaned_filename}")

    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=5.0, is_dense=True)

    vals = fit['vals']
    size_coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    tjs_gpu = cp.asarray(vals['tjs'].values, dtype=cp.float64)
    tis_gpu = cp.asarray(vals['tis'].values, dtype=cp.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    # Calculate smoothed size
    mean_expression_gpu = tjs_gpu / nc
    log_mean_expression_gpu = cp.log(mean_expression_gpu)
    smoothed_size_gpu = cp.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression_gpu)
    
    # [FIX] Clamp smoothed size to prevent instability
    smoothed_size_gpu = cp.maximum(smoothed_size_gpu, 1e-8)

    row_ps_gpu = cp.zeros(ng, dtype=cp.float64)
    col_ps_gpu = cp.zeros(nc, dtype=cp.float64)

    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]
        mu_chunk_gpu = tjs_gpu[:, cp.newaxis] * tis_chunk_gpu[cp.newaxis, :] / total
        
        # [FIX] Safer power calculation
        base = 1 + mu_chunk_gpu / smoothed_size_gpu[:, cp.newaxis]
        p_is_chunk_gpu = cp.power(base, -smoothed_size_gpu[:, cp.newaxis])
        
        # [FIX] Clamp probabilities to valid range [0, 1]
        p_is_chunk_gpu = cp.clip(p_is_chunk_gpu, 0.0, 1.0)
        p_is_chunk_gpu = cp.nan_to_num(p_is_chunk_gpu, nan=0.0)
        
        row_ps_gpu += p_is_chunk_gpu.sum(axis=1)
        col_ps_gpu[i:end_col] = p_is_chunk_gpu.sum(axis=0)
        
        del mu_chunk_gpu, p_is_chunk_gpu, base, tis_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()

    print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    row_ps_cpu = row_ps_gpu.get()
    col_ps_cpu = col_ps_gpu.get()
    djs_cpu = vals['djs'].values
    dis_cpu = vals['dis'].values

    gene_error = np.sum((djs_cpu - row_ps_cpu)**2)
    cell_error = np.sum((dis_cpu - col_ps_cpu)**2)
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'gene_error': gene_error,
        'cell_error': cell_error,
        'rowPs': pd.Series(row_ps_cpu, index=fit['vals']['tjs'].index),
        'colPs': pd.Series(col_ps_cpu, index=fit['vals']['tis'].index)
    }

def NBumiCompareModelsGPU(
    raw_filename: str,
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    OPTIMIZED VERSION (IN-MEMORY):
    - Calculates Basic Fit without writing 46GB file.
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=12.0, is_dense=False)

    print("Phase [1/3]: Calculating Basic Model (Depth-Normalized) variance on-the-fly...")
    
    # 1. Prepare Size Factors
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    nc, ng = stats['nc'], stats['ng']
    
    median_sum = np.median(tis[tis > 0])
    size_factors = np.ones_like(tis, dtype=np.float32)
    non_zero_mask = tis > 0
    size_factors[non_zero_mask] = tis[non_zero_mask] / median_sum
    
    sum_x_sq_gpu = cp.zeros(ng, dtype=cp.float64)
    sum_x_gpu = cp.zeros(ng, dtype=cp.float64)
    
    with h5py.File(cleaned_filename, 'r') as f_in:
        h5_indptr = f_in['X']['indptr']
        h5_data = f_in['X']['data']
        h5_indices = f_in['X']['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [1/3]: Processing: {end_row} of {nc} cells.", end='\r')
            
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue

            data_gpu = cp.asarray(h5_data[start_idx:end_idx], dtype=cp.float32)
            indices_gpu = cp.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cp.asarray(h5_indptr[i:end_row + 1] - start_idx)

            nnz_in_chunk = indptr_gpu[-1].item()
            cell_boundary_markers = cp.zeros(nnz_in_chunk, dtype=cp.int32)
            if len(indptr_gpu) > 1:
                cell_boundary_markers[indptr_gpu[:-1]] = 1
            row_indices = cp.cumsum(cell_boundary_markers, axis=0) - 1
            
            sf_chunk = cp.asarray(size_factors[i:end_row])
            
            # Normalize
            data_gpu /= sf_chunk[row_indices]
            
            # Accumulate
            cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            cp.add.at(sum_x_gpu, indices_gpu, data_gpu)
            
            del data_gpu, indices_gpu, indptr_gpu, row_indices, sf_chunk, cell_boundary_markers
            cp.get_default_memory_pool().free_all_blocks()

    print(f"Phase [1/3]: COMPLETE{' '*50}")

    mean_x_sq_gpu = sum_x_sq_gpu / nc
    mean_mu_gpu = sum_x_gpu / nc
    my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
    
    size_gpu = mean_mu_gpu**2 / (my_rowvar_gpu - mean_mu_gpu)
    
    max_size_val = cp.nanmax(size_gpu) * 10
    if cp.isnan(max_size_val): max_size_val = 1000
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    fit_basic = {
        'sizes': pd.Series(size_gpu.get(), index=stats['tjs'].index),
        'vals': stats,
        'var_obs': pd.Series(my_rowvar_gpu.get(), index=stats['tjs'].index)
    }
    
    print("Phase [2/3]: Evaluating fit errors on ORIGINAL data...")
    check_adjust = NBumiCheckFitFSGPU(cleaned_filename, fit_adjust, suppress_plot=True)
    check_basic = NBumiCheckFitFSGPU(cleaned_filename, fit_basic, suppress_plot=True)
    print("Phase [2/3]: COMPLETE")

    print("Phase [3/3]: Generating comparison...")
    
    nc_data = stats['nc']
    mean_expr = stats['tjs'] / nc_data
    observed_dropout = stats['djs'] / nc_data
    
    adj_dropout_fit = check_adjust['rowPs'] / nc_data
    bas_dropout_fit = check_basic['rowPs'] / nc_data
    
    err_adj = np.sum(np.abs(adj_dropout_fit - observed_dropout))
    err_bas = np.sum(np.abs(bas_dropout_fit - observed_dropout))
    
    comparison_df = pd.DataFrame({
        'mean_expr': mean_expr,
        'observed': observed_dropout,
        'adj_fit': adj_dropout_fit,
        'bas_fit': bas_dropout_fit
    })
    
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(mean_expr.values)
    
    plt.scatter(mean_expr.iloc[sorted_idx], observed_dropout.iloc[sorted_idx], 
                c='black', s=3, alpha=0.5, label='Observed')
    plt.scatter(mean_expr.iloc[sorted_idx], bas_dropout_fit.iloc[sorted_idx], 
                c='purple', s=3, alpha=0.6, label=f'Basic Fit (Error: {err_bas:.2f})')
    plt.scatter(mean_expr.iloc[sorted_idx], adj_dropout_fit.iloc[sorted_idx], 
                c='goldenrod', s=3, alpha=0.7, label=f'Depth-Adjusted Fit (Error: {err_adj:.2f})')
    
    plt.xscale('log')
    plt.xlabel("Mean Expression")
    plt.ylabel("Dropout Rate")
    plt.title("M3Drop Model Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Model comparison plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()
    
    plt.close()
    
    pipeline_end_time = time.time()
    print(f"Total time: {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    
    return {
        "errors": {"Depth-Adjusted": err_adj, "Basic": err_bas},
        "comparison_df": comparison_df
    }

def NBumiPlotDispVsMeanGPU(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a diagnostic plot of the dispersion vs. mean expression.
    """
    print("FUNCTION: NBumiPlotDispVsMean()")

    mean_expression = fit['vals']['tjs'].values / fit['vals']['nc']
    sizes = fit['sizes'].values
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]

    log_mean_expr_range = np.linspace(
        np.log(mean_expression[mean_expression > 0].min()),
        np.log(mean_expression.max()),
        100
    )
    log_fitted_sizes = intercept + slope * log_mean_expr_range
    fitted_sizes = np.exp(log_fitted_sizes)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_expression, sizes, label='Observed Dispersion', alpha=0.5, s=8)
    plt.plot(np.exp(log_mean_expr_range), fitted_sizes, color='red', label='Regression Fit', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean Expression')
    plt.ylabel('Dispersion Parameter (Sizes)')
    plt.title('Dispersion vs. Mean Expression')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Diagnostic plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()

    plt.close()
    print("FUNCTION: NBumiPlotDispVsMean() COMPLETE\n")
