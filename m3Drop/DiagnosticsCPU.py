import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import time
import pickle
import gc
from scipy import sparse
from scipy import stats
import anndata

import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# [FIX] Strict Relative Imports
from .ControlDeviceCPU import ControlDevice
from .CoreCPU import hidden_calc_valsCPU, NBumiFitModelCPU, NBumiFitDispVsMeanCPU, dropout_prob_kernel_cpu

# ==========================================
#        DIAGNOSTICS & COMPARISON (CPU)
# ==========================================

def NBumiFitBasicModelCPU(
    filename: str,
    stats: dict,
    mask_filename: str = None, 
    mode: str = "auto",
    manual_target: int = 3000,
    phase_label: str = "Phase [1/1]",
    desc_label: str = None
) -> dict:
    """
    Fits the Basic Model by calculating Normalized Variance ON-THE-FLY (CPU Optimized).
    STRICT FLOAT64 ENFORCEMENT.
    """
    # 1. Get Raw Dimensions & Setup ControlDevice
    with h5py.File(filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
        raw_ng = f['X'].attrs['shape'][1] 

    device = ControlDevice(
        indptr=indptr_cpu, 
        total_rows=total_rows, 
        n_genes=raw_ng, 
        mode=mode, 
        manual_target=manual_target
    )
    nc = device.total_rows

    if desc_label:
        print(f"{phase_label}: {desc_label}")

    # 2. Load Mask
    if mask_filename and os.path.exists(mask_filename):
        with open(mask_filename, 'rb') as f:
            mask = pickle.load(f)
    else:
        mask = np.ones(raw_ng, dtype=bool)

    filtered_ng = int(np.sum(mask))
    
    # 3. Pre-calculate Size Factors
    cell_sums = stats['tis'].values
    median_sum = np.median(cell_sums[cell_sums > 0])
    
    # [FLOAT64] Explicit enforcement
    size_factors = np.ones_like(cell_sums, dtype=np.float64)
    non_zero_mask = cell_sums > 0
    size_factors[non_zero_mask] = cell_sums[non_zero_mask] / median_sum
    
    # 4. Init Accumulators
    sum_norm_x = np.zeros(filtered_ng, dtype=np.float64)
    sum_norm_sq = np.zeros(filtered_ng, dtype=np.float64)

    with h5py.File(filename, 'r') as f_in:
        h5_indptr = f_in['X']['indptr']
        h5_data = f_in['X']['data']
        h5_indices = f_in['X']['indices']

        current_row = 0
        while current_row < nc:
            # CPU prefers dense chunks if they fit in L3, but sparse is safer for RAM.
            # We use 'dense' mode here because we convert to dense for normalization anyway.
            end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.5)
            if end_row is None or end_row <= current_row: break

            chunk_size = end_row - current_row
            print(f"{phase_label}: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            if start_idx == end_idx:
                current_row = end_row
                continue
            
            # [FLOAT64] Load Raw Chunk
            data = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices = np.array(h5_indices[start_idx:end_idx])
            indptr = np.array(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
            
            # Reconstruct CSR & Filter
            raw_chunk = sparse.csr_matrix((data, indices, indptr), shape=(chunk_size, raw_ng))
            filtered_chunk = raw_chunk[:, mask]

            # Normalization (Vectorized CPU)
            sf_chunk = size_factors[current_row:end_row]
            
            # Scipy sparse multiplication is efficient
            # D = diag(1/sf)
            recip_sf = 1.0 / sf_chunk
            D = sparse.diags(recip_sf)
            norm_chunk = D.dot(filtered_chunk)
            
            # Rounding (in-place on data array)
            np.round(norm_chunk.data, out=norm_chunk.data)
            
            # Accumulate
            # Convert to dense for summation if chunk is small (faster on CPU)
            # or keep sparse if very large. Given L3 optimization, dense is often fine.
            norm_dense = norm_chunk.toarray() 
            
            sum_norm_x += norm_dense.sum(axis=0)
            sum_norm_sq += (norm_dense ** 2).sum(axis=0)
            
            current_row = end_row

    # Final Calculations
    mean_norm = sum_norm_x / nc
    mean_sq_norm = sum_norm_sq / nc
    var_norm = mean_sq_norm - (mean_norm ** 2)
    
    denom = var_norm - mean_norm
    sizes = np.full(filtered_ng, 1000.0, dtype=np.float64)
    valid_mask = denom > 1e-6
    sizes[valid_mask] = mean_norm[valid_mask]**2 / denom[valid_mask]
    
    # Filtering outliers (Numpy version)
    with np.errstate(invalid='ignore'):
        max_size_val = np.nanmax(sizes[sizes < 1e6]) * 10
        
    if np.isnan(max_size_val) or max_size_val == 0: max_size_val = 1000.0
    sizes[np.isnan(sizes) | (sizes <= 0)] = max_size_val
    sizes[sizes < 1e-10] = 1e-10
    
    print("") 
    print(f"{phase_label}: COMPLETE")

    return {
        'var_obs': pd.Series(var_norm, index=stats['tjs'].index),
        'sizes': pd.Series(sizes, index=stats['tjs'].index),
        'vals': stats 
    }

def NBumiCheckFitFSCPU(
    filename: str,
    fit: dict,
    mode: str = "auto",
    manual_target: int = 3000,
    suppress_plot=False,
    plot_filename=None,
    phase_label="Phase [1/1]",
    desc_label: str = None
) -> dict:
    """
    Calculates expected dropouts using NUMBA KERNEL on CPU.
    """
    vals = fit['vals']
    ng = vals['ng']
    
    with h5py.File(filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
    
    device = ControlDevice(
        indptr=indptr_cpu, 
        total_rows=total_rows, 
        n_genes=ng, 
        mode=mode, 
        manual_target=manual_target
    )
    nc = device.total_rows

    if desc_label:
        print(f"{phase_label}: {desc_label}")

    size_coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)
    
    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    total = vals['total']

    mean_expression = tjs / nc
    log_mean_expression = np.zeros_like(mean_expression)
    valid_means = mean_expression > 0
    log_mean_expression[valid_means] = np.log(mean_expression[valid_means])
    smoothed_size = np.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression)

    row_ps = np.zeros(ng, dtype=np.float64)
    col_ps = np.zeros(nc, dtype=np.float64)

    current_row = 0
    while current_row < nc:
        # Use dense mode for Numba efficiency
        end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.1)
        if end_row is None or end_row <= current_row: break
        
        chunk_size = end_row - current_row
        print(f"{phase_label}: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

        tis_chunk = tis[current_row:end_row]
        
        # [CRITICAL] NUMBA KERNEL CALL
        # Prepare output buffer
        p_is_chunk = np.empty((chunk_size, ng), dtype=np.float64)
        
        dropout_prob_kernel_cpu(
            tjs,                 # Gene totals
            tis_chunk,           # Cell totals (1D array, broadcasting handled inside kernel)
            total,               # Grand total
            smoothed_size,       # Exp size
            p_is_chunk           # Output buffer
        )

        # Sanitize
        p_is_chunk = np.nan_to_num(p_is_chunk, nan=0.0, posinf=1.0, neginf=0.0)
        
        row_ps += p_is_chunk.sum(axis=0)
        col_ps[current_row:end_row] = p_is_chunk.sum(axis=1)
        
        current_row = end_row

    print("")
    print(f"{phase_label}: COMPLETE")

    return {
        'rowPs': pd.Series(row_ps, index=fit['vals']['tjs'].index),
        'colPs': pd.Series(col_ps, index=fit['vals']['tis'].index)
    }

def NBumiCompareModelsCPU(
    raw_filename: str,
    stats: dict,
    fit_adjust: dict,
    mask_filename: str = None, 
    mode: str = "auto",
    manual_target: int = 3000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    print(f"FUNCTION: NBumiCompareModelsCPU()")
    pipeline_start_time = time.time()

    # STEP 1: Fit Basic Model
    fit_basic = NBumiFitBasicModelCPU(
        raw_filename, 
        stats, 
        mask_filename=mask_filename,
        mode=mode, 
        manual_target=manual_target,
        phase_label="Phase [1/3]",
        desc_label="Fitting Basic Model (Virtual)..."
    )
    
    # STEP 2: Depth-Adjusted Dropout
    check_adjust = NBumiCheckFitFSCPU(
        raw_filename, 
        fit_adjust, 
        mode=mode, 
        manual_target=manual_target, 
        suppress_plot=True,
        phase_label="Phase [2/3]",
        desc_label="Calculating Depth-Adjusted Dropouts..."
    )
    
    # STEP 3: Basic Dropout
    stats_virtual = stats.copy()
    mean_depth = stats['total'] / stats['nc']
    stats_virtual['tis'] = pd.Series(
        np.full(stats['nc'], mean_depth), 
        index=stats['tis'].index
    )
    
    fit_basic_for_eval = {
        'sizes': fit_basic['sizes'],
        'vals': stats_virtual,
        'var_obs': fit_basic['var_obs']
    }
    
    check_basic = NBumiCheckFitFSCPU(
        raw_filename, 
        fit_basic_for_eval, 
        mode=mode, 
        manual_target=manual_target, 
        suppress_plot=True,
        phase_label="Phase [3/3]",
        desc_label="Calculating Basic Dropouts..."
    )

    # Calculation & Plotting
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
    
    # Plotting Logic (Standard Matplotlib)
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(mean_expr.values)
    plot_idx = sorted_idx[::2] if len(mean_expr) > 20000 else sorted_idx

    plt.scatter(mean_expr.iloc[plot_idx], observed_dropout.iloc[plot_idx], 
                c='black', s=3, alpha=0.5, label='Observed')
    
    plt.scatter(mean_expr.iloc[plot_idx], bas_dropout_fit.iloc[plot_idx], 
                c='purple', s=3, alpha=0.6, label=f'Basic Fit (Error: {err_bas:.2f})')
    
    plt.scatter(mean_expr.iloc[plot_idx], adj_dropout_fit.iloc[plot_idx], 
                c='goldenrod', s=3, alpha=0.7, label=f'Depth-Adjusted Fit (Error: {err_adj:.2f})')
    
    plt.xscale('log')
    plt.xlabel("Mean Expression")
    plt.ylabel("Dropout Rate")
    plt.title("M3Drop Model Comparison (CPU)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saving plot to: {plot_filename}")

    if not suppress_plot:
        plt.show()
    
    plt.close()

    pipeline_end_time = time.time()
    print(f"Total time: {pipeline_end_time - pipeline_start_time:.2f} seconds.\n")
    
    return {
        "errors": {"Depth-Adjusted": err_adj, "Basic": err_bas},
        "comparison_df": comparison_df
    }

def NBumiPlotDispVsMeanCPU(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    print("FUNCTION: NBumiPlotDispVsMeanCPU()")
    start_time = time.time()
        
    mean_expression = fit['vals']['tjs'].values / fit['vals']['nc']
    sizes = fit['sizes'].values
    
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)
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
    plt.title('Dispersion vs. Mean Expression (CPU)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saving plot to: {plot_filename}")
            
    if not suppress_plot:
        plt.show()

    plt.close()
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
