import numpy as np
import pandas as pd
import cupy as cp
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt
import h5py
import os
import time
import pickle
import psutil
import gc
from scipy import sparse
from scipy import stats
import anndata

# ==========================================
#      HYBRID IMPORT (PACKAGE VS LOCAL)
# ==========================================
try:
    # Case 1: Package
    from .ControlDeviceGPU import ControlDevice
    from .CoreGPU import (
        hidden_calc_valsGPU, 
        NBumiFitModelGPU, 
        NBumiFitDispVsMeanGPU, 
        dropout_prob_kernel
    )
except ImportError:
    # Case 2: Local
    try:
        from ControlDeviceGPU import ControlDevice
        from CoreGPU import (
            hidden_calc_valsGPU, 
            NBumiFitModelGPU, 
            NBumiFitDispVsMeanGPU, 
            dropout_prob_kernel
        )
    except ImportError:
        print("CRITICAL ERROR: Dependencies (ControlDeviceGPU, CoreGPU) not found.")
        sys.exit(1)

from cupy.sparse import csr_matrix as cp_csr_matrix
import scipy.sparse as sp
from scipy.sparse import csr_matrix as sp_csr_matrix

import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ==========================================
#        DIAGNOSTICS & COMPARISON
# ==========================================

def NBumiFitBasicModelGPU(
    filename: str,
    stats: dict,
    mask_filename: str = None, 
    mode: str = "auto",
    manual_target: int = 3000,
    phase_label: str = "Phase [1/1]",
    desc_label: str = None  # [UI FIX] Added for delayed printing
) -> dict:
    """
    Fits the Basic Model by calculating Normalized Variance ON-THE-FLY.
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

    # [UI FIX] Print description AFTER ControlDevice box
    if desc_label:
        print(f"{phase_label}: {desc_label}")

    # 2. Load Mask
    if mask_filename and os.path.exists(mask_filename):
        with open(mask_filename, 'rb') as f:
            mask_cpu = pickle.load(f)
    else:
        mask_cpu = np.ones(raw_ng, dtype=bool)

    filtered_ng = int(np.sum(mask_cpu))
    
    # 3. Pre-calculate Size Factors
    cell_sums = stats['tis'].values
    median_sum = np.median(cell_sums[cell_sums > 0])
    # [FLOAT64] Explicitly utilizing float64 for size factors
    size_factors = np.ones_like(cell_sums, dtype=np.float64)
    non_zero_mask = cell_sums > 0
    size_factors[non_zero_mask] = cell_sums[non_zero_mask] / median_sum
    
    # 4. Init GPU Arrays
    sum_norm_x_gpu = cp.zeros(filtered_ng, dtype=cp.float64)
    sum_norm_sq_gpu = cp.zeros(filtered_ng, dtype=cp.float64)

    with h5py.File(filename, 'r') as f_in:
        h5_indptr = f_in['X']['indptr']
        h5_data = f_in['X']['data']
        h5_indices = f_in['X']['indices']

        current_row = 0
        while current_row < nc:
            end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.5)
            if end_row is None or end_row <= current_row: break

            chunk_size = end_row - current_row
            # [UI] Phase-aware progress bar
            print(f"{phase_label}: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            if start_idx == end_idx:
                current_row = end_row
                continue
            
            # [FLOAT64] Load Raw Chunk as float64
            data_gpu = cp.asarray(h5_data[start_idx:end_idx], dtype=cp.float64)
            indices_gpu = cp.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cp.asarray(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
            
            # Reconstruct CSR & Filter
            raw_chunk = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(chunk_size, raw_ng))
            mask_gpu = cp.asarray(mask_cpu)
            filtered_chunk = raw_chunk[:, mask_gpu]

            # Fused Normalization
            # [FLOAT64] Size factors are already float64
            sf_chunk = cp.asarray(size_factors[current_row:end_row], dtype=cp.float64)
            recip_sf = 1.0 / sf_chunk
            D = csp.diags(recip_sf)
            norm_chunk = D.dot(filtered_chunk)
            norm_chunk.data = cp.round(norm_chunk.data)
            
            # Accumulate
            sum_norm_x_gpu += norm_chunk.sum(axis=0).ravel()
            norm_chunk.data **= 2
            sum_norm_sq_gpu += norm_chunk.sum(axis=0).ravel()
            
            del data_gpu, indices_gpu, raw_chunk, filtered_chunk, norm_chunk, D, sf_chunk, mask_gpu
            cp.get_default_memory_pool().free_all_blocks()
            current_row = end_row

    # Final Calculations
    mean_norm_gpu = sum_norm_x_gpu / nc
    mean_sq_norm_gpu = sum_norm_sq_gpu / nc
    var_norm_gpu = mean_sq_norm_gpu - (mean_norm_gpu ** 2)
    
    denom_gpu = var_norm_gpu - mean_norm_gpu
    size_gpu = cp.full(filtered_ng, 1000.0, dtype=cp.float64)
    valid_mask = denom_gpu > 1e-6
    size_gpu[valid_mask] = mean_norm_gpu[valid_mask]**2 / denom_gpu[valid_mask]
    
    max_size_val = cp.nanmax(size_gpu[size_gpu < 1e6]) * 10
    if cp.isnan(max_size_val) or max_size_val == 0: max_size_val = 1000.0
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    # [UI] Clean completion - Force Newline
    print("") 
    print(f"{phase_label}: COMPLETE")

    return {
        'var_obs': pd.Series(var_norm_gpu.get(), index=stats['tjs'].index),
        'sizes': pd.Series(size_gpu.get(), index=stats['tjs'].index),
        'vals': stats 
    }

def NBumiCheckFitFSGPU(
    filename: str,
    fit: dict,
    mode: str = "auto",
    manual_target: int = 3000,
    suppress_plot=False,
    plot_filename=None,
    phase_label="Phase [1/1]",
    desc_label: str = None  # [UI FIX] Added for delayed printing
) -> dict:
    """
    Calculates expected dropouts. Handles Real and Virtual Populations.
    Uses FUSED KERNEL to prevent OOM on large chunks.
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

    # [UI FIX] Print description AFTER ControlDevice box
    if desc_label:
        print(f"{phase_label}: {desc_label}")

    size_coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    
    tjs_gpu = cp.asarray(vals['tjs'].values, dtype=cp.float64)
    tis_gpu = cp.asarray(vals['tis'].values, dtype=cp.float64) 
    total = vals['total']

    mean_expression_gpu = tjs_gpu / nc
    log_mean_expression_gpu = cp.zeros_like(mean_expression_gpu)
    valid_means = mean_expression_gpu > 0
    log_mean_expression_gpu[valid_means] = cp.log(mean_expression_gpu[valid_means])
    smoothed_size_gpu = cp.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression_gpu)

    row_ps_gpu = cp.zeros(ng, dtype=cp.float64)
    col_ps_gpu = cp.zeros(nc, dtype=cp.float64)

    current_row = 0
    while current_row < nc:
        # [FIX] Keep overhead low (1.1) because we are using Fused Kernel
        end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.1)
        if end_row is None or end_row <= current_row: break
        
        chunk_size = end_row - current_row
        
        # [UI] Phase-aware progress bar
        print(f"{phase_label}: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

        tis_chunk_gpu = tis_gpu[current_row:end_row]
        
        # [CRITICAL] FUSED KERNEL PRESERVED (Supercomputer Fix)
        # Explicit float64 for the output buffer
        p_is_chunk_gpu = cp.empty((chunk_size, ng), dtype=cp.float64)
        
        dropout_prob_kernel(
            tjs_gpu,                 # Gene totals
            tis_chunk_gpu[:, None],  # Cell totals
            total,                   # Grand total
            smoothed_size_gpu,       # Exp size
            p_is_chunk_gpu           # Output
        )

        # [MEMORY FIX] Use copy=False to prevent doubling memory usage
        cp.nan_to_num(p_is_chunk_gpu, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        
        row_ps_gpu += p_is_chunk_gpu.sum(axis=0)
        col_ps_gpu[current_row:end_row] = p_is_chunk_gpu.sum(axis=1)
        
        del p_is_chunk_gpu, tis_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()
        current_row = end_row

    # [UI] Clean completion - Force Newline
    print("")
    print(f"{phase_label}: COMPLETE")

    row_ps_cpu = row_ps_gpu.get()
    col_ps_cpu = col_ps_gpu.get()
    
    return {
        'rowPs': pd.Series(row_ps_cpu, index=fit['vals']['tjs'].index),
        'colPs': pd.Series(col_ps_cpu, index=fit['vals']['tis'].index)
    }

def NBumiCompareModelsGPU(
    raw_filename: str,
    stats: dict,
    fit_adjust: dict,
    mask_filename: str = None, 
    mode: str = "auto",
    manual_target: int = 3000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    Orchestrates the Comparison Pipeline with standardized UI.
    """
    print(f"FUNCTION: NBumiCompareModelsGPU()")
    pipeline_start_time = time.time()

    # STEP 1: Fit Basic Model
    # [UI FIX] Removed early print, passed as desc_label
    fit_basic = NBumiFitBasicModelGPU(
        raw_filename, 
        stats, 
        mask_filename=mask_filename,
        mode=mode, 
        manual_target=manual_target, 
        phase_label="Phase [1/3]",
        desc_label="Fitting Basic Model (Virtual)..."
    )
    
    # STEP 2: Depth-Adjusted Dropout
    # [UI FIX] Removed early print, passed as desc_label
    check_adjust = NBumiCheckFitFSGPU(
        raw_filename, 
        fit_adjust, 
        mode=mode, 
        manual_target=manual_target, 
        suppress_plot=True,
        phase_label="Phase [2/3]",
        desc_label="Calculating Depth-Adjusted Dropouts..."
    )
    
    # STEP 3: Basic Dropout
    # [UI FIX] Removed early print, passed as desc_label
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
    
    check_basic = NBumiCheckFitFSGPU(
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
    
    # Plotting Logic
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
    plt.title("M3Drop Model Comparison")
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

def NBumiPlotDispVsMeanGPU(
    fit: dict,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    print("FUNCTION: NBumiPlotDispVsMean()")
    start_time = time.time()
        
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
        print(f"Saving plot to: {plot_filename}")
            
    if not suppress_plot:
        plt.show()

    plt.close()
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
