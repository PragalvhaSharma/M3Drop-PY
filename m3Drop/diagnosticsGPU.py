from .coreGPU import (
    hidden_calc_valsGPU, 
    NBumiFitModelGPU, 
    NBumiFitDispVsMeanGPU, 
    get_io_chunk_size, 
    get_compute_tile_size,
    get_optimal_chunk_size # Kept for safety, though unused
)
import cupy
import numpy as np
import anndata
import h5py
import pandas as pd
import time
import os

from cupy.sparse import csr_matrix as cp_csr_matrix
import scipy.sparse as sp
from scipy.sparse import csr_matrix as sp_csr_matrix

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def NBumiFitBasicModelGPU(
    cleaned_filename: str,
    stats: dict,
    is_logged=False,
    size_factors=None
) -> dict:
    """
    Fits a simpler, unadjusted NB model.
    OPTIMIZATION: Supports 'size_factors' for On-the-Fly normalization.
    If size_factors is provided, it normalizes and rounds data in VRAM 
    without creating a temporary file.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModel() | FILE: {cleaned_filename}")
    
    # Mode detection
    doing_normalization = size_factors is not None
    if doing_normalization:
        print("  > Mode: On-the-Fly Normalization & Rounding enabled.")

    # I/O Config
    chunk_size = get_io_chunk_size(cleaned_filename, target_mb=1024)

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    nc, ng = stats['nc'], stats['ng']

    # Accumulators
    sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
    
    # If normalizing, we must recalculate the means (tjs) from the normalized data
    if doing_normalization:
        sum_x_gpu = cupy.zeros(ng, dtype=cupy.float64)
    else:
        # If not normalizing, use the pre-calculated raw stats
        sum_x_gpu = cupy.asarray(stats['tjs'].values, dtype=cupy.float64)

    print("Phase [1/2]: COMPLETE")

    # --- Phase 2: Calculate Variance from Data Chunks ---
    print("Phase [2/2]: Calculating variance from data chunks...")
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx:
                continue
            
            # Load chunk
            data_slice = h5_data[start_idx:end_idx]
            indices_slice = h5_indices[start_idx:end_idx]
            indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

            data_gpu = cupy.asarray(data_slice, dtype=cupy.float64)
            indices_gpu = cupy.asarray(indices_slice)

            # --- ON-THE-FLY NORMALIZATION ---
            if doing_normalization:
                # Expand size factors to match data structure (CSR expansion)
                # 1. Get size factors for this chunk of cells
                sf_chunk = cupy.asarray(size_factors[i:end_row])
                
                # 2. Create row map for every non-zero value
                nnz_in_chunk = indptr_slice[-1]
                indptr_gpu = cupy.asarray(indptr_slice)
                
                # Efficient row expansion
                row_boundaries = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
                if len(indptr_gpu) > 1:
                    row_boundaries[indptr_gpu[:-1]] = 1
                row_indices_gpu = cupy.cumsum(row_boundaries, axis=0) - 1
                
                # 3. Apply Normalization & Rounding (Replicating original logic)
                # data = round(data / size_factor)
                data_gpu /= sf_chunk[row_indices_gpu]
                data_gpu = cupy.rint(data_gpu) # Round to nearest int

                # Accumulate Sum (New TJS)
                cupy.add.at(sum_x_gpu, indices_gpu, data_gpu)
                
                # Cleanup expansion vars
                del sf_chunk, row_indices_gpu, indptr_gpu

            # Accumulate Sum Squares
            cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            # Clean up
            del data_gpu, indices_gpu
            if i % (chunk_size * 2) == 0:
                cupy.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    # --- Final calculations on GPU ---
    if is_logged:
        raise NotImplementedError("Logged data variance calculation is not implemented for out-of-core.")
    else:
        # Variance of data: Var(X) = E[X^2] - E[X]^2
        mean_x_sq_gpu = sum_x_sq_gpu / nc
        mean_mu_gpu = sum_x_gpu / nc
        my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
        
        # Calculate dispersion ('size')
        # size = mu^2 / (var - mu)
        denominator_gpu = my_rowvar_gpu - mean_mu_gpu
        size_gpu = mean_mu_gpu**2 / denominator_gpu
    
    # Stability handling
    max_size_val = cupy.nanmax(size_gpu) * 10
    if cupy.isnan(max_size_val): 
        max_size_val = 1000
        
    mask_bad = cupy.isnan(size_gpu) | (size_gpu <= 0)
    size_gpu[mask_bad] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    # Move results to CPU
    my_rowvar_cpu = my_rowvar_gpu.get()
    sizes_cpu = size_gpu.get()
    
    # If we normalized, we have new stats (tjs has changed)
    if doing_normalization:
        # Create a shallow copy of stats with updated tjs
        new_stats = stats.copy()
        new_stats['tjs'] = pd.Series(sum_x_gpu.get(), index=stats['tjs'].index)
    else:
        new_stats = stats

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        'var_obs': pd.Series(my_rowvar_cpu, index=stats['tjs'].index),
        'sizes': pd.Series(sizes_cpu, index=stats['tjs'].index),
        'vals': new_stats
    }

def NBumiCheckFitFSGPU(
    cleaned_filename: str,
    fit: dict,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    Calculates expected dropout rates vs observed dropout rates.
    TILED IMPLEMENTATION: Prevents VRAM OOM by slicing inner loops.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFS() | FILE: {cleaned_filename}")

    # 1. READ CONFIG (Virtual I/O)
    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=1024)
    
    # 2. COMPUTE CONFIG (VRAM Safe)
    vals = fit['vals']
    # We need 3 matrices: mu, base, p_is (approx)
    # Using specific dense calculation logic
    compute_tile_size = get_compute_tile_size(n_genes=vals['ng'], vram_limit_gb=9.0)

    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    size_coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    # Must use float64 for precision
    tjs_gpu = cupy.asarray(vals['tjs'].values, dtype=cupy.float64)
    tis_gpu = cupy.asarray(vals['tis'].values, dtype=cupy.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    # Calculate smoothed size
    mean_expression_gpu = tjs_gpu / nc
    log_mean_expression_gpu = cupy.log(mean_expression_gpu)
    smoothed_size_gpu = cupy.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression_gpu)

    # Initialize result arrays
    row_ps_gpu = cupy.zeros(ng, dtype=cupy.float64)
    col_ps_gpu = cupy.zeros(nc, dtype=cupy.float64)
    print("Phase [1/2]: COMPLETE")

    # --- Phase 2: Calculate Expected Dropouts (TILED) ---
    print("Phase [2/2]: Calculating expected dropouts (Tiled)...")
    
    # Outer Loop: "I/O" Blocks (Though we don't read H5 here, we respect the structure)
    for i in range(0, nc, io_chunk_size):
        end_col = min(i + io_chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        # Inner Loop: GPU Tiles
        for j in range(i, end_col, compute_tile_size):
            tile_end = min(j + compute_tile_size, end_col)
            
            # Slice TIS
            tis_chunk_gpu = tis_gpu[j:tile_end]

            # BROADCAST OPERATION: Creates Dense Matrix (ng, tile_size)
            # mu = tjs * tis / total
            mu_chunk_gpu = tjs_gpu[:, cupy.newaxis] * tis_chunk_gpu[cupy.newaxis, :] / total
            
            # Calculate p_is
            # base = 1 + mu / size
            base = 1.0 + mu_chunk_gpu / smoothed_size_gpu[:, cupy.newaxis]
            p_is_chunk_gpu = cupy.power(base, -smoothed_size_gpu[:, cupy.newaxis])
            
            # Handle numerical instability
            p_is_chunk_gpu = cupy.nan_to_num(p_is_chunk_gpu, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Accumulate
            row_ps_gpu += p_is_chunk_gpu.sum(axis=1)
            col_ps_gpu[j:tile_end] = p_is_chunk_gpu.sum(axis=0)
            
            # Clean up VRAM immediately
            del mu_chunk_gpu, p_is_chunk_gpu, base, tis_chunk_gpu
            cupy.get_default_memory_pool().free_all_blocks()

    print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    # Move results to CPU
    row_ps_cpu = row_ps_gpu.get()
    col_ps_cpu = col_ps_gpu.get()
    djs_cpu = vals['djs'].values
    dis_cpu = vals['dis'].values

    # Plotting
    if not suppress_plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(djs_cpu, row_ps_cpu, alpha=0.5, s=10)
        plt.title("Gene-specific Dropouts (Smoothed)")
        plt.xlabel("Observed")
        plt.ylabel("Fit")
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="y=x line")
        plt.grid(True); plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(dis_cpu, col_ps_cpu, alpha=0.5, s=10)
        plt.title("Cell-specific Dropouts (Smoothed)")
        plt.xlabel("Observed")
        plt.ylabel("Expected")
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0, label="y=x line")
        plt.grid(True); plt.legend()
        
        plt.tight_layout()
        if plot_filename:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"STATUS: Diagnostic plot saved to '{plot_filename}'")
        plt.show()
        plt.close()

    # Calculate errors
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
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    Compares the Depth-Adjusted M3Drop model vs a Basic M3Drop model.
    OPTIMIZED: Uses On-the-Fly Normalization (No temporary files).
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    # --- Phase 1: Calculate Size Factors in RAM ---
    print("Phase [1/3]: Calculating size factors...")
    cell_sums = stats['tis'].values
    median_sum = np.median(cell_sums[cell_sums > 0])
    
    # Avoid division by zero
    size_factors = np.ones_like(cell_sums, dtype=np.float32)
    non_zero_mask = cell_sums > 0
    size_factors[non_zero_mask] = cell_sums[non_zero_mask] / median_sum
    print("Phase [1/3]: COMPLETE")

    # --- Phase 2: Fit Basic Model (On-the-Fly) ---
    print("Phase [2/3]: Fitting Basic Model (On-the-Fly Normalization)...")
    # We pass the size factors directly. The function handles normalization/rounding in VRAM.
    fit_basic = NBumiFitBasicModelGPU(
        cleaned_filename, 
        stats, 
        size_factors=size_factors
    )
    print("Phase [2/3]: COMPLETE")
    
    # --- Phase 3: Evaluate & Compare ---
    print("Phase [3/3]: Evaluating fits on ORIGINAL data structure...")
    
    # Check Adjusted Model
    check_adjust = NBumiCheckFitFSGPU(cleaned_filename, fit_adjust, suppress_plot=True)
    
    # Check Basic Model
    # Note: fit_basic['vals'] already contains the UPDATED stats (tjs) from the normalized data
    # because we passed size_factors to NBumiFitBasicModelGPU.
    check_basic = NBumiCheckFitFSGPU(cleaned_filename, fit_basic, suppress_plot=True)

    print("  > Generating comparison plot...")
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
    print("Phase [3/3]: COMPLETE")

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

    # --- 1. Extract data and regression coefficients ---
    mean_expression = fit['vals']['tjs'].values / fit['vals']['nc']
    sizes = fit['sizes'].values
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]

    # --- 2. Calculate the fitted line for plotting ---
    log_mean_expr_range = np.linspace(
        np.log(mean_expression[mean_expression > 0].min()),
        np.log(mean_expression.max()),
        100
    )
    log_fitted_sizes = intercept + slope * log_mean_expr_range
    fitted_sizes = np.exp(log_fitted_sizes)

    # --- 3. Create the plot ---
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
