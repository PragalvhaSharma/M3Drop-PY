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
    Fits a simpler, unadjusted NB model out-of-core using a GPU-accelerated
    algorithm. Designed to work with a standard (cell, gene) sparse matrix.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModel() | FILE: {cleaned_filename}")

    # [GOVERNOR INTEGRATION] Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=3.0, is_dense=True)

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    tjs = stats['tjs'].values
    nc, ng = stats['nc'], stats['ng']

    tjs_gpu = cp.asarray(tjs, dtype=cp.float64)
    sum_x_sq_gpu = cp.zeros(ng, dtype=cp.float64)
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
            
            # Process in smaller sub-chunks if needed
            max_elements = 5_000_000  # Process max 5M elements at a time
            
            if end_idx - start_idx > max_elements:
                # Process in sub-chunks
                for sub_start in range(start_idx, end_idx, max_elements):
                    sub_end = min(sub_start + max_elements, end_idx)
                    
                    data_slice = h5_data[sub_start:sub_end]
                    indices_slice = h5_indices[sub_start:sub_end]
                    
                    data_gpu = cp.asarray(data_slice, dtype=cp.float64)
                    indices_gpu = cp.asarray(indices_slice)
                    
                    # Accumulate the sum of squares for each gene
                    cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                    
                    # Free GPU memory
                    del data_gpu, indices_gpu
                    cp.get_default_memory_pool().free_all_blocks()
            else:
                # Original processing for smaller chunks
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]

                data_gpu = cp.asarray(data_slice, dtype=cp.float64)
                indices_gpu = cp.asarray(indices_slice)

                # Accumulate the sum of squares for each gene
                cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                
                # Clean up
                del data_gpu, indices_gpu
                cp.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/2]: COMPLETE                                       ")

    # --- Final calculations on GPU ---
    if is_logged:
        raise NotImplementedError("Logged data variance calculation is not implemented for out-of-core.")
    else:
        # Variance of raw data: Var(X) = E[X^2] - E[X]^2
        mean_x_sq_gpu = sum_x_sq_gpu / nc
        mean_mu_gpu = tjs_gpu / nc
        my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
        
        # Calculate dispersion ('size')
        size_gpu = mean_mu_gpu**2 / (my_rowvar_gpu - mean_mu_gpu)
    
    max_size_val = cp.nanmax(size_gpu) * 10
    if cp.isnan(max_size_val): 
        max_size_val = 1000
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    # Move results to CPU
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
    Calculates the fit errors (gene_error, cell_error) for a given model.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFS() | FILE: {cleaned_filename}")

    # [GOVERNOR INTEGRATION] Adaptive chunk sizing
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=5.0, is_dense=True)

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    vals = fit['vals']
    size_coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    # Must use float64 for precision
    tjs_gpu = cp.asarray(vals['tjs'].values, dtype=cp.float64)
    tis_gpu = cp.asarray(vals['tis'].values, dtype=cp.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    # Calculate smoothed size
    mean_expression_gpu = tjs_gpu / nc
    log_mean_expression_gpu = cp.log(mean_expression_gpu)
    smoothed_size_gpu = cp.exp(size_coeffs[0] + size_coeffs[1] * log_mean_expression_gpu)

    # Initialize result arrays
    row_ps_gpu = cp.zeros(ng, dtype=cp.float64)
    col_ps_gpu = cp.zeros(nc, dtype=cp.float64)
    print("Phase [1/2]: COMPLETE")

    # --- Phase 2: Calculate Expected Dropouts ---
    print(f"Phase [2/2]: Calculating expected dropouts (Chunk: {chunk_size})...")
    
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]

        # Standard calculation without errstate
        mu_chunk_gpu = tjs_gpu[:, cp.newaxis] * tis_chunk_gpu[cp.newaxis, :] / total
        
        # Calculate p_is directly - CuPy handles overflow internally
        base = 1 + mu_chunk_gpu / smoothed_size_gpu[:, cp.newaxis]
        p_is_chunk_gpu = cp.power(base, -smoothed_size_gpu[:, cp.newaxis])
        
        # Handle any inf/nan values that might have occurred
        p_is_chunk_gpu = cp.nan_to_num(p_is_chunk_gpu, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Sum results
        row_ps_gpu += p_is_chunk_gpu.sum(axis=1)
        col_ps_gpu[i:end_col] = p_is_chunk_gpu.sum(axis=0)
        
        # Clean up
        del mu_chunk_gpu, p_is_chunk_gpu, base, tis_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()

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
    raw_filename: str, # Kept for API compatibility, but functionally we use cleaned_filename for indices
    cleaned_filename: str,
    stats: dict,
    fit_adjust: dict,
    chunk_size: int = None,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    OPTIMIZED VERSION (IN-MEMORY):
    - Eliminates the 46GB '_basic_norm.h5ad' temporary file.
    - Performs depth normalization and variance calculation on-the-fly in GPU VRAM.
    - PRESERVED SCIENTIFIC LOGIC: Var(X) = E[X^2] - (E[X])^2 on normalized data.
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    # [GOVERNOR] High multiplier (12.0) because we hold Raw + Norm + Square in VRAM
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=12.0, is_dense=False)

    # --- Phase 1: In-Memory "Basic Fit" (Normalization + Variance) ---
    print("Phase [1/3]: Calculating Basic Model (Depth-Normalized) variance on-the-fly...")
    
    # 1. Prepare Size Factors (CPU)
    tjs = stats['tjs'].values # Gene sums
    tis = stats['tis'].values # Cell sums
    nc, ng = stats['nc'], stats['ng']
    
    median_sum = np.median(tis[tis > 0])
    size_factors = np.ones_like(tis, dtype=np.float32)
    non_zero_mask = tis > 0
    size_factors[non_zero_mask] = tis[non_zero_mask] / median_sum
    
    # 2. Prepare GPU Arrays
    sum_x_sq_gpu = cp.zeros(ng, dtype=cp.float64)
    sum_x_gpu = cp.zeros(ng, dtype=cp.float64) # Need sum(x) to calc mean(x) for variance
    
    # 3. GPU Loop (Raw Data -> Normalize -> Accumulate)
    # CRITICAL: We read CLEANED_FILENAME to ensure indices match 'stats'
    with h5py.File(cleaned_filename, 'r') as f_in:
        h5_indptr = f_in['X']['indptr']
        h5_data = f_in['X']['data']
        h5_indices = f_in['X']['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [1/3]: Processing: {end_row} of {nc} cells.", end='\r')
            
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue

            # Load Raw Chunk
            data_gpu = cp.asarray(h5_data[start_idx:end_idx], dtype=cp.float32)
            indices_gpu = cp.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cp.asarray(h5_indptr[i:end_row + 1] - start_idx)

            # Expand Size Factors to match Data Structure
            nnz_in_chunk = indptr_gpu[-1].item()
            cell_boundary_markers = cp.zeros(nnz_in_chunk, dtype=cp.int32)
            if len(indptr_gpu) > 1:
                cell_boundary_markers[indptr_gpu[:-1]] = 1
            # row_indices maps every data point to its cell index (0 to chunk_size)
            row_indices = cp.cumsum(cell_boundary_markers, axis=0) - 1
            
            # Get size factors for this chunk
            sf_chunk = cp.asarray(size_factors[i:end_row])
            
            # --- THE MAGIC: On-the-Fly Normalization ---
            # data_norm = data_raw / size_factor
            data_gpu /= sf_chunk[row_indices]
            
            # Accumulate for Variance: E[X^2] and E[X]
            cp.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            cp.add.at(sum_x_gpu, indices_gpu, data_gpu)
            
            # Clean up VRAM
            del data_gpu, indices_gpu, indptr_gpu, row_indices, sf_chunk, cell_boundary_markers
            cp.get_default_memory_pool().free_all_blocks()

    print(f"Phase [1/3]: COMPLETE{' '*50}")

    # 4. Finalize Basic Statistics
    # Var(X) = E[X^2] - (E[X])^2
    mean_x_sq_gpu = sum_x_sq_gpu / nc
    mean_mu_gpu = sum_x_gpu / nc
    my_rowvar_gpu = mean_x_sq_gpu - mean_mu_gpu**2
    
    # Dispersion = Mean^2 / (Var - Mean)
    size_gpu = mean_mu_gpu**2 / (my_rowvar_gpu - mean_mu_gpu)
    
    # Safety Clamping
    max_size_val = cp.nanmax(size_gpu) * 10
    if cp.isnan(max_size_val): max_size_val = 1000
    size_gpu[cp.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
    size_gpu[size_gpu < 1e-10] = 1e-10
    
    # Construct "Basic Fit" Object
    fit_basic = {
        'sizes': pd.Series(size_gpu.get(), index=stats['tjs'].index),
        'vals': stats,
        'var_obs': pd.Series(my_rowvar_gpu.get(), index=stats['tjs'].index)
    }
    
    # --- Phase 2: Check Fit (Calculate Errors) ---
    print("Phase [2/3]: Evaluating fit errors on ORIGINAL data...")
    
    # Check Adjust (M3Drop) - uses its own governor
    check_adjust = NBumiCheckFitFSGPU(
        cleaned_filename, fit_adjust, suppress_plot=True
    )
    
    # Check Basic (Depth-Norm) - uses its own governor
    check_basic = NBumiCheckFitFSGPU(
        cleaned_filename, fit_basic, suppress_plot=True
    )
    print("Phase [2/3]: COMPLETE")

    # --- Phase 3: Plotting & Comparison ---
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
