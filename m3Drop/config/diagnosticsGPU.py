from coreGPU import hidden_calc_valsGPU, NBumiFitModelGPU, NBumiFitDispVsMeanGPU
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
    chunk_size: int = 5000
) -> dict:
    """
    Fits a simpler, unadjusted NB model out-of-core using a GPU-accelerated
    algorithm. Designed to work with a standard (cell, gene) sparse matrix.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModel() | FILE: {cleaned_filename}")

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    tjs = stats['tjs'].values
    nc, ng = stats['nc'], stats['ng']

    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
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
                    
                    data_gpu = cupy.asarray(data_slice, dtype=cupy.float64)
                    indices_gpu = cupy.asarray(indices_slice)
                    
                    # Accumulate the sum of squares for each gene
                    cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                    
                    # Free GPU memory
                    del data_gpu, indices_gpu
                    cupy.get_default_memory_pool().free_all_blocks()
            else:
                # Original processing for smaller chunks
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]

                data_gpu = cupy.asarray(data_slice, dtype=cupy.float64)
                indices_gpu = cupy.asarray(indices_slice)

                # Accumulate the sum of squares for each gene
                cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                
                # Clean up
                del data_gpu, indices_gpu
                cupy.get_default_memory_pool().free_all_blocks()
    
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
    
    max_size_val = cupy.nanmax(size_gpu) * 10
    if cupy.isnan(max_size_val): 
        max_size_val = 1000
    size_gpu[cupy.isnan(size_gpu) | (size_gpu <= 0)] = max_size_val
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
    chunk_size: int = 5000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    FIXED VERSION - No cupy.errstate, proper GPU computation.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFS() | FILE: {cleaned_filename}")

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and arrays on GPU...")
    vals = fit['vals']
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

    # --- Phase 2: Calculate Expected Dropouts ---
    print("Phase [2/2]: Calculating expected dropouts from data chunks...")
    
    # Determine optimal chunk size
    mempool = cupy.get_default_memory_pool()
    free_mem_mb = cupy.cuda.Device().mem_info[0] / (1024**2)
    
    # Estimate memory per cell: 8 bytes * ng * 2 arrays
    bytes_per_cell = ng * 8 * 2
    optimal_chunk = min(chunk_size, int(free_mem_mb * 0.4 * 1024**2 / bytes_per_cell))
    optimal_chunk = max(100, optimal_chunk)
    
    print(f"  Using adaptive chunk size: {optimal_chunk} (original: {chunk_size})")
    
    for i in range(0, nc, optimal_chunk):
        end_col = min(i + optimal_chunk, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]

        # Standard calculation without errstate
        mu_chunk_gpu = tjs_gpu[:, cupy.newaxis] * tis_chunk_gpu[cupy.newaxis, :] / total
        
        # Calculate p_is directly - CuPy handles overflow internally
        base = 1 + mu_chunk_gpu / smoothed_size_gpu[:, cupy.newaxis]
        p_is_chunk_gpu = cupy.power(base, -smoothed_size_gpu[:, cupy.newaxis])
        
        # Handle any inf/nan values that might have occurred
        p_is_chunk_gpu = cupy.nan_to_num(p_is_chunk_gpu, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Sum results
        row_ps_gpu += p_is_chunk_gpu.sum(axis=1)
        col_ps_gpu[i:end_col] = p_is_chunk_gpu.sum(axis=0)
        
        # Clean up
        del mu_chunk_gpu, p_is_chunk_gpu, base, tis_chunk_gpu
        
        # Periodic memory cleanup
        if (i // optimal_chunk) % 10 == 0:
            mempool.free_all_blocks()

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
    chunk_size: int = 5000,
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    OPTIMIZED VERSION - Faster normalization and sparse matrix writing.
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    # --- Phase 1: OPTIMIZED Normalization ---
    print("Phase [1/4]: Creating temporary 'basic' normalized data file...")
    basic_norm_filename = cleaned_filename.replace('.h5ad', '_basic_norm.h5ad')

    # Read metadata. In 'backed' mode, this keeps a file handle open.
    adata_meta = anndata.read_h5ad(cleaned_filename, backed='r')
    nc, ng = adata_meta.shape
    obs_df = adata_meta.obs.copy()
    var_df = adata_meta.var.copy()
    
    cell_sums = stats['tis'].values
    median_sum = np.median(cell_sums[cell_sums > 0])
    
    # Avoid division by zero for cells with zero counts
    size_factors = np.ones_like(cell_sums, dtype=np.float32)
    non_zero_mask = cell_sums > 0
    size_factors[non_zero_mask] = cell_sums[non_zero_mask] / median_sum

    adata_out = anndata.AnnData(obs=obs_df, var=var_df)
    adata_out.write_h5ad(basic_norm_filename, compression="gzip")

    with h5py.File(basic_norm_filename, 'a') as f_out:
        if 'X' in f_out:
            del f_out['X']
        x_group_out = f_out.create_group('X')
        x_group_out.attrs['encoding-type'] = 'csr_matrix'
        x_group_out.attrs['encoding-version'] = '0.1.0'
        x_group_out.attrs['shape'] = np.array([nc, ng], dtype='int64')

        out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
        out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
        out_indptr = x_group_out.create_dataset('indptr', shape=(nc + 1,), dtype='int64')
        out_indptr[0] = 0
        current_nnz = 0

        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [1/4]: Normalizing: {end_row} of {nc} cells.", end='\r')
                
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                if start_idx == end_idx:
                    out_indptr[i + 1 : end_row + 1] = current_nnz
                    continue

                # Read data for the chunk
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row + 1] - start_idx
                
                # Move to GPU for fast normalization
                data_gpu = cupy.asarray(data_slice.copy(), dtype=cupy.float32)
                
                indptr_gpu = cupy.asarray(indptr_slice.copy())
                nnz_in_chunk = indptr_gpu[-1].item()
                cell_boundary_markers = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
                if len(indptr_gpu) > 1:
                    cell_boundary_markers[indptr_gpu[:-1]] = 1
                row_indices = cupy.cumsum(cell_boundary_markers, axis=0) - 1
                
                size_factors_for_chunk = cupy.asarray(size_factors[i:end_row])
                
                data_gpu /= size_factors_for_chunk[row_indices]
                
                data_cpu = np.round(data_gpu.get())

                num_cells_in_chunk = end_row - i
                chunk_sp = sp_csr_matrix((data_cpu, indices_slice, indptr_slice), 
                                         shape=(num_cells_in_chunk, ng))

                nnz_chunk = chunk_sp.nnz
                out_data.resize(current_nnz + nnz_chunk, axis=0)
                out_data[current_nnz:] = chunk_sp.data

                out_indices.resize(current_nnz + nnz_chunk, axis=0)
                out_indices[current_nnz:] = chunk_sp.indices

                new_indptr_list = chunk_sp.indptr[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list
                
                current_nnz += nnz_chunk

                del data_gpu, row_indices, size_factors_for_chunk, indptr_gpu
                cupy.get_default_memory_pool().free_all_blocks()

    print(f"Phase [1/4]: COMPLETE{' '*50}")

    print("Phase [2/4]: Fitting Basic Model on normalized data...")
    stats_basic = hidden_calc_valsGPU(basic_norm_filename, chunk_size=chunk_size)
    fit_basic = NBumiFitBasicModelGPU(basic_norm_filename, stats_basic, chunk_size=chunk_size)
    print("Phase [2/4]: COMPLETE")
    
    print("Phase [3/4]: Evaluating fits of both models on ORIGINAL data...")
    check_adjust = NBumiCheckFitFSGPU(cleaned_filename, fit_adjust, suppress_plot=True, chunk_size=chunk_size)
    
    fit_basic_for_eval = {
        'sizes': fit_basic['sizes'],
        'vals': stats,
        'var_obs': fit_basic['var_obs']
    }
    check_basic = NBumiCheckFitFSGPU(cleaned_filename, fit_basic_for_eval, suppress_plot=True, chunk_size=chunk_size)
    print("Phase [3/4]: COMPLETE")

    print("Phase [4/4]: Generating final comparison...")
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
    print("Phase [4/4]: COMPLETE")

    pipeline_end_time = time.time()
    
    # --- ADD THIS LINE TO FIX THE ERROR ---
    adata_meta.file.close() # Explicitly close the file handle
    
    os.remove(basic_norm_filename)
    print(f"STATUS: Temporary file '{basic_norm_filename}' removed.")
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

    Args:
        fit (dict): The 'fit' object from NBumiFitModelGPU.
        suppress_plot (bool): If True, the plot will not be displayed on screen.
        plot_filename (str, optional): Path to save the plot. If None, not saved.
    """
    print("FUNCTION: NBumiPlotDispVsMean()")

    # --- 1. Extract data and regression coefficients ---
    mean_expression = fit['vals']['tjs'].values / fit['vals']['nc']
    sizes = fit['sizes'].values
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    intercept, slope = coeffs[0], coeffs[1]

    # --- 2. Calculate the fitted line for plotting ---
    # Create a smooth, continuous line using the regression coefficients
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