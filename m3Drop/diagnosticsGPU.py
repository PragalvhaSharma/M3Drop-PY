from .coreGPU import hidden_calc_valsGPU, NBumiFitModelGPU, NBumiFitDispVsMeanGPU, get_optimal_chunk_size
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
    is_logged=False
) -> dict:
    """
    Fits a simpler, unadjusted NB model out-of-core using a GPU-accelerated
    algorithm. Designed to work with a standard (cell, gene) sparse matrix.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitBasicModel() | FILE: {cleaned_filename}")

    # --- HANDSHAKE ---
    # Multiplier 4.0: Sparse variance calculation (Sum of Squares + Expectation arrays)
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=4.0, is_dense=False)

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
            
            # Load chunk
            data_slice = h5_data[start_idx:end_idx]
            indices_slice = h5_indices[start_idx:end_idx]

            data_gpu = cupy.asarray(data_slice, dtype=cupy.float64)
            indices_gpu = cupy.asarray(indices_slice)

            # Accumulate the sum of squares for each gene
            cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            # Clean up
            del data_gpu, indices_gpu
            cupy.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/2]: COMPLETE{' ' * 50}")

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
    suppress_plot=False,
    plot_filename=None
) -> dict:
    """
    Calculates expected dropout rates vs observed dropout rates.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCheckFitFS() | FILE: {cleaned_filename}")

    # --- HANDSHAKE ---
    # Multiplier 3.0: 3x Full Dense Arrays. 
    # CRITICAL: is_dense=True. This function performs a dense broadcast (outer product).
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=3.0, is_dense=True)

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
    
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/2]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]

        # BROADCAST OPERATION: Creates Dense Matrix (ng, chunk_size)
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
    """
    pipeline_start_time = time.time()
    print(f"FUNCTION: NBumiCompareModels() | Comparing models for {cleaned_filename}")

    # --- HANDSHAKE ---
    # Multiplier 2.5: Normalization loop (Read, Divide, Write). Sparse I/O bound.
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=2.5, is_dense=False)

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
                
                # Load block
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Convert to GPU
                data_gpu = cupy.asarray(data_slice, dtype=cupy.float32)
                
                # Apply Size Factors (Slice appropriate size factors for this chunk)
                # Note: We need to expand size_factors to match the data structure (CSR)
                # Efficient approach: Repeat size factor for each non-zero element in the row
                
                # 1. Get size factors for this chunk of cells
                sf_chunk = size_factors[i:end_row]
                sf_chunk_gpu = cupy.asarray(sf_chunk, dtype=cupy.float32)
                
                # 2. Expand to match data array
                # Calculate row lengths to repeat
                row_lens = cupy.diff(cupy.asarray(indptr_slice))
                sf_expanded_gpu = cupy.repeat(sf_chunk_gpu, row_lens)
                
                # 3. Divide and Round
                data_gpu = data_gpu / sf_expanded_gpu
                data_gpu = cupy.rint(data_gpu)

                # Write back (CPU)
                filtered_data = data_gpu.get()
                
                # Append to H5
                out_data.resize(current_nnz + len(filtered_data), axis=0)
                out_data[current_nnz:] = filtered_data

                out_indices.resize(current_nnz + len(filtered_data), axis=0)
                out_indices[current_nnz:] = indices_slice

                new_indptr_list = indptr_slice[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list
                
                current_nnz += len(filtered_data)
                
                # Cleanup
                del data_gpu, sf_chunk_gpu, sf_expanded_gpu
                cupy.get_default_memory_pool().free_all_blocks()

    print(f"\nPhase [1/4]: COMPLETE | Saved: {basic_norm_filename}{' '*20}")

    # --- Phase 2: Fit Basic Model ---
    print("Phase [2/4]: Fitting Basic Model...")
    fit_basic = NBumiFitBasicModelGPU(
        cleaned_filename=basic_norm_filename,
        stats=stats
    )
    print("Phase [2/4]: COMPLETE")
    
    # --- Phase 3: Evaluate & Compare ---
    print("Phase [3/4]: Evaluating fits...")
    
    # Check Adjusted Model
    check_adjust = NBumiCheckFitFSGPU(cleaned_filename, fit_adjust, suppress_plot=True)
    
    # Check Basic Model
    # Note: We must use the basic_norm file for the basic fit check to match the fit derivation
    check_basic = NBumiCheckFitFSGPU(basic_norm_filename, fit_basic, suppress_plot=True)

    print("Phase [4/4]: Generating comparison plot...")
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
    
    # Cleanup temp file
    if os.path.exists(basic_norm_filename):
        os.remove(basic_norm_filename)
        print(f"STATUS: Removed temporary file '{basic_norm_filename}'")

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
