import cupy
import numpy as np
import anndata
import h5py
import pandas as pd
import time
import os

from cupy.sparse import csr_matrix as cp_csr_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

import pickle


def ConvertDataSparseGPU(
    input_filename: str,
    output_filename: str,
    row_chunk_size: int = 5000
):
    """
    Performs out-of-core data cleaning on a standard (cell, gene) sparse
    .h5ad file. It correctly identifies and removes genes with zero counts
    across all cells.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparse() | FILE: {input_filename}")

    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        # Correctly unpack dimensions: (cells, genes)
        n_cells, n_genes = x_group_in.attrs['shape']

        # --- PASS 1: EFFICIENTLY FIND GENES TO KEEP ---
        print("Phase [1/2]: Identifying genes with non-zero counts...")
        genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
        
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        for i in range(0, n_cells, row_chunk_size):
            end_row = min(i + row_chunk_size, n_cells)
            print(f"Phase [1/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx:
                continue

            indices_slice = h5_indices[start_idx:end_idx]
            unique_in_chunk = np.unique(indices_slice)
            genes_to_keep_mask[unique_in_chunk] = True

        n_genes_to_keep = np.sum(genes_to_keep_mask)
        print(f"\nPhase [1/2]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        # --- PASS 2: WRITE CLEANED DATA IN CHUNKS ---
        print("Phase [2/2]: Rounding up decimals and saving filtered output to disk...")
        adata_meta = anndata.read_h5ad(input_filename, backed='r')
        # Filter the .var dataframe (gene metadata) using our mask
        filtered_var_df = adata_meta.var[genes_to_keep_mask]
        
        # Create the new AnnData object with all original cells but only the kept genes
        adata_out_template = anndata.AnnData(obs=adata_meta.obs, var=filtered_var_df, uns=adata_meta.uns)
        # Write the metadata shell to the new file
        adata_out_template.write_h5ad(output_filename, compression="gzip")

        # Re-open the file to write the large data matrix chunk by chunk
        with h5py.File(output_filename, 'a') as f_out:
            # We are overwriting the empty 'X' group created by .write_h5ad
            if 'X' in f_out:
                del f_out['X']
            x_group_out = f_out.create_group('X')

            # Initialize datasets for the new sparse matrix
            out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
            out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
            out_indptr = x_group_out.create_dataset('indptr', shape=(n_cells + 1,), dtype='int64')
            out_indptr[0] = 0
            current_nnz = 0

            h5_data = x_group_in['data']

            # Process the matrix cell-by-cell (row-by-row)
            for i in range(0, n_cells, row_chunk_size):
                end_row = min(i + row_chunk_size, n_cells)
                print(f"Phase [2/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

                # Read a chunk of rows from the original file
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Create a sparse matrix for the chunk
                chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, n_genes))

                # Filter the chunk by removing unwanted columns (genes)
                filtered_chunk = chunk[:, genes_to_keep_mask]
                filtered_chunk.data = np.ceil(filtered_chunk.data).astype('float32')

                # Append the filtered data to the new file
                out_data.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_data[current_nnz:] = filtered_chunk.data

                out_indices.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_indices[current_nnz:] = filtered_chunk.indices

                # Append the new row pointers
                new_indptr_list = filtered_chunk.indptr[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list
                
                current_nnz += filtered_chunk.nnz

            # Set the final attributes for the new sparse matrix
            x_group_out.attrs['encoding-type'] = 'csr_matrix'
            x_group_out.attrs['encoding-version'] = '0.1.0'
            # Write the correct final shape: (n_cells, n_genes_to_keep)
            x_group_out.attrs['shape'] = np.array([n_cells, n_genes_to_keep], dtype='int64')
        print(f"\nPhase [2/2]: COMPLETE | Output: {output_filename} {' ' * 50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

def hidden_calc_valsGPU(
    filename: str,
    chunk_size: int = 5000
) -> dict:
    """
    Calculates key statistics from a large, sparse (cell, gene) .h5ad file
    using a memory-safe, GPU-accelerated, single-pass algorithm.

    Args:
        filename (str): Path to the cleaned sparse .h5ad file.
        chunk_size (int): The number of rows (cells) to process at a time.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")

    # --- 1. Initialization ---
    adata_meta = anndata.read_h5ad(filename, backed='r')
    print("Phase [1/3]: Finding nc and ng...")
    nc, ng = adata_meta.shape # nc = cells (rows), ng = genes (columns)
    print(f"Phase [1/3]: COMPLETE")

    # --- CPU arrays for cell (row) stats ---
    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')

    # --- GPU arrays for gene (column) stats ---
    tjs_gpu = cupy.zeros(ng, dtype=cupy.float32)
    gene_non_zeros_gpu = cupy.zeros(ng, dtype=cupy.int32)

    # --- 2. Single Pass Calculation ---
    print("Phase [2/3]: Calculating tis and tjs...")
    with h5py.File(filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')

            # Read CSR data for the current chunk from disk
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            data_slice = h5_data[start_idx:end_idx]
            indices_slice = h5_indices[start_idx:end_idx]
            indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

            # --- Move chunk to GPU ---
            # By calling .copy(), we ensure the memory buffer sent to CuPy is clean
            # and not subject to pre-existing memory maps from h5py.
            data_gpu = cupy.asarray(data_slice.copy(), dtype=cupy.float32)
            indices_gpu = cupy.asarray(indices_slice.copy())
            indptr_gpu = cupy.asarray(indptr_slice.copy())

            # Create a CuPy sparse matrix for the chunk
            chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(end_row-i, ng))

            # --- CALCULATE CELL (ROW) STATS ON GPU ---
            tis[i:end_row] = chunk_gpu.sum(axis=1).get().flatten()
            cell_non_zeros_chunk = cupy.diff(indptr_gpu)
            # ▼▼▼ THIS IS THE CORRECTED LINE ▼▼▼
            cell_non_zeros[i:end_row] = cell_non_zeros_chunk.get()

            # --- UPDATE GENE (COLUMN) STATS ON GPU ---
            cupy.add.at(tjs_gpu, indices_gpu, data_gpu)
            
            unique_indices_gpu, counts_gpu = cupy.unique(indices_gpu, return_counts=True)
            cupy.add.at(gene_non_zeros_gpu, unique_indices_gpu, counts_gpu)

    # --- 3. Finalization ---
    tjs = cupy.asnumpy(tjs_gpu)
    gene_non_zeros = cupy.asnumpy(gene_non_zeros_gpu)
    print(f"Phase [2/3]: COMPLETE{' ' * 50}")

    print("Phase [3/3]: Calculating dis, djs, and total...")
    dis = ng - cell_non_zeros
    djs = nc - gene_non_zeros
    total = tjs.sum()
    print("Phase [3/3]: COMPLETE")


    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return {
        "tis": pd.Series(tis, index=adata_meta.obs.index),
        "tjs": pd.Series(tjs, index=adata_meta.var.index),
        "dis": pd.Series(dis, index=adata_meta.obs.index),
        "djs": pd.Series(djs, index=adata_meta.var.index),
        "total": total,
        "nc": nc,
        "ng": ng
    }

def NBumiFitModelGPU(
    cleaned_filename: str,
    stats: dict,
    chunk_size: int = 5000
) -> dict:

    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    nc, ng = stats['nc'], stats['ng']
    total = stats['total']
    
    # Everything on GPU from the start
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    
    sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
    sum_2xmu_gpu = cupy.zeros(ng, dtype=cupy.float64)
    
    print("Phase [1/3]: Pre-calculating sum of squared expectations...")
    sum_tis_sq_gpu = cupy.sum(tis_gpu**2)
    sum_mu_sq_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
    print("Phase [1/3]: COMPLETE")
    
    print("Phase [2/3]: Calculating variance components from data chunks...")
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']
        
        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')
            
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx:
                continue
            
            # Load ALL data for this chunk at once - let GPU handle it
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])
            
            # GPU-accelerated operations
            cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            # ▼▼▼ CORRECTED CODE BLOCK START ▼▼▼
            # Efficient cell index calculation using a robust cumsum method
            nnz_in_chunk = indptr_gpu[-1].item()
            # Create an array that marks the start of each new cell's data
            cell_boundary_markers = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
            # indptr_gpu[:-1] gives the start indices of each cell's non-zero elements
            if len(indptr_gpu) > 1:
                cell_boundary_markers[indptr_gpu[:-1]] = 1
            # A cumulative sum will create a stepped array where each step corresponds to a cell index
            cell_indices_chunk = cupy.cumsum(cell_boundary_markers, axis=0) - 1
            cell_indices_gpu = cell_indices_chunk + i
            # ▲▲▲ CORRECTED CODE BLOCK END ▲▲▲
            
            # Vectorized calculation
            tis_per_nz = tis_gpu[cell_indices_gpu]
            tjs_per_nz = tjs_gpu[indices_gpu]
            term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
            cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
            
            # Clean up
            del data_gpu, indices_gpu, indptr_gpu, cell_indices_gpu
            del tis_per_nz, tjs_per_nz, term_vals
            
            # Only clear memory if actually needed
            if i % (chunk_size * 10) == 0:
                cupy.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/3]: COMPLETE {' ' * 50}")
    
    print("Phase [3/3]: Finalizing dispersion and variance calculations...")
    sum_sq_dev_gpu = sum_x_sq_gpu - sum_2xmu_gpu + sum_mu_sq_gpu
    var_obs_gpu = sum_sq_dev_gpu / (nc - 1)
    
    sizes_gpu = cupy.full(ng, 10000.0)
    numerator_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
    denominator_gpu = sum_sq_dev_gpu - tjs_gpu
    stable_mask = denominator_gpu > 1e-6
    sizes_gpu[stable_mask] = numerator_gpu[stable_mask] / denominator_gpu[stable_mask]
    sizes_gpu[sizes_gpu <= 0] = 10000.0
    
    var_obs_cpu = var_obs_gpu.get()
    sizes_cpu = sizes_gpu.get()
    print("Phase [3/3]: COMPLETE")
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    
    return {
        'var_obs': pd.Series(var_obs_cpu, index=stats['tjs'].index),
        'sizes': pd.Series(sizes_cpu, index=stats['tjs'].index),
        'vals': stats
    }

def NBumiFitDispVsMeanGPU(fit, suppress_plot=True):
    """
    Fits a linear model to the log-dispersion vs log-mean of gene expression.
    """
    vals = fit['vals']
    size_g = fit['sizes'].values
    tjs = vals['tjs'].values

    # --- CORRECTED LINES START HERE ---
    # Calculate mean expression before filtering.
    mean_expression = tjs / vals['nc']
    
    # Create a boolean mask to filter for valid data points.
    # 1. size_g must be finite and within a reasonable range (e.g., less than a large number like 1e5 or 1e6 to avoid the artifact cluster).
    # 2. mean_expression must be greater than the specified threshold (1e-3) to remove low-expression noise.
    # 3. size_g must be positive.
    forfit = (np.isfinite(size_g)) & (size_g < 1e6) & (mean_expression > 1e-3) & (size_g > 0)
    
    # Further filter for highly expressed genes if there are enough points.
    # This part of the logic is from the original M3Drop implementation and is optional but can improve the fit.
    log2_mean_expr = np.log2(mean_expression, where=(mean_expression > 0))
    higher = log2_mean_expr > 4
    if np.sum(higher & forfit) > 2000:
        forfit = higher & forfit
    # --- CORRECTED LINES END HERE ---

    # Prepare data for regression using the corrected filtered NumPy arrays
    y = np.log(size_g[forfit])
    x = np.log(mean_expression[forfit])
    
    # Add a constant for the intercept term
    X = sm.add_constant(x)

    # Fit the linear model
    model = sm.OLS(y, X).fit()
    
    # Optionally, create the diagnostic plot
    if not suppress_plot:
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.5, label="Data Points")
        plt.plot(x, model.fittedvalues, color='red', label='Regression Fit')
        plt.title('Dispersion vs. Mean Expression')
        plt.xlabel("Log Mean Expression")
        plt.ylabel("Log Size (Dispersion)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Return the regression coefficients
    return model.params

def NBumiFeatureSelectionHighVarGPU(fit: dict) -> pd.DataFrame:
    """
    Selects features (genes) with higher variance than expected.

    This function is already memory-efficient as it only operates on the
    small summary arrays contained within the 'fit' object.

    Args:
        fit (dict): The 'fit' object from NBumiFitModel.

    Returns:
        pd.DataFrame: A DataFrame of genes sorted by their residual,
                      indicating how much more variable they are than expected.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVar()")

    # This is a fast, in-memory call
    print("Phase [1/1]: Calculating residuals for high variance selection...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    # --- 2. Perform Calculations (CPU is optimal) ---
    mean_expression = vals['tjs'].values / vals['nc']

    with np.errstate(divide='ignore', invalid='ignore'):
        log_mean_expression = np.log(mean_expression)
        log_mean_expression[np.isneginf(log_mean_expression)] = 0
        exp_size = np.exp(coeffs[0] + coeffs[1] * log_mean_expression)

    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(fit['sizes'].values) - np.log(exp_size)

    results_df = pd.DataFrame({
        'Gene': fit['sizes'].index,
        'Residual': res
    })

    final_table = results_df.sort_values(by='Residual', ascending=True)
    print("Phase [1/1]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")

    return final_table

def NBumiFeatureSelectionCombinedDropGPU(
    fit: dict,
    cleaned_filename: str,
    chunk_size: int = 5000,
    method="fdr_bh",
    qval_thresh=0.05
) -> pd.DataFrame:
    """
    Selects features with a significantly higher dropout rate than expected,
    using an out-of-core, GPU-accelerated approach.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")

    # --- 1. Initialization and In-Memory Prep ---
    print("Phase [1/3]: Initializing arrays and calculating expected dispersion...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    tjs_gpu = cupy.asarray(vals['tjs'].values)
    tis_gpu = cupy.asarray(vals['tis'].values)
    total = vals['total']
    nc = vals['nc']
    ng = vals['ng']

    mean_expression_cpu = vals['tjs'].values / nc
    with np.errstate(divide='ignore'):
        exp_size_cpu = np.exp(coeffs[0] + coeffs[1] * np.log(mean_expression_cpu))

    exp_size_gpu = cupy.asarray(exp_size_cpu)

    p_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    p_var_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    print("Phase [1/3]: COMPLETE")

    # --- 2. Calculate Expected Dropouts in Chunks on the GPU ---
    print("Phase [2/3]: Calculating expected dropout sums from data chunks...")
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]

        mu_chunk_gpu = tjs_gpu[:, cupy.newaxis] * tis_chunk_gpu[cupy.newaxis, :] / total
        
        p_is_chunk_gpu = cupy.power(1 + mu_chunk_gpu / exp_size_gpu[:, cupy.newaxis], -exp_size_gpu[:, cupy.newaxis])
        
        p_var_is_chunk_gpu = p_is_chunk_gpu * (1 - p_is_chunk_gpu)
        
        p_sum_gpu += p_is_chunk_gpu.sum(axis=1)
        p_var_sum_gpu += p_var_is_chunk_gpu.sum(axis=1)
    
    print(f"Phase [2/3]: COMPLETE {' ' * 50}")

    # --- 3. Final Statistical Test on the CPU ---
    print("Phase [3/3]: Performing statistical test and adjusting p-values...")
    
    p_sum_cpu = p_sum_gpu.get()
    p_var_sum_cpu = p_var_sum_gpu.get()

    droprate_exp = p_sum_cpu / nc
    droprate_exp_err = np.sqrt(p_var_sum_cpu / (nc**2))

    droprate_obs = vals['djs'].values / nc
    
    diff = droprate_obs - droprate_exp
    combined_err = np.sqrt(droprate_exp_err**2 + (droprate_obs * (1 - droprate_obs) / nc))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Zed = diff / combined_err
    
    pvalue = norm.sf(Zed)

    results_df = pd.DataFrame({
        'Gene': vals['tjs'].index,
        'p.value': pvalue,
        'effect_size': diff
    })
    results_df = results_df.sort_values(by='p.value')

    qval = multipletests(results_df['p.value'].fillna(1), method=method)[1]
    results_df['q.value'] = qval
    final_table = results_df[results_df['q.value'] < qval_thresh]
    print("Phase [3/3]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    
    return final_table[['Gene', 'effect_size', 'p.value', 'q.value']]

def NBumiCombinedDropVolcanoGPU(
    results_df: pd.DataFrame,
    qval_thresh: float = 0.05,
    effect_size_thresh: float = 0.25,
    top_n_genes: int = 10,
    suppress_plot: bool = False,
    plot_filename: str = None
):
    """
    Generates a volcano plot from the results of feature selection.

    Args:
        results_df (pd.DataFrame): The DataFrame from NBumiFeatureSelectionCombinedDrop.
        qval_thresh (float): The q-value threshold for significance.
        effect_size_thresh (float): The effect size threshold for significance.
        top_n_genes (int): The number of top genes to label on the plot.
        suppress_plot (bool): If True, the plot will not be displayed on screen.
        plot_filename (str, optional): Path to save the plot. If None, plot is not saved.

    Returns:
        matplotlib.axes.Axes: The Axes object for the plot.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcano()")

    # --- Phase 1: Data Preparation ---
    print("Phase [1/1]: Preparing data for visualization...")
    
    df = results_df.copy()

    # Calculate -log10(q-value), handling cases where q-value is 0
    non_zero_min = df[df['q.value'] > 0]['q.value'].min()
    df['q.value'] = df['q.value'].replace(0, non_zero_min)
    df['-log10_qval'] = -np.log10(df['q.value'])

    # Categorize genes for coloring
    df['color'] = 'grey'
    sig_up = (df['q.value'] < qval_thresh) & (df['effect_size'] > effect_size_thresh)
    sig_down = (df['q.value'] < qval_thresh) & (df['effect_size'] < -effect_size_thresh)
    df.loc[sig_up, 'color'] = 'red'
    df.loc[sig_down, 'color'] = 'blue'

    print("Phase [1/1]: COMPLETE")

    # --- Phase 2: Plot Generation ---
    print("Phase [2/2]: Generating plot...")

    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot
    plt.scatter(df['effect_size'], df['-log10_qval'], c=df['color'], s=10, alpha=0.6)

    # Add threshold lines
    plt.axvline(x=effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axvline(x=-effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axhline(y=-np.log10(qval_thresh), linestyle='--', color='grey', linewidth=0.8)

    # Label top genes
    top_genes = df.nsmallest(top_n_genes, 'q.value')
    for i, row in top_genes.iterrows():
        plt.text(row['effect_size'], row['-log10_qval'], row['Gene'],
                 fontsize=9, ha='left', va='bottom', alpha=0.8)

    # Final styling
    plt.title('Volcano Plot of Dropout Feature Selection')
    plt.xlabel('Effect Size (Observed - Expected Dropout Rate)')
    plt.ylabel('-log10 (Adjusted p-value)')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    ax = plt.gca() # Get current axes

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"STATUS: Volcano plot saved to '{plot_filename}'")

    if not suppress_plot:
        plt.show()

    plt.close()
    
    print("Phase [2/2]: COMPLETE")
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    return ax