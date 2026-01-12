import time
import psutil
import h5py
import numpy as np
import anndata
import pandas as pd
import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix as sp_csr_matrix

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# Safe Import for GPU
try:
    import cupy
    import cupy.sparse as csp
    from cupy.sparse import csr_matrix as cp_csr_matrix
    HAS_GPU = True
    # Pin memory for faster CPU->GPU transfer if needed, though high-level API handles this mostly.
except ImportError:
    cupy = None
    HAS_GPU = False
    print(" [WARNING] CuPy not found. GPU acceleration disabled.")

# --- MEMORY MANAGEMENT UTILS ---

def get_io_chunk_size(filename: str, target_mb: int = 512) -> int:
    """
    Calculates a large chunk size for Disk I/O to satisfy parallel file systems (Lustre/GPFS).
    Targeting ~512MB per read ensures high throughput.
    """
    with h5py.File(filename, 'r') as f:
        x_group = f['X']
        shape = x_group.attrs['shape']
        n_cells, n_genes = shape[0], shape[1]
        
        # Estimate sparsity (default to 10% if unknown)
        if 'indptr' in x_group:
            nnz = x_group['indptr'][-1]
            bytes_per_row = (nnz / n_cells) * 12 # 4 bytes data + 4 bytes index + overhead
        else:
            # Fallback for dense-like estimation
            bytes_per_row = n_genes * 4 * 0.1 
            
    if bytes_per_row < 1: bytes_per_row = 1
    
    # Calculate rows to hit target MB
    target_bytes = target_mb * 1024 * 1024
    chunk_size = int(target_bytes / bytes_per_row)
    
    # Floor/Ceil safety
    chunk_size = max(5000, min(chunk_size, 200000))
    
    return chunk_size

def get_compute_tile_size(n_genes: int, vram_limit_gb: float = 9.0) -> int:
    """
    Calculates the max 'Micro-Batch' size for dense GPU operations 
    given the VRAM constraints (default 9GB safe limit for 10GB slice).
    
    Math: We need roughly 3 dense matrices of shape (tile, genes) in float64.
    Size = 3 * tile * genes * 8 bytes.
    """
    bytes_per_element = 8 # float64
    matrices_needed = 3   # mu, p_is, p_var (approx)
    
    bytes_per_row_dense = n_genes * bytes_per_element * matrices_needed
    
    total_vram_bytes = vram_limit_gb * 1024**3
    tile_size = int(total_vram_bytes / bytes_per_row_dense)
    
    # Safety margin: Cap at 20k to prevent timeouts/watchdogs
    return max(100, min(tile_size, 20000))

# --- CORE FUNCTIONS ---

def ConvertDataSparseGPU(input_filename: str, output_filename: str):
    """
    GPU-Accelerated Cleaning with High-Throughput I/O.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseGPU() | FILE: {input_filename}")

    # 1. READ IO: Large chunks for speed
    read_chunk_size = get_io_chunk_size(input_filename, target_mb=512)
    
    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        n_cells, n_genes = x_group_in.attrs['shape']
        
        print(f"Phase [1/2]: Identifying genes (Read Chunk: {read_chunk_size})...")
        
        if HAS_GPU:
            genes_to_keep_mask = cupy.zeros(n_genes, dtype=bool)
        else:
            genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
            
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        # READ LOOP
        for i in range(0, n_cells, read_chunk_size):
            end_row = min(i + read_chunk_size, n_cells)
            print(f"  > Processing rows {i} to {end_row}...", end='\r')

            # Load indices only (Fastest I/O)
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue
            
            indices_cpu = h5_indices[start_idx:end_idx]

            if HAS_GPU:
                # Move to GPU for fast `unique`
                indices_gpu = cupy.asarray(indices_cpu)
                unique_gpu = cupy.unique(indices_gpu)
                genes_to_keep_mask[unique_gpu] = True
                
                # Explicit cleanup
                del indices_gpu, unique_gpu
                if i % (read_chunk_size * 5) == 0:
                    cupy.get_default_memory_pool().free_all_blocks()
            else:
                unique_cpu = np.unique(indices_cpu)
                genes_to_keep_mask[unique_cpu] = True

        # Finalize Mask
        if HAS_GPU:
            genes_to_keep_mask_cpu = cupy.asnumpy(genes_to_keep_mask)
        else:
            genes_to_keep_mask_cpu = genes_to_keep_mask

        n_genes_to_keep = np.sum(genes_to_keep_mask_cpu)
        print(f"\nPhase [1/2]: COMPLETE | Retained {n_genes_to_keep} / {n_genes} genes.")

        # 2. WRITE IO: Large chunks to prevent Lustre stalls
        # Using same chunk size as read is usually safe for sparse-to-sparse write
        write_chunk_size = read_chunk_size 
        
        print(f"Phase [2/2]: Filtering & Saving (Write Chunk: {write_chunk_size})...")
        
        adata_meta = anndata.read_h5ad(input_filename, backed='r')
        filtered_var_df = adata_meta.var[genes_to_keep_mask_cpu]
        
        # Create output skeleton
        adata_out_template = anndata.AnnData(obs=adata_meta.obs, var=filtered_var_df, uns=adata_meta.uns)
        adata_out_template.write_h5ad(output_filename, compression="gzip")

        with h5py.File(output_filename, 'a') as f_out:
            if 'X' in f_out: del f_out['X']
            x_group_out = f_out.create_group('X')

            # Resizeable datasets
            out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
            out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
            out_indptr = x_group_out.create_dataset('indptr', shape=(n_cells + 1,), dtype='int64')
            out_indptr[0] = 0
            
            current_nnz = 0
            h5_data = x_group_in['data']

            for i in range(0, n_cells, write_chunk_size):
                end_row = min(i + write_chunk_size, n_cells)
                print(f"  > Processing rows {i} to {end_row}...", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                
                # Read Block
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Construct Sparse Matrix (CPU RAM is cheap, do construction here)
                chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, n_genes))
                
                # Filter Columns
                filtered_chunk = chunk[:, genes_to_keep_mask_cpu]
                
                # "Ceil" operation (Rounding up)
                # Note: Doing this on CPU is likely fine as it's O(NNZ), but could be GPU'd. 
                # Given I/O is bottleneck, CPU is safer for stability.
                filtered_chunk.data = np.ceil(filtered_chunk.data).astype('float32')

                # Append to H5
                out_data.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_data[current_nnz:] = filtered_chunk.data

                out_indices.resize(current_nnz + filtered_chunk.nnz, axis=0)
                out_indices[current_nnz:] = filtered_chunk.indices

                new_indptr_list = filtered_chunk.indptr[1:].astype(np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = new_indptr_list
                
                current_nnz += filtered_chunk.nnz

            x_group_out.attrs['encoding-type'] = 'csr_matrix'
            x_group_out.attrs['encoding-version'] = '0.1.0'
            x_group_out.attrs['shape'] = np.array([n_cells, n_genes_to_keep], dtype='int64')
            
    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds.\n")

def hidden_calc_valsGPU(filename: str) -> dict:
    """ 
    Calculates stats. 
    Optimization: Reads MASSIVE sparse chunks (100k+) because GPU sparse reduction is extremely memory efficient.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")

    # Chunking: Aggressive. Sparse matrices take very little VRAM.
    # 500MB sparse data can represent millions of cells.
    chunk_size = get_io_chunk_size(filename, target_mb=1024) 

    adata_meta = anndata.read_h5ad(filename, backed='r')
    nc, ng = adata_meta.shape 
    print(f"Dataset Shape: {nc} cells x {ng} genes")

    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    
    # Accumulators on GPU
    tjs_gpu = cupy.zeros(ng, dtype=cupy.float32)
    gene_non_zeros_gpu = cupy.zeros(ng, dtype=cupy.int32)

    with h5py.File(filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"  > Processing stats for rows {i} to {end_row}...", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            
            # Transfer to GPU
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float32)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])

            # Construct Cupy CSR
            chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(end_row-i, ng))

            # Row sums (tis) - retrieve immediately
            tis[i:end_row] = chunk_gpu.sum(axis=1).get().flatten()
            
            # Row NNZ
            cell_non_zeros[i:end_row] = cupy.diff(indptr_gpu).get()

            # Column Accumulation (tjs) - keep on GPU
            # sum(axis=0) returns a matrix, convert to 1D array
            current_sum = chunk_gpu.sum(axis=0)
            tjs_gpu += current_sum.ravel() # Flatten to 1D
            
            # Column NNZ
            # Trick: Convert data to 1s, then sum axis 0
            chunk_gpu.data[:] = 1
            gene_non_zeros_gpu += chunk_gpu.sum(axis=0).ravel()
            
            del data_gpu, indices_gpu, indptr_gpu, chunk_gpu
            # No aggressive free needed here, standard GC is usually fine for sparse

    tjs = cupy.asnumpy(tjs_gpu)
    gene_non_zeros = cupy.asnumpy(gene_non_zeros_gpu)
    
    dis = ng - cell_non_zeros
    djs = nc - gene_non_zeros
    total = tjs.sum()

    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds.\n")

    return {
        "tis": pd.Series(tis, index=adata_meta.obs.index),
        "tjs": pd.Series(tjs, index=adata_meta.var.index),
        "dis": pd.Series(dis, index=adata_meta.obs.index),
        "djs": pd.Series(djs, index=adata_meta.var.index),
        "total": total,
        "nc": nc,
        "ng": ng
    }

def NBumiFitModelGPU(cleaned_filename: str, stats: dict) -> dict:
    """
    Fits the model. Uses sparse operations for variance calculation.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
    
    # Sparse ops are memory efficient. Use large chunks.
    chunk_size = get_io_chunk_size(cleaned_filename, target_mb=1024)
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    nc, ng = stats['nc'], stats['ng']
    total = stats['total']
    
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    
    sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
    sum_2xmu_gpu = cupy.zeros(ng, dtype=cupy.float64)
    
    # Pre-calc constants
    sum_tis_sq_gpu = cupy.sum(tis_gpu**2)
    sum_mu_sq_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
    
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']
        
        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"  > Processing variance for rows {i} to {end_row}...", end='\r')
            
            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue
            
            # Sparse GPU Load
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])
            
            # 1. Sum X^2 (Element-wise square of sparse data)
            cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            # 2. Sum 2*X*mu
            # Need to map row_idx to every data point to get tis[row]
            # Efficient method: expand row indices
            nnz_in_chunk = indptr_gpu[-1].item()
            
            # Create row_indices vector (e.g. [0,0,0, 1,1, 2...])
            # Kernel-based expansion or diff trick
            # This consumes VRAM proportional to NNZ (safe)
            row_boundaries = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
            if len(indptr_gpu) > 1:
                row_boundaries[indptr_gpu[:-1]] = 1
            # Global row index = offset i + local index
            row_indices_gpu = (cupy.cumsum(row_boundaries, axis=0) - 1) + i
            
            tis_per_nz = tis_gpu[row_indices_gpu]
            tjs_per_nz = tjs_gpu[indices_gpu]
            
            # term = 2 * X * (tis*tjs/total)
            term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
            cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
            
            # Cleanup
            del data_gpu, indices_gpu, indptr_gpu, row_indices_gpu, tis_per_nz, tjs_per_nz, term_vals
            # Periodic GC
            if i % (chunk_size * 2) == 0:
                cupy.get_default_memory_pool().free_all_blocks()
    
    # Finalize
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
    
    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds.\n")
    
    return {
        'var_obs': pd.Series(var_obs_cpu, index=stats['tjs'].index),
        'sizes': pd.Series(sizes_cpu, index=stats['tjs'].index),
        'vals': stats
    }

def NBumiFitDispVsMeanGPU(fit, suppress_plot=True):
    vals = fit['vals']
    size_g = fit['sizes'].values
    tjs = vals['tjs'].values
    mean_expression = tjs / vals['nc']
    
    forfit = (np.isfinite(size_g)) & (size_g < 1e6) & (mean_expression > 1e-3) & (size_g > 0)
    log2_mean_expr = np.log2(mean_expression, where=(mean_expression > 0))
    higher = log2_mean_expr > 4
    if np.sum(higher & forfit) > 2000:
        forfit = higher & forfit

    y = np.log(size_g[forfit])
    x = np.log(mean_expression[forfit])
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    if not suppress_plot:
        plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, model.fittedvalues, color='red')
        plt.show()

    return model.params

def NBumiFeatureSelectionHighVarGPU(fit: dict) -> pd.DataFrame:
    # This is fast enough on CPU usually, but can use GPU for residuals
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVar()")

    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    
    # Use GPU for vector math
    mean_expression = cupy.asarray(vals['tjs'].values / vals['nc'])
    sizes = cupy.asarray(fit['sizes'].values)
    coeffs_gpu = cupy.asarray(coeffs)
    
    log_mean_expression = cupy.log(mean_expression)
    # Handle neginf
    log_mean_expression[cupy.isinf(log_mean_expression)] = 0
    
    exp_size = cupy.exp(coeffs_gpu[0] + coeffs_gpu[1] * log_mean_expression)
    res_gpu = cupy.log(sizes) - cupy.log(exp_size)
    
    res_cpu = res_gpu.get()
    
    results_df = pd.DataFrame({'Gene': fit['sizes'].index, 'Residual': res_cpu})
    final_table = results_df.sort_values(by='Residual', ascending=True)
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")

    return final_table

def NBumiFeatureSelectionCombinedDropGPU(fit: dict, cleaned_filename: str, method="fdr_bh", qval_thresh=0.05) -> pd.DataFrame:
    """
    TILED IMPLEMENTATION.
    Crucial fix for 10GB VRAM Limit.
    1. Reads BIG chunks from disk (to save I/O).
    2. Processes SMALL dense tiles on GPU (to save VRAM).
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")

    # 1. READ CONFIG: Read 100k rows (efficient I/O)
    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=512)
    
    # 2. COMPUTE CONFIG: Process ~5k-10k rows (efficient VRAM)
    vals = fit['vals']
    compute_tile_size = get_compute_tile_size(n_genes=vals['ng'], vram_limit_gb=9.0)
    
    print(f"  > I/O Chunk: {io_chunk_size} rows (High Bandwidth)")
    print(f"  > GPU Tile : {compute_tile_size} rows (VRAM Safe)")

    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    tjs_gpu = cupy.asarray(vals['tjs'].values)
    tis_gpu = cupy.asarray(vals['tis'].values)
    total = vals['total']
    nc = vals['nc']
    ng = vals['ng']

    # Pre-calc expected size vector
    mean_expression_gpu = tjs_gpu / nc
    exp_size_gpu = cupy.exp(cupy.asarray(coeffs[0]) + cupy.asarray(coeffs[1]) * cupy.log(mean_expression_gpu))
    
    # Global Accumulators
    p_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    p_var_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)

    # LOOP 1: DISK I/O
    for i in range(0, nc, io_chunk_size):
        end_col = min(i + io_chunk_size, nc)
        print(f"  > I/O Block {i} - {end_col}...", end='\r')
        
        # We don't actually need to read the data file for this step!
        # The formula depends ONLY on `tis` (which we have in RAM) and `tjs` (in RAM).
        # We are calculating EXPECTED dropout based on MARGINALS.
        # This is a massive optimization: No H5 reads required here.
        
        # We just iterate the `tis` array.
        
        # LOOP 2: GPU TILES
        for j in range(i, end_col, compute_tile_size):
            tile_end = min(j + compute_tile_size, end_col)
            
            # Sliced TIS
            tis_tile_gpu = tis_gpu[j:tile_end] # (tile_size,)
            
            # --- DENSE MATH (Broadcasting) ---
            # mu = tjs * tis / total
            # shape: (genes, 1) * (1, tile) -> (genes, tile) -> Transpose to (tile, genes)
            # Or (tile, 1) * (1, genes) -> (tile, genes)
            
            # Memory efficient: (tile, 1) * (1, genes)
            mu_chunk_gpu = (tis_tile_gpu[:, cupy.newaxis] * tjs_gpu[cupy.newaxis, :]) / total
            
            # p = (1 + mu / alpha)^-alpha
            # broadcasting alpha (1, genes)
            p_is_chunk_gpu = cupy.power(1.0 + mu_chunk_gpu / exp_size_gpu[cupy.newaxis, :], -exp_size_gpu[cupy.newaxis, :])
            
            # Accumulate
            p_sum_gpu += p_is_chunk_gpu.sum(axis=0)
            
            # Variance
            p_var_sum_gpu += (p_is_chunk_gpu * (1.0 - p_is_chunk_gpu)).sum(axis=0)
            
            # Free VRAM
            del mu_chunk_gpu, p_is_chunk_gpu, tis_tile_gpu
            cupy.get_default_memory_pool().free_all_blocks()
            
    print(f"\n  > Calculation complete. Performing tests...")

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
    results_df = pd.DataFrame({'Gene': vals['tjs'].index, 'p.value': pvalue, 'effect_size': diff})
    results_df = results_df.sort_values(by='p.value')
    qval = multipletests(results_df['p.value'].fillna(1), method=method)[1]
    results_df['q.value'] = qval
    final_table = results_df[results_df['q.value'] < qval_thresh]
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return final_table[['Gene', 'effect_size', 'p.value', 'q.value']]

def NBumiCombinedDropVolcanoGPU(results_df, qval_thresh=0.05, effect_size_thresh=0.25, top_n_genes=10, suppress_plot=False, plot_filename=None):
    # Visualization is CPU bound and fast, no changes needed
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcano()")

    df = results_df.copy()
    if df.empty:
        print(" [WARNING] No genes passed selection. Plotting skipped.")
        return None

    non_zero_min = df[df['q.value'] > 0]['q.value'].min()
    df['q.value'] = df['q.value'].replace(0, non_zero_min)
    df['-log10_qval'] = -np.log10(df['q.value'])
    df['color'] = 'grey'
    df.loc[(df['q.value'] < qval_thresh) & (df['effect_size'] > effect_size_thresh), 'color'] = 'red'
    df.loc[(df['q.value'] < qval_thresh) & (df['effect_size'] < -effect_size_thresh), 'color'] = 'blue'

    plt.figure(figsize=(10, 8))
    plt.scatter(df['effect_size'], df['-log10_qval'], c=df['color'], s=10, alpha=0.6)
    plt.axvline(x=effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axvline(x=-effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axhline(y=-np.log10(qval_thresh), linestyle='--', color='grey', linewidth=0.8)

    top_genes = df.nsmallest(top_n_genes, 'q.value')
    for i, row in top_genes.iterrows():
        plt.text(row['effect_size'], row['-log10_qval'], row['Gene'], fontsize=9)

    plt.title('Volcano Plot of Dropout Feature Selection')
    plt.xlabel('Effect Size (Observed - Expected Dropout Rate)')
    plt.ylabel('-log10 (Adjusted p-value)')
    plt.grid(True, linestyle='--', alpha=0.3)
    ax = plt.gca()

    if plot_filename: plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    if not suppress_plot: plt.show()
    plt.close()
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return ax
