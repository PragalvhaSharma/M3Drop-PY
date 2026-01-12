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
except ImportError:
    cupy = None
    HAS_GPU = False
    print(" [WARNING] CuPy not found. GPU acceleration disabled.")

# --- MEMORY MANAGEMENT UTILS ---

def get_io_chunk_size(filename: str, target_mb: int = 512) -> int:
    """
    Calculates a large chunk size for Disk I/O (Lustre/GPFS friendly).
    Targeting ~512MB-1GB per read ensures high throughput on Supercomputers.
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
    
    # Cap to reasonable limits for CPU RAM
    chunk_size = max(5000, min(chunk_size, 500000))
    
    return chunk_size

def get_compute_tile_size(n_genes: int, vram_limit_gb: float = 9.0) -> int:
    """
    Calculates the max 'Micro-Batch' size for dense GPU operations 
    given the VRAM constraints (default 9GB safe limit for 10GB slice).
    """
    bytes_per_element = 8 # float64 (Double Precision)
    matrices_needed = 3   # mu, p_is, p_var (approx active at once)
    
    bytes_per_row_dense = n_genes * bytes_per_element * matrices_needed
    
    total_vram_bytes = vram_limit_gb * 1024**3
    tile_size = int(total_vram_bytes / bytes_per_row_dense)
    
    # Safety margin: Cap at 50k (Empirically safe for A100/V100 10GB split)
    return max(100, min(tile_size, 50000))

# --- LEGACY ALIAS FOR BACKWARD COMPATIBILITY ---
def get_optimal_chunk_size(*args, **kwargs):
    """
    Legacy stub. Redirects to get_io_chunk_size.
    """
    if args and isinstance(args[0], str):
        return get_io_chunk_size(args[0])
    return 5000

# --- CORE FUNCTIONS ---

def ConvertDataSparseGPU(input_filename: str, output_filename: str):
    """
    HYBRID IMPLEMENTATION.
    Uses CPU (NumPy) for filtering. 
    REASON: This is an I/O bound task. Moving data to GPU for boolean masking 
    is slower than just doing it on the CPU due to PCIe latency.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseGPU() | FILE: {input_filename}")
    print("  > MODE: CPU Optimized (Bypassing GPU for I/O efficiency)")

    # 1. READ IO (Aggressive Chunking for CPU)
    read_chunk_size = get_io_chunk_size(input_filename, target_mb=1024)
    
    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        n_cells, n_genes = x_group_in.attrs['shape']
        
        print(f"Phase [1/2]: Identifying genes with non-zero counts... (Chunk: {read_chunk_size})")
        
        # CPU Mask
        genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
            
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        for i in range(0, n_cells, read_chunk_size):
            end_row = min(i + read_chunk_size, n_cells)
            print(f"Phase [1/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue
            
            # Load indices (Disk -> RAM)
            indices_cpu = h5_indices[start_idx:end_idx]
            
            # Compute Unique (Fast on CPU L3 Cache)
            unique_cpu = np.unique(indices_cpu)
            genes_to_keep_mask[unique_cpu] = True

        n_genes_to_keep = np.sum(genes_to_keep_mask)
        print(f"\nPhase [1/2]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        # 2. WRITE IO
        print(f"Phase [2/2]: Filtering & Saving (CPU Stream)...")
        
        adata_meta = anndata.read_h5ad(input_filename, backed='r')
        filtered_var_df = adata_meta.var[genes_to_keep_mask]
        
        adata_out_template = anndata.AnnData(obs=adata_meta.obs, var=filtered_var_df, uns=adata_meta.uns)
        adata_out_template.write_h5ad(output_filename, compression="gzip")

        with h5py.File(output_filename, 'a') as f_out:
            if 'X' in f_out: del f_out['X']
            x_group_out = f_out.create_group('X')

            out_data = x_group_out.create_dataset('data', shape=(0,), maxshape=(None,), dtype='float32')
            out_indices = x_group_out.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int32')
            out_indptr = x_group_out.create_dataset('indptr', shape=(n_cells + 1,), dtype='int64')
            out_indptr[0] = 0
            
            current_nnz = 0
            h5_data = x_group_in['data']

            # CPU Stream Loop
            for i in range(0, n_cells, read_chunk_size):
                end_row = min(i + read_chunk_size, n_cells)
                print(f"Phase [2/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                
                # Read Block
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Sparse Construct (CPU)
                chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, n_genes))
                
                # Filter (CPU is very fast at boolean indexing on sparse cols)
                filtered_chunk = chunk[:, genes_to_keep_mask]
                
                # Ceil (CPU Vectorized)
                filtered_chunk.data = np.ceil(filtered_chunk.data).astype('float32')

                # Write Block
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
    print(f"\nPhase [2/2]: COMPLETE | Output: {output_filename}")
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

def hidden_calc_valsGPU(filename: str) -> dict:
    """ 
    HYBRID IMPLEMENTATION.
    Uses CPU (NumPy) for summation.
    REASON: Calculating sums is memory-bound. GPU transfer overhead > Compute time.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")
    print("  > MODE: CPU Optimized (Stable)")

    chunk_size = get_io_chunk_size(filename, target_mb=1024) 

    adata_meta = anndata.read_h5ad(filename, backed='r')
    nc, ng = adata_meta.shape 
    print("Phase [1/3]: Finding nc and ng...")
    print(f"Phase [1/3]: COMPLETE")

    # CPU Accumulators
    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    tjs = np.zeros(ng, dtype='float64') 
    gene_non_zeros = np.zeros(ng, dtype='int64')
    
    print("Phase [2/3]: Calculating tis and tjs (CPU)...")
    with h5py.File(filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            
            # Read to RAM
            data_cpu = h5_data[start_idx:end_idx]
            indices_cpu = h5_indices[start_idx:end_idx]
            indptr_cpu = h5_indptr[i:end_row+1] - h5_indptr[i]

            # Construct Sparse Matrix
            chunk_cpu = sp_csr_matrix((data_cpu, indices_cpu, indptr_cpu), shape=(end_row-i, ng))

            # Row stats
            tis[i:end_row] = chunk_cpu.sum(axis=1).flatten()
            cell_non_zeros[i:end_row] = np.diff(indptr_cpu)

            # Col stats (Accumulate in L3 Cache)
            tjs += chunk_cpu.sum(axis=0).A1 
            
            # Col NNZ
            chunk_cpu.data[:] = 1
            gene_non_zeros += chunk_cpu.sum(axis=0).A1.astype(np.int64)

    print(f"\nPhase [2/3]: COMPLETE{' ' * 50}")

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

def NBumiFitModelGPU(cleaned_filename: str, stats: dict) -> dict:
    """
    GPU IMPLEMENTATION (Fixed Tiled).
    Prevents OOM by slicing the large I/O chunk into small GPU tiles.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
    print("  > MODE: GPU Accelerated (Tiled Memory Safety)")
    
    # Large chunk for Disk (1GB+)
    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=1024)
    # Small tile for GPU (50k rows or auto)
    compute_tile_size = get_compute_tile_size(stats['ng'], vram_limit_gb=9.0)
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    nc, ng = stats['nc'], stats['ng']
    total = stats['total']
    
    # Move global stats to GPU
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    
    sum_x_sq_gpu = cupy.zeros(ng, dtype=cupy.float64)
    sum_2xmu_gpu = cupy.zeros(ng, dtype=cupy.float64)
    
    print("Phase [1/3]: Pre-calculating sum of squared expectations...")
    sum_tis_sq_gpu = cupy.sum(tis_gpu**2)
    sum_mu_sq_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
    print("Phase [1/3]: COMPLETE")
    
    print(f"Phase [2/3]: Calculating variance (I/O: {io_chunk_size}, Tile: {compute_tile_size})...")
    with h5py.File(cleaned_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']
        
        # OUTER LOOP: Disk I/O (Chunked)
        for i in range(0, nc, io_chunk_size):
            end_row = min(i + io_chunk_size, nc)
            print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')
            
            start_idx_global, end_idx_global = h5_indptr[i], h5_indptr[end_row]
            if start_idx_global == end_idx_global: continue
            
            # Read Raw Data to RAM
            data_raw = h5_data[start_idx_global:end_idx_global]
            indices_raw = h5_indices[start_idx_global:end_idx_global]
            indptr_raw = h5_indptr[i:end_row+1] - h5_indptr[i]
            
            n_rows_in_chunk = end_row - i
            
            # INNER LOOP: GPU Compute (Tiled)
            for j in range(0, n_rows_in_chunk, compute_tile_size):
                tile_end_local = min(j + compute_tile_size, n_rows_in_chunk)
                
                # Identify data range for this tile
                p_start = indptr_raw[j]
                p_end = indptr_raw[tile_end_local]
                
                if p_start == p_end: continue

                # Load Tile to GPU
                data_gpu = cupy.asarray(data_raw[p_start:p_end], dtype=cupy.float64)
                indices_gpu = cupy.asarray(indices_raw[p_start:p_end])
                indptr_gpu = cupy.asarray(indptr_raw[j:tile_end_local+1] - indptr_raw[j])
                
                # 1. Sum X^2
                cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
                
                # 2. Sum 2*X*mu
                # Expand row indices for this tile
                nnz_tile = p_end - p_start
                row_boundaries = cupy.zeros(nnz_tile, dtype=cupy.int32)
                if len(indptr_gpu) > 1:
                    row_boundaries[indptr_gpu[:-1]] = 1
                
                # Global row indices for looking up 'tis'
                # row_offset = global_start (i) + local_tile_start (j)
                row_offset = i + j
                row_indices_gpu = (cupy.cumsum(row_boundaries, axis=0) - 1) + row_offset
                
                tis_per_nz = tis_gpu[row_indices_gpu]
                tjs_per_nz = tjs_gpu[indices_gpu]
                
                term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
                cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
                
                # Cleanup Tile
                del data_gpu, indices_gpu, indptr_gpu, row_indices_gpu, tis_per_nz, tjs_per_nz, term_vals
                cupy.get_default_memory_pool().free_all_blocks()
    
    print(f"\nPhase [2/3]: COMPLETE {' ' * 50}")
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
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVar()")
    
    print("Phase [1/1]: Calculating residuals...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    
    mean_expression = cupy.asarray(vals['tjs'].values / vals['nc'])
    sizes = cupy.asarray(fit['sizes'].values)
    coeffs_gpu = cupy.asarray(coeffs)
    
    log_mean_expression = cupy.log(mean_expression)
    log_mean_expression[cupy.isinf(log_mean_expression)] = 0
    
    exp_size = cupy.exp(coeffs_gpu[0] + coeffs_gpu[1] * log_mean_expression)
    res_gpu = cupy.log(sizes) - cupy.log(exp_size)
    
    res_cpu = res_gpu.get()
    
    results_df = pd.DataFrame({'Gene': fit['sizes'].index, 'Residual': res_cpu})
    final_table = results_df.sort_values(by='Residual', ascending=True)
    
    print("Phase [1/1]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")

    return final_table

def NBumiFeatureSelectionCombinedDropGPU(fit: dict, cleaned_filename: str, method="fdr_bh", qval_thresh=0.05) -> pd.DataFrame:
    """
    GPU IMPLEMENTATION (Tiled).
    REASON: Heavy dense matrix math (exp, power). 
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")
    print("  > MODE: GPU Accelerated (Dense Matrix Operations)")

    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=512)
    vals = fit['vals']
    compute_tile_size = get_compute_tile_size(n_genes=vals['ng'], vram_limit_gb=9.0)
    
    print("Phase [1/3]: Initializing arrays...")
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)
    tjs_gpu = cupy.asarray(vals['tjs'].values)
    tis_gpu = cupy.asarray(vals['tis'].values)
    total = vals['total']
    nc = vals['nc']
    ng = vals['ng']

    mean_expression_gpu = tjs_gpu / nc
    exp_size_gpu = cupy.exp(cupy.asarray(coeffs[0]) + cupy.asarray(coeffs[1]) * cupy.log(mean_expression_gpu))
    
    p_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    p_var_sum_gpu = cupy.zeros(ng, dtype=cupy.float64)
    print("Phase [1/3]: COMPLETE")

    print(f"Phase [2/3]: Calculating expected dropout sums (Tile: {compute_tile_size})...")
    # Outer Loop (IO)
    for i in range(0, nc, io_chunk_size):
        end_col = min(i + io_chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')
        
        # Inner Loop (GPU)
        for j in range(i, end_col, compute_tile_size):
            tile_end = min(j + compute_tile_size, end_col)
            
            tis_tile_gpu = tis_gpu[j:tile_end] 
            mu_chunk_gpu = (tis_tile_gpu[:, cupy.newaxis] * tjs_gpu[cupy.newaxis, :]) / total
            
            p_is_chunk_gpu = cupy.power(1.0 + mu_chunk_gpu / exp_size_gpu[cupy.newaxis, :], -exp_size_gpu[cupy.newaxis, :])
            
            p_sum_gpu += p_is_chunk_gpu.sum(axis=0)
            p_var_sum_gpu += (p_is_chunk_gpu * (1.0 - p_is_chunk_gpu)).sum(axis=0)
            
            del mu_chunk_gpu, p_is_chunk_gpu, tis_tile_gpu
            cupy.get_default_memory_pool().free_all_blocks()
            
    print(f"\nPhase [2/3]: COMPLETE{' ' * 50}")

    print("Phase [3/3]: Statistical testing...")
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
    
    print("Phase [3/3]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return final_table[['Gene', 'effect_size', 'p.value', 'q.value']]

def NBumiCombinedDropVolcanoGPU(results_df, qval_thresh=0.05, effect_size_thresh=0.25, top_n_genes=10, suppress_plot=False, plot_filename=None):
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
