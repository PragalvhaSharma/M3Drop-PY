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
    """
    with h5py.File(filename, 'r') as f:
        x_group = f['X']
        shape = x_group.attrs['shape']
        n_cells, n_genes = shape[0], shape[1]
        
        if 'indptr' in x_group:
            nnz = x_group['indptr'][-1]
            bytes_per_row = (nnz / n_cells) * 12 
        else:
            bytes_per_row = n_genes * 4 * 0.1 
            
    if bytes_per_row < 1: bytes_per_row = 1
    
    target_bytes = target_mb * 1024 * 1024
    chunk_size = int(target_bytes / bytes_per_row)
    chunk_size = max(5000, min(chunk_size, 200000))
    
    return chunk_size

def get_compute_tile_size(n_genes: int, vram_limit_gb: float = 9.0) -> int:
    """
    Calculates max 'Micro-Batch' size for dense GPU operations.
    """
    bytes_per_element = 8 
    matrices_needed = 3   
    
    bytes_per_row_dense = n_genes * bytes_per_element * matrices_needed
    total_vram_bytes = vram_limit_gb * 1024**3
    tile_size = int(total_vram_bytes / bytes_per_row_dense)
    
    return max(100, min(tile_size, 20000))

# --- LEGACY ALIAS ---
def get_optimal_chunk_size(*args, **kwargs):
    if args and isinstance(args[0], str):
        return get_io_chunk_size(args[0])
    return 5000

# --- CORE FUNCTIONS ---

def ConvertDataSparseGPU(input_filename: str, output_filename: str):
    """
    GPU-Accelerated Cleaning.
    FIX: Now uses Tiled GPU processing for the Write phase to offload CPU.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseGPU() | FILE: {input_filename}")

    # 1. READ IO
    read_chunk_size = get_io_chunk_size(input_filename, target_mb=512)
    
    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        n_cells, n_genes = x_group_in.attrs['shape']
        
        print(f"Phase [1/2]: Identifying genes with non-zero counts... (Chunk: {read_chunk_size})")
        
        if HAS_GPU:
            genes_to_keep_mask = cupy.zeros(n_genes, dtype=bool)
        else:
            genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
            
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        for i in range(0, n_cells, read_chunk_size):
            end_row = min(i + read_chunk_size, n_cells)
            print(f"Phase [1/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            if start_idx == end_idx: continue
            
            indices_cpu = h5_indices[start_idx:end_idx]

            if HAS_GPU:
                indices_gpu = cupy.asarray(indices_cpu)
                unique_gpu = cupy.unique(indices_gpu)
                genes_to_keep_mask[unique_gpu] = True
                del indices_gpu, unique_gpu
                if i % (read_chunk_size * 5) == 0:
                    cupy.get_default_memory_pool().free_all_blocks()
            else:
                unique_cpu = np.unique(indices_cpu)
                genes_to_keep_mask[unique_cpu] = True

        if HAS_GPU:
            genes_to_keep_mask_cpu = cupy.asnumpy(genes_to_keep_mask)
            genes_to_keep_mask_gpu = genes_to_keep_mask # Keep on GPU for Phase 2
        else:
            genes_to_keep_mask_cpu = genes_to_keep_mask

        n_genes_to_keep = np.sum(genes_to_keep_mask_cpu)
        print(f"\nPhase [1/2]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        # 2. WRITE IO
        # We read big chunks, but process on GPU in tiles to avoid OOM
        io_chunk_size = read_chunk_size 
        
        # Calculate safe tile size for sparse matrix construction on GPU
        # Sparse matrices are smaller, so we can use a larger tile than dense
        # But let's be safe. ~50k rows sparse is usually fine.
        compute_tile_size = 50000 
        
        print(f"Phase [2/2]: Filtering & Saving (I/O: {io_chunk_size}, GPU Tile: {compute_tile_size})...")
        
        adata_meta = anndata.read_h5ad(input_filename, backed='r')
        filtered_var_df = adata_meta.var[genes_to_keep_mask_cpu]
        
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

            # OUTER LOOP: DISK I/O
            for i in range(0, n_cells, io_chunk_size):
                end_row = min(i + io_chunk_size, n_cells)
                print(f"Phase [2/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

                start_idx_global, end_idx_global = h5_indptr[i], h5_indptr[end_row]
                
                # Read RAW Data (Fastest)
                data_raw = h5_data[start_idx_global:end_idx_global]
                indices_raw = h5_indices[start_idx_global:end_idx_global]
                indptr_raw = h5_indptr[i:end_row+1] - h5_indptr[i]
                
                # Accumulators for the processed chunk
                processed_data = []
                processed_indices = []
                processed_indptr = [0]
                chunk_nnz_accum = 0

                # INNER LOOP: GPU PROCESSING
                # We process the raw arrays in tiles to allow GPU filtering
                n_rows_in_chunk = end_row - i
                
                for j in range(0, n_rows_in_chunk, compute_tile_size):
                    tile_end_local = min(j + compute_tile_size, n_rows_in_chunk)
                    
                    # Slice the RAW arrays for this tile
                    # We need to find where in 'data_raw' this tile starts/ends
                    p_start = indptr_raw[j]
                    p_end = indptr_raw[tile_end_local]
                    
                    if p_start == p_end:
                        # Empty tile, just append zeros to indptr
                        processed_indptr.extend([chunk_nnz_accum] * (tile_end_local - j))
                        continue
                        
                    d_tile = cupy.asarray(data_raw[p_start:p_end])
                    i_tile = cupy.asarray(indices_raw[p_start:p_end])
                    p_tile = cupy.asarray(indptr_raw[j:tile_end_local+1] - indptr_raw[j])
                    
                    # Construct GPU CSR
                    tile_csr = cp_csr_matrix((d_tile, i_tile, p_tile), shape=(tile_end_local - j, n_genes))
                    
                    # FILTER (Fast GPU boolean indexing)
                    filtered_tile = tile_csr[:, genes_to_keep_mask_gpu]
                    
                    # CEIL (Fast GPU math)
                    filtered_tile.data = cupy.ceil(filtered_tile.data).astype(cupy.float32)
                    
                    # Collect results
                    processed_data.append(filtered_tile.data.get())
                    processed_indices.append(filtered_tile.indices.get())
                    
                    # Adjust indptr to be relative to the start of the chunk
                    new_ptrs = filtered_tile.indptr[1:].get() + chunk_nnz_accum
                    processed_indptr.extend(new_ptrs)
                    
                    chunk_nnz_accum += filtered_tile.nnz
                    
                    del d_tile, i_tile, p_tile, tile_csr, filtered_tile
                    cupy.get_default_memory_pool().free_all_blocks()

                # Concatenate and Write
                if chunk_nnz_accum > 0:
                    final_data = np.concatenate(processed_data)
                    final_indices = np.concatenate(processed_indices)
                    
                    out_data.resize(current_nnz + chunk_nnz_accum, axis=0)
                    out_data[current_nnz:] = final_data

                    out_indices.resize(current_nnz + chunk_nnz_accum, axis=0)
                    out_indices[current_nnz:] = final_indices
                
                # Write Indptr
                # processed_indptr contains relative pointers for the chunk. 
                # We need to add current_nnz to make them global.
                final_indptr = np.array(processed_indptr[1:], dtype=np.int64) + current_nnz
                out_indptr[i + 1 : end_row + 1] = final_indptr
                
                current_nnz += chunk_nnz_accum

            x_group_out.attrs['encoding-type'] = 'csr_matrix'
            x_group_out.attrs['encoding-version'] = '0.1.0'
            x_group_out.attrs['shape'] = np.array([n_cells, n_genes_to_keep], dtype='int64')
            
    end_time = time.perf_counter()
    print(f"\nPhase [2/2]: COMPLETE | Output: {output_filename}")
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

def hidden_calc_valsGPU(filename: str) -> dict:
    """ 
    Calculates stats. 
    FIX: Added explicit casting to Int32 for nnz calculation to prevent TypeErrors.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")

    chunk_size = get_io_chunk_size(filename, target_mb=1024) 

    adata_meta = anndata.read_h5ad(filename, backed='r')
    nc, ng = adata_meta.shape 
    print("Phase [1/3]: Finding nc and ng...")
    print(f"Phase [1/3]: COMPLETE")

    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    
    # Accumulators on GPU
    tjs_gpu = cupy.zeros(ng, dtype=cupy.float32)
    gene_non_zeros_gpu = cupy.zeros(ng, dtype=cupy.int32)
    
    print("Phase [2/3]: Calculating tis and tjs...")
    with h5py.File(filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        for i in range(0, nc, chunk_size):
            end_row = min(i + chunk_size, nc)
            print(f"Phase [2/3]: Processing: {end_row} of {nc} cells.", end='\r')

            start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
            
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float32)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])

            chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(end_row-i, ng))

            tis[i:end_row] = chunk_gpu.sum(axis=1).get().flatten()
            cell_non_zeros[i:end_row] = cupy.diff(indptr_gpu).get()

            tjs_gpu += chunk_gpu.sum(axis=0).ravel()
            
            chunk_gpu.data[:] = 1
            # !!! FIX: Explicit cast to int32 !!!
            gene_non_zeros_gpu += chunk_gpu.sum(axis=0).ravel().astype(cupy.int32)
            
            del data_gpu, indices_gpu, indptr_gpu, chunk_gpu

    tjs = cupy.asnumpy(tjs_gpu)
    gene_non_zeros = cupy.asnumpy(gene_non_zeros_gpu)
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

# ... [NBumiFitModelGPU, NBumiFitDispVsMeanGPU, NBumiFeatureSelectionHighVarGPU, NBumiFeatureSelectionCombinedDropGPU, NBumiCombinedDropVolcanoGPU remain unchanged from previous stable version] ...
# (Include them here or ensure they are preserved in your file)
# For brevity, I am assuming you have the previous working versions of these.
# If you need me to paste the ENTIRE file again with these changes, just ask.
# But `hidden_calc_valsGPU` and `ConvertDataSparseGPU` were the only modified functions above.
# To be safe, I will paste the REST of the file below to ensure a complete copy-paste.

def NBumiFitModelGPU(cleaned_filename: str, stats: dict) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
    
    chunk_size = get_io_chunk_size(cleaned_filename, target_mb=1024)
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    nc, ng = stats['nc'], stats['ng']
    total = stats['total']
    
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
            if start_idx == end_idx: continue
            
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[i:end_row+1] - h5_indptr[i])
            
            cupy.add.at(sum_x_sq_gpu, indices_gpu, data_gpu**2)
            
            nnz_in_chunk = indptr_gpu[-1].item()
            row_boundaries = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
            if len(indptr_gpu) > 1:
                row_boundaries[indptr_gpu[:-1]] = 1
            row_indices_gpu = (cupy.cumsum(row_boundaries, axis=0) - 1) + i
            
            tis_per_nz = tis_gpu[row_indices_gpu]
            tjs_per_nz = tjs_gpu[indices_gpu]
            
            term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
            cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
            
            del data_gpu, indices_gpu, indptr_gpu, row_indices_gpu, tis_per_nz, tjs_per_nz, term_vals
            if i % (chunk_size * 2) == 0:
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
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")

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

    print("Phase [2/3]: Calculating expected dropout sums...")
    for i in range(0, nc, io_chunk_size):
        end_col = min(i + io_chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')
        
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
