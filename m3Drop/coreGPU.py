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

# Safe Import for Local vs Supercomputer
try:
    import cupy
    import cupy.sparse as csp
    from cupy.sparse import csr_matrix as cp_csr_matrix
    HAS_GPU = True
except ImportError:
    cupy = None
    HAS_GPU = False
    print(" [WARNING] CuPy not found. GPU acceleration disabled.")

# --- (PING & GOVERNOR PROTOCOL) ---
def get_optimal_chunk_size(filename: str, multiplier: float, is_dense: bool = False, override_cap: int = 50000) -> int:
    """
    AUTO-TUNER ENGINE (PING & GOVERNOR).
    
    Sensors:
    1. Data Weight (Exact bytes per row)
    2. RAM Pressure (psutil)
    3. VRAM Pressure (cupy)
    4. Context (SLURM Check)
    
    Governor:
    - Cluster: Maximize Throughput (Min 5k rows, Ignore CPU Cache)
    - Local:   Maximize Responsiveness (Target 10MB Chunk, Protect CPU Cache)
    """
    
    # --- SENSOR A: DATA WEIGHT ---
    with h5py.File(filename, 'r') as f:
        x_group = f['X']
        shape = x_group.attrs['shape']
        n_cells, n_genes = shape[0], shape[1]
        
        # Detect exact byte size (4 for float32, 8 for float64)
        if 'data' in x_group:
            dtype_size = x_group['data'].dtype.itemsize
        else:
            dtype_size = 4 # Default safety
            
        # Calculate Load
        if is_dense:
            # Dense: Width * Bytes * Overhead
            bytes_per_row = n_genes * dtype_size * multiplier
        else:
            # Sparse: (Val + Col + Ptr) * Density
            if 'indptr' in x_group:
                nnz = x_group['indptr'][-1]
                density = nnz / (n_cells * n_genes)
            else:
                density = 0.1 # Safety default
            
            # Sparse Row = (Bytes_Data + 4_Index) * density * n_genes
            bytes_per_row = (n_genes * density * (dtype_size + 4)) * multiplier
    
    if bytes_per_row < 1: bytes_per_row = 1

    # --- SENSOR B: RAM CAPACITY ---
    avail_ram = psutil.virtual_memory().available
    limit_ram = int((avail_ram * 0.30) / bytes_per_row) # Cap at 30% RAM

    # --- SENSOR C: VRAM CAPACITY ---
    limit_vram = float('inf')
    if HAS_GPU:
        try:
            mempool = cupy.get_default_memory_pool()
            mempool.free_all_blocks()
            free_vram = cupy.cuda.Device(0).mem_info[0]
            limit_vram = int((free_vram * 0.60) / bytes_per_row) # Cap at 60% VRAM
        except:
            pass

    # --- SENSOR D: CONTEXT CHECK (SLURM) ---
    # This is the Ticket Stub check.
    is_cluster = "SLURM_JOB_ID" in os.environ

    # --- THE GOVERNOR ---
    
    if is_cluster and HAS_GPU:
        # SCENARIO 1: CLUSTER (Beast Mode)
        # Goal: Throughput. Ignore CPU Cache.
        optimal = min(limit_ram, limit_vram)
        
        # ANTI-STALL FLOOR: Force 5,000 rows minimum to overcome latency
        # (Unless the calculated limit is dangerously low, then respect safety)
        if optimal < 5000 and optimal > 100: optimal = 5000
        
        mode_msg = "CLUSTER (SLURM Detected)"
        
    else:
        # SCENARIO 2: LOCAL (Safe Harbor)
        # Goal: Responsiveness. Protect L3 Cache.
        
        # Sensor 4: CPU Cache Target (10MB)
        # 10MB fits in almost all L3 caches (preventing thrashing)
        target_10mb_rows = int(10_000_000 / bytes_per_row)
        
        optimal = min(limit_ram, limit_vram, target_10mb_rows)
        
        # ANTI-FREEZE FLOOR: Force 500 rows minimum
        if optimal < 500: optimal = 500
        
        mode_msg = "LOCAL (Safe Harbor)"

    # GLOBAL CAP (Transport Safety & Function Specific Override)
    if optimal > override_cap: optimal = override_cap
    
    # Cap at total file size
    if optimal > n_cells: optimal = n_cells

    # --- TELEMETRY OUTPUT ---
    print(f"\n------------------------------------------------------------")
    print(f" CHUNK SIZE OPTIMIZER (PING & GOVERNOR)            [{time.strftime('%H:%M:%S')}]")
    print(f"------------------------------------------------------------")
    print(f" CONTEXT      : {mode_msg}")
    print(f" DATA LOAD    : {int(bytes_per_row):,} bytes/row (dtype={dtype_size})")
    print(f" MULTIPLIER   : {multiplier}x")
    print(f" OVERRIDE CAP : {override_cap:,} rows")
    print(f" RAM LIMIT    : {limit_ram:,} rows")
    if HAS_GPU:
        print(f" VRAM LIMIT   : {limit_vram if limit_vram != float('inf') else 'N/A':,} rows")
    else:
        print(f" VRAM LIMIT   : N/A (No GPU)")
    print(f"------------------------------------------------------------")
    print(f" >> CHUNK SIZE  : {int(optimal):,} rows")
    print(f"------------------------------------------------------------\n")
    
    return int(optimal)


def ConvertDataSparseGPU(
    input_filename: str,
    output_filename: str
):
    """
    GPU-ACCELERATED CLEANING.
    Now properly shifts gears between Phase 1 (Fast Read) and Phase 2 (Slow Write).
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseGPU() | FILE: {input_filename}")

    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        n_cells, n_genes = x_group_in.attrs['shape']

        # --- GEAR 2: FAST READ (Phase 1) ---
        # We are only reading indices. No writing. Let it fly.
        # Max cap 50k to saturate PCIe bus without timeout.
        read_chunk_size = get_optimal_chunk_size(input_filename, multiplier=2.5, is_dense=False, override_cap=50000)
        
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
            if start_idx == end_idx:
                continue

            indices_cpu = h5_indices[start_idx:end_idx]

            if HAS_GPU:
                indices_gpu = cupy.asarray(indices_cpu)
                unique_gpu = cupy.unique(indices_gpu)
                genes_to_keep_mask[unique_gpu] = True
                del indices_gpu, unique_gpu
                cupy.get_default_memory_pool().free_all_blocks()
            else:
                unique_cpu = np.unique(indices_cpu)
                genes_to_keep_mask[unique_cpu] = True

        if HAS_GPU:
            genes_to_keep_mask_cpu = cupy.asnumpy(genes_to_keep_mask)
        else:
            genes_to_keep_mask_cpu = genes_to_keep_mask

        n_genes_to_keep = np.sum(genes_to_keep_mask_cpu)
        print(f"\nPhase [1/2]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        # --- GEAR 1: FORKLIFT WRITE (Phase 2) ---
        # We are writing to disk. We MUST slow down to 5,000 to save the hard drive.
        write_chunk_size = get_optimal_chunk_size(input_filename, multiplier=2.5, is_dense=False, override_cap=5000)
        
        print(f"Phase [2/2]: Rounding up decimals and saving filtered output to disk... (Chunk: {write_chunk_size})")
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

            for i in range(0, n_cells, write_chunk_size):
                end_row = min(i + write_chunk_size, n_cells)
                print(f"Phase [2/2]: Processing: {end_row} of {n_cells} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                chunk = sp_csr_matrix((data_slice, indices_slice, indptr_slice), shape=(end_row-i, n_genes))
                filtered_chunk = chunk[:, genes_to_keep_mask_cpu]
                filtered_chunk.data = np.ceil(filtered_chunk.data).astype('float32')

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
        print(f"\nPhase [2/2]: COMPLETE | Output: {output_filename} {' ' * 50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

def hidden_calc_valsGPU(filename: str) -> dict:
    """ Calculates key statistics using memory-safe, GPU-accelerated algorithm. """
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")

    # GEAR 3: CRUISER MODE (Transport Bound)
    # Simple math. Maximize throughput with 50k cap.
    chunk_size = get_optimal_chunk_size(filename, multiplier=3.0, is_dense=False, override_cap=50000)

    adata_meta = anndata.read_h5ad(filename, backed='r')
    print("Phase [1/3]: Finding nc and ng...")
    nc, ng = adata_meta.shape 
    print(f"Phase [1/3]: COMPLETE")

    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
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
            data_slice = h5_data[start_idx:end_idx]
            indices_slice = h5_indices[start_idx:end_idx]
            indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

            data_gpu = cupy.asarray(data_slice.copy(), dtype=cupy.float32)
            indices_gpu = cupy.asarray(indices_slice.copy())
            indptr_gpu = cupy.asarray(indptr_slice.copy())

            chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(end_row-i, ng))

            tis[i:end_row] = chunk_gpu.sum(axis=1).get().flatten()
            cell_non_zeros_chunk = cupy.diff(indptr_gpu)
            cell_non_zeros[i:end_row] = cell_non_zeros_chunk.get()

            cupy.add.at(tjs_gpu, indices_gpu, data_gpu)
            unique_indices_gpu, counts_gpu = cupy.unique(indices_gpu, return_counts=True)
            cupy.add.at(gene_non_zeros_gpu, unique_indices_gpu, counts_gpu)
            
            del data_gpu, indices_gpu, indptr_gpu, chunk_gpu
            cupy.get_default_memory_pool().free_all_blocks()

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

def NBumiFitModelGPU(cleaned_filename: str, stats: dict) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModel() | FILE: {cleaned_filename}")
    
    # GEAR 2: HEAVY LIFT MODE (Memory Bound)
    # High Multiplier (12.0) to account for heavy intermediate matrices (x, x^2, mean).
    # No artificial cap (50k) - Let it scale with VRAM.
    # If 12GB VRAM -> ~8k rows. If 80GB VRAM -> ~50k rows.
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=12.0, is_dense=False, override_cap=50000)
    
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
            cell_boundary_markers = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
            if len(indptr_gpu) > 1:
                cell_boundary_markers[indptr_gpu[:-1]] = 1
            cell_indices_gpu = (cupy.cumsum(cell_boundary_markers, axis=0) - 1) + i
            
            tis_per_nz = tis_gpu[cell_indices_gpu]
            tjs_per_nz = tjs_gpu[indices_gpu]
            term_vals = 2 * data_gpu * tjs_per_nz * tis_per_nz / total
            cupy.add.at(sum_2xmu_gpu, indices_gpu, term_vals)
            
            del data_gpu, indices_gpu, indptr_gpu, cell_indices_gpu
            del tis_per_nz, tjs_per_nz, term_vals
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
    mean_expression = vals['tjs'].values / vals['nc']

    with np.errstate(divide='ignore', invalid='ignore'):
        log_mean_expression = np.log(mean_expression)
        log_mean_expression[np.isneginf(log_mean_expression)] = 0
        exp_size = np.exp(coeffs[0] + coeffs[1] * log_mean_expression)
        res = np.log(fit['sizes'].values) - np.log(exp_size)

    results_df = pd.DataFrame({'Gene': fit['sizes'].index, 'Residual': res})
    final_table = results_df.sort_values(by='Residual', ascending=True)
    print("Phase [1/1]: COMPLETE")
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")

    return final_table

def NBumiFeatureSelectionCombinedDropGPU(fit: dict, cleaned_filename: str, method="fdr_bh", qval_thresh=0.05) -> pd.DataFrame:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {cleaned_filename}")

    # GEAR 4: DENSE MATH MODE (Memory Critical)
    # Multiplier 20.0x: 
    # 1. We assume data promotes to float64 (double memory).
    # 2. We broadcast dense matrices (ng * chunk).
    # 3. We hold ~5 copies (mu, exp_size, p_is, p_var, temp).
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=20.0, is_dense=True, override_cap=20000)

    print("Phase [1/3]: Initializing arrays...")
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

    print("Phase [2/3]: Calculating expected dropout sums...")
    for i in range(0, nc, chunk_size):
        end_col = min(i + chunk_size, nc)
        print(f"Phase [2/3]: Processing: {end_col} of {nc} cells.", end='\r')

        tis_chunk_gpu = tis_gpu[i:end_col]
        # Memory Intense: Creates dense (chunk x genes) float64 matrices
        mu_chunk_gpu = tjs_gpu[:, cupy.newaxis] * tis_chunk_gpu[cupy.newaxis, :] / total
        
        # Calculate p_is and p_var in steps to allow memory recycling if possible
        
        # [PATCH START] Restored safety clamping from CPU version to prevent NaN/Inf crashes
        base = 1 + mu_chunk_gpu / exp_size_gpu[:, cupy.newaxis]
        base = cupy.maximum(base, 1e-12) 
        
        p_is_chunk_gpu = cupy.power(base, -exp_size_gpu[:, cupy.newaxis])
        p_is_chunk_gpu = cupy.nan_to_num(p_is_chunk_gpu, nan=0.0, posinf=1.0, neginf=0.0)
        # [PATCH END]

        p_sum_gpu += p_is_chunk_gpu.sum(axis=1)
        
        # Calculate Variance
        p_var_is_chunk_gpu = p_is_chunk_gpu * (1 - p_is_chunk_gpu)
        p_var_sum_gpu += p_var_is_chunk_gpu.sum(axis=1)
        
        # Aggressive cleanup
        del mu_chunk_gpu, p_is_chunk_gpu, p_var_is_chunk_gpu, tis_chunk_gpu
        cupy.get_default_memory_pool().free_all_blocks()
    
    print(f"Phase [2/3]: COMPLETE {' ' * 50}")

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
