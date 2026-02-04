import time
import psutil
import h5py
import numpy as np
import anndata
import pandas as pd
import os
import sys
import pickle

try:
    import cupy
    import cupy.sparse as csp
    from cupy.sparse import csr_matrix as cp_csr_matrix
    import cupyx
except ImportError:
    print("CRITICAL ERROR: CuPy not found. This script requires a GPU.")
    sys.exit(1)

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

# ==========================================
#      HYBRID IMPORT (PACKAGE VS LOCAL)
# ==========================================
try:
    # Case 1: Running as an installed package (e.g. import m3drop.CoreGPU)
    from .ControlDeviceGPU import ControlDevice
except ImportError:
    # Case 2: Running locally (e.g. python CoreGPU.py)
    try:
        from ControlDeviceGPU import ControlDevice
    except ImportError:
        print("CRITICAL ERROR: 'ControlDeviceGPU.py' not found.")
        sys.exit(1)

# ==========================================
#        FUSED KERNELS
# ==========================================

nan_replace_kernel = cupy.ElementwiseKernel(
    'float64 x', 'float64 out',
    '''if (isnan(x)) { out = 0.0; } else if (isinf(x)) { out = (x > 0) ? 1.0 : 0.0; } else { out = x; }''',
    'nan_replace_kernel'
)

dropout_prob_kernel = cupy.ElementwiseKernel(
    'float64 tj, float64 ti, float64 total, float64 exp_size', 'float64 out',
    '''
    double mu = (tj * ti) / total;
    double base = (mu / exp_size) + 1.0;
    if (base < 1e-12) base = 1e-12;
    out = pow(base, -exp_size);
    if (isnan(out)) out = 0.0;
    else if (isinf(out)) out = (out > 0) ? 1.0 : 0.0;
    ''',
    'dropout_prob_kernel'
)

dropout_variance_inplace_kernel = cupy.ElementwiseKernel(
    'float64 p', 'float64 out',
    ''' out = p - (p * p); ''',
    'dropout_variance_inplace_kernel'
)

# ==========================================
#        STAGE 1: MASK GENERATION
# ==========================================

def ConvertDataSparseGPU(input_filename: str, output_mask_filename: str, mode: str = "auto", manual_target: int = 3000):
    """
    Scans RAW data to identify genes with non-zero counts.
    Saves a boolean mask to disk instead of rewriting the dataset.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseGPU() | FILE: {input_filename}")

    # Standard init is fine here (we don't know ng yet)
    device = ControlDevice.from_h5ad(input_filename, mode=mode, manual_target=manual_target)
    n_cells = device.total_rows
    n_genes = device.n_genes

    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        print(f"Phase [1/1]: identifying expressed genes...")
        genes_to_keep_mask_gpu = cupy.zeros(n_genes, dtype=bool)
        
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        current_row = 0
        while current_row < n_cells:
            end_row = device.get_next_chunk(current_row, mode='sparse', overhead_multiplier=1.1)
            if end_row is None or end_row <= current_row: break

            chunk_size = end_row - current_row
            print(f"Phase [1/1]: Scanning rows {end_row} of {n_cells} | Chunk: {chunk_size}", end='\r')

            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            if start_idx == end_idx:
                current_row = end_row
                continue

            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            unique_gpu = cupy.unique(indices_gpu)
            genes_to_keep_mask_gpu[unique_gpu] = True
            
            del indices_gpu, unique_gpu
            current_row = end_row

        n_genes_to_keep = int(cupy.sum(genes_to_keep_mask_gpu))
        print(f"\nPhase [1/1]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        print(f"Saving mask to {output_mask_filename}...")
        mask_cpu = cupy.asnumpy(genes_to_keep_mask_gpu)
        with open(output_mask_filename, 'wb') as f:
            pickle.dump(mask_cpu, f)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

# ==========================================
#        STAGE 2: STATISTICS
# ==========================================

def hidden_calc_valsGPU(filename: str, mask_filename: str, mode: str = "auto", manual_target: int = 3000) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_vals() | FILE: {filename}")

    # 1. Load Mask
    with open(mask_filename, 'rb') as f: mask_cpu = pickle.load(f)
    mask_gpu = cupy.asarray(mask_cpu)
    ng_filtered = int(cupy.sum(mask_gpu))
    
    # 2. Manual Device Init (Crucial for VRAM logic)
    with h5py.File(filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
        
    device = ControlDevice(
        indptr=indptr_cpu, 
        total_rows=total_rows, 
        n_genes=ng_filtered, # Force device to see real data size
        mode=mode, 
        manual_target=manual_target
    )
    nc = device.total_rows

    adata_meta = anndata.read_h5ad(filename, backed='r')
    tis = np.zeros(nc, dtype='int64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    tjs_gpu = cupy.zeros(ng_filtered, dtype=cupy.float64)
    gene_non_zeros_gpu = cupy.zeros(ng_filtered, dtype=cupy.int32)

    print("Phase [1/2]: Calculating statistics...")
    with h5py.File(filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']

        current_row = 0
        while current_row < nc:
            end_row = device.get_next_chunk(current_row, mode='sparse', overhead_multiplier=1.1)
            if end_row is None or end_row <= current_row: break

            chunk_size = end_row - current_row
            print(f"Phase [1/2]: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            data_gpu = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
            indices_gpu = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu = cupy.asarray(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])

            chunk_gpu = cp_csr_matrix((data_gpu, indices_gpu, indptr_gpu), shape=(chunk_size, len(mask_cpu)))
            
            # --- VIRTUAL FILTER + CEIL ---
            chunk_gpu = chunk_gpu[:, mask_gpu]
            chunk_gpu.data = cupy.ceil(chunk_gpu.data)
            # -----------------------------

            tis[current_row:end_row] = chunk_gpu.sum(axis=1).get().flatten()
            cell_non_zeros_chunk = cupy.diff(chunk_gpu.indptr)
            cell_non_zeros[current_row:end_row] = cell_non_zeros_chunk.get()

            cupy.add.at(tjs_gpu, chunk_gpu.indices, chunk_gpu.data)
            unique_indices_gpu, counts_gpu = cupy.unique(chunk_gpu.indices, return_counts=True)
            cupy.add.at(gene_non_zeros_gpu, unique_indices_gpu, counts_gpu)
            
            del data_gpu, indices_gpu, indptr_gpu, chunk_gpu
            cupy.get_default_memory_pool().free_all_blocks()
            current_row = end_row

    tjs = cupy.asnumpy(tjs_gpu)
    gene_non_zeros = cupy.asnumpy(gene_non_zeros_gpu)
    print(f"\nPhase [1/2]: COMPLETE{' ' * 50}")

    print("Phase [2/2]: Finalizing stats...")
    dis = ng_filtered - cell_non_zeros
    djs = nc - gene_non_zeros
    total = tjs.sum()
    print("Phase [2/2]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    filtered_var_index = adata_meta.var.index[mask_cpu]

    return {
        "tis": pd.Series(tis, index=adata_meta.obs.index),
        "tjs": pd.Series(tjs, index=filtered_var_index),
        "dis": pd.Series(dis, index=adata_meta.obs.index),
        "djs": pd.Series(djs, index=filtered_var_index),
        "total": total,
        "nc": nc,
        "ng": ng_filtered
    }


def NBumiFitModelGPU(raw_filename: str, mask_filename: str, stats: dict, mode: str = "auto", manual_target: int = 3000) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModelGPU() | FILE: {raw_filename}")

    with open(mask_filename, 'rb') as f: mask_cpu = pickle.load(f)
    mask_gpu = cupy.asarray(mask_cpu)
    ng_filtered = stats['ng']
    
    # MANUAL INIT
    with h5py.File(raw_filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    total = stats['total']
    
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    sum_x_sq_gpu = cupy.zeros(ng_filtered, dtype=cupy.float64)
    sum_2xmu_gpu = cupy.zeros(ng_filtered, dtype=cupy.float64)
    
    print("Phase [1/3]: Pre-calculating sum of squared expectations...")
    sum_tis_sq_gpu = cupy.sum(tis_gpu**2)
    sum_mu_sq_gpu = (tjs_gpu**2 / total**2) * sum_tis_sq_gpu
    print("Phase [1/3]: COMPLETE")
    
    print("Phase [2/3]: Calculating variance components...")
    with h5py.File(raw_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']
        
        current_row = 0
        while current_row < nc:
            end_row = device.get_next_chunk(current_row, mode='sparse', overhead_multiplier=1.1)
            if end_row is None or end_row <= current_row: break
            
            chunk_size = end_row - current_row
            print(f"Phase [2/3]: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')
            
            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            data_gpu_raw = cupy.asarray(h5_data[start_idx:end_idx], dtype=cupy.float64)
            indices_gpu_raw = cupy.asarray(h5_indices[start_idx:end_idx])
            indptr_gpu_raw = cupy.asarray(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
            
            chunk_gpu = cp_csr_matrix((data_gpu_raw, indices_gpu_raw, indptr_gpu_raw), shape=(chunk_size, len(mask_cpu)))
            
            # --- VIRTUAL FILTER + CEIL ---
            chunk_gpu = chunk_gpu[:, mask_gpu]
            chunk_gpu.data = cupy.ceil(chunk_gpu.data)
            # -----------------------------

            cupy.add.at(sum_x_sq_gpu, chunk_gpu.indices, chunk_gpu.data**2)
            
            nnz_in_chunk = chunk_gpu.indptr[-1].item()
            cell_boundary_markers = cupy.zeros(nnz_in_chunk, dtype=cupy.int32)
            if len(chunk_gpu.indptr) > 1:
                cell_boundary_markers[chunk_gpu.indptr[:-1]] = 1
            cell_indices_gpu = (cupy.cumsum(cell_boundary_markers, axis=0) - 1) + current_row
            
            term_vals = 2 * chunk_gpu.data * tjs_gpu[chunk_gpu.indices] * tis_gpu[cell_indices_gpu] / total
            cupy.add.at(sum_2xmu_gpu, chunk_gpu.indices, term_vals)
            
            del chunk_gpu, data_gpu_raw, indices_gpu_raw, indptr_gpu_raw, cell_indices_gpu, term_vals
            cupy.get_default_memory_pool().free_all_blocks()
            
            current_row = end_row
    
    print(f"\nPhase [2/3]: COMPLETE {' ' * 50}")
    
    print("Phase [3/3]: Finalizing dispersion...")
    sum_sq_dev_gpu = sum_x_sq_gpu - sum_2xmu_gpu + sum_mu_sq_gpu
    var_obs_gpu = sum_sq_dev_gpu / (nc - 1)
    
    sizes_gpu = cupy.full(ng_filtered, 10000.0)
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


def NBumiFitDispVsMeanGPU(fit: dict, suppress_plot=True):
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
        plt.scatter(x, y, alpha=0.5, s=1)
        plt.plot(x, model.fittedvalues, color='red')
        plt.show()

    return model.params


def NBumiFeatureSelectionHighVarGPU(fit: dict) -> pd.DataFrame:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVar()")

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
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.\n")
    return final_table


def NBumiFeatureSelectionCombinedDropGPU(
    fit: dict, 
    raw_filename: str, 
    # Mask not strictly needed for calc (uses vectors), 
    # but needed if we want consistent API. 
    # However, we DO need ng_filtered for ControlDevice.
    method="fdr_bh", 
    qval_thresh=0.05, 
    mode: str = "auto", 
    manual_target: int = 3000
) -> pd.DataFrame:
    
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDrop() | FILE: {raw_filename}")
    
    ng_filtered = fit['vals']['ng']

    # MANUAL INIT
    with h5py.File(raw_filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows

    print("Phase [1/3]: Initializing arrays...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanGPU(fit, suppress_plot=True)

    tjs_gpu = cupy.asarray(vals['tjs'].values, dtype=cupy.float64)
    tis_gpu = cupy.asarray(vals['tis'].values, dtype=cupy.float64)
    total = vals['total']

    mean_expression_cpu = vals['tjs'].values / nc
    with np.errstate(divide='ignore'):
        exp_size_cpu = np.exp(coeffs[0] + coeffs[1] * np.log(mean_expression_cpu))
    exp_size_gpu = cupy.asarray(exp_size_cpu, dtype=cupy.float64)

    p_sum_gpu = cupy.zeros(ng_filtered, dtype=cupy.float64)
    p_var_sum_gpu = cupy.zeros(ng_filtered, dtype=cupy.float64)
    print("Phase [1/3]: COMPLETE")

    print("Phase [2/3]: Calculating dropout stats (Virtual)...")
    
    current_row = 0
    while current_row < nc:
        # Dense mode check is safe here because device sees ng_filtered
        end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.1)
        if end_row is None or end_row <= current_row: break
        
        chunk_size = end_row - current_row
        print(f"Phase [2/3]: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

        tis_chunk_gpu = tis_gpu[current_row:end_row]
        work_matrix = cupy.empty((chunk_size, ng_filtered), dtype=cupy.float64)
        
        dropout_prob_kernel(
            tjs_gpu,
            tis_chunk_gpu[:, cupy.newaxis],
            total,
            exp_size_gpu,
            work_matrix 
        )

        p_sum_gpu += work_matrix.sum(axis=0) 
        dropout_variance_inplace_kernel(work_matrix, work_matrix)
        p_var_sum_gpu += work_matrix.sum(axis=0)
        
        del work_matrix, tis_chunk_gpu
        cupy.get_default_memory_pool().free_all_blocks()
        
        current_row = end_row
    
    print(f"\nPhase [2/3]: COMPLETE {' ' * 50}")

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

def NBumiCombinedDropVolcanoGPU(results_df: pd.DataFrame, qval_thresh=0.05, effect_size_thresh=0.25, top_n_genes=10, suppress_plot=False, plot_filename=None):
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcano()")

    df = results_df.copy()
    if (df['q.value'] == 0).any():
        non_zero_min = df[df['q.value'] > 0]['q.value'].min()
        df['q.value'] = df['q.value'].replace(0, non_zero_min)

    df['-log10_qval'] = -np.log10(df['q.value'])
    df['color'] = 'grey'
    df.loc[(df['q.value'] < qval_thresh) & (df['effect_size'] > effect_size_thresh), 'color'] = 'red'
    df.loc[(df['q.value'] < qval_thresh) & (df['effect_size'] < -effect_size_thresh), 'color'] = 'blue'

    plt.figure(figsize=(10, 8))
    plt.scatter(df['effect_size'], df['-log10_qval'], c=df['color'], s=10, alpha=0.6, edgecolors='none')
    
    plt.axvline(x=effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axvline(x=-effect_size_thresh, linestyle='--', color='grey', linewidth=0.8)
    plt.axhline(y=-np.log10(qval_thresh), linestyle='--', color='grey', linewidth=0.8)

    top_genes = df.nsmallest(top_n_genes, 'q.value')
    for i, row in top_genes.iterrows():
        plt.text(row['effect_size'], row['-log10_qval'], row['Gene'], fontsize=9, fontweight='bold')

    plt.title('Volcano Plot: Dropout Rate vs Significance')
    plt.xlabel('Effect Size (Observed - Expected Dropout Rate)')
    plt.ylabel('-log10 (FDR Adjusted p-value)')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    ax = plt.gca()

    if plot_filename:
        print(f"Saving plot to: {plot_filename}")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
    if not suppress_plot:
        plt.show()
        
    plt.close()
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    return ax
