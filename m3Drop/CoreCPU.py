import time
import psutil
import h5py
import numpy as np
import anndata
import pandas as pd
import os
import sys
import pickle

# [OPTIMIZATION] Use Numba for near-C++ speed on CPU
try:
    import numba
    from numba import jit, prange
except ImportError:
    print("CRITICAL ERROR: 'numba' not found. Please install it (pip install numba) for CPU optimization.")
    sys.exit(1)

import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import sparse
from statsmodels.stats.multitest import multipletests

# ==========================================
#        HYBRID IMPORT (PACKAGE VS LOCAL)
# ==========================================
try:
    # Case 1: Running as an installed package
    from .ControlDeviceCPU import ControlDevice
except ImportError:
    # Case 2: Running locally
    try:
        from ControlDeviceCPU import ControlDevice
    except ImportError:
        print("CRITICAL ERROR: 'ControlDeviceCPU.py' not found.")
        sys.exit(1)

# ==========================================
#        NUMBA KERNELS (CPU OPTIMIZED)
# ==========================================

@jit(nopython=True, cache=True)
def nan_replace_cpu(x):
    """Replaces NaNs with 0 and Infs with 0 or 1."""
    flat = x.ravel()
    for i in range(flat.size):
        val = flat[i]
        if np.isnan(val):
            flat[i] = 0.0
        elif np.isinf(val):
            flat[i] = 1.0 if val > 0 else 0.0
    return x.reshape(x.shape)

@jit(nopython=True, parallel=True, fastmath=True)
def dropout_prob_kernel_cpu(tj, ti, total, exp_size, out_matrix):
    """
    Calculates dropout probabilities using Negative Binomial logic.
    Parallelized across CPU cores.
    """
    rows = out_matrix.shape[0]
    cols = out_matrix.shape[1]
    
    # Numba handles the broadcasting loops explicitly for max speed
    for r in prange(rows):
        ti_val = ti[r]
        for c in range(cols):
            mu = (tj[c] * ti_val) / total
            size_val = exp_size[c]
            
            base = (mu / size_val) + 1.0
            if base < 1e-12:
                base = 1e-12
            
            # pow(base, -size_val)
            val = base ** (-size_val)
            
            if np.isnan(val):
                out_matrix[r, c] = 0.0
            elif np.isinf(val):
                out_matrix[r, c] = 1.0 if val > 0 else 0.0
            else:
                out_matrix[r, c] = val

@jit(nopython=True, cache=True)
def dropout_variance_inplace_cpu(p):
    """Calculates variance p * (1 - p) in-place."""
    flat = p.ravel()
    for i in range(flat.size):
        val = flat[i]
        flat[i] = val - (val * val)

# ==========================================
#        STAGE 1: MASK GENERATION
# ==========================================

def ConvertDataSparseCPU(input_filename: str, output_mask_filename: str, mode: str = "auto", manual_target: int = 3000):
    start_time = time.perf_counter()
    print(f"FUNCTION: ConvertDataSparseCPU() | FILE: {input_filename}")

    device = ControlDevice.from_h5ad(input_filename, mode=mode, manual_target=manual_target)
    n_cells = device.total_rows
    n_genes = device.n_genes

    with h5py.File(input_filename, 'r') as f_in:
        x_group_in = f_in['X']
        print(f"Phase [1/1]: identifying expressed genes...")
        genes_to_keep_mask = np.zeros(n_genes, dtype=bool)
        
        h5_indptr = x_group_in['indptr']
        h5_indices = x_group_in['indices']

        current_row = 0
        while current_row < n_cells:
            # Overhead 1.0 is fine for sparse scan on CPU
            end_row = device.get_next_chunk(current_row, mode='sparse', overhead_multiplier=1.0)
            if end_row is None or end_row <= current_row: break

            chunk_size = end_row - current_row
            print(f"Phase [1/1]: Scanning rows {end_row} of {n_cells} | Chunk: {chunk_size}", end='\r')

            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            if start_idx == end_idx:
                current_row = end_row
                continue

            indices = h5_indices[start_idx:end_idx]
            unique_indices = np.unique(indices)
            genes_to_keep_mask[unique_indices] = True
            
            current_row = end_row

        n_genes_to_keep = int(np.sum(genes_to_keep_mask))
        print(f"\nPhase [1/1]: COMPLETE | Result: {n_genes_to_keep} / {n_genes} genes retained.")

        print(f"Saving mask to {output_mask_filename}...")
        with open(output_mask_filename, 'wb') as f:
            pickle.dump(genes_to_keep_mask, f)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

# ==========================================
#        STAGE 2: STATISTICS
# ==========================================

def hidden_calc_valsCPU(filename: str, mask_filename: str, mode: str = "auto", manual_target: int = 3000) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: hidden_calc_valsCPU() | FILE: {filename}")

    # 1. Load Mask
    with open(mask_filename, 'rb') as f: mask = pickle.load(f)
    ng_filtered = int(np.sum(mask))
    
    # 2. Init Device
    with h5py.File(filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
        
    device = ControlDevice(
        indptr=indptr_cpu, 
        total_rows=total_rows, 
        n_genes=ng_filtered, 
        mode=mode, 
        manual_target=manual_target
    )
    nc = device.total_rows

    adata_meta = anndata.read_h5ad(filename, backed='r')
    tis = np.zeros(nc, dtype='float64')
    cell_non_zeros = np.zeros(nc, dtype='int64')
    tjs = np.zeros(ng_filtered, dtype=np.float64)
    gene_non_zeros = np.zeros(ng_filtered, dtype=np.int32)

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
            data = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices = np.array(h5_indices[start_idx:end_idx])
            indptr = np.array(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])

            # Use Scipy CSR for CPU operations
            chunk_csr = sparse.csr_matrix((data, indices, indptr), shape=(chunk_size, len(mask)))
            
            # --- VIRTUAL FILTER + CEIL ---
            chunk_csr = chunk_csr[:, mask]
            chunk_csr.data = np.ceil(chunk_csr.data)
            # -----------------------------

            tis[current_row:end_row] = np.array(chunk_csr.sum(axis=1)).flatten()
            cell_non_zeros[current_row:end_row] = np.diff(chunk_csr.indptr)

            # Numpy 'add.at' equivalent for sparse accumulation
            np.add.at(tjs, chunk_csr.indices, chunk_csr.data)
            
            unique_indices, counts = np.unique(chunk_csr.indices, return_counts=True)
            np.add.at(gene_non_zeros, unique_indices, counts)
            
            current_row = end_row

    print(f"\nPhase [1/2]: COMPLETE{' ' * 50}")

    print("Phase [2/2]: Finalizing stats...")
    dis = ng_filtered - cell_non_zeros
    djs = nc - gene_non_zeros
    total = tjs.sum()
    print("Phase [2/2]: COMPLETE")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")

    filtered_var_index = adata_meta.var.index[mask]

    return {
        "tis": pd.Series(tis, index=adata_meta.obs.index),
        "tjs": pd.Series(tjs, index=filtered_var_index),
        "dis": pd.Series(dis, index=adata_meta.obs.index),
        "djs": pd.Series(djs, index=filtered_var_index),
        "total": total,
        "nc": nc,
        "ng": ng_filtered
    }

def NBumiFitModelCPU(raw_filename: str, mask_filename: str, stats: dict, mode: str = "auto", manual_target: int = 3000) -> dict:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFitModelCPU() | FILE: {raw_filename}")

    with open(mask_filename, 'rb') as f: mask = pickle.load(f)
    ng_filtered = stats['ng']
    
    with h5py.File(raw_filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows
    
    tjs = stats['tjs'].values
    tis = stats['tis'].values
    total = stats['total']
    
    # Numpy arrays
    sum_x_sq = np.zeros(ng_filtered, dtype=np.float64)
    sum_2xmu = np.zeros(ng_filtered, dtype=np.float64)
    
    print("Phase [1/3]: Pre-calculating sum of squared expectations...")
    sum_tis_sq = np.sum(tis**2)
    sum_mu_sq = (tjs**2 / total**2) * sum_tis_sq
    print("Phase [1/3]: COMPLETE")
    
    print("Phase [2/3]: Calculating variance components...")
    with h5py.File(raw_filename, 'r') as f_in:
        x_group = f_in['X']
        h5_indptr = x_group['indptr']
        h5_data = x_group['data']
        h5_indices = x_group['indices']
        
        current_row = 0
        while current_row < nc:
            # L3 optimization is critical here for CPU performance
            end_row = device.get_next_chunk(current_row, mode='sparse', overhead_multiplier=1.1)
            if end_row is None or end_row <= current_row: break
            
            chunk_size = end_row - current_row
            print(f"Phase [2/3]: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')
            
            start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
            data = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
            indices = np.array(h5_indices[start_idx:end_idx])
            indptr = np.array(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
            
            chunk_csr = sparse.csr_matrix((data, indices, indptr), shape=(chunk_size, len(mask)))
            chunk_csr = chunk_csr[:, mask]
            chunk_csr.data = np.ceil(chunk_csr.data)

            # Accumulate X^2
            np.add.at(sum_x_sq, chunk_csr.indices, chunk_csr.data**2)
            
            # Vectorized term calculation for 2 * x * mu
            # To avoid expanding dense matrices, we iterate over CSR structure manually or use broadcasting
            # For CPU, iterating over the non-zeros is efficient enough
            
            # Map row indices to global cell indices
            row_indices = np.repeat(np.arange(chunk_size), np.diff(chunk_csr.indptr)) + current_row
            global_tis = tis[row_indices]
            
            term_vals = 2 * chunk_csr.data * tjs[chunk_csr.indices] * global_tis / total
            np.add.at(sum_2xmu, chunk_csr.indices, term_vals)
            
            current_row = end_row
    
    print(f"\nPhase [2/3]: COMPLETE {' ' * 50}")
    
    print("Phase [3/3]: Finalizing dispersion...")
    sum_sq_dev = sum_x_sq - sum_2xmu + sum_mu_sq
    var_obs = sum_sq_dev / (nc - 1)
    
    sizes = np.full(ng_filtered, 10000.0)
    numerator = (tjs**2 / total**2) * sum_tis_sq
    denominator = sum_sq_dev - tjs
    
    stable_mask = denominator > 1e-6
    sizes[stable_mask] = numerator[stable_mask] / denominator[stable_mask]
    sizes[sizes <= 0] = 10000.0
    
    print("Phase [3/3]: COMPLETE")
    
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
    
    return {
        'var_obs': pd.Series(var_obs, index=stats['tjs'].index),
        'sizes': pd.Series(sizes, index=stats['tjs'].index),
        'vals': stats
    }

def NBumiFitDispVsMeanCPU(fit: dict, suppress_plot=True):
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

def NBumiFeatureSelectionHighVarCPU(fit: dict) -> pd.DataFrame:
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionHighVarCPU()")

    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)
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

def NBumiFeatureSelectionCombinedDropCPU(
    fit: dict, 
    raw_filename: str, 
    method="fdr_bh", 
    qval_thresh=0.05, 
    mode: str = "auto", 
    manual_target: int = 3000
) -> pd.DataFrame:
    
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiFeatureSelectionCombinedDropCPU() | FILE: {raw_filename}")
    
    ng_filtered = fit['vals']['ng']

    with h5py.File(raw_filename, 'r') as f:
        indptr_cpu = f['X']['indptr'][:]
        total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows

    print("Phase [1/3]: Initializing arrays...")
    vals = fit['vals']
    coeffs = NBumiFitDispVsMeanCPU(fit, suppress_plot=True)

    tjs = vals['tjs'].values
    tis = vals['tis'].values
    total = vals['total']

    mean_expression = vals['tjs'].values / nc
    with np.errstate(divide='ignore'):
        exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(mean_expression))
    
    # Pre-allocate accumulators
    p_sum = np.zeros(ng_filtered, dtype=np.float64)
    p_var_sum = np.zeros(ng_filtered, dtype=np.float64)
    print("Phase [1/3]: COMPLETE")

    print("Phase [2/3]: Calculating dropout stats (Virtual)...")
    
    current_row = 0
    while current_row < nc:
        # Dense mode allows Numba to rip through the data
        end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=1.1)
        if end_row is None or end_row <= current_row: break
        
        chunk_size = end_row - current_row
        print(f"Phase [2/3]: Processing {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

        tis_chunk = tis[current_row:end_row]
        work_matrix = np.empty((chunk_size, ng_filtered), dtype=np.float64)
        
        # CALL NUMBA KERNEL
        dropout_prob_kernel_cpu(
            tjs,
            tis_chunk,
            total,
            exp_size,
            work_matrix 
        )

        p_sum += work_matrix.sum(axis=0) 
        
        # In-place variance calc
        dropout_variance_inplace_cpu(work_matrix)
        p_var_sum += work_matrix.sum(axis=0)
        
        current_row = end_row
    
    print(f"\nPhase [2/3]: COMPLETE {' ' * 50}")

    print("Phase [3/3]: Statistical testing...")
    
    droprate_exp = p_sum / nc
    droprate_exp_err = np.sqrt(p_var_sum / (nc**2))
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

def NBumiCombinedDropVolcanoCPU(results_df: pd.DataFrame, qval_thresh=0.05, effect_size_thresh=0.25, top_n_genes=10, suppress_plot=False, plot_filename=None):
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiCombinedDropVolcanoCPU()")
    
    # Standard Matplotlib code - safe for CPU
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

    plt.title('Volcano Plot: Dropout Rate vs Significance (CPU)')
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
