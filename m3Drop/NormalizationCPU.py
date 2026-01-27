import pickle
import time
import sys
import numpy as np
import h5py
import anndata
import pandas as pd
import os
from scipy import sparse

try:
    from numba import jit, prange
except ImportError:
    print("CRITICAL ERROR: 'numba' not found. Please install it (pip install numba).")
    sys.exit(1)

# [FIX] Strict Relative Import
from .ControlDeviceCPU import ControlDevice

# ==========================================
#        NUMBA KERNELS (CPU)
# ==========================================

@jit(nopython=True, parallel=True, fastmath=True)
def pearson_residual_kernel_cpu(counts, tj, ti, theta, total, out_matrix):
    """
    Calculates Pearson residuals using Negative Binomial logic.
    Parallelized across CPU cores.
    """
    rows = counts.shape[0]
    cols = counts.shape[1]
    
    for r in prange(rows):
        ti_val = ti[r]
        for c in range(cols):
            count_val = counts[r, c]
            mu = (tj[c] * ti_val) / total
            
            # theta is vector of size cols (genes)
            theta_val = theta[c]
            
            denom_sq = mu + ((mu * mu) / theta_val)
            denom = np.sqrt(denom_sq)
            
            if denom < 1e-12:
                out_matrix[r, c] = 0.0
            else:
                out_matrix[r, c] = (count_val - mu) / denom

@jit(nopython=True, parallel=True, fastmath=True)
def pearson_approx_kernel_cpu(counts, tj, ti, total, out_matrix):
    """
    Calculates Approximate Pearson residuals (Poisson limit).
    """
    rows = counts.shape[0]
    cols = counts.shape[1]
    
    for r in prange(rows):
        ti_val = ti[r]
        for c in range(cols):
            count_val = counts[r, c]
            mu = (tj[c] * ti_val) / total
            
            denom = np.sqrt(mu)
            
            if denom < 1e-12:
                out_matrix[r, c] = 0.0
            else:
                out_matrix[r, c] = (count_val - mu) / denom

# ==========================================
#        NORMALIZATION FUNCTION
# ==========================================

def NBumiPearsonResidualsCombinedCPU(
    raw_filename: str, 
    mask_filename: str, 
    fit_filename: str, 
    stats_filename: str,
    output_filename_full: str,
    output_filename_approx: str,
    mode: str = "auto",
    manual_target: int = 3000
):
    """
    CPU-Optimized: Calculates Full and Approximate residuals in a SINGLE PASS.
    Uses Numba for acceleration on L3-sized dense chunks.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsCombinedCPU() | FILE: {raw_filename}")

    # 1. Load Mask
    with open(mask_filename, 'rb') as f: mask = pickle.load(f)
    ng_filtered = int(np.sum(mask))

    # 2. Init Device
    with h5py.File(raw_filename, 'r') as f: indptr_cpu = f['X']['indptr'][:]; total_rows = len(indptr_cpu) - 1
    device = ControlDevice(indptr=indptr_cpu, total_rows=total_rows, n_genes=ng_filtered, mode=mode, manual_target=manual_target)
    nc = device.total_rows

    print("Phase [1/2]: Initializing parameters...")
    # Load parameters
    with open(fit_filename, 'rb') as f: fit = pickle.load(f)
    with open(stats_filename, 'rb') as f: stats = pickle.load(f)
    
    # Common params (Numpy Arrays)
    total = fit['vals']['total']
    tjs = fit['vals']['tjs'].values.astype(np.float64)
    tis = fit['vals']['tis'].values.astype(np.float64)
    
    # Specific params
    sizes = fit['sizes'].values.astype(np.float64) # For Full

    # Setup Output Files
    adata_in = anndata.read_h5ad(raw_filename, backed='r')
    filtered_var = adata_in.var[mask]
    
    # Create skeletons
    adata_out_full = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_full.write_h5ad(output_filename_full, compression=None)
    
    adata_out_approx = anndata.AnnData(obs=adata_in.obs, var=filtered_var)
    adata_out_approx.write_h5ad(output_filename_approx, compression=None)
    
    # Calculate appropriate H5 storage chunks
    storage_chunk_rows = int(1_000_000_000 / (ng_filtered * 8)) 
    if storage_chunk_rows < 1: storage_chunk_rows = 1
    
    # Open both files for writing simultaneously
    with h5py.File(output_filename_full, 'a') as f_full, h5py.File(output_filename_approx, 'a') as f_approx:
        if 'X' in f_full: del f_full['X']
        if 'X' in f_approx: del f_approx['X']
        
        # Float64 output
        out_x_full = f_full.create_dataset(
            'X', shape=(nc, ng_filtered), chunks=(storage_chunk_rows, ng_filtered), dtype='float64'
        )
        out_x_approx = f_approx.create_dataset(
            'X', shape=(nc, ng_filtered), chunks=(storage_chunk_rows, ng_filtered), dtype='float64'
        )

        with h5py.File(raw_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']
            
            current_row = 0
            while current_row < nc:
                # Dense mode is faster for Numba
                end_row = device.get_next_chunk(current_row, mode='dense', overhead_multiplier=3.0) 
                if end_row is None or end_row <= current_row: break

                chunk_size = end_row - current_row
                print(f"Phase [2/2]: Processing rows {end_row} of {nc} | Chunk: {chunk_size}", end='\r')

                start_idx, end_idx = h5_indptr[current_row], h5_indptr[end_row]
                
                # Load & Filter
                data = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
                indices = np.array(h5_indices[start_idx:end_idx])
                indptr = np.array(h5_indptr[current_row:end_row+1] - h5_indptr[current_row])
                
                chunk_csr = sparse.csr_matrix((data, indices, indptr), shape=(chunk_size, len(mask)))
                chunk_csr = chunk_csr[:, mask]
                chunk_csr.data = np.ceil(chunk_csr.data)

                # Convert to Dense for Numba (faster than sparse iteration for dense ops)
                counts_dense = chunk_csr.toarray()
                
                # --- CALC 1: APPROX ---
                approx_out = np.empty_like(counts_dense)
                pearson_approx_kernel_cpu(
                    counts_dense,
                    tjs,
                    tis[current_row:end_row], 
                    total,
                    approx_out 
                )
                out_x_approx[current_row:end_row, :] = approx_out
                del approx_out

                # --- CALC 2: FULL (In-place on counts_dense) ---
                # We can reuse the counts_dense buffer for output to save RAM
                pearson_residual_kernel_cpu(
                    counts_dense,
                    tjs,
                    tis[current_row:end_row], 
                    sizes,
                    total,
                    counts_dense # Overwrite input
                )
                out_x_full[current_row:end_row, :] = counts_dense
                
                current_row = end_row
        
        print(f"\nPhase [2/2]: COMPLETE{' '*50}")
    
    if hasattr(adata_in, "file") and adata_in.file is not None: adata_in.file.close()
    print(f"Total time: {time.perf_counter() - start_time:.2f} seconds.\n")
