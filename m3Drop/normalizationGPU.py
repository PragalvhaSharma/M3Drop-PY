try:
    from .coreGPU import get_optimal_chunk_size
except ImportError:
    from coreGPU import get_optimal_chunk_size

import pickle
import time
import cupy
import numpy as np
import h5py
import anndata
import pandas as pd
from cupy.sparse import csr_matrix as cp_csr_matrix
import os

def NBumiPearsonResidualsGPU(
    cleaned_filename: str,
    fit_filename: str,
    output_filename: str
):
    """
    Calculates Pearson residuals. Safe Mode: Multiplier increased to 10.0.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResiduals() | FILE: {cleaned_filename}")

    # --- SAFETY UPDATE ---
    # Multiplier 10.0 (Was 6.0): Accounts for Float64 precision (8 bytes) vs Governor default (4 bytes).
    # 4 matrices * 8 bytes = 32 bytes/cell. Governor 10 * 4 = 40 bytes. Safe buffer established.
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=10.0, is_dense=True)

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and preparing output file...")
    with open(fit_filename, 'rb') as f:
        fit = pickle.load(f)

    vals = fit['vals']
    tjs = vals['tjs'].values
    tis = vals['tis'].values
    sizes = fit['sizes'].values
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    sizes_gpu = cupy.asarray(sizes, dtype=cupy.float64)

    # Create Output H5 (Identical structure to cleaned input)
    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip") 
    
    with h5py.File(output_filename, 'a') as f_out:
        if 'X' in f_out:
            del f_out['X']
        # Create dataset for dense matrix output (float32)
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')

        print("Phase [1/2]: COMPLETE")

        # --- Phase 2: Calculate Residuals ---
        print("Phase [2/2]: Calculating Pearson residuals from data chunks...")
        
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                # Load Chunk
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Convert to Dense GPU Matrix
                # We construct sparse first, then densify on GPU to save bandwidth
                counts_chunk_sparse_gpu = cp_csr_matrix((
                    cupy.asarray(data_slice, dtype=cupy.float64),
                    cupy.asarray(indices_slice),
                    cupy.asarray(indptr_slice)
                ), shape=(end_row-i, ng))
                
                counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                # Calculate Residuals
                tis_chunk_gpu = tis_gpu[i:end_row]
                mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total
                
                denominator_gpu = cupy.sqrt(mus_chunk_gpu + mus_chunk_gpu**2 / sizes_gpu[cupy.newaxis, :])
                
                # --- LOGIC RESTORED: Prevent Division by Zero ---
                denominator_gpu = cupy.where(denominator_gpu == 0, 1, denominator_gpu)

                # (Counts - Mu) / Sqrt(V)
                pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu
                
                # Write to Disk
                # [OPTIMIZATION] Cast to float32 on GPU to halve PCIe transfer time
                out_x[i:end_row, :] = pearson_chunk_gpu.astype(cupy.float32).get()
                
                # Cleanup
                del counts_chunk_dense_gpu, counts_chunk_sparse_gpu, mus_chunk_gpu, pearson_chunk_gpu, denominator_gpu
                cupy.get_default_memory_pool().free_all_blocks()
        
        print(f"Phase [2/2]: COMPLETE{' '*50}")

    # --- LOGIC RESTORED: Explicit File Cleanup ---
    if hasattr(adata_in, "file") and adata_in.file is not None:
        adata_in.file.close()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")


def NBumiPearsonResidualsApproxGPU(
    cleaned_filename: str,
    stats_filename: str,
    output_filename: str
):
    """
    Calculates approximate Pearson residuals.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsApprox() | FILE: {cleaned_filename}")

    # --- HANDSHAKE ---
    # Multiplier 10.0: Same safety logic as Full residuals.
    chunk_size = get_optimal_chunk_size(cleaned_filename, multiplier=10.0, is_dense=True)

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and preparing output file...")
    with open(stats_filename, 'rb') as f:
        stats = pickle.load(f)

    vals = stats
    tjs = vals['tjs'].values
    tis = vals['tis'].values
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)

    # Create Output H5
    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip") 
    
    with h5py.File(output_filename, 'a') as f_out:
        if 'X' in f_out:
            del f_out['X']
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')

        print("Phase [1/2]: COMPLETE")

        # --- Phase 2: Calculate Residuals ---
        print("Phase [2/2]: Calculating approx residuals from data chunks...")
        
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                counts_chunk_sparse_gpu = cp_csr_matrix((
                    cupy.asarray(data_slice, dtype=cupy.float64),
                    cupy.asarray(indices_slice),
                    cupy.asarray(indptr_slice)
                ), shape=(end_row-i, ng))
                
                counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                tis_chunk_gpu = tis_gpu[i:end_row]
                mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total
                
                # Approx: Denom = Sqrt(Mu)
                denominator_gpu = cupy.sqrt(mus_chunk_gpu)
                
                # --- LOGIC RESTORED: Prevent Division by Zero ---
                denominator_gpu = cupy.where(denominator_gpu == 0, 1, denominator_gpu)
                
                pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu
                
                # [OPTIMIZATION] Cast to float32 on GPU to halve PCIe transfer time
                out_x[i:end_row, :] = pearson_chunk_gpu.astype(cupy.float32).get()
                
                del counts_chunk_dense_gpu, counts_chunk_sparse_gpu, mus_chunk_gpu, pearson_chunk_gpu, denominator_gpu
                cupy.get_default_memory_pool().free_all_blocks()
        
        print(f"Phase [2/2]: COMPLETE{' '*50}")

    # --- LOGIC RESTORED: Explicit File Cleanup ---
    if hasattr(adata_in, "file") and adata_in.file is not None:
        adata_in.file.close()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
