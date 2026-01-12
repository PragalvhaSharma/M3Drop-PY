try:
    from .coreGPU import get_io_chunk_size, get_compute_tile_size
except ImportError:
    from coreGPU import get_io_chunk_size, get_compute_tile_size

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
    Calculates Pearson residuals.
    FIX: Uses Tiled & Buffered I/O to prevent VRAM OOM on 10GB slice.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResiduals() | FILE: {cleaned_filename}")

    # CONFIG: IO=Big, GPU=Small
    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=512)
    
    # We are densifying the tile, so we need to be strict with VRAM.
    # A dense tile of 5k rows x 30k cols x 4 bytes = 600MB. 
    # Plus overheads. 5000 is safe for 10GB.
    compute_tile_size = 5000 

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

    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip") 
    
    with h5py.File(output_filename, 'a') as f_out:
        if 'X' in f_out: del f_out['X']
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(io_chunk_size, ng), dtype='float32')

        print("Phase [1/2]: COMPLETE")

        print(f"Phase [2/2]: Calculating Pearson residuals (I/O: {io_chunk_size}, Tile: {compute_tile_size})...")
        
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            # OUTER LOOP: Disk I/O
            for i in range(0, nc, io_chunk_size):
                end_row = min(i + io_chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                # Read raw data
                start_idx_global, end_idx_global = h5_indptr[i], h5_indptr[end_row]
                data_raw = h5_data[start_idx_global:end_idx_global]
                indices_raw = h5_indices[start_idx_global:end_idx_global]
                indptr_raw = h5_indptr[i:end_row+1] - h5_indptr[i]

                # CPU Buffer for the results of this entire chunk
                # We do this to perform one big write at the end
                chunk_rows = end_row - i
                result_buffer_cpu = np.zeros((chunk_rows, ng), dtype=np.float32)

                # INNER LOOP: GPU Tiles
                for j in range(0, chunk_rows, compute_tile_size):
                    tile_end_local = min(j + compute_tile_size, chunk_rows)
                    
                    p_start = indptr_raw[j]
                    p_end = indptr_raw[tile_end_local]
                    
                    if p_start == p_end: continue # Empty tile
                    
                    # Move small tile to GPU
                    d_tile = cupy.asarray(data_raw[p_start:p_end], dtype=cupy.float64)
                    i_tile = cupy.asarray(indices_raw[p_start:p_end])
                    p_tile = cupy.asarray(indptr_raw[j:tile_end_local+1] - indptr_raw[j])
                    
                    counts_chunk_sparse_gpu = cp_csr_matrix((d_tile, i_tile, p_tile), shape=(tile_end_local - j, ng))
                    counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                    # Calculate Residuals
                    # Adjust 'tis' slice to global index
                    global_row_start = i + j
                    global_row_end = i + tile_end_local
                    tis_chunk_gpu = tis_gpu[global_row_start:global_row_end]
                    
                    mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total
                    denominator_gpu = cupy.sqrt(mus_chunk_gpu + mus_chunk_gpu**2 / sizes_gpu[cupy.newaxis, :])
                    
                    pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu
                    
                    # Store in buffer
                    result_buffer_cpu[j:tile_end_local, :] = pearson_chunk_gpu.get()
                    
                    del counts_chunk_sparse_gpu, counts_chunk_dense_gpu, mus_chunk_gpu, pearson_chunk_gpu, denominator_gpu, d_tile, i_tile, p_tile
                    cupy.get_default_memory_pool().free_all_blocks()

                # Write Buffer
                out_x[i:end_row, :] = result_buffer_cpu
        
        print(f"Phase [2/2]: COMPLETE{' '*50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")


def NBumiPearsonResidualsApproxGPU(
    cleaned_filename: str,
    stats_filename: str,
    output_filename: str
):
    """
    Calculates approximate Pearson residuals.
    FIX: Uses Tiled & Buffered I/O.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsApprox() | FILE: {cleaned_filename}")

    io_chunk_size = get_io_chunk_size(cleaned_filename, target_mb=512)
    compute_tile_size = 5000 

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

    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip") 
    
    with h5py.File(output_filename, 'a') as f_out:
        if 'X' in f_out: del f_out['X']
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(io_chunk_size, ng), dtype='float32')
        print("Phase [1/2]: COMPLETE")

        print(f"Phase [2/2]: Calculating approx residuals (I/O: {io_chunk_size}, Tile: {compute_tile_size})...")
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, io_chunk_size):
                end_row = min(i + io_chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                start_idx_global, end_idx_global = h5_indptr[i], h5_indptr[end_row]
                data_raw = h5_data[start_idx_global:end_idx_global]
                indices_raw = h5_indices[start_idx_global:end_idx_global]
                indptr_raw = h5_indptr[i:end_row+1] - h5_indptr[i]

                chunk_rows = end_row - i
                result_buffer_cpu = np.zeros((chunk_rows, ng), dtype=np.float32)

                for j in range(0, chunk_rows, compute_tile_size):
                    tile_end_local = min(j + compute_tile_size, chunk_rows)
                    
                    p_start = indptr_raw[j]
                    p_end = indptr_raw[tile_end_local]
                    
                    if p_start == p_end: continue

                    d_tile = cupy.asarray(data_raw[p_start:p_end], dtype=cupy.float64)
                    i_tile = cupy.asarray(indices_raw[p_start:p_end])
                    p_tile = cupy.asarray(indptr_raw[j:tile_end_local+1] - indptr_raw[j])
                    
                    counts_chunk_sparse_gpu = cp_csr_matrix((d_tile, i_tile, p_tile), shape=(tile_end_local - j, ng))
                    counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                    global_row_start = i + j
                    global_row_end = i + tile_end_local
                    tis_chunk_gpu = tis_gpu[global_row_start:global_row_end]
                    
                    mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total
                    denominator_gpu = cupy.sqrt(mus_chunk_gpu)
                    pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu
                    
                    result_buffer_cpu[j:tile_end_local, :] = pearson_chunk_gpu.get()
                    
                    del counts_chunk_sparse_gpu, counts_chunk_dense_gpu, mus_chunk_gpu, pearson_chunk_gpu, denominator_gpu, d_tile, i_tile, p_tile
                    cupy.get_default_memory_pool().free_all_blocks()

                out_x[i:end_row, :] = result_buffer_cpu
        
        print(f"Phase [2/2]: COMPLETE{' '*50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
