import time
import pickle
from typing import Optional

# Lazy GPU imports with clear error if unavailable
try:
    import cupy
    from cupy.sparse import csr_matrix as cp_csr_matrix
except Exception as e:  # pragma: no cover - optional dependency
    cupy = None
    cp_csr_matrix = None
    _CUPY_IMPORT_ERROR: Optional[Exception] = e
else:
    _CUPY_IMPORT_ERROR = None

import numpy as np
import h5py
import anndata


def NBumiPearsonResiduals_h5ad_gpu(
    cleaned_filename: str,
    fit_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
    """
    Calculates Pearson residuals in an out-of-core, GPU-accelerated manner.
    The output is a dense matrix of residuals stored in an .h5ad file.

    Args:
        cleaned_filename: Path to the sparse, cleaned .h5ad input file.
        fit_filename: Path to the saved 'fit' object from NBumiFitModel (pickle).
        output_filename: Path to save the output .h5ad file.
        chunk_size: The number of cells to process at a time.
    """
    if cupy is None or cp_csr_matrix is None:
        raise ImportError(
            f"cupy is required for NBumiPearsonResiduals_h5ad_gpu but could not be imported: {_CUPY_IMPORT_ERROR}"
        )

    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResiduals_h5ad_gpu() | FILE: {cleaned_filename}")

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

    # Move necessary arrays to GPU
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)
    sizes_gpu = cupy.asarray(sizes, dtype=cupy.float64)

    # Prepare the output file on disk preserving obs/var
    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip")

    # Re-open with h5py to write X data in chunks (cells x genes)
    with h5py.File(output_filename, 'a') as f_out:
        # Create a chunked, resizable dataset for X
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')

        print("Phase [1/2]: COMPLETE")

        # --- Phase 2: Calculate Residuals in Chunks ---
        print("Phase [2/2]: Calculating Pearson residuals in chunks...")
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                # Read one sparse chunk from disk (CSR rows are cells)
                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = h5_data[start_idx:end_idx]
                indices_slice = h5_indices[start_idx:end_idx]
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                # Move to GPU and convert to dense for calculation
                counts_chunk_sparse_gpu = cp_csr_matrix((
                    cupy.asarray(data_slice, dtype=cupy.float64),
                    cupy.asarray(indices_slice),
                    cupy.asarray(indptr_slice)
                ), shape=(end_row - i, ng))
                counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                # Calculate expected counts ('mus') for the chunk
                tis_chunk_gpu = tis_gpu[i:end_row]
                mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total

                # Calculate the Pearson residuals for the chunk
                denominator_gpu = cupy.sqrt(mus_chunk_gpu + mus_chunk_gpu**2 / sizes_gpu[cupy.newaxis, :])
                pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu

                # Write the dense chunk of residuals to the output file (cells x genes)
                out_x[i:end_row, :] = pearson_chunk_gpu.get().astype(np.float32, copy=False)

        print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")


def NBumiPearsonResidualsApprox_h5ad_gpu(
    cleaned_filename: str,
    stats_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
    """
    Calculates approximate Pearson residuals (Poisson approximation) out-of-core on GPU.
    The output is a dense matrix of residuals stored in an .h5ad file.

    Args:
        cleaned_filename: Path to the sparse, cleaned .h5ad input file.
        stats_filename: Path to the saved 'stats' object (pickle) with keys tjs, tis, total, nc, ng.
        output_filename: Path to save the output .h5ad file.
        chunk_size: The number of cells to process at a time.
    """
    if cupy is None or cp_csr_matrix is None:
        raise ImportError(
            f"cupy is required for NBumiPearsonResidualsApprox_h5ad_gpu but could not be imported: {_CUPY_IMPORT_ERROR}"
        )

    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsApprox_h5ad_gpu() | FILE: {cleaned_filename}")

    # --- Phase 1: Initialization ---
    print("Phase [1/2]: Initializing parameters and preparing output file...")
    with open(stats_filename, 'rb') as f:
        stats = pickle.load(f)

    vals = stats
    tjs = vals['tjs'].values
    tis = vals['tis'].values
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    # Move necessary arrays to GPU
    tjs_gpu = cupy.asarray(tjs, dtype=cupy.float64)
    tis_gpu = cupy.asarray(tis, dtype=cupy.float64)

    # Prepare the output file on disk preserving obs/var
    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip")

    with h5py.File(output_filename, 'a') as f_out:
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')
        print("Phase [1/2]: COMPLETE")

        # --- Phase 2: Calculate Approximate Residuals in Chunks ---
        print("Phase [2/2]: Calculating approximate Pearson residuals in chunks...")
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
                ), shape=(end_row - i, ng))
                counts_chunk_dense_gpu = counts_chunk_sparse_gpu.todense()

                tis_chunk_gpu = tis_gpu[i:end_row]
                mus_chunk_gpu = tjs_gpu[cupy.newaxis, :] * tis_chunk_gpu[:, cupy.newaxis] / total

                # Calculate the approximate Pearson residuals (simpler denominator)
                denominator_gpu = cupy.sqrt(mus_chunk_gpu)
                pearson_chunk_gpu = (counts_chunk_dense_gpu - mus_chunk_gpu) / denominator_gpu

                out_x[i:end_row, :] = pearson_chunk_gpu.get().astype(np.float32, copy=False)

        print(f"Phase [2/2]: COMPLETE{' ' * 50}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")


