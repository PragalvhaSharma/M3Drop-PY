import pickle
import time
import numpy as np
import h5py
import anndata
from scipy.sparse import csr_matrix as sp_csr_matrix


def NBumiPearsonResidualsCPU(
    cleaned_filename: str,
    fit_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
    """
    Calculates Pearson residuals in an out-of-core, CPU-only manner.
    The output is a dense matrix of residuals.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsCPU() | FILE: {cleaned_filename}")

    print("Phase [1/2]: Initializing parameters and preparing output file...")
    with open(fit_filename, 'rb') as f:
        fit = pickle.load(f)

    vals = fit['vals']
    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    sizes = fit['sizes'].values.astype(np.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip")

    with h5py.File(output_filename, 'a') as f_out:
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')
        print("Phase [1/2]: COMPLETE")

        print("Phase [2/2]: Calculating Pearson residuals in chunks...")
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
                indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int64)
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                counts_chunk_sparse = sp_csr_matrix(
                    (data_slice, indices_slice, indptr_slice),
                    shape=(end_row - i, ng)
                )
                counts_chunk_dense = counts_chunk_sparse.toarray()

                tis_chunk = tis[i:end_row]
                mus_chunk = tjs[np.newaxis, :] * tis_chunk[:, np.newaxis] / total

                denominator = np.sqrt(mus_chunk + mus_chunk**2 / sizes[np.newaxis, :])
                denominator = np.where(denominator == 0, 1, denominator)
                pearson_chunk = (counts_chunk_dense - mus_chunk) / denominator

                out_x[i:end_row, :] = pearson_chunk.astype(np.float32)

        print(f"Phase [2/2]: COMPLETE{' '*50}")

    if hasattr(adata_in, "file") and adata_in.file is not None:
        adata_in.file.close()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")


def NBumiPearsonResidualsApproxCPU(
    cleaned_filename: str,
    stats_filename: str,
    output_filename: str,
    chunk_size: int = 5000
):
    """
    Calculates approximate Pearson residuals in an out-of-core, CPU-only manner.
    """
    start_time = time.perf_counter()
    print(f"FUNCTION: NBumiPearsonResidualsApproxCPU() | FILE: {cleaned_filename}")

    print("Phase [1/2]: Initializing parameters and preparing output file...")
    with open(stats_filename, 'rb') as f:
        stats = pickle.load(f)

    vals = stats
    tjs = vals['tjs'].values.astype(np.float64)
    tis = vals['tis'].values.astype(np.float64)
    total = vals['total']
    nc, ng = vals['nc'], vals['ng']

    adata_in = anndata.read_h5ad(cleaned_filename, backed='r')
    adata_out = anndata.AnnData(obs=adata_in.obs, var=adata_in.var)
    adata_out.write_h5ad(output_filename, compression="gzip")

    with h5py.File(output_filename, 'a') as f_out:
        out_x = f_out.create_dataset('X', shape=(nc, ng), chunks=(chunk_size, ng), dtype='float32')
        print("Phase [1/2]: COMPLETE")

        print("Phase [2/2]: Calculating approximate Pearson residuals in chunks...")
        with h5py.File(cleaned_filename, 'r') as f_in:
            h5_indptr = f_in['X']['indptr']
            h5_data = f_in['X']['data']
            h5_indices = f_in['X']['indices']

            for i in range(0, nc, chunk_size):
                end_row = min(i + chunk_size, nc)
                print(f"Phase [2/2]: Processing: {end_row} of {nc} cells.", end='\r')

                start_idx, end_idx = h5_indptr[i], h5_indptr[end_row]
                data_slice = np.array(h5_data[start_idx:end_idx], dtype=np.float64)
                indices_slice = np.array(h5_indices[start_idx:end_idx], dtype=np.int64)
                indptr_slice = h5_indptr[i:end_row+1] - h5_indptr[i]

                counts_chunk_sparse = sp_csr_matrix(
                    (data_slice, indices_slice, indptr_slice),
                    shape=(end_row - i, ng)
                )
                counts_chunk_dense = counts_chunk_sparse.toarray()

                tis_chunk = tis[i:end_row]
                mus_chunk = tjs[np.newaxis, :] * tis_chunk[:, np.newaxis] / total

                denominator = np.sqrt(mus_chunk)
                denominator = np.where(denominator == 0, 1, denominator)
                pearson_chunk = (counts_chunk_dense - mus_chunk) / denominator

                out_x[i:end_row, :] = pearson_chunk.astype(np.float32)

        print(f"Phase [2/2]: COMPLETE{' '*50}")

    if hasattr(adata_in, "file") and adata_in.file is not None:
        adata_in.file.close()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds.\n")
