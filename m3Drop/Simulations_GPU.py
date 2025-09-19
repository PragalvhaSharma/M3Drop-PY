import time
from typing import Optional

# Lazy GPU imports with clear error if unavailable
try:
    import cupy
except Exception as e:  # optional dependency
    cupy = None
    _CUPY_IMPORT_ERROR: Optional[Exception] = e
else:
    _CUPY_IMPORT_ERROR = None

import numpy as np
import pandas as pd
import anndata
import h5py
from scipy.sparse import csr_matrix


def _default_mean2disp_vectorized(mu, coeffs=np.array([3.967816, -1.855054])):
    """
    Vectorized mean-to-dispersion function.
    Returns size parameter (1/dispersion) for each mean in mu.
    """
    log_mu = np.log10(mu)
    cv2 = np.exp(coeffs[0] + coeffs[1] * log_mu)
    variance = cv2 * (mu ** 2)
    disp = mu ** 2 / (variance - mu)
    return 1 / disp


def _ensure_cupy():
    if cupy is None:
        raise ImportError(
            f"cupy is required for GPU simulations but could not be imported: {_CUPY_IMPORT_ERROR}"
        )


def _add_dropouts_gpu(counts_chunk, means_chunk, K):
    """
    Adds dropouts to a count matrix chunk on the GPU.
    counts_chunk: CuPy array (genes x cells)
    means_chunk: 1D array-like of gene means (length = genes)
    K: dropout parameter
    """
    counts_chunk_gpu = cupy.asarray(counts_chunk)
    means_chunk_gpu = cupy.asarray(means_chunk)
    p_drop = K / (K + means_chunk_gpu[:, cupy.newaxis])
    toss = cupy.random.rand(*counts_chunk_gpu.shape)
    counts_chunk_gpu[toss < p_drop] = 0
    return counts_chunk_gpu


def MakeSimSparseData(
    filename: str,
    n_genes: int = 25000,
    n_cells: int = 100000,
    gene_chunk_size: int = 2500,
    K: float = 10.3,
    dispersion_factor: float = 1.0,
):
    """
    Generate large, realistic, sparse single-cell dataset out-of-core using GPU and save as .h5ad.
    Orientation: X is CSR with rows=cells, cols=genes for efficient row slicing by cell.
    """
    _ensure_cupy()
    start_time = time.perf_counter()
    print(f"Starting generation of sparse data: {n_genes} genes, {n_cells} cells.")

    # Metadata: obs=cells, var=genes to match rows=cells orientation
    print("Initializing output file and metadata...")
    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata_template = anndata.AnnData(obs=obs_df, var=var_df)
    adata_template.write_h5ad(filename)

    base_means = 10 ** np.random.normal(loc=1, scale=1, size=n_genes)

    # Precreate sparse group for CSR storage
    with h5py.File(filename, 'a') as f:
        if 'X' in f:
            del f['X']
        f.create_group('X')

    print("Generating data in chunks using the GPU...")
    with h5py.File(filename, 'a') as f:
        x_group = f['X']

        all_data = []
        all_indices = []
        all_indptr = [0]
        current_nnz = 0

        # We generate in gene chunks, resulting matrix shape will be genes x cells
        # Convert to CSR with rows=cells at the end via transpose
        for i in range(0, n_genes, gene_chunk_size):
            end_row = min(i + gene_chunk_size, n_genes)
            current_chunk_genes = end_row - i
            print(f"  -> Processing genes {i} to {end_row - 1}...")

            chunk_means = base_means[i:end_row]
            chunk_disp = _default_mean2disp_vectorized(chunk_means)
            chunk_size_param = 1 / (dispersion_factor * chunk_disp)

            n_param = cupy.asarray(chunk_size_param)[:, cupy.newaxis]
            mu_param = cupy.asarray(chunk_means)[:, cupy.newaxis]
            p_param = n_param / (n_param + mu_param)

            counts_gpu = cupy.random.negative_binomial(
                n=n_param, p=p_param, size=(current_chunk_genes, n_cells)
            )
            counts_with_dropouts_gpu = _add_dropouts_gpu(counts_gpu, chunk_means, K)
            sparse_chunk_cpu = csr_matrix(counts_with_dropouts_gpu.get())

            all_data.append(sparse_chunk_cpu.data)
            all_indices.append(sparse_chunk_cpu.indices)
            new_indptr = sparse_chunk_cpu.indptr[1:] + current_nnz
            all_indptr.extend(new_indptr.tolist())
            current_nnz += sparse_chunk_cpu.nnz

        print("Finalizing and writing sparse data to disk...")
        # Concatenate gene-chunk CSR components for genes x cells matrix
        data_gxC = np.concatenate(all_data, axis=0).astype('int32')
        indices_gxC = np.concatenate(all_indices, axis=0).astype('int32')
        indptr_gxC = np.array(all_indptr, dtype='int64')

        # Build genes x cells CSR and transpose to cells x genes CSR
        gxC = csr_matrix((data_gxC, indices_gxC, indptr_gxC), shape=(n_genes, n_cells))
        Cxg = gxC.transpose().tocsr()

        x_group.create_dataset('data', data=Cxg.data.astype('int32'))
        x_group.create_dataset('indices', data=Cxg.indices.astype('int32'))
        x_group.create_dataset('indptr', data=Cxg.indptr.astype('int64'))
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = np.array([n_cells, n_genes], dtype='int64')

    end_time = time.perf_counter()
    print(f"\nSparse data generation complete. File '{filename}' created.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")


def MakeSimSparseDataDE(
    filename: str,
    n_genes: int = 25000,
    n_cells: int = 100000,
    gene_chunk_size: int = 2500,
    frac_change: float = 0.1,
    fold_change: float = 10.0,
    sub_pop_frac: float = 0.5,
    K: float = 10.3,
    dispersion_factor: float = 1.0,
):
    """
    Generate sparse dataset with two cell populations and DE, GPU accelerated, as .h5ad.
    Orientation: CSR rows=cells, cols=genes.
    """
    _ensure_cupy()
    start_time = time.perf_counter()
    print(f"Starting DE simulation: {n_genes} genes, {n_cells} cells.")

    print("Setting up populations and identifying DE genes...")
    n_cells_pop2 = int(n_cells * sub_pop_frac)
    n_cells_pop1 = n_cells - n_cells_pop2
    cell_labels = np.array([1] * n_cells_pop1 + [2] * n_cells_pop2)

    n_tp_genes = int(n_genes * frac_change)
    tp_indices = np.random.choice(n_genes, size=n_tp_genes, replace=False)

    base_means = 10 ** np.random.normal(loc=1, scale=1, size=n_genes)
    changed_means = base_means.copy()
    changed_means[tp_indices] *= fold_change

    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs_df['population'] = cell_labels.astype(str)
    var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata_template = anndata.AnnData(obs=obs_df, var=var_df)
    adata_template.uns['DE_genes'] = [f"gene_{i}" for i in tp_indices]
    adata_template.write_h5ad(filename)
    with h5py.File(filename, 'a') as f:
        if 'X' in f:
            del f['X']
        f.create_group('X')

    print("Generating data in chunks using the GPU...")
    with h5py.File(filename, 'a') as f:
        x_group = f['X']
        all_data, all_indices, all_indptr = [], [], [0]
        current_nnz = 0

        for i in range(0, n_genes, gene_chunk_size):
            end_row = min(i + gene_chunk_size, n_genes)
            current_chunk_genes = end_row - i
            print(f"  -> Processing genes {i} to {end_row - 1}...")

            chunk_means1 = base_means[i:end_row]
            chunk_disp1 = _default_mean2disp_vectorized(chunk_means1)
            chunk_size1 = 1 / (dispersion_factor * chunk_disp1)
            n1 = cupy.asarray(chunk_size1)[:, cupy.newaxis]
            mu1 = cupy.asarray(chunk_means1)[:, cupy.newaxis]
            p1 = n1 / (n1 + mu1)
            counts1_gpu = cupy.random.negative_binomial(
                n=n1, p=p1, size=(current_chunk_genes, n_cells_pop1)
            )
            counts1_drop_gpu = _add_dropouts_gpu(counts1_gpu, chunk_means1, K)

            chunk_means2 = changed_means[i:end_row]
            chunk_disp2 = _default_mean2disp_vectorized(chunk_means2)
            chunk_size2 = 1 / (dispersion_factor * chunk_disp2)
            n2 = cupy.asarray(chunk_size2)[:, cupy.newaxis]
            mu2 = cupy.asarray(chunk_means2)[:, cupy.newaxis]
            p2 = n2 / (n2 + mu2)
            counts2_gpu = cupy.random.negative_binomial(
                n=n2, p=p2, size=(current_chunk_genes, n_cells_pop2)
            )
            counts2_drop_gpu = _add_dropouts_gpu(counts2_gpu, chunk_means2, K)

            combined_chunk_cpu = np.concatenate(
                (counts1_drop_gpu.get(), counts2_drop_gpu.get()), axis=1
            )
            sparse_chunk_cpu = csr_matrix(combined_chunk_cpu)

            all_data.append(sparse_chunk_cpu.data)
            all_indices.append(sparse_chunk_cpu.indices)
            new_indptr = sparse_chunk_cpu.indptr[1:] + current_nnz
            all_indptr.extend(new_indptr.tolist())
            current_nnz += sparse_chunk_cpu.nnz

        print("Finalizing and writing sparse data to disk...")
        data_gxC = np.concatenate(all_data, axis=0).astype('int32')
        indices_gxC = np.concatenate(all_indices, axis=0).astype('int32')
        indptr_gxC = np.array(all_indptr, dtype='int64')
        gxC = csr_matrix((data_gxC, indices_gxC, indptr_gxC), shape=(n_genes, n_cells))
        Cxg = gxC.transpose().tocsr()

        x_group.create_dataset('data', data=Cxg.data.astype('int32'))
        x_group.create_dataset('indices', data=Cxg.indices.astype('int32'))
        x_group.create_dataset('indptr', data=Cxg.indptr.astype('int64'))
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = np.array([n_cells, n_genes], dtype='int64')

    end_time = time.perf_counter()
    print(f"\nDE sparse data generation complete. File '{filename}' created.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")


def MakeSimSparseDataHVar(
    filename: str,
    n_genes: int = 25000,
    n_cells: int = 100000,
    gene_chunk_size: int = 2500,
    frac_change: float = 0.1,
    fold_change: float = 10.0,
    K: float = 10.3,
    base_dispersion_factor: float = 1.0,
):
    """
    Generate sparse dataset with a subset of highly variable genes (HVG), GPU accelerated, as .h5ad.
    Orientation: CSR rows=cells, cols=genes.
    """
    _ensure_cupy()
    start_time = time.perf_counter()
    print(f"Starting HVar simulation: {n_genes} genes, {n_cells} cells.")

    print("Setting up and identifying HVG genes...")
    n_tp_genes = int(n_genes * frac_change)
    tp_indices = np.random.choice(n_genes, size=n_tp_genes, replace=False)

    base_means = 10 ** np.random.normal(loc=1, scale=1, size=n_genes)
    dispersion_factors = np.full(n_genes, base_dispersion_factor, dtype=np.float64)
    dispersion_factors[tp_indices] *= fold_change

    obs_df = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

    adata_template = anndata.AnnData(obs=obs_df, var=var_df)
    adata_template.uns['HVar_genes'] = [f"gene_{i}" for i in tp_indices]
    adata_template.write_h5ad(filename)
    with h5py.File(filename, 'a') as f:
        if 'X' in f:
            del f['X']
        f.create_group('X')

    print("Generating data in chunks using the GPU...")
    with h5py.File(filename, 'a') as f:
        x_group = f['X']
        all_data, all_indices, all_indptr = [], [], [0]
        current_nnz = 0

        for i in range(0, n_genes, gene_chunk_size):
            end_row = min(i + gene_chunk_size, n_genes)
            current_chunk_genes = end_row - i
            print(f"  -> Processing genes {i} to {end_row - 1}...")

            chunk_means = base_means[i:end_row]
            chunk_disp_factors = dispersion_factors[i:end_row]
            chunk_disp = _default_mean2disp_vectorized(chunk_means)
            chunk_size_param = 1 / (chunk_disp_factors * chunk_disp)

            n_param = cupy.asarray(chunk_size_param)[:, cupy.newaxis]
            mu_param = cupy.asarray(chunk_means)[:, cupy.newaxis]
            p_param = n_param / (n_param + mu_param)

            counts_gpu = cupy.random.negative_binomial(
                n=n_param, p=p_param, size=(current_chunk_genes, n_cells)
            )
            counts_with_dropouts_gpu = _add_dropouts_gpu(counts_gpu, chunk_means, K)
            sparse_chunk_cpu = csr_matrix(counts_with_dropouts_gpu.get())

            all_data.append(sparse_chunk_cpu.data)
            all_indices.append(sparse_chunk_cpu.indices)
            new_indptr = sparse_chunk_cpu.indptr[1:] + current_nnz
            all_indptr.extend(new_indptr.tolist())
            current_nnz += sparse_chunk_cpu.nnz

        print("Finalizing and writing sparse data to disk...")
        data_gxC = np.concatenate(all_data, axis=0).astype('int32')
        indices_gxC = np.concatenate(all_indices, axis=0).astype('int32')
        indptr_gxC = np.array(all_indptr, dtype='int64')
        gxC = csr_matrix((data_gxC, indices_gxC, indptr_gxC), shape=(n_genes, n_cells))
        Cxg = gxC.transpose().tocsr()

        x_group.create_dataset('data', data=Cxg.data.astype('int32'))
        x_group.create_dataset('indices', data=Cxg.indices.astype('int32'))
        x_group.create_dataset('indptr', data=Cxg.indptr.astype('int64'))
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = np.array([n_cells, n_genes], dtype='int64')

    end_time = time.perf_counter()
    print(f"\nHVar sparse data generation complete. File '{filename}' created.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")


def verify_sparse_h5ad(filename: str):
    """
    Verify sparse .h5ad (CSR rows=cells). Avoid full materialization.
    """
    print(f"--- Verifying '{filename}' ---")
    try:
        adata = anndata.read_h5ad(filename, backed='r')
        print("File opened successfully.")
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    shape = adata.shape
    print(f"Shape: {shape}")
    # Check small slice type
    small = adata[:1, :1].X
    print(f"Small slice type: {type(small)}")

    # Compute sparsity by iterating over row chunks (cells)
    total = shape[0] * shape[1]
    nnz = 0
    with h5py.File(filename, 'r') as f:
        indptr = f['X']['indptr']
        for i in range(0, shape[0], 10000):
            j = min(i + 10000, shape[0])
            nnz += int(indptr[j] - indptr[i])
    sparsity = 1 - (nnz / total)
    print(f"Sparsity: {sparsity:.2%}")


def verify_sparse_DE_h5ad(filename: str):
    print(f"--- Verifying DE simulation file '{filename}' ---")
    try:
        adata = anndata.read_h5ad(filename, backed='r')
        print("File opened successfully.")
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    print(f"Shape: {adata.shape}")
    if 'population' in adata.obs.columns:
        counts = adata.obs['population'].value_counts()
        print(f"Population counts: {counts.to_dict()}")
    else:
        print("Missing 'population' in obs.")

    if 'DE_genes' in adata.uns:
        print(f"DE genes: {len(adata.uns['DE_genes'])}")
    else:
        print("Missing 'DE_genes' in uns.")

    with h5py.File(filename, 'r') as f:
        indptr = f['X']['indptr']
        nnz = int(indptr[-1])
    total = adata.shape[0] * adata.shape[1]
    sparsity = 1 - (nnz / total)
    print(f"Sparsity: {sparsity:.2%}")


def verify_cleaned_DE_h5ad(filename: str):
    print(f"--- Verifying CLEANED file '{filename}' ---")
    adata = anndata.read_h5ad(filename, backed='r')
    print("File opened successfully.")
    print(f"Shape: {adata.shape}")
    small = adata[:1, :1].X
    print(f"Small slice dtype: {getattr(small, 'dtype', type(small))}")
    kept = True
    with h5py.File(filename, 'r') as f:
        indptr = f['X']['indptr']
        # If any row has zero nnz, difference between consecutive indptr is zero
        row_nnzs = np.diff(indptr[:])
        kept = bool(np.all(row_nnzs > 0))
    print("All rows non-zero:" if kept else "Zero rows present:", kept)


