import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp
import h5py


def M3DropConvertData(input_data, is_log=False, is_counts=False, pseudocount=1, preserve_sparse=False):
    """
    Converts various data formats to a normalized, non-log-transformed matrix.

    Recognizes a variety of object types, extracts expression matrices, and
    converts them to a format suitable for M3Drop functions.

    Parameters
    ----------
    input_data : AnnData, pd.DataFrame, np.ndarray
        The input data.
    is_log : bool, default=False
        Whether the data has been log-transformed.
    is_counts : bool, default=False
        Whether the data is raw, unnormalized counts.
    pseudocount : float, default=1
        Pseudocount added before log-transformation.
    preserve_sparse : bool, default=True
        Whether to preserve sparse matrix format for memory efficiency.

    Returns
    -------
    pd.DataFrame or scipy.sparse matrix
        A normalized, non-log-transformed matrix. By default returns a pandas
        DataFrame. If preserve_sparse=True and input is sparse, returns a
        sparse matrix wrapper that preserves metadata.
    """
    def remove_undetected_genes(mat, gene_names=None, cell_names=None):
        """Helper to filter out genes with no expression across all cells"""
        if sp.issparse(mat):
            # For sparse matrices, use efficient sparse operations
            detected = np.array(mat.sum(axis=1)).flatten() > 0
            if np.sum(~detected) > 0:
                print(f"Removing {np.sum(~detected)} undetected genes.")
            filtered_mat = mat[detected, :]
            if gene_names is not None:
                gene_names = gene_names[detected]
            return filtered_mat, gene_names
        elif isinstance(mat, pd.DataFrame):
            detected = mat.sum(axis=1) > 0
            if np.sum(~detected) > 0:
                print(f"Removing {np.sum(~detected)} undetected genes.")
            return mat[detected], None
        else:
            # numpy array
            detected = np.sum(mat, axis=1) > 0
            if np.sum(~detected) > 0:
                print(f"Removing {np.sum(~detected)} undetected genes.")
            filtered_mat = mat[detected, :]
            if gene_names is not None:
                gene_names = gene_names[detected]
            return filtered_mat, gene_names

    from scipy.sparse import issparse
    
    # Store gene and cell names for later use
    gene_names = None
    cell_names = None
    
    # 1. Handle Input Type and convert to appropriate format
    if isinstance(input_data, AnnData):
        # Robustly handle both AnnData and AnnData.T
        # We always return genes × cells. Decide which axis represents genes.
        if input_data.n_vars >= input_data.n_obs:
            # Standard orientation: obs=cells, var=genes
            gene_names = input_data.var_names.copy()
            cell_names = input_data.obs_names.copy()
            if issparse(input_data.X) and preserve_sparse:
                counts = input_data.X.T.tocsr()  # genes × cells
            else:
                if issparse(input_data.X):
                    counts = pd.DataFrame(input_data.X.toarray().T, index=gene_names, columns=cell_names)
                else:
                    counts = pd.DataFrame(input_data.X.T, index=gene_names, columns=cell_names)
        else:
            # Transposed orientation: obs=genes, var=cells
            gene_names = input_data.obs_names.copy()
            cell_names = input_data.var_names.copy()
            if issparse(input_data.X) and preserve_sparse:
                counts = input_data.X.tocsr()  # already genes × cells
            else:
                if issparse(input_data.X):
                    counts = pd.DataFrame(input_data.X.toarray(), index=gene_names, columns=cell_names)
                else:
                    counts = pd.DataFrame(input_data.X, index=gene_names, columns=cell_names)
    elif isinstance(input_data, pd.DataFrame):
        counts = input_data
    elif isinstance(input_data, np.ndarray):
        if preserve_sparse:
            # Convert to sparse for memory efficiency
            counts = sp.csr_matrix(input_data)
            gene_names = np.array([f"Gene_{i}" for i in range(input_data.shape[0])])
            cell_names = np.array([f"Cell_{i}" for i in range(input_data.shape[1])])
        else:
            counts = pd.DataFrame(input_data)
    elif issparse(input_data):
        if preserve_sparse:
            counts = input_data.tocsr()  # Ensure CSR format
        else:
            counts = pd.DataFrame(input_data.toarray())
    else:
        raise TypeError(f"Unrecognized input format: {type(input_data)}")

    # 2. Handle log-transformation
    if is_log:
        if issparse(counts):
            # Handle sparse log transformation
            counts = counts.copy()
            counts.data = 2**counts.data - pseudocount
        elif isinstance(counts, pd.DataFrame):
            counts = 2**counts - pseudocount
        else:
            counts = 2**counts - pseudocount
    
    # 3. Handle normalization for raw counts
    if is_counts:
        if issparse(counts):
            # Efficient sparse normalization
            sf = np.array(counts.sum(axis=0)).flatten()
            sf[sf == 0] = 1  # Avoid division by zero
            # Normalize to CPM (counts per million) - sparse matrix operations
            sf_cpm = 1e6 / sf
            # Create diagonal matrix for efficient multiplication
            sf_diag = sp.diags(sf_cpm, format='csr')
            norm_counts = counts @ sf_diag
            
            # Filter undetected genes
            filtered_counts, filtered_gene_names = remove_undetected_genes(norm_counts, gene_names, cell_names)
            
            if preserve_sparse:
                # Return sparse matrix with metadata if possible
                return SparseMat3Drop(filtered_counts, gene_names=filtered_gene_names, cell_names=cell_names)
            else:
                # Convert to DataFrame for compatibility
                if gene_names is not None and cell_names is not None:
                    filtered_gene_names = gene_names if filtered_gene_names is None else filtered_gene_names
                    return pd.DataFrame(filtered_counts.toarray(), 
                                      index=filtered_gene_names, 
                                      columns=cell_names)
                else:
                    return pd.DataFrame(filtered_counts.toarray())
        else:
            # DataFrame/array normalization as before
            sf = counts.sum(axis=0)
            sf[sf == 0] = 1  # Avoid division by zero
            norm_counts = (counts / sf) * 1e6
            filtered_result, _ = remove_undetected_genes(norm_counts)
            return filtered_result
    
    # 4. If data is already normalized (not raw counts), just filter
    filtered_result, filtered_gene_names = remove_undetected_genes(counts, gene_names, cell_names)
    
    if preserve_sparse and issparse(filtered_result):
        return SparseMat3Drop(filtered_result, gene_names=filtered_gene_names, cell_names=cell_names)
    else:
        return filtered_result


class SparseMat3Drop:
    """
    Wrapper class for sparse matrices with gene/cell name metadata.
    Maintains memory efficiency while preserving essential metadata.
    """
    def __init__(self, matrix, gene_names=None, cell_names=None):
        self.matrix = matrix
        self.gene_names = gene_names
        self.cell_names = cell_names
        self.shape = matrix.shape
    
    def __getattr__(self, name):
        # Delegate to the underlying sparse matrix
        return getattr(self.matrix, name)
    
    def toarray(self):
        """Convert to dense array"""
        return self.matrix.toarray()

    def to_dataframe(self):
        """Convert to pandas DataFrame with proper indices"""
        if self.gene_names is not None and self.cell_names is not None:
            return pd.DataFrame(self.matrix.toarray(), 
                              index=self.gene_names, 
                              columns=self.cell_names)
        else:
            return pd.DataFrame(self.matrix.toarray())
    
    def sum(self, axis=None):
        """Sum operation maintaining sparse efficiency"""
        return self.matrix.sum(axis=axis)
    
    def mean(self, axis=None):
        """Mean operation maintaining sparse efficiency"""
        return self.matrix.mean(axis=axis)


def _ensure_index(gene_names, n_genes):
    """Return a pandas Index of gene names, falling back to sequential labels."""
    if gene_names is None:
        return pd.Index([f"Gene_{i}" for i in range(n_genes)])
    if isinstance(gene_names, pd.Index):
        return gene_names.astype(str)
    return pd.Index(np.asarray(gene_names, dtype=object).astype(str))


def ann_data_to_sparse_gene_matrix(adata, layer=None, use_raw=False, dtype=np.float32):
    """Generate a ``SparseMat3Drop`` from an AnnData object without densifying the matrix."""
    if not isinstance(adata, AnnData):
        raise TypeError("Expected an AnnData object.")

    if use_raw:
        if adata.raw is None:
            raise ValueError("Requested raw matrix, but `adata.raw` is empty.")
        matrix_data = adata.raw.X
        gene_names = adata.raw.var_names
        cell_names = adata.raw.obs_names
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in AnnData object.")
        matrix_data = adata.layers[layer]
        gene_names = adata.var_names
        cell_names = adata.obs_names
    else:
        matrix_data = adata.X
        gene_names = adata.var_names
        cell_names = adata.obs_names

    if sp.issparse(matrix_data):
        matrix = matrix_data.T.tocsr()
        if dtype is not None:
            matrix = matrix.astype(dtype, copy=False)
    else:
        matrix = sp.csr_matrix(np.asarray(matrix_data.T, dtype=dtype))

    gene_index = _ensure_index(gene_names, matrix.shape[0])
    cell_index = pd.Index(np.asarray(cell_names, dtype=object).astype(str))

    return SparseMat3Drop(matrix, gene_names=gene_index, cell_names=cell_index)


def _iter_sparse_rows(matrix, chunk_size=64, dtype=np.float32):
    """Yield dense blocks of rows from a sparse matrix, chunked to limit memory usage."""
    matrix = matrix.tocsr()
    n_rows = matrix.shape[0]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        block = matrix[start:end, :].toarray()
        if dtype is not None:
            block = block.astype(dtype, copy=False)
        yield start, end, block


def compute_row_mean_and_var(expr_mat, ddof=1):
    """Compute per-gene mean and variance for dense or sparse matrices without densifying fully."""
    if isinstance(expr_mat, SparseMat3Drop):
        matrix = expr_mat.matrix
        gene_index = _ensure_index(expr_mat.gene_names, expr_mat.shape[0])
        is_sparse = True
    elif sp.issparse(expr_mat):
        matrix = expr_mat
        gene_index = _ensure_index(None, expr_mat.shape[0])
        is_sparse = True
    elif isinstance(expr_mat, pd.DataFrame):
        matrix = expr_mat.values
        gene_index = _ensure_index(expr_mat.index, expr_mat.shape[0])
        is_sparse = False
    elif isinstance(expr_mat, np.ndarray):
        matrix = expr_mat
        gene_index = _ensure_index(None, expr_mat.shape[0])
        is_sparse = False
    else:
        raise TypeError("Unsupported matrix type for compute_row_mean_and_var.")

    if is_sparse:
        n_cells = matrix.shape[1]
        if n_cells == 0:
            zeros = np.zeros(matrix.shape[0], dtype=np.float32)
            return pd.Series(zeros, index=gene_index), pd.Series(zeros, index=gene_index)

        sum_counts = np.array(matrix.sum(axis=1)).ravel()
        sum_sq = np.array(matrix.multiply(matrix).sum(axis=1)).ravel()
        means = sum_counts / n_cells

        denom = n_cells - ddof
        if denom <= 0:
            vars_ = np.zeros_like(means)
        else:
            vars_ = (sum_sq - (sum_counts ** 2) / n_cells) / denom
            vars_ = np.maximum(vars_, 0)

        return pd.Series(means, index=gene_index), pd.Series(vars_, index=gene_index)

    # Dense case
    means = matrix.mean(axis=1)
    vars_ = matrix.var(axis=1, ddof=ddof)
    return pd.Series(means, index=gene_index), pd.Series(vars_, index=gene_index)


def compute_gene_statistics_h5ad(filename, chunk_size=5000):
    """Compute gene-level statistics for large .h5ad files without
    materialising the full matrix in memory.

    The function streams the sparse matrix in cell chunks and computes the
    quantities required by ``bg__calc_variables``: mean expression (``s``),
    dropout rate (``p``), and their standard errors. Counts are normalised to
    counts per million (CPM) to mirror ``M3DropConvertData``.

    Parameters
    ----------
    filename : str
        Path to the .h5ad file containing a sparse count matrix.
    chunk_size : int, default=5000
        Number of cells to load per chunk when streaming from disk. Adjust to
        balance memory use and throughput.

    Returns
    -------
    tuple(dict, int)
        A tuple where the first element is a ``gene_info`` style dictionary
        with keys ``s``, ``s_stderr``, ``p`` and ``p_stderr`` (all pandas
        Series indexed by gene names), and the second element is the total
        number of cells in the dataset.
    """

    def _read_axis_index(group):
        if group is None:
            return None
        for key in ("_index", "index", "names"):
            if key in group:
                data = group[key][...]
                break
        else:
            return None

        if isinstance(data, np.ndarray):
            if data.dtype.kind in {"S", "O"}:
                decoded = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in data]
            else:
                decoded = data.astype(str).tolist()
        else:
            decoded = [str(data)]
        return pd.Index(decoded)

    with h5py.File(filename, "r") as f:
        if "X" not in f:
            raise ValueError(f"Expected dataset 'X' in {filename} but none was found.")

        X_group = f["X"]
        encoding_type = X_group.attrs.get("encoding-type")
        if encoding_type != "csr_matrix":
            raise ValueError(
                f"Only CSR-encoded sparse matrices are supported for streaming; found '{encoding_type}'."
            )

        shape = X_group.attrs.get("shape")
        if shape is None or len(shape) != 2:
            raise ValueError("Missing or invalid shape attribute for dataset 'X'.")

        n_cells, n_genes = map(int, shape)
        indptr = X_group["indptr"][...].astype(np.int64, copy=False)
        indices_ds = X_group["indices"]
        data_ds = X_group["data"]

        gene_names = _read_axis_index(f.get("var"))
        if gene_names is None or len(gene_names) != n_genes:
            gene_names = pd.Index([f"Gene_{i}" for i in range(n_genes)])

        sum_per_gene = np.zeros(n_genes, dtype=np.float64)
        sum_sq_per_gene = np.zeros(n_genes, dtype=np.float64)
        nonzero_per_gene = np.zeros(n_genes, dtype=np.int64)

        for cell_start in range(0, n_cells, chunk_size):
            cell_end = min(cell_start + chunk_size, n_cells)
            nnz_start = indptr[cell_start]
            nnz_end = indptr[cell_end]
            if nnz_end - nnz_start == 0:
                continue

            indices_chunk = indices_ds[nnz_start:nnz_end]
            data_chunk = data_ds[nnz_start:nnz_end]

            rel_indptr = indptr[cell_start:cell_end + 1] - nnz_start
            chunk_matrix = sp.csr_matrix(
                (data_chunk.astype(np.float64, copy=False), indices_chunk, rel_indptr),
                shape=(cell_end - cell_start, n_genes)
            )

            cell_sums = np.array(chunk_matrix.sum(axis=1)).ravel()
            scaling = np.zeros_like(cell_sums, dtype=np.float64)
            valid_cells = cell_sums > 0
            scaling[valid_cells] = 1e6 / cell_sums[valid_cells]

            scaled_chunk = chunk_matrix.multiply(scaling[:, None])

            sum_per_gene += np.array(scaled_chunk.sum(axis=0)).ravel()
            sum_sq_per_gene += np.array(scaled_chunk.power(2).sum(axis=0)).ravel()
            nonzero_per_gene += np.asarray(chunk_matrix.getnnz(axis=0)).ravel()

    n_cells_float = float(n_cells)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_gene = sum_per_gene / n_cells_float
        mean_sq_per_gene = sum_sq_per_gene / n_cells_float
        variance_term = np.maximum(mean_sq_per_gene - mean_per_gene**2, 0.0)
        s_stderr = np.sqrt(variance_term / n_cells_float)

    p = 1.0 - nonzero_per_gene.astype(np.float64) / n_cells_float
    p_stderr = np.sqrt(p * (1.0 - p) / n_cells_float)

    detected = nonzero_per_gene > 0
    if np.sum(~detected) > 0:
        print(f"Removing {np.sum(~detected)} undetected genes.")

    mean_per_gene = mean_per_gene[detected]
    s_stderr = s_stderr[detected]
    p = p[detected]
    p_stderr = p_stderr[detected]
    gene_index = gene_names[detected]

    gene_info = {
        's': pd.Series(mean_per_gene, index=gene_index),
        's_stderr': pd.Series(s_stderr, index=gene_index),
        'p': pd.Series(p, index=gene_index),
        'p_stderr': pd.Series(p_stderr, index=gene_index)
    }

    return gene_info, n_cells


def bg__calc_variables(expr_mat):
    """
    Calculates a suite of gene-specific variables including: mean, dropout rate,
    and their standard errors. Updated to match R implementation behavior and 
    handle sparse matrices efficiently.
    """
    # Handle different input types
    if hasattr(expr_mat, 'matrix') and hasattr(expr_mat, 'gene_names'):
        # SparseMat3Drop object
        expr_mat_values = expr_mat.matrix
        gene_names = expr_mat.gene_names if expr_mat.gene_names is not None else pd.RangeIndex(start=0, stop=expr_mat.shape[0], step=1)
        is_sparse = True
    elif isinstance(expr_mat, pd.DataFrame):
        expr_mat_values = expr_mat.values
        gene_names = expr_mat.index
        is_sparse = False
    elif sp.issparse(expr_mat):
        expr_mat_values = expr_mat
        gene_names = pd.RangeIndex(start=0, stop=expr_mat.shape[0], step=1)
        is_sparse = True
    else:
        expr_mat_values = expr_mat
        gene_names = pd.RangeIndex(start=0, stop=expr_mat.shape[0], step=1)
        is_sparse = False

    # Check for NA values
    if is_sparse:
        # For sparse matrices, only check non-zero values
        if np.sum(np.isnan(expr_mat_values.data)) > 0:
            raise ValueError("Error: Expression matrix contains NA values")
    else:
        if np.sum(np.isnan(expr_mat_values)) > 0:
            raise ValueError("Error: Expression matrix contains NA values")
    
    # Check for negative values
    if is_sparse:
        lowest = np.min(expr_mat_values.data) if expr_mat_values.nnz > 0 else 0
    else:
        lowest = np.min(expr_mat_values)
        
    if lowest < 0:
        raise ValueError("Error: Expression matrix cannot contain negative values! Has the matrix been log-transformed?")
    
    # Deal with strangely normalized data (no zeros)
    if lowest > 0:
        print("Warning: No zero values (dropouts) detected will use minimum expression value instead.")
        min_val = lowest + 0.05
        if is_sparse:
            # For sparse matrices, we need to handle this differently
            expr_mat_values.data[expr_mat_values.data == min_val] = 0
            expr_mat_values.eliminate_zeros()
        else:
            expr_mat_values[expr_mat_values == min_val] = 0
    
    # Check if we have enough zeros (efficient for sparse matrices)
    if is_sparse:
        total_elements = expr_mat_values.shape[0] * expr_mat_values.shape[1]
        non_zero_elements = expr_mat_values.nnz
        sum_zero = total_elements - non_zero_elements
    else:
        sum_zero = np.prod(expr_mat_values.shape) - np.sum(expr_mat_values > 0)
    
    total_elements = np.prod(expr_mat_values.shape)
    if sum_zero < 0.1 * total_elements:
        print("Warning: Expression matrix contains few zero values (dropouts) this may lead to poor performance.")

    # Calculate dropout rate efficiently
    if is_sparse:
        # For sparse matrices, count non-zeros per row
        non_zero_per_gene = np.array((expr_mat_values > 0).sum(axis=1)).flatten()
    else:
        non_zero_per_gene = np.sum(expr_mat_values > 0, axis=1)
    
    p = 1 - non_zero_per_gene / expr_mat_values.shape[1]
    
    # Remove undetected genes
    if np.sum(p == 1) > 0:
        print(f"Warning: Removing {np.sum(p == 1)} undetected genes.")
        detected = p < 1
        if is_sparse:
            expr_mat_values = expr_mat_values[detected, :]
        else:
            expr_mat_values = expr_mat_values[detected, :]
        if isinstance(gene_names, pd.Index):
            gene_names = gene_names[detected]
        else:
            gene_names = gene_names[detected] if hasattr(gene_names, '__getitem__') else np.arange(expr_mat_values.shape[0])
        p = 1 - non_zero_per_gene[detected] / expr_mat_values.shape[1]

    if expr_mat_values.shape[0] == 0:
        return {
            's': pd.Series(dtype=float),
            's_stderr': pd.Series(dtype=float),
            'p': pd.Series(dtype=float),
            'p_stderr': pd.Series(dtype=float)
        }

    # Calculate mean expression efficiently
    if is_sparse:
        s = np.array(expr_mat_values.mean(axis=1)).flatten()
        # Calculate variance for sparse matrices
        mean_sq = np.array((expr_mat_values.multiply(expr_mat_values)).mean(axis=1)).flatten()
        s_stderr = np.sqrt((mean_sq - s**2) / expr_mat_values.shape[1])
    else:
        s = np.mean(expr_mat_values, axis=1)
        s_stderr = np.sqrt((np.mean(expr_mat_values**2, axis=1) - s**2) / expr_mat_values.shape[1])
    
    p_stderr = np.sqrt(p * (1 - p) / expr_mat_values.shape[1])

    return {
        's': pd.Series(s, index=gene_names),
        's_stderr': pd.Series(s_stderr, index=gene_names),
        'p': pd.Series(p, index=gene_names),
        'p_stderr': pd.Series(p_stderr, index=gene_names)
    }


def hidden__invert_MM(K, p):
    """
    Helper function for Michaelis-Menten inversion.
    """
    return K * (1 - p) / p


def bg__horizontal_residuals_MM_log10(K, p, s):
    """
    Calculate horizontal residuals for Michaelis-Menten model in log10 space.
    """
    return np.log10(s) - np.log10(hidden__invert_MM(K, p))


def hidden_getAUC(gene, labels):
    """
    Original AUC calculation function (alternative to fast version).
    Uses ROCR-style AUC calculation like the R implementation.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    
    labels = np.array(labels)
    ranked = np.argsort(np.argsort(gene)) + 1  # Rank calculation
    
    # Get average score for each cluster
    unique_labels = np.unique(labels)
    mean_scores = {}
    for label in unique_labels:
        mean_scores[label] = np.mean(ranked[labels == label])
    
    # Get cluster with highest average score
    max_score = max(mean_scores.values())
    posgroups = [k for k, v in mean_scores.items() if v == max_score]
    
    if len(posgroups) > 1:
        return [-1, -1, -1]  # Return negatives if there is a tie
    
    posgroup = posgroups[0]
    
    # Create truth vector for predictions
    truth = (labels == posgroup).astype(int)
    
    try:
        # Calculate AUC using sklearn
        auc = roc_auc_score(truth, ranked)
        # Calculate p-value using Wilcoxon test
        _, pval = mannwhitneyu(gene[truth == 1], gene[truth == 0], alternative='two-sided')
    except ValueError:
        return [0, posgroup, 1]
    
    return [auc, posgroup, pval]


def hidden_fast_AUC_m3drop(expression_vec, labels):
    """
    Fast AUC calculation for M3Drop marker identification.
    """
    from scipy.stats import mannwhitneyu
    
    R = np.argsort(np.argsort(expression_vec)) + 1  # Rank calculation
    labels = np.array(labels)
    
    # Get average rank for each cluster
    unique_labels = np.unique(labels)
    mean_ranks = {}
    for label in unique_labels:
        mean_ranks[label] = np.mean(R[labels == label])
    
    # Find cluster with highest average score
    max_rank = max(mean_ranks.values())
    posgroups = [k for k, v in mean_ranks.items() if v == max_rank]
    
    if len(posgroups) > 1:
        return [-1, -1, -1]  # Tie for highest score
    
    posgroup = posgroups[0]
    truth = labels == posgroup
    
    if np.sum(truth) == 0 or np.sum(~truth) == 0:
        return [0 if np.sum(truth) == 0 else 1, posgroup, 1]
    
    try:
        stat, pval = mannwhitneyu(expression_vec[truth], expression_vec[~truth], alternative='two-sided')
    except ValueError:
        return [0, posgroup, 1]
    
    # Calculate AUC using Mann-Whitney U statistic
    N1 = np.sum(truth)
    N2 = np.sum(~truth)
    U2 = np.sum(R[~truth]) - N2 * (N2 + 1) / 2
    AUC = 1 - U2 / (N1 * N2)
    
    return [AUC, posgroup, pval]


def _compute_markers_from_sparse(matrix, labels, gene_index, chunk_size=64):
    labels = np.asarray(labels)
    if labels.shape[0] != matrix.shape[1]:
        raise ValueError("Length of labels does not match number of cells.")

    results = []
    for start, end, block in _iter_sparse_rows(matrix, chunk_size=chunk_size):
        for offset, row in enumerate(block):
            results.append(hidden_fast_AUC_m3drop(row, labels))

    auc_df = pd.DataFrame(results, index=gene_index, columns=['AUC', 'Group', 'pval'])
    auc_df['AUC'] = pd.to_numeric(auc_df['AUC'])
    auc_df['pval'] = pd.to_numeric(auc_df['pval'])
    auc_df['Group'] = auc_df['Group'].astype(str)
    auc_df.loc[auc_df['Group'] == '-1', 'Group'] = "Ambiguous"
    auc_df = auc_df[auc_df['AUC'] > 0]
    auc_df = auc_df.sort_values(by='AUC', ascending=False)
    return auc_df


def M3DropGetMarkers(expr_mat, labels, chunk_size=64):
    """
    Identifies marker genes using the area under the ROC curve.

    Calculates area under the ROC curve for each gene to predict the best
    group of cells from all other cells.

    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Normalized expression values.
    labels : array-like
        Group IDs for each cell/sample.

    Returns
    -------
    pd.DataFrame
        DataFrame with AUC, group, and p-value for each gene.
    """
    if isinstance(expr_mat, SparseMat3Drop):
        gene_index = _ensure_index(expr_mat.gene_names, expr_mat.shape[0])
        return _compute_markers_from_sparse(expr_mat.matrix, labels, gene_index, chunk_size=chunk_size)

    if sp.issparse(expr_mat):
        gene_index = _ensure_index(None, expr_mat.shape[0])
        return _compute_markers_from_sparse(expr_mat, labels, gene_index, chunk_size=chunk_size)

    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
    elif not isinstance(expr_mat, pd.DataFrame):
        raise TypeError("expr_mat must be a pandas DataFrame, ndarray, sparse matrix, or SparseMat3Drop.")

    if len(labels) != expr_mat.shape[1]:
        raise ValueError("Length of labels does not match number of cells.")

    gene_index = _ensure_index(expr_mat.index, expr_mat.shape[0])
    labels_array = np.asarray(labels)

    # Apply the fast AUC function to each gene
    aucs = expr_mat.apply(lambda gene: hidden_fast_AUC_m3drop(gene.values.astype(np.float32, copy=False), labels_array), axis=1)

    # Convert results to DataFrame
    auc_df = pd.DataFrame(aucs.tolist(), index=gene_index, columns=['AUC', 'Group', 'pval'])

    # Convert data types
    auc_df['AUC'] = pd.to_numeric(auc_df['AUC'])
    auc_df['pval'] = pd.to_numeric(auc_df['pval'])
    auc_df['Group'] = auc_df['Group'].astype(str)

    # Handle ambiguous cases
    auc_df.loc[auc_df['Group'] == '-1', 'Group'] = "Ambiguous"

    # Filter and sort
    auc_df = auc_df[auc_df['AUC'] > 0]
    auc_df = auc_df.sort_values(by='AUC', ascending=False)

    return auc_df
