import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp


def M3DropConvertData(input_data, is_log=False, is_counts=False, pseudocount=1, preserve_sparse=True):
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
        A normalized, non-log-transformed matrix. Returns sparse matrix if 
        preserve_sparse=True and input is sparse, otherwise DataFrame.
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
        # AnnData stores data as cells x genes, we need genes x cells for M3Drop
        # So var_names are the genes, obs_names are the cells
        gene_names = input_data.var_names.copy()  # These are the actual gene names
        cell_names = input_data.obs_names.copy()  # These are the actual cell names
        
        if issparse(input_data.X) and preserve_sparse:
            # Keep as sparse matrix but note that AnnData is cells x genes, we need genes x cells
            counts = input_data.X.T.tocsr()  # Transpose to genes x cells
        else:
            # Convert to DataFrame as before
            if issparse(input_data.X):
                # AnnData: cells x genes -> transpose to genes x cells for DataFrame
                counts = pd.DataFrame(input_data.X.toarray().T, index=input_data.var_names, columns=input_data.obs_names)
            else:
                counts = pd.DataFrame(input_data.X.T, index=input_data.var_names, columns=input_data.obs_names)
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


def M3DropGetMarkers(expr_mat, labels):
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
    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
    
    if len(labels) != expr_mat.shape[1]:
        raise ValueError("Length of labels does not match number of cells.")

    # Apply the fast AUC function to each gene
    aucs = expr_mat.apply(lambda gene: hidden_fast_AUC_m3drop(gene.values, labels), axis=1)
    
    # Convert results to DataFrame
    auc_df = pd.DataFrame(aucs.tolist(), index=expr_mat.index, columns=['AUC', 'Group', 'pval'])
    
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
