import numpy as np
import pandas as pd
from anndata import AnnData


def M3DropConvertData(input_data, is_log=False, is_counts=False, pseudocount=1):
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

    Returns
    -------
    pd.DataFrame
        A normalized, non-log-transformed matrix.
    """
    def remove_undetected_genes(mat):
        # Helper to filter out genes with no expression across all cells
        if not isinstance(mat, pd.DataFrame):
             raise TypeError("remove_undetected_genes expects a pandas DataFrame.")
        detected = mat.sum(axis=1) > 0
        if np.sum(~detected) > 0:
            print(f"Removing {np.sum(~detected)} undetected genes.")
        return mat[detected]

    from scipy.sparse import issparse
    
    # 1. Handle Input Type and convert to a pandas DataFrame `counts`
    if isinstance(input_data, AnnData):
        if issparse(input_data.X):
            counts = pd.DataFrame(input_data.X.toarray(), index=input_data.obs_names, columns=input_data.var_names)
        else:
            counts = pd.DataFrame(input_data.X, index=input_data.obs_names, columns=input_data.var_names)
    elif isinstance(input_data, pd.DataFrame):
        counts = input_data
    elif isinstance(input_data, np.ndarray):
        counts = pd.DataFrame(input_data)
    else:
        raise TypeError(f"Unrecognized input format: {type(input_data)}")

    # 2. Handle log-transformation (corrected to match R implementation)
    if is_log:
        # R uses 2^lognorm-pseudocount, equivalent to expm1 for log1p transformed data
        counts = 2**counts - pseudocount
    
    # 3. Handle normalization for raw counts (corrected to match R implementation)
    if is_counts:
        sf = counts.sum(axis=0)
        sf[sf == 0] = 1 # Avoid division by zero
        # Normalize using median scaling factor (as in R)
        norm_counts = (counts / sf) * np.median(sf)
        return remove_undetected_genes(norm_counts)
    
    # 4. If data is already normalized (not raw counts), just filter
    return remove_undetected_genes(counts)


def bg__calc_variables(expr_mat):
    """
    Calculates a suite of gene-specific variables including: mean, dropout rate,
    and their standard errors. Updated to match R implementation behavior.
    """
    if isinstance(expr_mat, pd.DataFrame):
        expr_mat_values = expr_mat.values
        gene_names = expr_mat.index
    else:
        expr_mat_values = expr_mat
        gene_names = pd.RangeIndex(start=0, stop=expr_mat.shape[0], step=1)

    # Check for NA values
    if np.sum(np.isnan(expr_mat_values)) > 0:
        raise ValueError("Error: Expression matrix contains NA values.")
    
    # Check for negative values
    lowest = np.min(expr_mat_values)
    if lowest < 0:
        raise ValueError("Error: Expression matrix cannot contain negative values! Has the matrix been log-transformed?")
    
    # Deal with strangely normalized data (no zeros)
    if lowest > 0:
        print("Warning: No zero values (dropouts) detected will use minimum expression value instead.")
        min_val = lowest + 0.05
        expr_mat_values[expr_mat_values == min_val] = 0
    
    # Check if we have enough zeros
    sum_zero = np.prod(expr_mat_values.shape) - np.sum(expr_mat_values > 0)
    if sum_zero < 0.1 * np.prod(expr_mat_values.shape):
        print("Warning: Expression matrix contains few zero values (dropouts) this may lead to poor performance.")

    # Remove undetected genes
    p = 1 - np.sum(expr_mat_values > 0, axis=1) / expr_mat_values.shape[1]
    
    if np.sum(p == 1) > 0:
        print(f"Warning: Removing {np.sum(p == 1)} undetected genes.")
        detected = p < 1
        expr_mat_values = expr_mat_values[detected, :]
        if isinstance(gene_names, pd.Index):
            gene_names = gene_names[detected]
        else:
            gene_names = np.arange(expr_mat_values.shape[0])
        p = 1 - np.sum(expr_mat_values > 0, axis=1) / expr_mat_values.shape[1]

    if expr_mat_values.shape[0] == 0:
        return {
            's': pd.Series(dtype=float),
            's_stderr': pd.Series(dtype=float),
            'p': pd.Series(dtype=float),
            'p_stderr': pd.Series(dtype=float)
        }

    s = np.mean(expr_mat_values, axis=1)
    
    # Calculate standard error using sparse matrix friendly method
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
