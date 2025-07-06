import numpy as np
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

def hidden_calc_vals(counts):
    """
    Calculates various summary statistics from the expression matrix.
    
    This is a helper function for `NBumiFitModel`.
    """
    if np.sum(counts < 0) > 0:
        raise ValueError("Expression matrix contains negative values! Please provide raw UMI counts.")
    if np.sum(counts % 1 != 0) > 0:
        raise ValueError("Expression matrix is not integers! Please provide raw UMI counts.")
    
    if hasattr(counts, 'index') and hasattr(counts, 'columns'):
        if counts.index is None:
            counts.index = [str(i) for i in range(counts.shape[0])]
    
    tjs = np.sum(counts, axis=1).A1 if issparse(counts) else np.sum(counts, axis=1) # Total molecules/gene
    no_detect = np.sum(tjs <= 0)
    if no_detect > 0:
        raise ValueError(f"Error: contains {no_detect} undetected genes.")
    
    tis = np.sum(counts, axis=0).A1 if issparse(counts) else np.sum(counts, axis=0) # Total molecules/cell
    if np.sum(tis <= 0) > 0:
        raise ValueError("Error: all cells must have at least one detected molecule.")
        
    djs = counts.shape[1] - (np.sum(counts > 0, axis=1).A1 if issparse(counts) else np.sum(counts > 0, axis=1)) # Observed Dropouts per gene
    dis = counts.shape[0] - (np.sum(counts > 0, axis=0).A1 if issparse(counts) else np.sum(counts > 0, axis=0)) # Observed Dropouts per cell
    
    nc = counts.shape[1] # Number of cells
    ng = counts.shape[0] # Number of genes
    total = np.sum(tis) # Total molecules sampled
    
    return {'tis': tis, 'tjs': tjs, 'dis': dis, 'djs': djs, 'total': total, 'nc': nc, 'ng': ng}

def NBumiConvertToInteger(mat):
    """
    Converts a matrix to integers and removes rows with all zeros.
    """
    mat = np.ceil(mat).astype(int)
    if issparse(mat):
        mat = mat[np.sum(mat, axis=1).A1 > 0, :]
    else:
        mat = mat[np.sum(mat, axis=1) > 0, :]
    return mat

def NBumiFitModel(counts):
    """
    Fits a negative binomial model to the UMI count data.
    """
    vals = hidden_calc_vals(counts)
    
    min_size = 1e-10
    
    if issparse(counts):
        my_rowvar = np.array([np.var(counts[i, :].toarray() - (vals['tjs'][i] * vals['tis'] / vals['total'])) for i in range(counts.shape[0])])
    else:
        mu_is = np.outer(vals['tjs'], vals['tis'] / vals['total'])
        my_rowvar = np.var(counts - mu_is, axis=1)

    size = (vals['tjs']**2 * (np.sum(vals['tis']**2) / vals['total']**2)) / ((vals['nc'] - 1) * my_rowvar - vals['tjs'])
    
    max_size = 10 * np.nanmax(size[size > 0]) if np.any(size > 0) else 1
    size[size < 0] = max_size
    size[size < min_size] = min_size
    
    return {'var_obs': my_rowvar, 'sizes': size, 'vals': vals}

def NBumiCheckFit(counts, fit, suppress_plot=False):
    """
    Checks the fit of the negative binomial model.
    
    Compares the observed dropouts to the expected dropouts from the fitted model.
    
    Parameters:
    - counts: The expression matrix.
    - fit: The fitted model from `NBumiFitModel`.
    - suppress_plot (bool): If True, suppresses the generation of plots.
    
    Returns:
    - dict: A dictionary containing gene_error, cell_error, rowPs, and colPs.
    """
    vals = fit['vals']
    
    row_ps = np.zeros(counts.shape[0])
    col_ps = np.zeros(counts.shape[1])
    
    for i in range(counts.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / fit['sizes'][i])**(-fit['sizes'][i])
        row_ps[i] = np.sum(p_is)
        col_ps += p_is
    
    if not suppress_plot:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(vals['djs'], row_ps)
        plt.plot([min(vals['djs']), max(vals['djs'])], [min(vals['djs']), max(vals['djs'])], 'r-')
        plt.xlabel("Observed")
        plt.ylabel("Fit")
        plt.title("Gene-specific Dropouts")
        
        plt.subplot(1, 2, 2)
        plt.scatter(vals['dis'], col_ps)
        plt.plot([min(vals['dis']), max(vals['dis'])], [min(vals['dis']), max(vals['dis'])], 'r-')
        plt.xlabel("Observed")
        plt.ylabel("Expected")
        plt.title("Cell-specific Dropouts")
        
        plt.tight_layout()
        plt.show()
        
    gene_error = np.sum((vals['djs'] - row_ps)**2)
    cell_error = np.sum((vals['dis'] - col_ps)**2)
    
    return {'gene_error': gene_error, 'cell_error': cell_error, 'rowPs': row_ps, 'colPs': col_ps}

def NBumiFeatureSelection(fit, mt_method="fdr_bh", mt_threshold=0.01):
    """
    Performs feature selection based on the negative binomial model.
    
    Identifies significantly variable genes by comparing their observed dropout 
    rate to the expected rate from the fitted model.
    
    Parameters:
    - fit: The fitted model from `NBumiFitModel`.
    - mt_method (str): The multiple testing correction method to use.
    - mt_threshold (float): The significance threshold.
    
    Returns:
    - pandas.DataFrame: A DataFrame of significantly variable genes.
    """
    
    vals = fit['vals']
    
    # Calculate expected dropouts
    check = NBumiCheckFit(np.zeros((vals['ng'], vals['nc'])), fit, suppress_plot=True)
    row_ps = check['rowPs']
    
    # Likelihood ratio test
    # Log-likelihood of the saturated model (observed data)
    loglik_saturated = (vals['djs'] * np.log(vals['djs'] / vals['nc'])) + \
                       ((vals['nc'] - vals['djs']) * np.log(1 - (vals['djs'] / vals['nc'])))
    # Log-likelihood of the fitted model
    loglik_fitted = (vals['djs'] * np.log(row_ps / vals['nc'])) + \
                    ((vals['nc'] - vals['djs']) * np.log(1 - (row_ps / vals['nc'])))

    # Replace NaNs with 0 (for cases where djs is 0 or nc)
    loglik_saturated[np.isnan(loglik_saturated)] = 0
    loglik_fitted[np.isnan(loglik_fitted)] = 0
    
    # Test statistic
    test_stat = 2 * (loglik_saturated - loglik_fitted)
    
    # P-values from chi-squared distribution
    from scipy.stats import chi2
    p_values = chi2.sf(test_stat, 1)

    # Multiple testing correction
    from statsmodels.stats.multitest import multipletests
    reject, q_values, _, _ = multipletests(p_values, alpha=mt_threshold, method=mt_method)
    
    # Filter for significant genes with lower dropout than expected
    gene_names = fit.get('gene_names')
    if gene_names is None and 'vals' in fit and 'ng' in fit['vals']:
        gene_names = np.arange(fit['vals']['ng'])

    significant_genes = pd.DataFrame({
        'Gene': gene_names,
        'p.value': p_values,
        'q.value': q_values
    })
    
    significant_genes = significant_genes[
        (significant_genes['q.value'] < mt_threshold) & 
        (vals['djs'] < row_ps)
    ].sort_values(by='q.value')
    
    return significant_genes 

def NBumiKnnDrop(counts, fit, features, k=5, cutoff=0.5):
    """
    Imputes dropout events using K-Nearest Neighbors.
    
    For each cell, it finds the k-nearest neighbors based on the expression
    of a set of feature genes. It then predicts the probability of a dropout
    for each gene in each cell. If the observed value is zero but the
    predicted dropout probability is low, the value is imputed.
    
    Parameters:
    - counts: The expression matrix.
    - fit: The fitted model from `NBumiFitModel`.
    - features (list or array): A list of feature genes to use for KNN.
    - k (int): The number of nearest neighbors to use.
    - cutoff (float): The dropout probability cutoff for imputation.
    
    Returns:
    - scipy.sparse.csr_matrix or numpy.ndarray: The imputed expression matrix.
    """
    
    # Find K-nearest neighbors
    feature_indices = [i for i, gene in enumerate(fit.get('gene_names', [])) if gene in features]
    if not feature_indices:
        # Fallback to using all genes if feature matching fails
        feature_indices = np.arange(counts.shape[0])
        
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(counts[feature_indices, :].T)
    distances, indices = nn.kneighbors(counts[feature_indices, :].T)
    
    imputed_counts = counts.copy()
    vals = fit['vals']
    
    for i in range(counts.shape[1]): # For each cell
        knn_indices = indices[i, :]
        
        # Average expression of neighbors
        knn_avg_expr = np.mean(counts[:, knn_indices], axis=1)
        
        # Dropout probability
        dropout_prob = (1 + knn_avg_expr / fit['sizes'])**(-fit['sizes'])
        
        # Impute dropouts
        impute_mask = (counts[:, i] == 0) & (dropout_prob < cutoff)
        if np.any(impute_mask):
            imputed_counts[impute_mask, i] = knn_avg_expr[impute_mask]
            
    return imputed_counts 