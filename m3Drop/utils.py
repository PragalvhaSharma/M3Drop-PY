import numpy as np
import pandas as pd
from scipy.stats import norm, nbinom, poisson
from scipy.optimize import minimize
from scipy import sparse
import warnings
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def bg__calc_variables(expr_mat):
    """
    Calculates a suite of gene-specific variables including: mean, dropout rate,
    and their standard errors.
    """
    if isinstance(expr_mat, pd.DataFrame):
        expr_mat_values = expr_mat.values
        gene_names = expr_mat.index
    else:
        expr_mat_values = expr_mat
        gene_names = pd.RangeIndex(start=0, stop=expr_mat.shape[0], step=1)

    # Remove undetected genes
    detected = np.sum(expr_mat_values > 0, axis=1) > 0
    if not np.all(detected):
        expr_mat_values = expr_mat_values[detected, :]
        if isinstance(gene_names, pd.Index):
            gene_names = gene_names[detected]
        else: # RangeIndex
            gene_names = np.arange(expr_mat.shape[0])[detected]


    if expr_mat_values.shape[0] == 0:
        return {
            's': pd.Series(dtype=float),
            's_stderr': pd.Series(dtype=float),
            'p': pd.Series(dtype=float),
            'p_stderr': pd.Series(dtype=float)
        }

    s = np.mean(expr_mat_values, axis=1)
    p = np.sum(expr_mat_values == 0, axis=1) / expr_mat_values.shape[1]

    s_stderr = np.std(expr_mat_values, axis=1, ddof=1) / np.sqrt(expr_mat_values.shape[1])
    p_stderr = np.sqrt(p * (1 - p) / expr_mat_values.shape[1])

    return {
        's': pd.Series(s, index=gene_names),
        's_stderr': pd.Series(s_stderr, index=gene_names),
        'p': pd.Series(p, index=gene_names),
        'p_stderr': pd.Series(p_stderr, index=gene_names)
    }

def bg__fit_MM(p, s):
    """
    Fits the modified Michaelis-Menten equation to the relationship between
    mean expression and dropout-rate.
    """
    s_clean = s[~p.isna() & ~s.isna()]
    p_clean = p[~p.isna() & ~s.isna()]

    def neg_log_likelihood(params):
        K, sd = params
        if K <= 0 or sd <= 0:
            return np.inf

        predictions = K / (s_clean + K)
        log_likelihood = np.sum(norm.logpdf(p_clean, loc=predictions, scale=sd))
        return -log_likelihood

    initial_params = [np.median(s_clean), 0.1]

    result = minimize(
        neg_log_likelihood,
        initial_params,
        method='L-BFGS-B',
        bounds=[(1e-9, None), (1e-9, None)]
    )

    K, sd = result.x

    predictions = K / (s + K)
    ssr = np.sum((p - predictions)**2)

    return {
        'K': K,
        'sd': sd,
        'predictions': pd.Series(predictions, index=s.index),
        'SSr': ssr,
        'model': f"Michaelis-Menten (K={K:.2f})"
    }

def bg__horizontal_residuals_MM_log10(K, p, s):
    """
    Calculates horizontal residuals from the Michaelis-Menten Function.
    """
    res_series = pd.Series(np.nan, index=s.index)
    
    valid_indices = (p > 0) & (p < 1) & (s > 0)
    if not valid_indices.any():
        return res_series

    p_valid = p[valid_indices]
    s_valid = s[valid_indices]

    s_pred = K * (1 - p_valid) / p_valid

    epsilon = 1e-9
    residuals = np.log10(s_valid + epsilon) - np.log10(s_pred + epsilon)

    res_series[valid_indices] = residuals
    return res_series

def hidden_calc_vals(counts):
    """
    Calculate basic statistics from count matrix.
    """
    # Convert to numpy array if needed
    if isinstance(counts, pd.DataFrame):
        counts_matrix = counts.values
        gene_names = counts.index
        cell_names = counts.columns
    else:
        counts_matrix = np.array(counts)
        gene_names = pd.RangeIndex(start=0, stop=counts_matrix.shape[0], step=1)
        cell_names = pd.RangeIndex(start=0, stop=counts_matrix.shape[1], step=1)
    
    # Check for negative values
    if np.sum(counts_matrix < 0) > 0:
        raise ValueError("Expression matrix contains negative values! Please provide raw UMI counts!")
    
    # Check for integers
    if not np.allclose(counts_matrix, counts_matrix.astype(int)):
        raise ValueError("Error: Expression matrix is not integers! Please provide raw UMI counts.")
    
    # Set row names if missing
    if gene_names is None:
        gene_names = pd.RangeIndex(start=0, stop=counts_matrix.shape[0], step=1)
    
    # Calculate statistics
    if sparse.issparse(counts_matrix):
        tjs = np.array(counts_matrix.sum(axis=1)).flatten()  # Total molecules/gene
        tis = np.array(counts_matrix.sum(axis=0)).flatten()  # Total molecules/cell
        djs = counts_matrix.shape[1] - np.array((counts_matrix > 0).sum(axis=1)).flatten()  # Dropouts per gene
        dis = counts_matrix.shape[0] - np.array((counts_matrix > 0).sum(axis=0)).flatten()  # Dropouts per cell
    else:
        tjs = np.sum(counts_matrix, axis=1)  # Total molecules/gene
        tis = np.sum(counts_matrix, axis=0)  # Total molecules/cell
        djs = counts_matrix.shape[1] - np.sum(counts_matrix > 0, axis=1)  # Dropouts per gene
        dis = counts_matrix.shape[0] - np.sum(counts_matrix > 0, axis=0)  # Dropouts per cell
    
    # Check for undetected genes
    no_detect = np.sum(tjs <= 0)
    if no_detect > 0:
        raise ValueError(f"Error: contains {no_detect} undetected genes.")
    
    # Check for empty cells
    if np.sum(tis <= 0) > 0:
        raise ValueError("Error: all cells must have at least one detected molecule.")
    
    nc = counts_matrix.shape[1]  # Number of cells
    ng = counts_matrix.shape[0]  # Number of genes
    total = np.sum(tis)  # Total molecules sampled
    
    return {
        'tis': tis,
        'tjs': tjs,
        'dis': dis,
        'djs': djs,
        'total': total,
        'nc': nc,
        'ng': ng
    }

def NBumiFitModel(counts):
    """
    Fits the depth-adjusted negative binomial model.
    """
    vals = hidden_calc_vals(counts)
    
    # Convert to dense array if sparse
    if sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.array(counts)
    
    min_size = 1e-10
    
    # Calculate row-wise variance
    my_rowvar = np.zeros(counts_dense.shape[0])
    for i in range(counts_dense.shape[0]):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        my_rowvar[i] = np.var(counts_dense[i, :] - mu_is)
    
    # Calculate size parameters
    size = (vals['tjs']**2 * (np.sum(vals['tis']**2) / vals['total']**2) / 
            ((vals['nc'] - 1) * my_rowvar - vals['tjs']))
    
    max_size = 10 * np.max(size[size > 0])
    size[size < 0] = max_size
    size[size < min_size] = min_size
    
    return {
        'var_obs': my_rowvar,
        'sizes': size,
        'vals': vals
    }

def NBumiFitBasicModel(counts):
    """
    Fits a basic negative binomial model.
    """
    vals = hidden_calc_vals(counts)
    
    # Convert to dense array if sparse
    if sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.array(counts)
    
    mus = vals['tjs'] / vals['nc']
    gm = np.mean(counts_dense, axis=1)
    v = np.sum((counts_dense - gm[:, np.newaxis])**2, axis=1) / (counts_dense.shape[1] - 1)
    
    errs = v < mus
    v[errs] = mus[errs] + 1e-10
    size = mus**2 / (v - mus)
    max_size = np.max(mus)**2
    size[errs] = max_size
    
    my_rowvar = np.zeros(counts_dense.shape[0])
    for i in range(counts_dense.shape[0]):
        my_rowvar[i] = np.var(counts_dense[i, :] - mus[i])
    
    return {
        'var_obs': my_rowvar,
        'sizes': size,
        'vals': vals
    }

def NBumiCheckFit(counts, fit, suppress_plot=False):
    """
    Checks the fit quality of the NBumi model.
    """
    vals = fit['vals']
    
    # Convert to dense array if sparse
    if sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.array(counts)
    
    # Convert pandas Series to numpy arrays to avoid FutureWarnings
    tjs = np.array(vals['tjs']) if hasattr(vals['tjs'], 'values') else vals['tjs']
    tis = np.array(vals['tis']) if hasattr(vals['tis'], 'values') else vals['tis']
    sizes = np.array(fit['sizes']) if hasattr(fit['sizes'], 'values') else fit['sizes']
    total = vals['total']
    
    row_ps = np.zeros(vals['ng'])
    col_ps = np.zeros(vals['nc'])
    
    for i in range(vals['ng']):
        mu_is = tjs[i] * tis / total
        p_is = (1 + mu_is / sizes[i])**(-sizes[i])
        row_ps[i] = np.sum(p_is)
        col_ps += p_is
    
    if not suppress_plot:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(vals['djs'], row_ps)
        plt.plot([0, max(vals['djs'])], [0, max(vals['djs'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Fit')
        plt.title('Gene-specific Dropouts')
        
        plt.subplot(1, 2, 2)
        plt.scatter(vals['dis'], col_ps)
        plt.plot([0, max(vals['dis'])], [0, max(vals['dis'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Expected')
        plt.title('Cell-specific Dropouts')
        
        plt.tight_layout()
        plt.show()
    
    gene_error = np.sum((vals['djs'] - row_ps)**2)
    cell_error = np.sum((vals['dis'] - col_ps)**2)
    
    return {
        'gene_error': gene_error,
        'cell_error': cell_error,
        'rowPs': row_ps,
        'colPs': col_ps
    }

def NBumiCheckFitFS(counts, fit, suppress_plot=False):
    """
    Checks the fit quality for feature selection using smoothed size parameters.
    """
    # Import here to avoid circular import
    from .normalization import NBumiFitDispVsMean
    
    vals = fit['vals']
    size_coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    
    # Convert pandas Series to numpy arrays to avoid FutureWarnings
    tjs = np.array(vals['tjs']) if hasattr(vals['tjs'], 'values') else vals['tjs']
    smoothed_size = np.exp(size_coeffs[0] + size_coeffs[1] * np.log(tjs / vals['nc']))
    
    # Convert to dense array if sparse
    if sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.array(counts)
    
    # Convert pandas Series to numpy arrays to avoid FutureWarnings
    tis = np.array(vals['tis']) if hasattr(vals['tis'], 'values') else vals['tis']
    total = vals['total']
    
    row_ps = np.zeros(vals['ng'])
    col_ps = np.zeros(vals['nc'])
    
    for i in range(vals['ng']):
        mu_is = tjs[i] * tis / total
        p_is = (1 + mu_is / smoothed_size[i])**(-smoothed_size[i])
        row_ps[i] = np.sum(p_is)
        col_ps += p_is
    
    if not suppress_plot:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(vals['djs'], row_ps)
        plt.plot([0, max(vals['djs'])], [0, max(vals['djs'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Fit')
        plt.title('Gene-specific Dropouts')
        
        plt.subplot(1, 2, 2)
        plt.scatter(vals['dis'], col_ps)
        plt.plot([0, max(vals['dis'])], [0, max(vals['dis'])], 'r-')
        plt.xlabel('Observed')
        plt.ylabel('Expected')
        plt.title('Cell-specific Dropouts')
        
        plt.tight_layout()
        plt.show()
    
    gene_error = np.sum((vals['djs'] - row_ps)**2)
    cell_error = np.sum((vals['dis'] - col_ps)**2)
    
    return {
        'gene_error': gene_error,
        'cell_error': cell_error,
        'rowPs': row_ps,
        'colPs': col_ps
    }

def NBumiCompareModels(counts, size_factor=None):
    """
    Compares the fit of different negative binomial models.
    """
    # Import here to avoid circular import
    from .normalization import NBumiConvertToInteger
    
    # Convert to dense array if sparse
    if sparse.issparse(counts):
        counts_dense = counts.toarray()
    else:
        counts_dense = np.array(counts)
    
    if size_factor is None:
        size_factor = np.sum(counts_dense, axis=0) / np.median(np.sum(counts_dense, axis=0))
    
    if np.max(counts_dense) < np.max(size_factor):
        raise ValueError("Error: size factors are too large")
    
    # Normalize counts
    norm = NBumiConvertToInteger((counts_dense / size_factor[np.newaxis, :]).T).T
    
    # Fit both models
    fit_adjust = NBumiFitModel(counts_dense)
    fit_basic = NBumiFitBasicModel(norm)
    
    check_adjust = NBumiCheckFitFS(counts_dense, fit_adjust, suppress_plot=True)
    check_basic = NBumiCheckFitFS(norm, fit_basic, suppress_plot=True)
    
    nc = fit_adjust['vals']['nc']
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    
    xes = np.log10(fit_adjust['vals']['tjs'] / nc)
    dropout_obs = fit_adjust['vals']['djs'] / nc
    
    plt.scatter(xes, dropout_obs, c='black', s=16, alpha=0.7, label='Observed')
    plt.scatter(xes, check_adjust['rowPs'] / nc, c='orange', s=4, label='Depth-Adjusted')
    plt.scatter(xes, check_basic['rowPs'] / nc, c='purple', s=4, label='Basic')
    
    plt.xlabel('log10(expression)')
    plt.ylabel('Dropout Rate')
    plt.xscale('log')
    
    err_adj = np.sum(np.abs(check_adjust['rowPs'] / nc - fit_adjust['vals']['djs'] / nc))
    err_bas = np.sum(np.abs(check_basic['rowPs'] / nc - fit_adjust['vals']['djs'] / nc))
    
    plt.legend([f'Depth-Adjusted\nError: {err_adj:.1f}', f'Basic\nError: {err_bas:.1f}'], 
              loc='lower left')
    plt.show()
    
    errors = {'Depth-Adjusted': err_adj, 'Basic': err_bas}
    
    return {
        'errors': errors,
        'basic_fit': fit_basic,
        'adjusted_fit': fit_adjust
    }

def bg__fit_size_to_var(obs, mu_vec, max_size, min_size=1e-10, convergence=0.001):
    """
    Internal function to fit size parameter to variance.
    """
    step_size = 1
    size_fit = 1
    last = 0
    last2 = 0
    
    for iteration in range(1000):
        if size_fit < min_size:
            size_fit = min_size
        
        expect = mu_vec + mu_vec * mu_vec / size_fit
        diff = np.sum(expect) / (len(mu_vec) - 1) - obs
        
        if abs(diff) < convergence:
            return size_fit
        
        if diff > 0:
            size_fit = size_fit + step_size
            if last < 0:
                step_size = step_size / 2
            else:
                if last2 > 0:
                    step_size = step_size * 2
        else:
            size_fit = size_fit - step_size
            if last > 0:
                step_size = step_size / 2
            else:
                if last2 < 0:
                    step_size = step_size * 2
        
        last2 = last
        last = diff
        
        if size_fit > max_size:
            return max_size
    
    warnings.warn("Fitting size did not converge.")
    return size_fit

def hidden_shift_size(mu_all, size_all, mu_group, coeffs):
    """
    Shift size parameter based on mean expression change.
    """
    b = np.log(size_all) - coeffs[1] * np.log(mu_all)
    size_group = np.exp(coeffs[1] * np.log(mu_group) + b)
    return size_group
