import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

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
