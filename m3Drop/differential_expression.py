import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import warnings
import functools
from scipy.stats import mannwhitneyu

from .utils import bg__calc_variables, bg__fit_MM, bg__horizontal_residuals_MM_log10

def unfinished(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is not fully implemented and should not be used.", UserWarning)
        return func(*args, **kwargs)
    return wrapper

def obsolete(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"Call to obsolete function {func.__name__}.", category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

def bg__test_DE_K_equiv(gene_info, fit=None):
    """
    Internal function to test for differentially expressed genes based on K equivalence.
    """
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    s = gene_info['s']
    p = gene_info['p']
    s_err = gene_info['s_stderr']
    p_err = gene_info['p_stderr']

    p_safe = p.copy()
    p_safe[p_safe == 1] = 1 - 1e-9
    k_gene = (s * p_safe) / (1 - p_safe)

    var_log_k = (s_err / s)**2 + (p_err / (p_safe * (1 - p_safe)))**2
    var_log_k[(s == 0) | (p == 0) | (p == 1)] = np.inf
    std_log_k = np.sqrt(var_log_k)

    log_k_gene = np.log(k_gene.replace(0, 1e-9))
    log_k_fit = np.log(fit['K'])
    
    z_score = (log_k_gene - log_k_fit) / std_log_k
    
    p_values = norm.sf(z_score)

    fold_change = k_gene / fit['K']
    
    return {'pval': pd.Series(p_values, index=s.index), 'fold_change': fold_change}

def M3DropDifferentialExpression(expr_mat, mt_method="fdr_bh", mt_threshold=0.05, suppress_plot=True):
    """
    Finds differentially expressed genes using the Michaelis-Menten curve.
    
    This function was previously misnamed M3DropFeatureSelection.
    """
    gene_info = bg__calc_variables(expr_mat)
    if gene_info['s'].empty:
        return pd.DataFrame({'Gene': [], 'p.value': [], 'q.value': []})
        
    fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    de_output = bg__test_DE_K_equiv(gene_info, fit)
    pvals = de_output['pval'].dropna()

    if not pvals.empty:
        _, qvals, _, _ = multipletests(pvals, alpha=mt_threshold, method=mt_method)
        qvals = pd.Series(qvals, index=pvals.index)
        sig_genes = qvals[qvals < mt_threshold]
    else:
        sig_genes = pd.Series(dtype=float)

    if not sig_genes.empty:
        result_df = pd.DataFrame({
            'Gene': sig_genes.index,
            'p.value': pvals.loc[sig_genes.index],
            'q.value': sig_genes
        }).sort_values('q.value')
    else:
        result_df = pd.DataFrame({'Gene': [], 'p.value': [], 'q.value': []})

    if not suppress_plot:
        print("Plotting for M3DropDifferentialExpression is not implemented to avoid circular dependencies.")
        
    return result_df

def M3DropTestShift(expr_mat, genes_to_test, name="", background=None, suppress_plot=False):
    """
    Tests whether a given set of genes are significantly shifted to the left or
    right of the Michaelis-Menten curve.
    """
    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
        
    if background is None:
        background = expr_mat.index.tolist()

    gene_info = bg__calc_variables(expr_mat)
    fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    all_residuals = bg__horizontal_residuals_MM_log10(fit['K'], gene_info['p'], gene_info['s']).dropna()
    
    # Ensure genes_to_test and background are sets for efficient operations
    genes_to_test_set = set(genes_to_test)
    background_set = set(background)
    
    # The background for comparison should not include the genes being tested
    background_for_test_set = background_set - genes_to_test_set
    
    test_residuals = all_residuals[all_residuals.index.isin(genes_to_test_set)]
    background_residuals = all_residuals[all_residuals.index.isin(background_for_test_set)]

    if len(test_residuals) < 1 or len(background_residuals) < 1:
        return pd.DataFrame({
            'sample': [np.nan],
            'pop': [np.nan],
            'p.value': [np.nan],
            'stat': [np.nan]
        }, index=[name])

    stat, p_val = mannwhitneyu(test_residuals, background_residuals, alternative='two-sided')

    if not suppress_plot:
        print("Plotting for M3DropTestShift is not yet implemented.")

    return pd.DataFrame({
        'sample': [test_residuals.median()],
        'pop': [background_residuals.median()],
        'p.value': [p_val],
        'stat': [stat]
    }, index=[name])

@unfinished
def bg__m3dropTraditionalDE(counts, groups, batches=None):
    warnings.warn("bg__m3dropTraditionalDE is not implemented.", UserWarning)
    return pd.DataFrame()

@unfinished
def bg__m3dropTraditionalDEShiftDisp(counts, groups, batches=None):
    warnings.warn("bg__m3dropTraditionalDEShiftDisp is not implemented.", UserWarning)
    return pd.DataFrame()

@unfinished
def bg__nbumiGroupDE(counts, fit, groups):
    warnings.warn("bg__nbumiGroupDE is not implemented.", UserWarning)
    return pd.DataFrame()
