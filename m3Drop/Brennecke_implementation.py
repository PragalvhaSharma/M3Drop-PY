import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.sparse as sp
from scipy.stats import chi2

from .basics import SparseMat3Drop, compute_row_mean_and_var

def BrenneckeGetVariableGenes(expr_mat, spikes=None, suppress_plot=False, fdr=0.1, mt_method="fdr_bh", mt_threshold=0.01, minBiolDisp=0.5, fitMeanQuantile=0.8):
    """
    Implements the method of Brennecke et al. (2013) to identify highly
    variable genes.

    Parameters
    ----------
    expr_mat : pd.DataFrame
        Normalized or raw (not log-transformed) expression values.
        Columns = samples, rows = genes.
    spikes : list or np.ndarray, optional
        Gene names or row numbers of spike-in genes.
    suppress_plot : bool, default=False
        Whether to make a plot.
    fdr : float, default=0.1
        FDR to identify significantly highly variable genes.
    mt_method : str, default="fdr_bh"
        Multiple testing correction method.
    mt_threshold : float, default=0.01
        Multiple testing threshold.
    minBiolDisp : float, default=0.5
        Minimum percentage of variance due to biological factors.
    fitMeanQuantile : float, default=0.8
        Threshold for genes to be used in fitting.

    Returns
    -------
    pd.DataFrame
        DataFrame of highly variable genes.
    """
    
    # Use mt_threshold if provided, otherwise use fdr
    threshold = mt_threshold if mt_threshold != 0.01 or fdr == 0.1 else fdr

    matrix_input = expr_mat
    if isinstance(expr_mat, np.ndarray):
        matrix_input = pd.DataFrame(expr_mat)
    elif isinstance(expr_mat, pd.DataFrame):
        matrix_input = expr_mat
    elif isinstance(expr_mat, (SparseMat3Drop, sp.spmatrix)):
        matrix_input = expr_mat
    else:
        raise TypeError("Unsupported input type for expr_mat.")

    means_all, vars_all = compute_row_mean_and_var(matrix_input, ddof=1)

    if spikes is not None:
        if isinstance(spikes[0], str):
            spike_mask = means_all.index.isin(spikes)
        elif isinstance(spikes[0], (int, np.integer)):
            spike_mask = np.zeros(len(means_all), dtype=bool)
            spike_mask[np.asarray(spikes, dtype=int)] = True
        else:
            raise TypeError("Spike identifiers must be strings or integers.")

        meansSp = means_all[spike_mask]
        varsSp = vars_all[spike_mask]
        meansGenes = means_all[~spike_mask]
        varsGenes = vars_all[~spike_mask]
    else:
        meansSp = means_all
        varsSp = vars_all
        meansGenes = means_all
        varsGenes = vars_all

    def safe_cv2(vars_series, mean_series):
        cv2 = vars_series / (mean_series.replace(0, np.nan) ** 2)
        return cv2.replace([np.inf, -np.inf], np.nan).fillna(0)

    cv2Sp = safe_cv2(varsSp, meansSp)
    cv2Genes = safe_cv2(varsGenes, meansGenes)

    # Fit Model
    minMeanForFit = np.quantile(meansSp[cv2Sp > 0.3], fitMeanQuantile) if np.sum(cv2Sp > 0.3) > 0 else 0
    useForFit = meansSp >= minMeanForFit
    
    if np.sum(useForFit) < 20:
        print("Too few spike-ins exceed minMeanForFit, recomputing using all genes.")
        meansAll = pd.concat([meansGenes, meansSp])
        cv2All = pd.concat([cv2Genes, cv2Sp])
        minMeanForFit = np.quantile(meansAll[cv2All > 0.3], 0.80)
        useForFit = meansSp >= minMeanForFit

    if np.sum(useForFit) < 30:
        print(f"Only {np.sum(useForFit)} spike-ins to be used in fitting, may result in poor fit.")

    # GLM fit
    glm_data = pd.DataFrame({'cv2': cv2Sp[useForFit], 'mean': meansSp[useForFit]})
    glm_data['a1tilde'] = 1 / glm_data['mean']
    
    fit = sm.GLM(
        glm_data['cv2'], 
        sm.add_constant(glm_data['a1tilde']), 
        family=sm.families.Gamma(link=sm.families.links.identity())
    ).fit()
    
    a0 = fit.params['const']
    a1 = fit.params['a1tilde']

    res = cv2Genes - (a0 + a1 / meansGenes)
    
    # Test
    psia1theta = a1
    minBiolDisp_sq = minBiolDisp**2
    m = matrix_input.shape[1]
    cv2th = a0 + minBiolDisp_sq + a0 * minBiolDisp_sq
    testDenom = (meansGenes * psia1theta + meansGenes**2 * cv2th) / (1 + cv2th / m)
    
    p = pd.Series(1 - chi2.cdf(varsGenes * (m - 1) / testDenom, m - 1), index=varsGenes.index)
    
    # FDR adjustment
    p_df = pd.DataFrame({'p': p, 'gene': p.index})
    p_df = p_df.sort_values(by='p')
    p_df['i'] = np.arange(1, len(p_df) + 1)
    p_df['p_adj'] = p_df['p'] * len(p_df) / p_df['i']
    padj = p_df.set_index('gene')['p_adj']
    padj = padj.reindex(p.index)

    sig = padj < threshold
    sig[sig.isna()] = False

    # Create result table
    table = pd.DataFrame({
        'Gene': meansGenes.index[sig],
        'effect.size': res[sig],
        'p.value': p[sig],
        'q.value': padj[sig]
    })
    table = table.sort_values(by='effect.size', ascending=False)
    
    return table
