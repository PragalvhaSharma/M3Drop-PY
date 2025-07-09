import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

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

    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)

    if spikes is not None:
        if isinstance(spikes[0], str):
            sp = expr_mat.index.isin(spikes)
            countsSp = expr_mat.loc[sp]
            countsGenes = expr_mat.loc[~sp]
        elif isinstance(spikes[0], (int, np.integer)):
            countsSp = expr_mat.iloc[spikes]
            countsGenes = expr_mat.drop(expr_mat.index[spikes])
    else:
        countsSp = expr_mat
        countsGenes = expr_mat

    meansSp = countsSp.mean(axis=1)
    varsSp = countsSp.var(axis=1, ddof=1)
    cv2Sp = varsSp / (meansSp**2)
    
    meansGenes = countsGenes.mean(axis=1)
    varsGenes = countsGenes.var(axis=1, ddof=1)
    cv2Genes = varsGenes / (meansGenes**2)

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
    m = expr_mat.shape[1]
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