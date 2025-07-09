import numpy as np
import pandas as pd

def NBumiHVG(counts, fit, fdr_thresh=0.05, suppress_plot=False, method="DANB"):
    """
    Tests for significantly high variability in droplet-based datasets.

    Parameters
    ----------
    counts : pd.DataFrame or np.ndarray
        Raw count matrix.
    fit : dict
        Output from `NBumiFitModel`.
    fdr_thresh : float, default=0.05
        Multiple testing correction threshold.
    suppress_plot : bool, default=False
        Whether to plot mean vs variance.
    method : {"DANB", "basic"}, default="DANB"
        Whether to use DANB dispersions or raw sample variances.

    Returns
    -------
    pd.DataFrame
        DataFrame of highly variable genes.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm

    # Ensure we have a DataFrame with proper index
    if isinstance(counts, np.ndarray):
        gene_names = [f"Gene_{i}" for i in range(counts.shape[0])]
        counts = pd.DataFrame(counts, index=gene_names)
    elif not hasattr(counts, 'index'):
        gene_names = [f"Gene_{i}" for i in range(counts.shape[0])]
        counts = pd.DataFrame(counts, index=gene_names)

    n = counts.shape[1]

    if method == "DANB":
        mu_obs = fit['vals']['tjs'] / n
        v_obs = mu_obs + mu_obs**2 / fit['sizes']
    else: # basic
        mu_obs = counts.mean(axis=1)
        v_obs = counts.var(axis=1, ddof=1)

    # Fit GLM to get dispersion
    tmp = mu_obs**2
    glm_fit = sm.GLM(v_obs - mu_obs, tmp, family=sm.families.Gaussian()).fit()
    disp = glm_fit.params[0]
    
    sigma2 = mu_obs + disp * mu_obs**2 # v_fitted in R code

    # Negative binomial parameters from mean and variance
    p = mu_obs / sigma2
    r = mu_obs * p / (1 - p)
    
    # Central moments of NB distribution
    mu4 = r * (1 - p) * (6 - 6 * p + p**2 + 3 * r - 3 * p * r) / (p**4)
    
    # Variance of sample variance
    v_of_v = mu4 * (n - 1)**2 / n**3 - (sigma2**2 * (n - 3) * (n - 1)) / (n**3)
    
    z = (v_obs - sigma2) / np.sqrt(v_of_v)
    pvals = norm.sf(z)
    
    qvals = multipletests(pvals[~np.isnan(pvals)], method='fdr_bh')[1]
    
    eff = v_obs - sigma2
    
    tab = pd.DataFrame({
        'Gene': counts.index,
        'effect.size': eff,
        'p.value': pvals,
    })
    
    q_series = pd.Series(np.nan, index=counts.index)
    q_series[~np.isnan(pvals)] = qvals
    tab['q.value'] = q_series
    
    tab = tab.dropna(subset=['p.value'])
    tab = tab.sort_values(by=['q.value', 'effect.size'], ascending=[False, False])
    
    return tab[tab['q.value'] < fdr_thresh]

