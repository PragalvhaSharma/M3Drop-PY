import numpy as np
import pandas as pd
from scipy import stats
from .Curve_fitting import bg__fit_MM
from .basics import bg__calc_variables, hidden__invert_MM, bg__horizontal_residuals_MM_log10
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def hidden__test_DE_K_equiv_raw(expr_mat, fit=None):
    """
    Raw implementation of K equivalence test for differential expression.
    """
    gene_info = bg__calc_variables(expr_mat)
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    p_obs = gene_info['p']
    N = expr_mat.shape[1]
    p_err = gene_info['p_stderr']
    S_mean = gene_info['s']
    S_err = gene_info['s_stderr']
    K_err = fit.get('Kerr', 0.1)  # Use Kerr if available, otherwise default
    
    K_equiv = p_obs * S_mean / (1 - p_obs)
    K_equiv_err = p_obs / (1 - p_obs) * S_err
    
    Z = (K_equiv - fit['K']) / np.sqrt(K_equiv_err**2 + K_err**2)
    pval = stats.norm.sf(Z)  # Right tail test
    effect_size = K_equiv / fit['K']
    
    return {'pval': pval, 'fold_change': effect_size}


def bg__test_DE_K_equiv(gene_info, fit=None):
    """
    Internal function to test for differentially expressed genes based on K equivalence.
    """
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    p_obs = gene_info['p'].copy()
    always_detected = p_obs == 0
    
    # Handle zero dropout rates
    min_p = np.min(p_obs[p_obs > 0]) if np.any(p_obs > 0) else 1e-10
    p_obs[p_obs == 0] = min_p / 2
    
    p_err = gene_info['p_stderr']
    S_mean = gene_info['s']
    S_err = gene_info['s_stderr']
    K_err = fit.get('Kerr', 0.1)
    K_obs = fit['K']
    
    K_equiv = p_obs * S_mean / (1 - p_obs)
    K_equiv_err = np.abs(K_equiv) * np.sqrt((S_err / S_mean)**2 + (p_err / p_obs)**2)
    
    K_equiv_log = np.log(K_equiv)
    thing = K_equiv - K_equiv_err
    thing[thing <= 0] = 1e-100
    K_equiv_err_log = np.abs(np.log(thing) - K_equiv_log)
    K_equiv_err_log[K_equiv - K_equiv_err <= 0] = 1e10
    
    K_obs_log = np.log(fit['K'])
    K_err_log = np.std(K_equiv_log - K_obs_log) / np.sqrt(len(K_equiv_log))
    
    Z = (K_equiv_log - K_obs_log) / np.sqrt(K_equiv_err_log**2 + K_err_log**2)
    pval = stats.norm.sf(Z)
    
    pval[always_detected] = 1
    pval[np.isnan(pval)] = 1
    effect_size = K_equiv / fit['K']
    effect_size[np.isnan(effect_size)] = 1
    
    return {'pval': pval, 'fold_change': effect_size}


def hidden__test_DE_P_equiv(expr_mat, fit=None):
    """
    Test for differential expression based on P equivalence.
    """
    gene_info = bg__calc_variables(expr_mat)
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    p_obs = gene_info['p']
    N = expr_mat.shape[1]
    p_err = gene_info['p_stderr']
    S_mean = gene_info['s']
    S_err = gene_info['s_stderr']
    K_err = fit.get('Kerr', 0.1)
    
    p_equiv = fit['predictions']
    propagated_err_p_equiv = p_equiv * np.sqrt(((S_err + K_err) / (S_mean + fit['K']))**2 + (K_err / fit['K'])**2)
    fitted_err_p_equiv = fit.get('fitted_err', propagated_err_p_equiv)
    
    Z = (p_equiv - p_obs) / fitted_err_p_equiv
    pval = stats.norm.cdf(Z)  # Left tail test
    effect_size = p_obs / p_equiv
    
    return {'pval': pval, 'fold_change': effect_size}


def hidden__test_DE_S_equiv(expr_mat, fit=None, method="propagate"):
    """
    Test for differential expression based on S equivalence.
    """
    gene_info = bg__calc_variables(expr_mat)
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    p_obs = gene_info['p']
    N = expr_mat.shape[1]
    p_err = gene_info['p_stderr']
    S_mean = gene_info['s']
    S_err = gene_info['s_stderr']
    K_err = fit.get('Kerr', 0.1)
    
    S_equiv = hidden__invert_MM(fit['K'], p_obs)
    
    if method == "MC":
        # Monte Carlo method (simplified)
        def MC_err(p_base_val, p_err_val):
            n_sims = 10000  # Match R implementation
            p_rand = np.random.normal(p_base_val, p_err_val, n_sims)
            p_rand = p_rand[(p_rand > 0) & (p_rand < 1)]
            if len(p_rand) == 0:
                return 0.1  # Default error if no valid samples
            K_rand = np.random.normal(fit['K'], K_err, len(p_rand))
            K_rand[K_rand < 1] = 1
            S_equiv_rand = hidden__invert_MM(K_rand, p_rand)
            return np.std(S_equiv_rand)
        
        S_equiv_err = pd.Series([MC_err(p_val, e_val) for p_val, e_val in zip(p_obs, p_err)], 
                               index=p_obs.index)
    else:
        S_equiv_err = S_equiv * np.sqrt(2 * (p_err / p_obs)**2 + (K_err / fit['K'])**2)
    
    Z = (S_equiv - S_mean) / np.sqrt(S_err**2 + S_equiv_err**2)
    pval = stats.norm.cdf(Z) * 2  # Two-tailed test
    effect_size = (S_mean - S_equiv) / S_equiv
    
    return {'pval': pval, 'effect': effect_size}


def bg__get_extreme_residuals(expr_mat, fit=None, fdr_threshold=0.1, percent=None, v_threshold=(0.05, 0.95), direction="right", suppress_plot=False):
    """
    Internal function to get outliers from the Michaelis-Menten curve.
    """
    gene_info = bg__calc_variables(expr_mat)
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    res = bg__horizontal_residuals_MM_log10(fit['K'], gene_info['p'], gene_info['s'])
    
    valid_res = res[(gene_info['p'] < max(v_threshold)) & (gene_info['p'] > min(v_threshold))].dropna()

    if percent is None:
        mu = valid_res.mean()
        sigma = valid_res.std()
        
        # Bi-modality check - should be 0.68 theoretically
        if np.sum((valid_res > mu - sigma) & (valid_res < mu + sigma)) < 0.5 * len(valid_res):
            mu = valid_res[valid_res > np.quantile(valid_res, 0.33)].mean()
            sigma = valid_res[valid_res > np.quantile(valid_res, 0.33)].std()

        if direction == "right":
            pvals = stats.norm.sf((valid_res - mu) / sigma)
        else:
            pvals = stats.norm.cdf((valid_res - mu) / sigma)

        # FDR correction using Benjamini-Hochberg procedure
        qvals = multipletests(pvals, alpha=fdr_threshold, method='fdr_bh')[1]
        sig = qvals < fdr_threshold

        return valid_res.index[sig].tolist()
    else:
        if direction == "right":
            cut_off = np.quantile(valid_res, 1 - percent)
            return valid_res.index[valid_res > cut_off].tolist()
        else:
            cut_off = np.quantile(valid_res, percent)
            return valid_res.index[valid_res < cut_off].tolist()


def M3DropFeatureSelection(expr_mat, mt_method="bon", mt_threshold=0.05, suppress_plot=False, xlim=None):
    """
    Identifies features with significantly different expression than expected
    by using the Michaelis-Menten model.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression matrix with genes as rows and cells as columns.
    mt_method : str, default="bon"
        Multiple testing correction method. "bon" for Bonferroni, "fdr_bh" for FDR.
    mt_threshold : float, default=0.05
        Significance threshold after multiple testing correction.
    suppress_plot : bool, default=False
        Whether to suppress plotting.
    xlim : tuple, optional
        X-axis limits for plotting.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with significant genes, their effect sizes, p-values, and q-values.
    """
    from .Curve_fitting import bg__dropout_plot_base, bg__add_model_to_plot
    
    # Placeholder plotting functions for now
    def bg__dropout_plot_base_placeholder(mat, xlim=None, suppress_plot=False):
        return {'gene_info': bg__calc_variables(mat)}
    
    def bg__add_model_to_plot_placeholder(fit, base_plot, **kwargs):
        return None
    
    def bg__highlight_genes_placeholder(base_plot, expr_mat, genes):
        return None
    
    BasePlot = bg__dropout_plot_base_placeholder(expr_mat, xlim=xlim, suppress_plot=suppress_plot)
    MM = bg__fit_MM(BasePlot['gene_info']['p'], BasePlot['gene_info']['s'])
    
    if not suppress_plot:
        bg__add_model_to_plot_placeholder(MM, BasePlot)
    
    DEoutput = bg__test_DE_K_equiv(BasePlot['gene_info'], fit=MM)
    
    # Multiple testing correction - convert method names
    if mt_method == "bon":
        method = "bonferroni"
    elif mt_method == "fdr":
        method = "fdr_bh"
    else:
        method = mt_method
    
    qvals = multipletests(DEoutput['pval'], alpha=mt_threshold, method=method)[1]
    
    sig = qvals < mt_threshold
    # Use the filtered gene names from the processed data instead of original expr_mat.index
    filtered_gene_names = BasePlot['gene_info']['p'].index
    DEgenes = filtered_gene_names[sig]
    DEgenes = DEgenes[~pd.isna(DEgenes)]
    
    if not suppress_plot:
        bg__highlight_genes_placeholder(BasePlot, expr_mat, DEgenes)
    
    TABLE = pd.DataFrame({
        'Gene': DEgenes,
        'effect.size': DEoutput['fold_change'][sig],
        'p.value': DEoutput['pval'][sig],
        'q.value': qvals[sig]
    })
    TABLE = TABLE.sort_values('effect.size', ascending=False)
    
    return TABLE


def M3DropGetExtremes(expr_mat, fdr_threshold=0.1, percent=None, v_threshold=(0.05, 0.95), suppress_plot=False):
    """
    Identifies outliers left and right of a fitted Michaelis-Menten curve.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Normalized (not log-transformed) expression values.
    fdr_threshold : float, default=0.1
        Threshold for identifying significant outliers.
    percent : float, optional
        Identify this percentage of data that is most extreme.
    v_threshold : tuple of float, default=(0.05, 0.95)
        Restrict to this range of dropout rates.
    suppress_plot : bool, default=False
        Whether to plot the fitted curve.

    Returns
    -------
    dict
        A dictionary with 'left' and 'right' extreme genes.
    """
    # Placeholders for plotting functions
    def bg__dropout_plot_base(mat, suppress_plot=False, **kwargs): 
        return {'gene_info': bg__calc_variables(mat)}
    def bg__add_model_to_plot(fit, base_plot, **kwargs): 
        return
    def bg__highlight_genes(base_plot, mat, genes, **kwargs): 
        return

    base_plot = bg__dropout_plot_base(expr_mat, suppress_plot=suppress_plot)
    MM = bg__fit_MM(base_plot['gene_info']['p'], base_plot['gene_info']['s'])

    if not suppress_plot:
        bg__add_model_to_plot(MM, base_plot)
    
    # Match R implementation parameter handling
    if percent is None:
        shifted_right = bg__get_extreme_residuals(expr_mat, fit=MM, fdr_threshold=fdr_threshold, 
                                                 v_threshold=v_threshold, direction="right", suppress_plot=True)
        shifted_left = bg__get_extreme_residuals(expr_mat, fit=MM, fdr_threshold=fdr_threshold, 
                                                v_threshold=v_threshold, direction="left", suppress_plot=True)
    else:
        shifted_right = bg__get_extreme_residuals(expr_mat, fit=MM, percent=percent, 
                                                 v_threshold=v_threshold, direction="right", suppress_plot=True)
        shifted_left = bg__get_extreme_residuals(expr_mat, fit=MM, percent=percent, 
                                                v_threshold=v_threshold, direction="left", suppress_plot=True)
    
    if not suppress_plot:
        bg__highlight_genes(base_plot, expr_mat, shifted_right)
        bg__highlight_genes(base_plot, expr_mat, shifted_left)
        
    return {'left': shifted_left, 'right': shifted_right}


def M3DropTestShift(expr_mat, genes_to_test, name="", background=None, suppress_plot=False):
    """
    Tests whether a given set of genes are significantly shifted to the left or
    right of the Michaelis-Menten curve.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression matrix with genes as rows and cells as columns.
    genes_to_test : list
        List of gene names to test.
    name : str, default=""
        Name for the test (used in output).
    background : list, optional
        Background gene set for comparison. If None, uses all genes.
    suppress_plot : bool, default=False
        Whether to suppress plotting.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with test results including sample median, background median, and p-value.
    """
    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
        
    # Set default background to all genes like in R version
    if background is None:
        background = expr_mat.index.tolist()

    # Placeholders for plotting functions
    def bg__dropout_plot_base(mat, suppress_plot=False, **kwargs): 
        return {'gene_info': bg__calc_variables(mat)}
    def bg__add_model_to_plot(fit, base_plot, **kwargs): 
        return
    def bg__highlight_genes(base_plot, mat, genes, **kwargs): 
        return

    BasePlot = bg__dropout_plot_base(expr_mat, suppress_plot=suppress_plot)
    MM = bg__fit_MM(BasePlot['gene_info']['p'], BasePlot['gene_info']['s'])
    
    if not suppress_plot:
        bg__add_model_to_plot(MM, BasePlot)
        bg__highlight_genes(BasePlot, expr_mat, genes_to_test)

    res = bg__horizontal_residuals_MM_log10(MM['K'], BasePlot['gene_info']['p'], BasePlot['gene_info']['s'])
    res[np.isinf(res)] = np.nan

    # Calculate medians - use the filtered gene names from res instead of original expr_mat.index
    # to ensure mask length matches res length
    background_mask = res.index.isin([str(g) for g in background])
    test_mask = res.index.isin([str(g) for g in genes_to_test])
    
    mu = np.nanmedian(res[background_mask])
    s_mu = np.nanmedian(res[test_mask])
    
    # Wilcoxon test using Mann-Whitney U
    test_residuals = res[test_mask].dropna()
    background_residuals = res[background_mask].dropna()
    
    if len(test_residuals) == 0 or len(background_residuals) == 0:
        pval = np.nan
    else:
        try:
            _, pval = stats.mannwhitneyu(test_residuals, background_residuals, alternative='two-sided')
        except ValueError:
            pval = np.nan

    return pd.DataFrame({
        'sample': [s_mu],
        'background': [mu], 
        'p.value': [pval]
    }, index=[name])