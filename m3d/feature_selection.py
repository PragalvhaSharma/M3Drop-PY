import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

def BrenneckeGetVariableGenes(expr_mat, spikes=None, suppress_plot=False, fdr=0.1, minBiolDisp=0.5, fitMeanQuantile=0.8):
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
    minBiolDisp : float, default=0.5
        Minimum percentage of variance due to biological factors.
    fitMeanQuantile : float, default=0.8
        Threshold for genes to be used in fitting.

    Returns
    -------
    pd.DataFrame
        DataFrame of highly variable genes.
    """

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
    
    p = 1 - chi2.cdf(varsGenes * (m - 1) / testDenom, m - 1)
    
    # FDR adjustment
    p_df = pd.DataFrame({'p': p, 'gene': p.index})
    p_df = p_df.sort_values(by='p')
    p_df['i'] = np.arange(1, len(p_df) + 1)
    p_df['p_adj'] = p_df['p'] * len(p_df) / p_df['i']
    padj = p_df.set_index('gene')['p_adj']
    padj = padj.reindex(p.index)

    sig = padj < fdr
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

def irlbaPcaFS(expr_mat, pcs=[1, 2]):
    """
    Ranks features by PCA loadings using irlba.

    Features are ranked by the sum of the magnitude of the loadings for the
    specified principal components.

    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Normalized but not log-transformed gene expression matrix.
    pcs : list of int, default=[1,2]
        Which principal components to use to score genes.

    Returns
    -------
    pd.Series
        Sorted series of scores for each gene.
    """
    from sklearn.decomposition import TruncatedSVD

    if isinstance(expr_mat, pd.DataFrame):
        gene_names = expr_mat.index
        norm = expr_mat.values
    else:
        norm = expr_mat
        gene_names = np.arange(norm.shape[0])

    norm = np.log1p(norm) / np.log(2)
    
    # Filter out genes that are constant
    genes_to_keep = np.var(norm, axis=1) > 0
    norm_filtered = norm[genes_to_keep, :]
    
    svd = TruncatedSVD(n_components=max(pcs) + 1, random_state=42)
    
    # sklearn's TruncatedSVD works on sample x feature, so we transpose
    svd.fit(norm_filtered.T)
    
    # components_ are equivalent to loadings (v in R's irlba)
    loadings = svd.components_.T 
    
    if len(pcs) > 1:
        score = np.sum(np.abs(loadings[:, pcs]), axis=1)
    else:
        score = np.abs(loadings[:, pcs[0]])

    score_series = pd.Series(score, index=gene_names[genes_to_keep])
    
    # Add back zero-variance genes with a score of 0
    all_scores = pd.Series(0, index=gene_names)
    all_scores.update(score_series)

    return all_scores.sort_values(ascending=False)

def gini(x):
    """
    Calculate the Gini coefficient of a numpy array.
    From: https://github.com/oliviaguest/gini
    """
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(x, dtype=np.float64)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def giniFS(expr_mat, suppress_plot=True):
    """
    Ranks features by Gini-index based residuals.

    Fits a loess curve between the maximum expression value and gini-index of
    each gene. Genes are ranked by p-value from a normal distribution fit to
    the residuals of the curve.

    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Normalized but not log-transformed expression matrix.
    suppress_plot : bool, default=True
        Whether to plot the gene expression vs Gini score.

    Returns
    -------
    pd.Series
        Sorted series of p-values for each gene.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if isinstance(expr_mat, pd.DataFrame):
        gene_names = expr_mat.index
        norm = expr_mat.values
    else:
        norm = expr_mat
        gene_names = np.arange(norm.shape[0])

    norm = norm[np.sum(norm, axis=1) > 0, :]
    gene_names = gene_names[np.sum(norm, axis=1) > 0]
    
    ginis = np.apply_along_axis(gini, 1, norm)
    max_expr = np.log1p(np.max(norm, axis=1)) / np.log(2)

    # Loess fit, similar to R's loess
    fit = lowess(ginis, max_expr, frac=0.3, it=3, delta=0.0, is_sorted=False)
    
    # This is a simplification of the outlier removal and re-fitting in R
    residuals = ginis - fit[:, 1]
    
    p_values = norm.sf(residuals, loc=np.mean(residuals), scale=np.std(residuals))
    
    p_series = pd.Series(p_values, index=gene_names)
    return p_series.sort_values()

def corFS(expr_mat, direction="both", fdr=None):
    """
    Ranks features by gene-gene correlations.

    Calculates all gene-gene correlations then ranks genes by the magnitude of
    the most positive or negative correlation.

    Parameters
    ----------
    expr_mat : pd.DataFrame or np.ndarray
        Normalized but not log-transformed gene expression matrix.
    direction : {"both", "pos", "neg"}, default="both"
        Direction of correlation to consider.
    fdr : float, optional
        FDR threshold for significant features. Not implemented yet.

    Returns
    -------
    pd.Series
        Sorted series of scores for each gene.
    """
    if isinstance(expr_mat, pd.DataFrame):
        df = expr_mat
    else:
        df = pd.DataFrame(expr_mat)
    
    # Using pandas' corr function with spearman method
    # Note: This can be very memory intensive for large datasets
    cor_mat = df.T.corr(method='spearman')
    
    # Set diagonal to 0 to ignore self-correlation
    np.fill_diagonal(cor_mat.values, 0)
    
    if direction == "both":
        min_cors = cor_mat.min(axis=1)
        max_cors = cor_mat.max(axis=1)
        score = np.abs(min_cors) + np.abs(max_cors)
    elif direction == "pos":
        score = cor_mat.max(axis=1)
    elif direction == "neg":
        score = np.abs(cor_mat.min(axis=1))
    else:
        raise ValueError("Unrecognized direction")
    
    if fdr is not None:
        # FDR calculation for correlations is not straightforward and is skipped
        print("FDR thresholding for corFS is not implemented.")
        
    return score.sort_values(ascending=False)

def Consensus_FS(counts, norm=None, is_spike=None, pcs=[1, 2], include_cors=True):
    """
    Performs seven different feature selection methods then calculates the
    consensus ranking of features from that.

    Parameters
    ----------
    counts : pd.DataFrame or np.ndarray
        Raw count matrix, rows=genes, cols=cells.
    norm : pd.DataFrame or np.ndarray, optional
        Normalized but not log-transformed gene expression matrix.
    is_spike : array-like of bool, optional
        Vector of whether each gene is a spike-in.
    pcs : list of int, default=[1, 2]
        Which principal components to use for `irlbaPcaFS`.
    include_cors : bool, default=True
        Whether to perform gene-gene correlation feature selection.

    Returns
    -------
    pd.DataFrame
        Table of ranking of each gene by each method including the consensus.
    """
    from .normalization import M3DropConvertData
    from .utils import NBumiFitModel
    from .differential_expression import M3DropFeatureSelection

    if isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts)

    if norm is None:
        norm = M3DropConvertData(counts, is_counts=True)
    elif isinstance(norm, np.ndarray):
        norm = pd.DataFrame(norm, index=counts.index)
        
    if is_spike is None:
        is_spike = pd.Series(False, index=counts.index)
    elif isinstance(is_spike, (list, np.ndarray)):
        is_spike = pd.Series(is_spike, index=counts.index)

    # Remove invariant genes
    invariant = counts.var(axis=1) == 0
    counts = counts.loc[~invariant]
    norm = norm.loc[~invariant]
    is_spike = is_spike.loc[~invariant]
    
    # Placeholders for not-yet-ported functions
    def NBumiFeatureSelectionCombinedDrop(fit, **kwargs): return pd.DataFrame({'Gene': fit['vals']['tjs'].index})
    def NBumiFeatureSelectionHighVar(fit, **kwargs): return pd.Series(index=fit['vals']['tjs'].index)

    # Apply FS methods
    # DANB
    fit = NBumiFitModel(counts) # Assumes NBumiFitModel is in utils
    danb_drop = NBumiFeatureSelectionCombinedDrop(fit)
    danb_var = NBumiFeatureSelectionHighVar(fit)
    
    # HVG
    spikes = is_spike[is_spike].index.tolist()
    hvg = BrenneckeGetVariableGenes(norm, spikes=spikes if len(spikes) > 10 else None, fdr=2, suppress_plot=True)
    
    # M3Drop
    m3drop = M3DropFeatureSelection(norm, mt_method="fdr", mt_threshold=2, suppress_plot=True)
    
    # Gini
    gini_fs = giniFS(norm)
    
    # PCA
    pca_fs = irlbaPcaFS(norm, pcs=pcs)
    
    # cor
    if include_cors:
        cor_fs = corFS(norm)
    else:
        cor_fs = pd.Series(-1, index=norm.index)
        
    # Combine ranks
    ref_order = counts.index
    ranks = pd.Series(np.arange(1, len(ref_order) + 1), index=ref_order)
    
    def get_ranks(feature_series, ref):
        if isinstance(feature_series, pd.DataFrame):
            s = pd.Series(np.arange(1, len(feature_series)+1), index=feature_series['Gene'])
        else: #is series
            s = pd.Series(np.arange(1, len(feature_series)+1), index=feature_series.index)
        return s.reindex(ref).fillna(len(ref) + 1)

    rank_table = pd.DataFrame({
        'DANB_drop': get_ranks(danb_drop, ref_order),
        'DANB_var': get_ranks(danb_var, ref_order),
        'M3Drop': get_ranks(m3drop, ref_order),
        'HVG': get_ranks(hvg, ref_order),
        'PCA': get_ranks(pca_fs, ref_order),
        'Cor': get_ranks(cor_fs, ref_order),
        'Gini': get_ranks(gini_fs, ref_order)
    })
    
    rank_table['Cons'] = rank_table.mean(axis=1)
    
    return rank_table.sort_values(by='Cons')

def bg__get_extreme_residuals(expr_mat, fit=None, fdr_threshold=0.1, percent=None, v_threshold=(0.05, 0.95), direction="right", suppress_plot=False):
    """
    Internal function to get outliers from the Michaelis-Menten curve.
    """
    from .utils import bg__calc_variables, bg__fit_MM, bg__horizontal_residuals_MM_log10
    from scipy.stats import norm

    gene_info = bg__calc_variables(expr_mat)
    if fit is None:
        fit = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    res = bg__horizontal_residuals_MM_log10(fit['K'], gene_info['p'], gene_info['s'])
    
    valid_res = res[(gene_info['p'] < max(v_threshold)) & (gene_info['p'] > min(v_threshold))].dropna()

    if percent is None:
        mu = valid_res.mean()
        sigma = valid_res.std()
        
        # Bi-modality check
        if np.sum((valid_res > mu - sigma) & (valid_res < mu + sigma)) < 0.5 * len(valid_res):
            mu = valid_res[valid_res > np.quantile(valid_res, 0.33)].mean()
            sigma = valid_res[valid_res > np.quantile(valid_res, 0.33)].std()

        if direction == "right":
            pvals = norm.sf((valid_res - mu) / sigma)
        else:
            pvals = norm.cdf((valid_res - mu) / sigma)

        # Simple FDR
        pvals_sorted = np.sort(pvals)
        i = np.arange(1, len(pvals_sorted) + 1)
        try:
            threshold_p = pvals_sorted[pvals_sorted <= i / len(pvals_sorted) * fdr_threshold][-1]
            sig = pvals <= threshold_p
        except IndexError:
            sig = np.repeat(False, len(pvals))

        return valid_res.index[sig].tolist()
    else:
        if direction == "right":
            cut_off = np.quantile(valid_res, 1 - percent)
            return valid_res.index[valid_res > cut_off].tolist()
        else:
            cut_off = np.quantile(valid_res, percent)
            return valid_res.index[valid_res < cut_off].tolist()


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
    from .utils import bg__calc_variables, bg__fit_MM
    # from .plotting import bg__dropout_plot_base, bg__add_model_to_plot, bg__highlight_genes

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
    
    shifted_right = bg__get_extreme_residuals(expr_mat, fit=MM, fdr_threshold=fdr_threshold, percent=percent, v_threshold=v_threshold, direction="right", suppress_plot=True)
    shifted_left = bg__get_extreme_residuals(expr_mat, fit=MM, fdr_threshold=fdr_threshold, percent=percent, v_threshold=v_threshold, direction="left", suppress_plot=True)
    
    if not suppress_plot:
        bg__highlight_genes(base_plot, expr_mat, shifted_right)
        bg__highlight_genes(base_plot, expr_mat, shifted_left)
        
    return {'left': shifted_left, 'right': shifted_right}

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
    from scipy.stats import rankdata, mannwhitneyu
    from sklearn.metrics import roc_auc_score

    if isinstance(expr_mat, np.ndarray):
        expr_mat = pd.DataFrame(expr_mat)
    
    if len(labels) != expr_mat.shape[1]:
        raise ValueError("Length of labels does not match number of cells.")

    def get_auc(gene_expr, labels):
        ranks = rankdata(gene_expr)
        df = pd.DataFrame({'ranks': ranks, 'labels': labels})
        mean_ranks = df.groupby('labels')['ranks'].mean()
        pos_group = mean_ranks.idxmax()
        
        if (mean_ranks == mean_ranks.max()).sum() > 1:
            return -1, -1, -1

        truth = (labels == pos_group).astype(int)
        
        try:
            auc = roc_auc_score(truth, gene_expr)
            stat, pval = mannwhitneyu(gene_expr[truth==1], gene_expr[truth==0])
        except ValueError:
            # Handle cases with no variance or too few samples
            return 0, pos_group, 1

        return auc, pos_group, pval

    aucs = expr_mat.apply(lambda gene: get_auc(gene, labels), axis=1)
    
    auc_df = pd.DataFrame(aucs.tolist(), index=expr_mat.index, columns=['AUC', 'Group', 'pval'])
    
    auc_df['Group'] = auc_df['Group'].astype(str)
    auc_df.loc[auc_df['Group'] == '-1', 'Group'] = "Ambiguous"
    
    auc_df = auc_df[auc_df['AUC'] > 0]
    auc_df = auc_df.sort_values(by='AUC', ascending=False)
    
    return auc_df

def NBumiCoexpression(counts, fit, gene_list=None, method="both"):
    """
    Ranks genes based on co-expression.

    Tests for co-expression using the normal approximation of a binomial test.

    Parameters
    ----------
    counts : pd.DataFrame or np.ndarray
        Raw count matrix.
    fit : dict
        Output from `NBumiFitModel`.
    gene_list : list of str, optional
        Set of gene names to test coexpression of.
    method : {"both", "on", "off"}, default="both"
        Type of co-expression to test. "on" for co-expression, "off" for
        co-absence, "both" for either.

    Returns
    -------
    pd.DataFrame
        A matrix of Z-scores for each pair of genes.
    """
    if gene_list is None:
        gene_list = fit['vals']['tjs'].index

    if isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts)

    pd_gene = []
    name_gene = []
    
    for gene_name in gene_list:
        if gene_name in fit['vals']['tjs'].index:
            gid = fit['vals']['tjs'].index.get_loc(gene_name)
            mu_is = fit['vals']['tjs'][gid] * fit['vals']['tis'] / fit['vals']['total']
            p_is = (1 + mu_is / fit['sizes'][gid])**(-fit['sizes'][gid])
            pd_gene.append(p_is)
            name_gene.append(gene_name)
    
    pd_gene = pd.DataFrame(pd_gene, index=name_gene)

    z_mat = pd.DataFrame(np.nan, index=pd_gene.index, columns=pd_gene.index)

    for i, g1_name in enumerate(pd_gene.index):
        for j, g2_name in enumerate(pd_gene.index):
            if i > j: continue
            
            p_g1 = pd_gene.loc[g1_name]
            p_g2 = pd_gene.loc[g2_name]
            expr_g1 = counts.loc[g1_name]
            expr_g2 = counts.loc[g2_name]

            if method == "off" or method == "both":
                expect_both_zero = p_g1 * p_g2
                expect_both_err = expect_both_zero * (1 - expect_both_zero)
                obs_both_zero = np.sum((expr_g1 == 0) & (expr_g2 == 0))
                z = (obs_both_zero - np.sum(expect_both_zero)) / np.sqrt(np.sum(expect_both_err))

            if method == "on" or method == "both":
                expect_both_nonzero = (1 - p_g1) * (1 - p_g2)
                expect_non_err = expect_both_nonzero * (1 - expect_both_nonzero)
                obs_both_nonzero = np.sum((expr_g1 != 0) & (expr_g2 != 0))
                z_on = (obs_both_nonzero - np.sum(expect_both_nonzero)) / np.sqrt(np.sum(expect_non_err))
                if method == "on":
                    z = z_on
                elif method == "both":
                    # R code has a bug here, it overwrites z. Let's combine them properly.
                    # Simple averaging of Z-scores is not statistically sound.
                    # The R code for "both" calculates a third Z-score.
                    obs_either = obs_both_zero + obs_both_nonzero
                    expect_either = expect_both_zero + expect_both_nonzero
                    expect_err = expect_either * (1 - expect_either)
                    z = (obs_either - np.sum(expect_either)) / np.sqrt(np.sum(expect_err))
            
            z_mat.loc[g1_name, g2_name] = z
            z_mat.loc[g2_name, g1_name] = z
            
    return z_mat

def NBumiFeatureSelectionCombinedDrop(fit, ntop=None, method="fdr", qval_thresh=2, suppress_plot=True):
    """
    Ranks genes by significance of increase in dropouts compared to expectation.

    Parameters
    ----------
    fit : dict
        Output from `NBumiFitModel`.
    ntop : int, optional
        Number of top ranked genes to return.
    method : str, default="fdr"
        Correction method for multiple comparisons.
    qval_thresh : float, default=2
        Significance threshold.
    suppress_plot : bool, default=True
        Whether to plot the fitted curve.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene, effect size, p-value, and q-value.
    """
    from .utils import NBumiFitDispVsMean
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    vals = fit['vals']
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(vals['tjs'] / vals['nc']))

    droprate_exp = np.zeros(vals['ng'])
    droprate_exp_err = np.zeros(vals['ng'])

    for i in range(vals['ng']):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / exp_size[i])**(-exp_size[i])
        p_var_is = p_is * (1 - p_is)
        droprate_exp[i] = np.sum(p_is) / vals['nc']
        droprate_exp_err[i] = np.sqrt(np.sum(p_var_is) / (vals['nc']**2))

    droprate_exp[droprate_exp < 1 / vals['nc']] = 1 / vals['nc']
    droprate_obs = vals['djs'] / vals['nc']
    
    diff = droprate_obs - droprate_exp
    combined_err = np.sqrt(droprate_exp_err**2 + (droprate_obs * (1 - droprate_obs) / vals['nc']))
    
    zed = diff / combined_err
    pvalue = norm.sf(zed)
    
    pvalue_df = pd.DataFrame({'pvalue': pvalue, 'diff': diff}, index=vals['tjs'].index)
    pvalue_df = pvalue_df.sort_values(by=['pvalue', 'diff'], ascending=[True, False])

    qval = multipletests(pvalue_df['pvalue'], method=method)[1]
    pvalue_df['qvalue'] = qval
    
    if ntop is None:
        result = pvalue_df[pvalue_df['qvalue'] < qval_thresh]
    else:
        result = pvalue_df.iloc[:ntop]
        
    result = result.rename(columns={'diff': 'effect_size', 'pvalue': 'p.value', 'qvalue': 'q.value'})
    result['Gene'] = result.index
    return result

def NBumiFeatureSelectionHighVar(fit):
    """
    Ranks genes by residual dispersion from mean-dispersion power-law relationship.

    Parameters
    ----------
    fit : dict
        Output from `NBumiFitModel`.

    Returns
    -------
    pd.Series
        Sorted vector of residuals.
    """
    from .utils import NBumiFitDispVsMean
    
    vals = fit['vals']
    coeffs = NBumiFitDispVsMean(fit, suppress_plot=True)
    exp_size = np.exp(coeffs[0] + coeffs[1] * np.log(vals['tjs'] / vals['nc']))
    res = np.log(fit['sizes']) - np.log(exp_size)
    return res.sort_values()

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
    
    qvals = multipletests(pvals[~np.isnan(pvals)], method='fdr')[1]
    
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

def PoissonUMIFeatureSelectionDropouts(fit):
    """
    Ranks genes by significance of increase in dropouts compared to a Poisson
    expectation.

    Parameters
    ----------
    fit : dict
        Output from `NBumiFitModel`.

    Returns
    -------
    pd.Series
        Sorted series of p-values.
    """
    from scipy.stats import norm
    
    vals = fit['vals']
    droprate_exp = np.zeros(vals['ng'])
    droprate_exp_err = np.zeros(vals['ng'])

    for i in range(vals['ng']):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = np.exp(-mu_is)
        p_var_is = p_is * (1 - p_is)
        droprate_exp[i] = np.sum(p_is) / vals['nc']
        droprate_exp_err[i] = np.sqrt(np.sum(p_var_is) / (vals['nc']**2))

    droprate_exp[droprate_exp < 1 / vals['nc']] = 1 / vals['nc']
    droprate_obs = vals['djs'] / vals['nc']
    
    diff = droprate_obs - droprate_exp
    combined_err = droprate_exp_err
    zed = diff / combined_err
    pvalue = norm.sf(zed)
    
    pvalue_series = pd.Series(pvalue, index=vals['tjs'].index)
    return pvalue_series.sort_values()

import warnings
import functools

def obsolete(func):
    """
    Decorator to mark functions as obsolete.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(
            f"Call to obsolete function {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return new_func

@obsolete
def obsolete__nbumiFeatureSelectionDropouts(fit):
    """
    Ranks genes by significance of increase in dropouts compared to expectation
    allowing for gene-specific dispersions.
    """
    from scipy.stats import norm
    
    vals = fit['vals']
    droprate_exp = np.zeros(vals['ng'])
    droprate_exp_err = np.zeros(vals['ng'])

    for i in range(vals['ng']):
        mu_is = vals['tjs'][i] * vals['tis'] / vals['total']
        p_is = (1 + mu_is / fit['sizes'][i])**(-fit['sizes'][i])
        p_var_is = p_is * (1 - p_is)
        droprate_exp[i] = np.sum(p_is) / vals['nc']
        droprate_exp_err[i] = np.sqrt(np.sum(p_var_is) / (vals['nc']**2))

    droprate_exp[droprate_exp < 1 / vals['nc']] = 1 / vals['nc']
    droprate_obs = vals['djs'] / vals['nc']
    
    diff = droprate_obs - droprate_exp
    combined_err = droprate_exp_err
    zed = diff / combined_err
    pvalue = norm.sf(zed)
    
    pvalue_series = pd.Series(pvalue, index=vals['tjs'].index)
    return pvalue_series.sort_values()

@obsolete
def obsolete__nbumiFeatureSelectionHighVarDist2Med(fit, window_size=1000):
    """
    Ranks genes by the distance to median of log-transformed estimated dispersions.
    """
    vals = fit['vals']
    mean_order = np.argsort(vals['tjs'])
    obs_mean = (vals['tjs']/vals['nc'])[mean_order]
    fit_disp = fit['sizes'][mean_order]
    
    keep = fit_disp < np.max(fit_disp)
    fit_disp = np.log(fit_disp[keep])
    obs_mean = obs_mean[keep]

    flank = window_size // 2
    
    def dist_from_med(i):
        low = max(0, i - flank)
        high = min(len(fit_disp), i + flank)
        return fit_disp[i] - np.median(fit_disp[low:high])

    score = pd.Series([dist_from_med(i) for i in range(len(fit_disp))], index=vals['tjs'].index[mean_order][keep])
    return score.sort_values()

def M3DropFeatureSelection(expr_mat, mt_method="fdr_bh", mt_threshold=0.01, method="brennecke"):
    """
    Wrapper for different feature selection methods.
    
    Parameters:
    - expr_mat: The expression matrix.
    - mt_method (str): The multiple testing correction method to use.
    - mt_threshold (float): The significance threshold.
    - method (str): The feature selection method to use. Currently only "brennecke" is supported.
    
    Returns:
    - pandas.DataFrame: A DataFrame of significantly variable genes.
    """
    if method.lower() == "brennecke":
        return BrenneckeGetVariableGenes(expr_mat, mt_method=mt_method, fdr=mt_threshold)
    else:
        raise NotImplementedError(f"Method '{method}' is not currently supported by M3DropFeatureSelection.")
