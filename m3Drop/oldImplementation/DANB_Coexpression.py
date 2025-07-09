import numpy as np
import pandas as pd

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