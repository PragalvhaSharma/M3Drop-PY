import numpy as np
import pandas as pd
import warnings

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
    # Set up
    if gene_list is None:
        gene_list = list(fit['vals']['tjs'].index)

    if isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts)

    # Initialize matrix for gene probabilities
    pd_gene = np.full((len(gene_list), counts.shape[1]), -1.0)
    name_gene = [""] * len(gene_list)
    
    for i, gene_name in enumerate(gene_list):
        if gene_name in fit['vals']['tjs'].index:
            gid = fit['vals']['tjs'].index.get_loc(gene_name)
            mu_is = fit['vals']['tjs'].iloc[gid] * fit['vals']['tis'] / fit['vals']['total']
            p_is = (1 + mu_is / fit['sizes'][gid])**(-fit['sizes'][gid])
            pd_gene[i, :] = p_is
            name_gene[i] = gene_name

    # Remove genes that weren't found
    if sum(name == "" for name in name_gene) > 0:
        missing_count = sum(name == "" for name in name_gene)
        warnings.warn(f"Warning: {missing_count} genes not found, check your gene list is correct.")
        exclude = [i for i, name in enumerate(name_gene) if name == ""]
        pd_gene = np.delete(pd_gene, exclude, axis=0)
        name_gene = [name for name in name_gene if name != ""]
    
    # Convert to DataFrame for easier indexing
    pd_gene = pd.DataFrame(pd_gene, index=name_gene)
    
    # Initialize Z-score matrix
    n_genes = len(pd_gene)
    z_mat = np.full((n_genes, n_genes), -1.0)

    for i in range(n_genes):
        for j in range(i, n_genes):
            p_g1 = pd_gene.iloc[i, :]
            p_g2 = pd_gene.iloc[j, :]
            
            gene1_name = pd_gene.index[i]
            gene2_name = pd_gene.index[j]
            
            expr_g1 = counts.loc[gene1_name, :]
            expr_g2 = counts.loc[gene2_name, :]

            if method == "off" or method == "both":
                # Both zero
                expect_both_zero = p_g1 * p_g2
                expect_both_err = expect_both_zero * (1 - expect_both_zero)
                obs_both_zero = np.sum((expr_g1 == 0) & (expr_g2 == 0))
                z = (obs_both_zero - np.sum(expect_both_zero)) / np.sqrt(np.sum(expect_both_err))

            if method == "on" or method == "both":
                # Both nonzero
                obs_both_nonzero = np.sum((expr_g1 != 0) & (expr_g2 != 0))
                expect_both_nonzero = (1 - p_g1) * (1 - p_g2)
                expect_non_err = expect_both_nonzero * (1 - expect_both_nonzero)
                z = (obs_both_nonzero - np.sum(expect_both_nonzero)) / np.sqrt(np.sum(expect_non_err))

            if method == "both":
                # Either (this overwrites the previous z calculation, matching R behavior)
                obs_either = obs_both_zero + obs_both_nonzero
                expect_either = expect_both_zero + expect_both_nonzero
                expect_err = expect_either * (1 - expect_either)
                z = (obs_either - np.sum(expect_either)) / np.sqrt(np.sum(expect_err))
            
            z_mat[i, j] = z_mat[j, i] = z
            
    # Convert to DataFrame with proper row/column names
    z_mat = pd.DataFrame(z_mat, index=pd_gene.index, columns=pd_gene.index)
    
    return z_mat