import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, nbinom
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import minimize

def hidden_pca_fs(expr_mat, pcs=[1, 2]):
    """
    PCA-based feature selection - matches R implementation exactly
    """
    # Convert to numpy array if needed
    if sparse.issparse(expr_mat):
        expr_mat = expr_mat.toarray()
    
    # Log2 transform
    log_expr = np.log2(expr_mat + 1)
    
    # R does: pca <- prcomp(log(expr_mat+1)/log(2))
    # This treats genes as observations (rows), cells as variables (columns)
    pca = PCA()
    pca_scores = pca.fit_transform(log_expr)  # genes x cells -> genes x PCs
    
    # R does: pca$x[,pcs] - select specific PCs for each gene
    if len(pcs) > 1:
        score = np.sum(np.abs(pca_scores[:, np.array(pcs) - 1]), axis=1)  # Sum across specified PCs
    else:
        score = np.abs(pca_scores[:, pcs[0] - 1])  # Single PC
    
    # Create series with gene names as index
    if hasattr(expr_mat, 'index'):
        gene_names = expr_mat.index
    else:
        gene_names = [f'gene_{i}' for i in range(expr_mat.shape[0])]
    
    score_series = pd.Series(score, index=gene_names)
    return score_series.sort_values(ascending=False)

def irlba_pca_fs(expr_mat, pcs=[2, 3]):
    """
    PCA feature selection using sparse matrix operations (equivalent to irlba)
    """
    # Convert to sparse matrix if not already
    if not sparse.issparse(expr_mat):
        expr_mat = sparse.csr_matrix(expr_mat)
    
    # Get non-zero genes
    nz_genes = np.where(np.array(expr_mat.sum(axis=1)).flatten() != 0)[0]
    
    # Log2 normalize
    norm = expr_mat.copy().astype(float)
    norm.data = np.log2(norm.data + 1)
    
    gene_names = getattr(expr_mat, 'index', [f'gene_{i}' for i in range(expr_mat.shape[0])])
    nc = norm.shape[1]
    
    # Calculate means and variances for sparse matrix
    expression_means = np.array(norm.mean(axis=1)).flatten()
    
    # Calculate variance efficiently for sparse matrix
    mean_sq = np.array((norm.multiply(norm)).mean(axis=1)).flatten()
    expression_vars = (mean_sq - expression_means**2) * (nc / (nc - 1))
    
    # Filter constant genes
    genes_to_keep = expression_vars > 0
    norm_filtered = norm[genes_to_keep, :]
    expression_means_filtered = expression_means[genes_to_keep]
    expression_vars_filtered = expression_vars[genes_to_keep]
    
    # Use TruncatedSVD for sparse PCA equivalent to irlba
    n_components = max(pcs) if len(pcs) > 1 else pcs[0]
    
    # Center and scale the data
    norm_centered = norm_filtered.copy()
    norm_centered.data -= np.repeat(expression_means_filtered, np.diff(norm_centered.indptr))
    
    # Apply truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(norm_centered.T)
    
    # Get the right singular vectors (equivalent to PCA loadings)
    components = svd.components_
    
    if len(pcs) > 1:
        score = np.sum(np.abs(components[np.array(pcs) - 1, :]), axis=0)
    else:
        score = np.abs(components[pcs[0] - 1, :])
    
    # Map back to original gene order
    full_score = np.zeros(len(gene_names))
    full_score[genes_to_keep] = score
    
    score_series = pd.Series(full_score, index=gene_names)
    return score_series.sort_values(ascending=False)

def hidden_grun_fs(expr_mat, spikes):
    """
    Grun's feature selection method using spike-ins
    """
    if sparse.issparse(expr_mat):
        expr_mat = expr_mat.toarray()
    
    spike_indices = [i for i, gene in enumerate(expr_mat.index) if gene in spikes]
    spike_expr = expr_mat[spike_indices, :]
    
    spike_T = np.mean(spike_expr, axis=1)
    
    # Fit linear models for each cell
    beta2 = []
    for j in range(expr_mat.shape[1]):
        cell_spikes = spike_expr[:, j]
        reg = LinearRegression().fit(spike_T.reshape(-1, 1), cell_spikes)
        beta2.append(reg.coef_[0])
    
    # Model 3
    beta3 = spike_expr / spike_T[:, np.newaxis]
    
    def fit_gamma(x):
        mean_x = np.mean(x)
        var_x = np.var(x)
        if var_x == 0:
            return [1.0, 1.0]
        b = var_x / mean_x
        a = mean_x / b
        return [a, 1/b]
    
    gammas = np.array([fit_gamma(beta3[i, :]) for i in range(beta3.shape[0])]).T
    
    # Fit linear models for gamma parameters
    a_fit = LinearRegression().fit(spike_T.reshape(-1, 1), gammas[0, :])
    b_fit = LinearRegression().fit(spike_T.reshape(-1, 1), gammas[1, :])
    
    # Log space fitting
    log_gammas = np.log2(gammas)
    a_fit_l = LinearRegression().fit(spike_T.reshape(-1, 1), log_gammas[0, :])
    b_fit_l = LinearRegression().fit(spike_T.reshape(-1, 1), log_gammas[1, :])
    
    ka = a_fit_l.intercept_
    kb = b_fit_l.intercept_
    fa = a_fit_l.coef_[0]
    fb = b_fit_l.coef_[0]
    
    def get_nb_params_l(mu):
        r = 2**(ka + fa*(kb - ka)/(1 + fa - fb)) * mu**((fa)/(1 + fa - fb))
        return r
    
    def get_bio_disp(g):
        u_g = np.mean(g)
        v_g = np.var(g)
        if v_g <= u_g:
            v_g = u_g + 1e-10
        r_g = u_g**2 / (v_g - u_g)
        
        def min_fun(bio):
            u_bio, r_bio = bio
            ns = np.round(g).astype(int)
            
            def diff(n):
                ms = np.arange(1, 2*n + 1)
                deconvolve = np.sum([nbinom.pmf(n, n=get_nb_params_l(m), p=1/(1+get_nb_params_l(m))) * 
                                   nbinom.pmf(m, n=r_bio, p=r_bio/(r_bio + u_bio)) for m in ms])
                return abs(nbinom.pmf(n, n=r_g, p=r_g/(r_g + u_g)) - deconvolve)
            
            return sum([diff(n) for n in ns])
        
        try:
            result = minimize(min_fun, [u_g, r_g], method='BFGS')
            return result.x[1]
        except:
            return r_g
    
    bio_disp = [get_bio_disp(expr_mat[i, :]) for i in range(expr_mat.shape[0])]
    
    gene_names = getattr(expr_mat, 'index', [f'gene_{i}' for i in range(expr_mat.shape[0])])
    return pd.Series(bio_disp, index=gene_names)

def hidden_ginifs_simple(expr_mat):
    """
    Simple Gini coefficient feature selection
    """
    def gini_coefficient(x):
        """Calculate Gini coefficient"""
        sorted_x = np.sort(x)
        n = len(x)
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    if sparse.issparse(expr_mat):
        expr_mat = expr_mat.toarray()
    
    ginis = [gini_coefficient(expr_mat[i, :]) for i in range(expr_mat.shape[0])]
    d = np.mean(expr_mat > 0, axis=1)
    
    # Linear regression
    reg = LinearRegression().fit(d.reshape(-1, 1), ginis)
    score = ginis - reg.predict(d.reshape(-1, 1))
    
    gene_names = getattr(expr_mat, 'index', [f'gene_{i}' for i in range(expr_mat.shape[0])])
    score_series = pd.Series(score, index=gene_names)
    return score_series.sort_values(ascending=False)

def gini_fs(expr_mat, suppress_plot=True):
    """
    GiniClust feature selection
    """
    def gini_coefficient(x):
        """Calculate Gini coefficient"""
        sorted_x = np.sort(x)
        n = len(x)
        if n == 0 or np.sum(x) == 0:
            return 0
        cumsum = np.cumsum(sorted_x)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    # Store original gene names
    original_gene_names = getattr(expr_mat, 'index', [f'gene_{i}' for i in range(expr_mat.shape[0])])
    
    # Remove genes with zero expression
    if sparse.issparse(expr_mat):
        gene_sums = np.array(expr_mat.sum(axis=1)).flatten()
        non_zero_genes = gene_sums > 0
        expr_mat = expr_mat[non_zero_genes, :].toarray()
        gene_names = [original_gene_names[i] for i in range(len(original_gene_names)) if non_zero_genes[i]]
    else:
        gene_sums = np.sum(expr_mat, axis=1)
        non_zero_genes = gene_sums > 0
        # Fix the boolean indexing - use proper DataFrame indexing
        if isinstance(expr_mat, pd.DataFrame):
            expr_mat_filtered = expr_mat.loc[non_zero_genes, :]
            gene_names = expr_mat_filtered.index.tolist()
            expr_mat = expr_mat_filtered.values  # Convert to numpy for row access
        else:
            expr_mat = expr_mat[non_zero_genes, :]
            gene_names = [original_gene_names[i] for i in range(len(original_gene_names)) if non_zero_genes[i]]
    
    ginis = np.array([gini_coefficient(expr_mat[i, :]) for i in range(expr_mat.shape[0])])
    max_expr = np.log2(np.max(expr_mat, axis=1) + 1)
    
    # Use lowess for smoothing (equivalent to loess in R)
    fit = lowess(ginis, max_expr, frac=0.3, return_sorted=False)
    residuals = ginis - fit
    outliers = np.abs(residuals) > np.percentile(np.abs(residuals), 75)
    
    # Fit without outliers
    fit2 = lowess(ginis[~outliers], max_expr[~outliers], frac=0.3, return_sorted=False)
    
    norm_ginis = np.full(len(ginis), np.nan)
    norm_ginis[~outliers] = ginis[~outliers] - fit2
    
    # Impute outliers
    for i in np.where(outliers)[0]:
        distances = np.abs(max_expr - max_expr[i])
        closest_non_outlier = np.where(~outliers)[0][np.argmin(distances[~outliers])]
        imputed = fit2[np.where(~outliers)[0] == closest_non_outlier][0]
        norm_ginis[i] = ginis[i] - imputed
    
    # Calculate p-values
    p = 1 - norm.cdf(norm_ginis, loc=np.mean(norm_ginis), scale=np.std(norm_ginis))
    
    if len(gene_names) != len(p):
        gene_names = [f'gene_{i}' for i in range(len(p))]
    
    return pd.Series(p, index=gene_names).sort_values()

def cor_fs(expr_mat, direction="both", fdr=None):
    """
    Correlation-based feature selection
    """
    # Store original gene names
    original_gene_names = getattr(expr_mat, 'index', [f'gene_{i}' for i in range(expr_mat.shape[0])])
    
    if sparse.issparse(expr_mat):
        expr_mat = expr_mat.toarray()
    elif isinstance(expr_mat, pd.DataFrame):
        expr_mat = expr_mat.values  # Convert to numpy for row access
    
    # Calculate Spearman correlation matrix (like R version)
    from scipy.stats import spearmanr
    n_genes = expr_mat.shape[0]
    
    if fdr is not None:
        # Calculate correlation with p-values
        cor_mat = np.zeros((n_genes, n_genes))
        p_mat = np.zeros((n_genes, n_genes))
        
        for i in range(n_genes):
            for j in range(n_genes):
                if i != j:
                    corr, p_val = spearmanr(expr_mat[i, :], expr_mat[j, :])
                    cor_mat[i, j] = corr
                    p_mat[i, j] = p_val
        
        # Apply FDR correction
        from statsmodels.stats.multitest import multipletests
        p_flat = p_mat.flatten()
        _, p_adj, _, _ = multipletests(p_flat, method='fdr_bh')
        p_adj_mat = p_adj.reshape(p_mat.shape)
        
    else:
        # Just correlation without p-values
        cor_mat = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(n_genes):
                if i != j:
                    corr, _ = spearmanr(expr_mat[i, :], expr_mat[j, :])
                    cor_mat[i, j] = corr
    
    np.fill_diagonal(cor_mat, 0)
    
    if direction == "both":
        score = np.sum([np.abs(np.min(cor_mat, axis=1)), np.abs(np.max(cor_mat, axis=1))], axis=0)
    elif direction == "pos":
        score = np.max(cor_mat, axis=1)
    elif direction == "neg":
        score = np.abs(np.min(cor_mat, axis=1))
    else:
        raise ValueError("Unrecognized direction")
    
    if fdr is not None:
        # Filter by FDR threshold
        sig_genes = np.min(p_adj_mat, axis=1) < fdr
        score = score[sig_genes]
        gene_names = [original_gene_names[i] for i in range(len(original_gene_names)) if sig_genes[i]]
    else:
        gene_names = original_gene_names
    
    score_series = pd.Series(score, index=gene_names)
    return score_series.sort_values(ascending=False)

def Consensus_fs(counts, norm=None, is_spike=None, pcs=[2, 3], include_cors=True):
    """
    Consensus feature selection combining multiple methods
    """
    # Import real M3Drop functions
    from .Extremes import M3DropFeatureSelection
    from .Brennecke_implementation import BrenneckeGetVariableGenes
    
    if is_spike is None:
        is_spike = np.repeat(False, counts.shape[0])
    
    # Handle normalization
    if norm is None:
        if not np.any(is_spike):
            sf = np.array(counts.sum(axis=0)).flatten()
        else:
            sf = np.array(counts[~is_spike, :].sum(axis=0)).flatten()
        norm = counts / sf * np.median(sf)
    
    # Remove invariant genes
    if sparse.issparse(counts):
        row_vars = np.array(counts.sum(axis=1)).flatten()
    else:
        if counts.shape[1] < 1000:
            row_vars = np.var(counts, axis=1)
        else:
            row_vars = np.sum(counts, axis=1)
    
    invariant = row_vars == 0
    # Fix the boolean indexing - use proper DataFrame indexing
    if isinstance(counts, pd.DataFrame):
        counts = counts.loc[~invariant, :]
        norm = norm.loc[~invariant, :]
    else:
        counts = counts[~invariant, :]
        norm = norm[~invariant, :]
    
    # Apply feature selection methods
    try:
        m3drop_result = M3DropFeatureSelection(norm, mt_method="fdr", mt_threshold=2.0, suppress_plot=True)
        m3drop_genes = m3drop_result['Gene'].tolist() if 'Gene' in m3drop_result.columns else m3drop_result.index.tolist()
    except Exception as e:
        print(f"Warning: M3Drop failed, using random fallback: {e}")
        m3drop_genes = np.random.choice(norm.index if hasattr(norm, 'index') else range(norm.shape[0]), 
                                       size=min(1000, norm.shape[0]), replace=False)
    
    # HVG using Brennecke method
    try:
        if np.sum(is_spike) > 10:
            spikes = np.where(is_spike)[0]
            hvg_result = BrenneckeGetVariableGenes(norm, spikes=spikes, fdr=2.0, suppress_plot=True)
        else:
            hvg_result = BrenneckeGetVariableGenes(norm, fdr=2.0, suppress_plot=True)
        hvg_genes = hvg_result['Gene'].tolist() if 'Gene' in hvg_result.columns else hvg_result.index.tolist()
    except Exception as e:
        print(f"Warning: HVG Brennecke failed, using variance fallback: {e}")
        hvg_score = np.var(norm, axis=1) / np.mean(norm, axis=1)
        hvg_genes = hvg_score.sort_values(ascending=False).index.tolist()
    
    # Gini
    gini_result = gini_fs(norm)
    
    # PCA
    pca_result = irlba_pca_fs(norm, pcs=pcs)
    
    # Correlation
    if include_cors:
        cor_result = cor_fs(norm)
    else:
        cor_result = pd.Series(np.repeat(-1, norm.shape[0]), 
                              index=getattr(norm, 'index', range(norm.shape[0])))
    
    # Create output table
    gene_names = getattr(counts, 'index', [f'gene_{i}' for i in range(counts.shape[0])])
    ranks = np.arange(1, counts.shape[0] + 1)
    
    # Create ranking dictionaries
    m3drop_rank_dict = {gene: i+1 for i, gene in enumerate(m3drop_genes)}
    hvg_rank_dict = {gene: i+1 for i, gene in enumerate(hvg_genes)}
    
    out_table = pd.DataFrame({
        'M3Drop': [m3drop_rank_dict.get(gene, counts.shape[0]) for gene in gene_names],
        'HVG': [hvg_rank_dict.get(gene, counts.shape[0]) for gene in gene_names],
        'PCA': [np.where(pca_result.index == gene)[0][0] + 1 if gene in pca_result.index else counts.shape[0] 
                for gene in gene_names],
        'Cor': [np.where(cor_result.index == gene)[0][0] + 1 if gene in cor_result.index else counts.shape[0] 
                for gene in gene_names] if include_cors else np.repeat(-1, counts.shape[0]),
        'Gini': [np.where(gini_result.index == gene)[0][0] + 1 if gene in gini_result.index else counts.shape[0] 
                 for gene in gene_names]
    }, index=gene_names)
    
    if not include_cors:
        out_table['Cor'] = -1
    
    # Calculate consensus score
    consensus_score = out_table.mean(axis=1)
    out_table = out_table.loc[consensus_score.sort_values().index]
    out_table['Cons'] = np.arange(1, len(out_table) + 1)
    
    return out_table
