import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import nbinom, chi2
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
import warnings

# Import the required functions from other modules
from .basics import bg__calc_variables
from .Curve_fitting import bg__fit_MM


def hidden_get_K(expr_mat):
    """
    Calculate K parameter using background functions.
    """
    # Calculate gene-specific variables (mean expression, dropout rate, etc.)
    gene_info = bg__calc_variables(expr_mat)
    
    # Fit Michaelis-Menten model to get K parameter
    fit_result = bg__fit_MM(gene_info['p'], gene_info['s'])
    
    return fit_result['K']


def bg__get_mean2disp(expr_mat):
    """
    Calculate mean-to-dispersion relationship function.
    
    Parameters:
    expr_mat : numpy.ndarray
        Expression matrix (genes x cells)
    
    Returns:
    function : mean2disp_fun
        Function that maps mean expression to dispersion
    """
    # Calculate coefficient of variation squared
    row_vars = np.var(expr_mat, axis=1)
    row_means = np.mean(expr_mat, axis=1)
    
    # Avoid division by zero
    row_means_safe = row_means.copy()
    row_means_safe[row_means_safe == 0] = np.nan
    
    cv2 = row_vars / (row_means_safe ** 2)
    
    # Log transform means
    xes = np.log10(row_means)
    
    # Fit linear regression (only for positive means)
    valid_idx = (xes > 0) & (~np.isnan(cv2)) & (~np.isinf(cv2))
    
    if np.sum(valid_idx) < 2:
        raise ValueError("Not enough valid data points for regression")
    
    X = xes[valid_idx].reshape(-1, 1)
    y = np.log(cv2[valid_idx])
    
    reg = LinearRegression().fit(X, y)
    intercept = reg.intercept_
    slope = reg.coef_[0]
    
    def mean2disp_fun(mu):
        if mu <= 0:
            return 1e-10
        cv2_pred = np.exp(intercept + slope * np.log10(mu))
        variance = cv2_pred * (mu ** 2)
        if variance <= mu:
            variance = 1.01 * mu
        disp = mu ** 2 / (variance - mu)
        return 1 / disp
    
    return mean2disp_fun


def bg__fitdispersion(expr_mat):
    """
    Fit dispersion parameters.
    
    Parameters:
    expr_mat : numpy.ndarray
        Expression matrix (genes x cells)
    
    Returns:
    float : slope coefficient from regression
    """
    V = np.var(expr_mat, axis=1)
    mu = np.mean(expr_mat, axis=1)
    xes = np.log10(mu)
    
    # Adjust variance to be > mean
    V[V <= mu] = mu[V <= mu] + 1e-10
    nb_size = mu ** 2 / (V - mu)
    
    # Fit regression for positive means
    valid_idx = (xes > 0) & (~np.isnan(nb_size)) & (~np.isinf(nb_size)) & (nb_size > 0)
    
    if np.sum(valid_idx) < 2:
        raise ValueError("Not enough valid data points for regression")
    
    X = xes[valid_idx].reshape(-1, 1)
    y = np.log(nb_size[valid_idx])
    
    reg = LinearRegression().fit(X, y)
    return reg.coef_[0]  # Return slope coefficient


def hidden__cv2coeffs(expr_mat):
    """
    Calculate CV2 coefficients.
    
    Parameters:
    expr_mat : numpy.ndarray
        Expression matrix (genes x cells)
    
    Returns:
    list : [intercept, slope] coefficients
    """
    row_vars = np.var(expr_mat, axis=1)
    row_means = np.mean(expr_mat, axis=1)
    
    # Avoid division by zero
    row_means_safe = row_means.copy()
    row_means_safe[row_means_safe == 0] = np.nan
    
    cv2 = row_vars / (row_means_safe ** 2)
    xes = np.log10(row_means)
    
    # Fit regression for positive means
    valid_idx = (xes > 0) & (~np.isnan(cv2)) & (~np.isinf(cv2))
    
    if np.sum(valid_idx) < 2:
        raise ValueError("Not enough valid data points for regression")
    
    X = xes[valid_idx].reshape(-1, 1)
    y = np.log(cv2[valid_idx])
    
    reg = LinearRegression().fit(X, y)
    return [reg.intercept_, reg.coef_[0]]


def hidden_calc_p(obs, mu, K, disp):
    """
    Calculate probability using negative binomial distribution.
    
    Parameters:
    obs : float
        Observed count
    mu : float
        Mean parameter
    K : float
        K parameter
    disp : float
        Dispersion parameter
    
    Returns:
    float : probability
    """
    if mu == 0 and obs != 0:
        raise ValueError("Error: non-zero obs has zero mean")
    
    if obs == 0:
        p = 1 - mu / (mu + K)
    else:
        # For negative binomial: n = 1/disp, p = n/(n+mu)
        n = 1 / disp
        prob_param = n / (n + mu)
        p = nbinom.pmf(int(round(obs)), int(round(n)), prob_param)
        if p < 1e-200:
            p = 1e-200
    
    return p


def unfinished__m3dTraditionalDE(expr_mat, groups, batches=None, fdr=0.05):
    """
    Traditional differential expression analysis.
    
    Parameters:
    expr_mat : numpy.ndarray or pandas.DataFrame
        Expression matrix (genes x cells)
    groups : array-like
        Group assignments for each cell
    batches : array-like, optional
        Batch assignments for each cell (default: all same batch)
    fdr : float
        False discovery rate threshold (default: 0.05)
    
    Returns:
    pandas.DataFrame : Differential expression results
    """
    # Convert to numpy array if needed
    if isinstance(expr_mat, pd.DataFrame):
        gene_names = expr_mat.index
        expr_mat = expr_mat.values
    elif hasattr(expr_mat, 'index'):
        gene_names = expr_mat.index
        expr_mat = np.array(expr_mat)
    else:
        expr_mat = np.array(expr_mat)
        gene_names = [f"Gene_{i}" for i in range(expr_mat.shape[0])]
    
    # Set default batches
    if batches is None:
        batches = np.ones(expr_mat.shape[1])
    
    # Convert to pandas categorical for easier handling
    batches = pd.Categorical(batches)
    groups = pd.Categorical(groups)
    
    # Check input dimensions
    if len(batches) != len(groups) or len(batches) != expr_mat.shape[1]:
        raise ValueError("Error: length of groups and batches must match number of cells (columns of expr_mat)")
    
    # Fit batches
    batch_levels = batches.categories
    Ks = {}
    DispFits = {}
    
    for b in batch_levels:
        batch_mask = batches == b
        batch_expr = expr_mat[:, batch_mask]
        
        Ks[b] = hidden_get_K(batch_expr)
        
        # Get dispersion fitting function
        DispFits[b] = bg__get_mean2disp(batch_expr)
    
    # Calculate overall means
    Ms = np.mean(expr_mat, axis=1)
    
    # Calculate group means
    Mis = {}
    for group in groups.categories:
        group_mask = groups == group
        Mis[group] = np.mean(expr_mat[:, group_mask], axis=1)
    
    # Main analysis loop
    AllOut = []
    
    for g in range(expr_mat.shape[0]):  # For each gene
        probs = []
        
        for i in range(expr_mat.shape[1]):  # For each cell
            obs = expr_mat[g, i]
            M = Ms[g]
            b = batches[i]
            group = groups[i]
            Mi = Mis[group][g]
            
            p1 = hidden_calc_p(round(obs), M, Ks[b], DispFits[b](M))
            p2 = hidden_calc_p(round(obs), Mi, Ks[b], DispFits[b](Mi))
            probs.append([p1, p2])
        
        probs = np.array(probs)
        
        # Calculate likelihood ratio test statistic
        D = -2 * (np.sum(np.log(probs[:, 0])) - np.sum(np.log(probs[:, 1])))
        df = len(groups.categories) - 1
        pval = chi2.sf(D, df)
        
        # Calculate group and batch means for this gene
        group_means = [np.mean(expr_mat[g, groups == group]) for group in groups.categories]
        batch_means = [np.mean(expr_mat[g, batches == batch]) for batch in batch_levels]
        
        output = group_means + batch_means + [pval]
        AllOut.append(output)
    
    # Convert to DataFrame
    AllOut = np.array(AllOut)
    
    # FDR correction
    pvals = AllOut[:, -1]
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    
    # Combine results
    result = np.column_stack([AllOut, qvals])
    
    # Create column names
    col_names = list(groups.categories) + list(batch_levels) + ['p.value', 'q.value']
    
    # Create DataFrame with gene names
    result_df = pd.DataFrame(result, columns=col_names, index=gene_names)
    
    # Filter by FDR
    significant_mask = qvals < fdr
    
    return result_df[significant_mask]


def unfinished__m3dTraditionalDEShiftDisp(expr_mat, groups, batches=None, fdr=0.05):
    """
    Traditional DE analysis with shifted dispersion.
    
    Parameters:
    expr_mat : numpy.ndarray or pandas.DataFrame
        Expression matrix (genes x cells)
    groups : array-like
        Group assignments for each cell
    batches : array-like, optional
        Batch assignments for each cell (default: all same batch)
    fdr : float
        False discovery rate threshold (default: 0.05)
    
    Returns:
    pandas.DataFrame : Differential expression results
    """
    # Convert to numpy array if needed
    if isinstance(expr_mat, pd.DataFrame):
        gene_names = expr_mat.index
        expr_mat = expr_mat.values
    elif hasattr(expr_mat, 'index'):
        gene_names = expr_mat.index
        expr_mat = np.array(expr_mat)
    else:
        expr_mat = np.array(expr_mat)
        gene_names = [f"Gene_{i}" for i in range(expr_mat.shape[0])]
    
    # Set default batches
    if batches is None:
        batches = np.ones(expr_mat.shape[1])
    
    # Convert to pandas categorical for easier handling
    batches = pd.Categorical(batches)
    groups = pd.Categorical(groups)
    
    # Check input dimensions
    if len(batches) != len(groups) or len(batches) != expr_mat.shape[1]:
        raise ValueError("Error: length of groups and batches must match number of cells (columns of expr_mat)")
    
    # Fit batches
    batch_levels = batches.categories
    Ks = {}
    DispFits = {}
    
    for b in batch_levels:
        batch_mask = batches == b
        batch_expr = expr_mat[:, batch_mask]
        
        Ks[b] = hidden_get_K(batch_expr)
        
        # Fit mean-variance relationship for each batch
        DispFits[b] = bg__fitdispersion(batch_expr)
    
    # Calculate overall means and variance
    Ms = np.mean(expr_mat, axis=1)
    V = np.var(expr_mat, axis=1)
    V[V <= Ms] = Ms[V <= Ms] + 1e-10
    nb_size = Ms ** 2 / (V - Ms)  # Gene-specific dataset-wide dispersion
    
    # Calculate group means
    Mis = {}
    for group in groups.categories:
        group_mask = groups == group
        Mis[group] = np.mean(expr_mat[:, group_mask], axis=1)
    
    # Main analysis loop
    AllOut = []
    
    for g in range(expr_mat.shape[0]):  # For each gene
        probs = []
        
        for i in range(expr_mat.shape[1]):  # For each cell
            obs = expr_mat[g, i]
            M = Ms[g]
            b = batches[i]
            group = groups[i]
            Mi = Mis[group][g]
            
            # Shift dispersion
            slope = DispFits[b]
            disp1 = nb_size[g]
            if disp1 <= 0:
                disp1 = 1e-10
            
            tmp_intercept = np.log(disp1) - slope * np.log(M) if M > 0 else np.log(disp1)
            disp2 = np.exp(slope * np.log(Mi) + tmp_intercept) if Mi > 0 else disp1
            if disp2 <= 0:
                disp2 = 1e-10
            
            # Calculate probabilities
            p1 = hidden_calc_p(round(obs), M, Ks[b], 1/disp1)
            p2 = hidden_calc_p(round(obs), Mi, Ks[b], 1/disp2)
            probs.append([p1, p2])
        
        probs = np.array(probs)
        
        # Calculate likelihood ratio test statistic
        D = -2 * (np.sum(np.log(probs[:, 0])) - np.sum(np.log(probs[:, 1])))
        df = len(groups.categories) - 1
        pval = chi2.sf(D, df)
        
        # Calculate group and batch means for this gene
        group_means = [np.mean(expr_mat[g, groups == group]) for group in groups.categories]
        batch_means = [np.mean(expr_mat[g, batches == batch]) for batch in batch_levels]
        
        output = group_means + batch_means + [pval]
        AllOut.append(output)
    
    # Convert to DataFrame
    AllOut = np.array(AllOut)
    
    # FDR correction
    pvals = AllOut[:, -1]
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
    
    # Combine results
    result = np.column_stack([AllOut, qvals])
    
    # Create column names
    col_names = list(groups.categories) + list(batch_levels) + ['p.value', 'q.value']
    
    # Create DataFrame with gene names
    result_df = pd.DataFrame(result, columns=col_names, index=gene_names)
    
    # Filter by FDR
    significant_mask = qvals < fdr
    
    return result_df[significant_mask]