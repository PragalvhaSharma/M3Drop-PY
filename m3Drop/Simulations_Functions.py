import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import nbinom, gamma, uniform
import warnings

### Functions ###

def hidden_add_dropouts(x, mu, K):
    """Add dropouts to expression data"""
    x = np.array(x)
    p_drop = 1 - mu / (mu + K)
    expect_pos = mu * len(x) / np.sum(x > 0)
    p_drop = K / expect_pos
    # p_new_drop = (p_drop * len(x) - np.sum(x == 0)) / np.sum(x > 0)
    toss = np.random.uniform(size=len(x))
    x[toss < p_drop] = 0
    return x

def hidden_amplification1(x, rounds=12, efficiency=0.97):
    """Simulate PCR amplification"""
    x = np.array(x)
    tot = np.sum(x)
    if tot == 0:
        return x  # Nothing to be done if all zero already
    
    # Amplify
    for i in range(rounds):
        x = np.array([y + np.random.binomial(y, efficiency) for y in x])
    
    amped = np.sum(x)
    # Downsample back to starting amounts
    x = np.array([np.random.binomial(z, 1/((1+efficiency)**rounds)) for z in x])
    return x

def bg__default_mean2disp(mu, coeffs=np.array([3.967816, -1.855054])):
    """Calculate dispersion from mean"""
    cv2 = np.exp(coeffs[0] + coeffs[1] * (np.log(mu) / np.log(10)))
    variance = cv2 * (mu**2)
    disp = mu**2 / (variance - mu)
    return 1 / disp

def bg__MakeSimData(dispersion_fun=bg__default_mean2disp, n_cells=300, dispersion_factor=1, 
                   base_means=None, K=10.3):
    """Make Simulated Matrix"""
    if base_means is None:
        base_means = 10**np.random.normal(1, 1, 25000)
    
    n_genes = len(base_means)
    expr_mat = np.zeros((n_genes, n_cells))
    
    for x in range(n_genes):
        size = 1 / (dispersion_factor * dispersion_fun(base_means[x]))
        base = np.random.negative_binomial(size, size/(size + base_means[x]), n_cells)
        if K is not None:
            base = hidden_add_dropouts(base, base_means[x], K)
        expr_mat[x, :] = base
    
    return expr_mat

def bg__MakeSimDE(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, 
                 n_cells=300, sub_pop=0.5, dispersion_factor=1, base_means=None, K=10.3):
    """Make simulated differential expression data"""
    if base_means is None:
        base_means = 10**np.random.normal(1, 1, 25000)
    
    n_genes = len(base_means)
    TP = np.random.choice(n_genes, int(frac_change * n_genes), replace=False)
    sub_pop_size = int(sub_pop * n_cells)
    Pop_lab = np.concatenate([np.ones(n_cells - sub_pop_size), np.full(sub_pop_size, 2)])
    
    # Base Population
    base = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                          n_cells=int(np.sum(Pop_lab == 1)), 
                          dispersion_factor=dispersion_factor, 
                          base_means=base_means, K=K)
    
    changed_means = base_means.copy()
    changed_means[TP] = base_means[TP] * fold_change
    
    # Changed Subpopulation
    sub_pop_data = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                                  n_cells=int(np.sum(Pop_lab == 2)), 
                                  dispersion_factor=dispersion_factor, 
                                  base_means=changed_means, K=K)
    
    data = np.concatenate([base, sub_pop_data], axis=1)
    return {'data': data, 'cell_labels': Pop_lab, 'TP': TP}

def bg__MakeSimDVar(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, 
                   n_cells=300, sub_pop=0.5, dispersion_factor=1, base_means=None, K=10.3):
    """Make simulated differential variance data"""
    if base_means is None:
        base_means = 10**np.random.normal(1, 1, 25000)
    
    n_genes = len(base_means)
    TP = np.random.choice(n_genes, int(frac_change * n_genes), replace=False)
    sub_pop_size = int(sub_pop * n_cells)
    Pop_lab = np.concatenate([np.ones(n_cells - sub_pop_size), np.full(sub_pop_size, 2)])
    
    # Whole Population
    base = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                          n_cells=n_cells, 
                          dispersion_factor=dispersion_factor, 
                          base_means=base_means, K=K)
    
    # Changed Vals
    subpop = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                            n_cells=int(np.sum(Pop_lab == 2)), 
                            dispersion_factor=fold_change * dispersion_factor, 
                            base_means=base_means[TP], K=K)
    
    base[np.ix_(TP, Pop_lab == 2)] = subpop
    return {'data': base, 'cell_labels': Pop_lab, 'TP': TP}

def bg__MakeSimHVar(dispersion_fun=bg__default_mean2disp, fold_change=10, frac_change=0.1, 
                   n_cells=300, dispersion_factor=1, base_means=None, K=10.3):
    """Make simulated high variance data"""
    if base_means is None:
        base_means = 10**np.random.normal(1, 1, 25000)
    
    n_genes = len(base_means)
    TP = np.random.choice(n_genes, int(frac_change * n_genes), replace=False)
    Pop_lab = np.ones(n_cells)
    
    # Whole Population
    base = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                          n_cells=n_cells, 
                          dispersion_factor=dispersion_factor, 
                          base_means=base_means, K=K)
    
    # Changed Vals
    subpop = bg__MakeSimData(dispersion_fun=dispersion_fun, 
                            n_cells=n_cells, 
                            dispersion_factor=fold_change * dispersion_factor, 
                            base_means=base_means[TP], K=K)
    
    base[TP, :] = subpop
    return {'data': base, 'cell_labels': Pop_lab, 'TP': TP}

def bg__get_stats(sig, TP, ngenes):
    """Calculate FDR and FNR statistics"""
    sig = np.array(sig)
    TP = np.array(TP)
    
    TPs = np.sum(np.isin(sig, TP))
    FPs = len(sig) - TPs
    FNs = len(TP) - TPs
    TNs = ngenes - len(TP) - FPs
    
    if (TPs + FPs) == 0:
        FDR = 0
    else:
        FDR = FPs / (TPs + FPs)
    
    if (FNs + TPs) == 0:
        FNR = 0
    else:
        FNR = FNs / (FNs + TPs)
    
    return np.array([FDR, FNR])

def hidden__calc_DE_stats_simplified_singular(expr_mat, TP, DE):
    """Calculate DE stats for a single method"""
    qvals = DE['q.value'].values if hasattr(DE['q.value'], 'values') else DE['q.value']
    if not np.issubdtype(qvals.dtype, np.number):
        qvals = qvals.astype(float)
    
    pvals = DE['p.value'].values if hasattr(DE['p.value'], 'values') else DE['p.value']
    if not np.issubdtype(pvals.dtype, np.number):
        pvals = pvals.astype(float)
    
    sig = DE[qvals < 0.05]
    stats = bg__get_stats(sig.iloc[:, 0], TP, ngenes=expr_mat.shape[0])
    
    # Calculate AUC using sklearn instead of ROCR
    try:
        from sklearn.metrics import roc_auc_score
        DE_truth = np.zeros(len(DE))
        DE_truth[np.isin(DE.iloc[:, 0], TP)] = 1
        val = roc_auc_score(DE_truth, 1 - pvals)
    except ImportError:
        # Fallback if sklearn not available
        val = 0.5
    
    output = np.array([val, stats[0], stats[1]])
    return {'summary': output, 'per_expr': None}

def bg__var_vs_drop(pop_size, fixed_mean, K=10.3, dispersion_from_mean=bg__default_mean2disp, 
                   suppress_plot=True):
    """Analyze variance vs dropout relationship"""
    fc = np.arange(1, 101)
    labels = np.concatenate([np.ones(pop_size), np.full(pop_size, 2)])
    
    def lowmean_fun(fc):
        return 2 * fixed_mean / (1 + fc)
    
    def test_func(f):
        low_mean = lowmean_fun(f)
        high_mean = low_mean * f
        
        size_low = 1 / dispersion_from_mean(low_mean)
        base = np.random.negative_binomial(size_low, size_low/(size_low + low_mean), pop_size)
        
        size_high = 1 / dispersion_from_mean(high_mean)
        subpop = np.random.negative_binomial(size_high, size_high/(size_high + high_mean), pop_size)
        
        base = hidden_add_dropouts(base, low_mean, K)
        subpop = hidden_add_dropouts(subpop, high_mean * f, K)
        return np.concatenate([base, subpop])
    
    test = np.array([test_func(f) for f in fc]).T
    
    def var_btw_fun(x):
        # Simple between-group variance calculation
        group1 = x[labels == 1]
        group2 = x[labels == 2]
        mean1, mean2 = np.mean(group1), np.mean(group2)
        overall_mean = np.mean(x)
        var_btw = pop_size * (mean1 - overall_mean)**2 + pop_size * (mean2 - overall_mean)**2
        return var_btw / (2 * pop_size - 1)
    
    def var_within_fun(x):
        # Simple within-group variance calculation
        group1 = x[labels == 1]
        group2 = x[labels == 2]
        var_within = (np.sum((group1 - np.mean(group1))**2) + 
                     np.sum((group2 - np.mean(group2))**2))
        return var_within / (2 * pop_size - 1)
    
    Vbtw = np.array([var_btw_fun(test[:, i]) for i in range(test.shape[1])])
    Vwithin = np.array([var_within_fun(test[:, i]) for i in range(test.shape[1])])
    vars = np.var(test, axis=0)
    drops = np.sum(test == 0, axis=0) / test.shape[0]
    
    if not suppress_plot:
        import matplotlib.pyplot as plt
        
        # Variance plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fc, vars, 'k-', label='Total')
        plt.plot(fc, Vbtw, 'b-', label='Between')
        plt.plot(fc, Vwithin, 'r-', label='Within')
        plt.xlabel('Fold Change')
        plt.ylabel('Variance')
        plt.title(f'mu = {fixed_mean}, n = {2*pop_size}')
        plt.legend()
        
        # Dropout plot
        plt.subplot(1, 2, 2)
        plt.plot(fc, drops, 'ko-')
        plt.xlabel('Fold Change')
        plt.ylabel('Dropout Rate')
        plt.title(f'mu = {fixed_mean}, n = {2*pop_size}')
        plt.tight_layout()
        plt.show()
    
    var_r = np.corrcoef(vars, fc)[0, 1]
    drop_r = np.corrcoef(drops, fc)[0, 1]
    
    return {
        'var_r': var_r, 
        'drop_r': drop_r, 
        'vars': vars, 
        'drops': drops, 
        'fc': fc, 
        'Vbtw': Vbtw, 
        'Vwithin': Vwithin
    }

def bg__fit_gamma(x):
    """Fit gamma distribution parameters"""
    x = np.array(x)
    s = np.var(x) / np.mean(x)
    a = np.mean(x) / s
    return {'shape': a, 'scale': s}

def bg__shift_size(mu_all, size_all, mu_group, coeffs):
    """Shift size parameter"""
    b = np.log(size_all) - coeffs[1] * np.log(mu_all)
    size_group = np.exp(coeffs[1] * np.log(mu_group) + b)
    return size_group

def NBumiSimulationTrifecta(original_data, n_genes=25000, n_cells=250, sub_pop_prop=0.5):
    """NBumi simulation trifecta - requires M3Drop functions that aren't implemented here"""
    # This function requires NBumiFitModel and NBumiFitDispVsMean which are not in this file
    # Would need to be implemented separately or imported from M3Drop package
    raise NotImplementedError("This function requires additional M3Drop functions not available in this file")

def M3DropSimulationTrifecta(original_data, n_genes=25000, n_cells=250, sub_pop_prop=0.5):
    """M3Drop simulation trifecta - requires M3Drop functions that aren't implemented here"""
    # This function requires bg__calc_variables, bg__fit_MM, bg__get_mean2disp which are not in this file
    # Would need to be implemented separately or imported from M3Drop package
    raise NotImplementedError("This function requires additional M3Drop functions not available in this file")

def Make_Sim(model_params, n_genes=25000, pop_params=None, dV_params=None, dE_params=None):
    """
    Make simulation with specified parameters
    
    model_params should include:
        slope & intercept of log-log relationship between mean & variance
        mean & sd of distribution of mean expression per gene
        shape & scale of gamma distribution of total counts per cell (in 100,000s reads)
        K of MM dropouts
    
    Parameters for each cell population:
        number of cells
        relative size of the cells (affects total counts per cell)
        heterogeneity of the population (multiply FC in variance) - setting to 0 = no change in variance
        distinctness of the population (multiply FC in mean expression) - setting to 0 = no change in mean expression
    
    The "reference" population is never seen, (unless hetero=0, distinct=0, size=1)
    DE & DV are added to every population therefore actual distribution of FC is 
    sum of distribution of FC between two populations, therefore sd=1 is actually sd=2
    
    Does not support batch effects -> would require one population to have same dV/dE as another 
    but with an additional effect on top.
    """
    if pop_params is None:
        pop_params = pd.DataFrame({
            'cells': [125, 125],
            'size': [1, 1],
            'hetero': [0, 0],
            'distinct': [0, 0]
        })
    
    if dV_params is None:
        dV_params = {'mu': 0, 'sd': 1}
    
    if dE_params is None:
        dE_params = {'mu': 0, 'sd': 1}
    
    # Implementation would depend on the specific model_params structure
    raise NotImplementedError("Full implementation requires specification of model_params structure")
