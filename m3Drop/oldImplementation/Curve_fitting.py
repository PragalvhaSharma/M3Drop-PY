import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize


def bg__fit_MM(p, s):
    """
    Fits the modified Michaelis-Menten equation to the relationship between
    mean expression and dropout-rate.
    """
    s_clean = s[~p.isna() & ~s.isna()]
    p_clean = p[~p.isna() & ~s.isna()]

    def neg_log_likelihood(params):
        K, sd = params
        if K <= 0 or sd <= 0:
            return np.inf

        predictions = K / (s_clean + K)
        log_likelihood = np.sum(norm.logpdf(p_clean, loc=predictions, scale=sd))
        return -log_likelihood

    initial_params = [np.median(s_clean), 0.1]

    result = minimize(
        neg_log_likelihood,
        initial_params,
        method='L-BFGS-B',
        bounds=[(1e-9, None), (1e-9, None)]
    )

    K, sd = result.x

    predictions = K / (s + K)
    ssr = np.sum((p - predictions)**2)

    return {
        'K': K,
        'sd': sd,
        'predictions': pd.Series(predictions, index=s.index),
        'SSr': ssr,
        'model': f"Michaelis-Menten (K={K:.2f})"
    }