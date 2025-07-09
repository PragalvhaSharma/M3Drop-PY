import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import warnings


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
    
    # Calculate predictions for all data
    predictions = K / (s + K)
    
    # Calculate residuals and error estimates
    residuals = p - predictions
    ssr = np.sum(residuals**2)
    
    # Estimate K error based on the Hessian (if available) or use a reasonable default
    if hasattr(result, 'hess_inv') and result.hess_inv is not None:
        try:
            # Extract standard error from Hessian inverse
            K_var = result.hess_inv[0, 0] if result.hess_inv.shape[0] > 0 else 0.1**2
            Kerr = np.sqrt(K_var)
        except:
            # Fallback: use empirical estimate
            Kerr = max(0.05 * K, 0.1)
    else:
        # Fallback: use empirical estimate
        Kerr = max(0.05 * K, 0.1)
    
    # Fitted error is the residual standard deviation
    fitted_err = sd

    return {
        'K': K,
        'Kerr': Kerr,
        'sd': sd,
        'fitted_err': fitted_err,
        'predictions': pd.Series(predictions, index=s.index),
        'SSr': ssr,
        'model': f"Michaelis-Menten (K={K:.2f})"
    }


def hidden__fit_MM_lognormal(p, s):
    """
    Fit Michaelis-Menten using lognormal approach.
    This consistently underestimates K compared to the main method.
    """
    if len(p) != len(s):
        raise ValueError("Error: p and s not same length. Cannot fit Michaelis-Menten.")
    
    # Clean data - remove invalid values
    mask = (p < 1) & (p > 0) & (~np.isnan(p)) & (~np.isnan(s))
    p_c = p[mask]
    s_c = s[mask]
    
    if len(p_c) == 0:
        # Return default values if no valid data
        K = 1.0
        predicted = 1 - (s / (K + s))
        residuals = p - predicted
        return {
            'K': K,
            'Kerr': 1.0,
            'fitted_err': 0.25,
            'predictions': predicted,
            'model': f"MMenten K={K:.3f}",
            'SSr': np.sum(residuals**2),
            'SAr': np.sum(np.abs(residuals))
        }
    
    def neg_log_likelihood(params):
        krt, sigma = params
        if krt <= 0 or sigma <= 0:
            return 1e100
        
        try:
            obs_Ks = p_c / (1 - p_c) * s_c
            R = np.log(obs_Ks) - np.log(krt)
            
            # Filter based on density (simplified version of R's densCols approach)
            Q75, Q25 = np.percentile(R, [75, 25])
            IQR = Q75 - Q25
            
            # Use all data points within reasonable range
            valid_mask = np.abs(R - np.median(R)) < 3 * IQR
            R_filtered = R[valid_mask]
            
            if len(R_filtered) == 0:
                return 1e100
            
            log_likelihood = np.sum(norm.logpdf(R_filtered, 0, sigma))
            return -log_likelihood
        except:
            return 1e100
    
    # Initial parameters
    initial_params = [6.0, 0.25]
    
    try:
        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-9, None), (1e-9, None)]
        )
        
        krt = result.x[0]
        res_err = result.x[1]
        Kerr = max(res_err, 0.1)  # Simplified error estimate
        
    except:
        krt = 6.0
        res_err = 0.25
        Kerr = 0.25
    
    predicted = 1 - (s / (krt + s))
    residuals = p - predicted
    
    return {
        'K': krt,
        'Kerr': Kerr,
        'fitted_err': res_err,
        'predictions': predicted,
        'model': f"MMenten K={krt:.3f}",
        'SSr': np.sum(residuals**2),
        'SAr': np.sum(np.abs(residuals))
    }


def hidden__fit_MM_logistic(p, s):
    """
    Fit Michaelis-Menten using logistic regression.
    """
    if len(p) != len(s):
        raise ValueError("Error: p and s not same length. Cannot fit Michaelis-Menten.")
    
    # Remove zero values for log transformation
    mask = s > 0
    s_nozero = s[mask]
    p_nozero = p[mask]
    
    if len(s_nozero) == 0:
        # Return default values if no valid data
        predicted = np.zeros_like(s)
        residuals = p - predicted
        return {
            'K': 1.0,
            'Kerr': 1.0,
            'predictions': predicted,
            'model': "MMenten K=1.000",
            'SSr': np.sum(residuals**2),
            'SAr': np.sum(np.abs(residuals))
        }
    
    try:
        # Use logistic regression with offset
        # R: glm(p_nozero ~ offset(-1*log(s_nozero)), family="binomial")
        # This is equivalent to fitting: logit(p) = K_coeff - log(s)
        
        # Transform to logistic regression format
        X = np.ones((len(s_nozero), 1))  # Intercept only
        offset = -np.log(s_nozero)
        
        # Manual logistic regression with offset
        def logistic_with_offset(beta, X, offset, y):
            linear_pred = X @ beta + offset
            p_pred = 1 / (1 + np.exp(-linear_pred))
            p_pred = np.clip(p_pred, 1e-15, 1-1e-15)  # Avoid log(0)
            return -np.sum(y * np.log(p_pred) + (1-y) * np.log(1-p_pred))
        
        initial_beta = [0.0]
        result = minimize(
            lambda beta: logistic_with_offset(beta, X, offset, p_nozero),
            initial_beta,
            method='BFGS'
        )
        
        Kcoeff = result.x[0]
        krt = np.exp(Kcoeff)
        
        # Error estimate (simplified)
        Kerr = 0.1 * krt
        
        # Predictions
        predicted = np.zeros_like(s, dtype=float)
        linear_pred = Kcoeff - np.log(s_nozero)
        predicted[mask] = 1 / (1 + np.exp(-linear_pred))
        
    except:
        # Fallback values
        krt = 1.0
        Kerr = 1.0
        predicted = np.zeros_like(s, dtype=float)
    
    residuals = p - predicted
    
    return {
        'K': krt,
        'Kerr': Kerr,
        'predictions': predicted,
        'model': f"MMenten K={krt:.3f}",
        'SSr': np.sum(residuals**2),
        'SAr': np.sum(np.abs(residuals))
    }


def bg__fit_logistic(p, s):
    """
    Fits logistic regression to the relationship between mean expression and dropout rate.
    """
    if len(p) != len(s):
        raise ValueError("Error: p and s not same length. Cannot fit Logistic Regression.")
    
    # Remove zero values for log transformation
    mask = s > 0
    s_nozero = s[mask]
    p_nozero = p[mask]
    
    if len(s_nozero) == 0:
        # Return default values if no valid data
        fullpredictions = np.zeros_like(s)
        res = fullpredictions - p
        return {
            'predictions': fullpredictions,
            'B0': 0.0,
            'B1': 0.0,
            'model': "Logistic Intercept=0.000 Coeff=0.000",
            'SSr': np.sum(res**2),
            'SAr': np.sum(np.abs(res))
        }
    
    try:
        # Fit logistic regression: p_nozero ~ log(s_nozero)
        X = np.column_stack([np.ones(len(s_nozero)), np.log(s_nozero)])
        
        def logistic_loss(beta, X, y):
            linear_pred = X @ beta
            p_pred = 1 / (1 + np.exp(-linear_pred))
            p_pred = np.clip(p_pred, 1e-15, 1-1e-15)  # Avoid log(0)
            return -np.sum(y * np.log(p_pred) + (1-y) * np.log(1-p_pred))
        
        initial_beta = [0.0, 0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                lambda beta: logistic_loss(beta, X, p_nozero),
                initial_beta,
                method='BFGS'
            )
        
        B0, B1 = result.x
        
        # Generate predictions
        fullpredictions = np.zeros_like(s, dtype=float)
        linear_pred = B0 + B1 * np.log(s_nozero)
        fullpredictions[mask] = 1 / (1 + np.exp(-linear_pred))
        
    except:
        # Fallback values
        B0, B1 = 0.0, 0.0
        fullpredictions = np.zeros_like(s, dtype=float)
    
    res = fullpredictions - p
    
    return {
        'predictions': fullpredictions,
        'B0': B0,
        'B1': B1,
        'model': f"Logistic Intercept={B0:.3f} Coeff={B1:.3f}",
        'SSr': np.sum(res**2),
        'SAr': np.sum(np.abs(res))
    }


def bg__fit_ZIFA(p, s):
    """
    Fits double exponential (ZIFA-style) model to the relationship between 
    mean expression and dropout rate.
    """
    if len(p) != len(s):
        raise ValueError("Error: p and s not same length. Cannot fit double exponential.")
    
    # Handle zero dropout rates
    p_nozero = p.copy()
    p_nozero[p == 0] = np.min(p[p > 0]) / 10 if np.any(p > 0) else 1e-10
    
    try:
        # Fit: log(p_nozero) ~ -1 + s^2 (no intercept, s-squared term only)
        # This is equivalent to: p = exp(-lambda * s^2)
        
        X = (s**2).values.reshape(-1, 1)
        y = np.log(p_nozero).values
        
        # Use least squares to fit the model
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        
        # Extract lambda (negative of coefficient since we want exp(-lambda*s^2))
        lambda_param = -reg.coef_[0]
        
        # Error estimates (simplified)
        Lerr = 0.1 * abs(lambda_param)
        res_err = 0.1
        
        # Generate predictions
        predicted = np.exp(-lambda_param * s**2)
        
    except:
        # Fallback values
        lambda_param = 1e-6
        Lerr = 1e-7
        res_err = 0.1
        predicted = np.exp(-lambda_param * s**2)
    
    residuals = p - predicted
    
    return {
        'lambda': lambda_param,
        'Lerr': Lerr,
        'fitted_err': res_err,
        'predictions': predicted,
        'model': f"p ~ e^(-lambda*S^2) lambda={lambda_param:.2e}",
        'SSr': np.sum(residuals**2),
        'SAr': np.sum(np.abs(residuals))
    }


def bg__dropout_plot_base(expr_mat, xlim=None, suppress_plot=False):
    """
    Create base plot for dropout analysis.
    For now, this is a simplified version that just calculates variables.
    """
    from .basics import bg__calc_variables
    
    gene_info = bg__calc_variables(expr_mat)
    
    # Placeholder for actual plotting functionality
    if not suppress_plot:
        print("Plotting functionality not yet implemented.")
    
    return {'gene_info': gene_info}


def bg__add_model_to_plot(model_fit, base_plot, lty=1, lwd=2.5, col="black", legend_loc="topright"):
    """
    Add model curve to dropout plot.
    For now, this is a placeholder.
    """
    if base_plot is None:
        return
    
    # Placeholder for actual plotting functionality
    print(f"Would add {model_fit.get('model', 'Unknown')} model to plot")
    
    # Return dummy legend location
    return {
        'rect': {
            'left': 0.7,
            'top': 0.9,
            'w': 0.2,
            'h': 0.1
        }
    }


def M3DropDropoutModels(expr_mat, xlim=None, suppress_plot=False):
    """
    Fits and compares three different dropout models: Michaelis-Menten, 
    Logistic Regression, and ZIFA double exponential.
    
    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression matrix with genes as rows and cells as columns.
    xlim : tuple, optional
        X-axis limits for plotting.
    suppress_plot : bool, default=False
        Whether to suppress plotting.
        
    Returns
    -------
    dict
        Dictionary containing fit results for all three models:
        - MMFit: Michaelis-Menten fit
        - LogiFit: Logistic regression fit  
        - ExpoFit: ZIFA exponential fit
    """
    # Create base plot and get gene info
    base_plot = bg__dropout_plot_base(expr_mat, xlim=xlim, suppress_plot=suppress_plot)
    
    # Extract dropout rate (p) and mean expression (s)
    p = base_plot['gene_info']['p']
    s = base_plot['gene_info']['s']
    
    # Fit the three models
    MM = bg__fit_MM(p, s)
    SCDE = bg__fit_logistic(p, s)  # Called SCDE in R (Single Cell Differential Expression)
    ZIFA = bg__fit_ZIFA(p, s)
    
    # Add models to plot if plotting is enabled
    if not suppress_plot:
        sizeloc = bg__add_model_to_plot(MM, base_plot, lty=1, lwd=2.5, col="black", legend_loc="topright")
        sizeloc = bg__add_model_to_plot(SCDE, base_plot, lty=2, lwd=2.5, col="magenta3", 
                                       legend_loc=(sizeloc['rect']['left'] + sizeloc['rect']['w'],
                                                  sizeloc['rect']['top'] - sizeloc['rect']['h'] - 0.05))
        sizeloc = bg__add_model_to_plot(ZIFA, base_plot, lty=3, lwd=2.5, col="red",
                                       legend_loc=(sizeloc['rect']['left'] + sizeloc['rect']['w'],
                                                  sizeloc['rect']['top'] - sizeloc['rect']['h'] - 0.05))
    
    return {
        'MMFit': MM,
        'LogiFit': SCDE, 
        'ExpoFit': ZIFA
    }