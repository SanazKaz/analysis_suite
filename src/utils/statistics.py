"""Statistical functions for molecular analysis."""

import numpy as np


def cliffs_delta(x, y) -> float:
    """
    Calculate Cliff's delta effect size: P(X>Y) - P(X<Y).
    
    Args:
        x: First array of values
        y: Second array of values
        
    Returns:
        Cliff's delta value (-1 to 1)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    
    # Use full computation for smaller arrays, sampling for larger
    if nx * ny <= 100000:
        diff = x[:, None] - y[None, :]
        return (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)
    
    rng = np.random.default_rng(12345)
    xs = rng.choice(x, size=min(nx, 500), replace=False)
    ys = rng.choice(y, size=min(ny, 500), replace=False)
    diff = xs[:, None] - ys[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (xs.size * ys.size)


def pvalue_to_asterisks(pvalue: float) -> str:
    """
    Convert a p-value to a standard significance string.
    
    Args:
        pvalue: P-value from statistical test
        
    Returns:
        String representation of significance level
    """
    if pvalue <= 0.0001:
        return '****'
    elif pvalue <= 0.001:
        return '***'
    elif pvalue <= 0.01:
        return '**'
    elif pvalue <= 0.05:
        return '*'
    else:
        return 'ns'