import numpy as np
from scipy.stats import norm


def bs(S, K, T, sigma, r, type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) \
        if type == 'call' else (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
