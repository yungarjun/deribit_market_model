import numpy as np
from numpy import sqrt, log
from scipy.special import erf

def _phi(x):  # standard normal pdf
    return np.exp(-0.5*x*x) / np.sqrt(2*np.pi)

def _Phi(x):  # standard normal cdf
    return 0.5*(1.0 + erf(x/np.sqrt(2)))

def bs_call_norm(m, tau, sigma):
    """ Black call normalized: c = C/F, inputs: m=ln(K/F), tau in years, sigma >= 0 """
    m, tau, sigma = np.asarray(m), np.asarray(tau), np.asarray(sigma)
    srt = np.maximum(np.sqrt(np.maximum(tau, 0.0)), 1e-12)
    sigsrt = np.maximum(sigma*srt, 1e-12)
    d1 = (-m)/sigsrt + 0.5*sigsrt
    d2 = d1 - sigsrt
    return _Phi(d1) - np.exp(m)*_Phi(d2)

def bs_vega_norm(m, tau, sigma):
    """ Normalized vega: ∂c/∂σ = φ(d1) * sqrt(tau) """
    m, tau, sigma = np.asarray(m), np.asarray(tau), np.asarray(sigma)
    srt = np.maximum(np.sqrt(np.maximum(tau, 0.0)), 1e-12)
    sigsrt = np.maximum(sigma*srt, 1e-12)
    d1 = (-m)/sigsrt + 0.5*sigsrt
    return _phi(d1) * srt

def implied_vol_from_c_norm(c, m, tau, lo=1e-6, hi=5.0, tol=1e-8, maxit=100):
    """ Brent-like bisection for IV from normalized call price. Vectorized. """
    c = np.asarray(c); m = np.asarray(m); tau = np.asarray(tau)
    lo_arr = np.full_like(c, lo, dtype=float)
    hi_arr = np.full_like(c, hi, dtype=float)
    clo = bs_call_norm(m, tau, lo_arr)
    chi = bs_call_norm(m, tau, hi_arr)
    # clamp target into [clo, chi] to avoid NaNs on degenerate quotes
    target = np.clip(c, clo + 1e-12, chi - 1e-12)
    for _ in range(maxit):
        mid = 0.5*(lo_arr + hi_arr)
        cmid = bs_call_norm(m, tau, mid)
        go_lo = cmid < target
        lo_arr = np.where(go_lo, mid, lo_arr)
        hi_arr = np.where(go_lo, hi_arr, mid)
        if np.max(hi_arr - lo_arr) < tol:
            break
    return 0.5*(lo_arr + hi_arr)