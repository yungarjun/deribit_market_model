import numpy as np
import scipy.sparse as sp
from utils.black_scholes import _phi, _Phi, bs_call_norm, bs_vega_norm, implied_vol_from_c_norm


def mape(C_true, C_hat, eps=1e-8):
    C_true = np.asarray(C_true, float); C_hat = np.asarray(C_hat, float)
    return np.mean(np.abs(C_true - C_hat) / (np.abs(C_true) + eps))

def psas(C_hat, A, b, tol=1e-3):
    A_ = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b_ = np.asarray(b, float).reshape(-1)
    lhs = A_ @ np.asarray(C_hat, float).T
    ok = (lhs >= (b_[:, None] - tol)).all(axis=0)
    return 1.0 - ok.mean()

def pda_from_pca(Z, pca_dyn):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    total_var = Zc.var(axis=0, ddof=1).sum()
    explained = pca_dyn.explained_variance_.sum()
    return 1.0 - (explained / total_var)

def vega_weighted_mape(C_true, C_hat, nodes, tau_vec=None, power=1.0, eps=1e-8,
                       clip_pct=(1, 99), normalize='global'):
    """
    C_true, C_hat: arrays shape (T, N) of normalized calls (C/F)
    nodes: array shape (N, 2) with columns [tau, m] for each column
    power: weight = vega**power (1.0 is standard; try 0.5 to temper ATM dominance)
    clip_pct: winsorize vegas to de-noise tails
    normalize: 'global' (sum weights = 1 over all T,N) or 'per_time' (weights sum to 1 per row)
    """
    C_true = np.asarray(C_true, float); C_hat = np.asarray(C_hat, float)
    assert C_true.shape == C_hat.shape
    T, N = C_true.shape
    tau = nodes[:, 0]; m = nodes[:, 1]

    # Build per-node (τ,m) arrays
    tauN = np.broadcast_to(tau, (T, N))
    mN   = np.broadcast_to(m,   (T, N))

    # IV per (t,node) from true prices (safer for weights)
    sigma_iv = implied_vol_from_c_norm(C_true, mN, tauN)
    vega = bs_vega_norm(mN, tauN, sigma_iv)

    # Winsorise to avoid exploding ATM short-maturity weights
    lo, hi = np.percentile(vega[~np.isnan(vega)], clip_pct)
    vega = np.clip(vega, lo, hi)

    W = np.power(np.maximum(vega, eps), power)

    if normalize == 'per_time':
        W = W / (W.sum(axis=1, keepdims=True) + eps)
    else:
        W = W / (W.sum() + eps)

    err = np.abs(C_true - C_hat) / (np.abs(C_true) + eps)
    return float(np.sum(W * err))