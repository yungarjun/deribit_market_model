import numpy as np
import scipy.sparse as sp
import pybobyqa
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA


def reconstruct_prices(G0, Xi_dyn, G_dyn, Xi_stat, G_stat, Xi_sa=None, G_sa=None):
    C = (np.asarray(G0, float)[None, :]
         + np.asarray(Xi_dyn, float) @ np.asarray(G_dyn, float)
         + np.asarray(Xi_stat, float) @ np.asarray(G_stat, float))
    if Xi_sa is not None and G_sa is not None:
        C = C + np.asarray(Xi_sa, float) @ np.asarray(G_sa, float)
    return C

def count_violations(C: np.ndarray, A: sp.csr_matrix, b: np.ndarray, tol=1e-6) -> int:
    A_ = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    lhs = A_ @ C.T  # (m, T)
    return int((lhs < (b[:, None] - tol)).sum())

def hinge_penalty(C: np.ndarray, A: sp.csr_matrix, b: np.ndarray) -> float:
    """ Σ_t Σ_i max(0, b_i - (A c_t)_i ) """
    A_ = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    viol = b[:, None] - (A_ @ C.T)   # (m, T)
    return np.maximum(viol, 0.0).sum()

def decode_static_arb_hinge(R_sa_train: np.ndarray,         # residual after dyn+stat removal
                            G0: np.ndarray,
                            Recon_sofar: np.ndarray,         # dyn + stat reconstruction
                            A: sp.csr_matrix, b: np.ndarray, # constraints
                            n_sa=2, n_PC=8,
                            lam_rec=1e-3,
                            lam_hinge=4,
                            maxfun=8000,
                            seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy static‑factor decoding:
      For i=1..n_sa:
        - PCA(R_current) → G_sub (n_PC×N), scores ξ_PC (T×n_PC).
        - Optimize unit vector w∈R^{n_PC} to minimize:
              lam_hinge * Hinge( G0 + Recon_sofar + ξw·(wᵀG_sub) ) + lam_rec * ||R_current - ξw·(wᵀG_sub)||_F
        - Update residuals and store factor/shocks.
    Uses derivative‑free BOBYQA on the sphere (implemented via box + normalization).
    """
    rng = np.random.default_rng(seed)
    T, N = R_sa_train.shape
    R = R_sa_train.copy()
    Rhat = Recon_sofar.copy()

    G_sa = np.zeros((n_sa, N))
    Xi_sa = np.zeros((T, n_sa))
    W = np.zeros((n_sa, n_PC))

    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A, float)
    b_vec = np.asarray(b, float).reshape(-1)
    G0r = np.asarray(G0, float)[None, :]

    def objective_factory(G_sub, Xi_PC):
        # return objective over w∈[-1,1]^{n_PC} (we’ll renormalize inside)
        def obj(w):
            w = np.asarray(w, float)
            if np.allclose(w, 0.0):
                return 1e9
            w = w / np.linalg.norm(w)
            G_cand = w @ G_sub            # (N,)
            xi_cand = Xi_PC @ w           # (T,)
            C_cand = G0r + Rhat + np.outer(xi_cand, G_cand)   # (T, N)
            # penalties
            H = hinge_penalty(C_cand, A_dense, b_vec)
            rec = np.linalg.norm(R - np.outer(xi_cand, G_cand))
            return lam_hinge * H + lam_rec * rec
        return obj

    for i in range(n_sa):
        pca = PCA(n_components=n_PC, random_state=seed+i).fit(R)
        G_sub = pca.components_               # (n_PC, N)
        Xi_PC = pca.transform(R)              # (T, n_PC)

        obj = objective_factory(G_sub, Xi_PC)
        w0 = rng.normal(size=n_PC); w0 /= np.linalg.norm(w0)
        lb, ub = -np.ones(n_PC), np.ones(n_PC)

        sol = pybobyqa.solve(obj, w0, bounds=(lb, ub),
                             rhobeg=0.5, rhoend=1e-4,
                             maxfun=maxfun, seek_global_minimum=True, print_progress=False)
        w_opt = sol.x / np.linalg.norm(sol.x)

        G_i  = w_opt @ G_sub
        xi_i = Xi_PC @ w_opt

        # update running residual and reconstruction
        R    = R    - np.outer(xi_i, G_i)
        Rhat = Rhat + np.outer(xi_i, G_i)

        # store
        G_sa[i]   = G_i
        Xi_sa[:, i] = xi_i
        W[i] = w_opt

    return G_sa, Xi_sa, W
