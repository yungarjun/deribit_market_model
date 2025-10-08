import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import torch
from tqdm import tqdm

# Script for functions to enforce Friedman-Pinsky Conditions on drift and diffusion
# Ensures Stochastic Differential Equation will remain in convex polytope


# ------------------ Diffusion Shrinkage ------------------

# Function to compute polytopes of static arb free factor space
def compute_factor_polytope_vertices(
    outs,
    xi_builder=None,          # optional: a function like neural_sde.loss.build_xi_training_data
    k_box: float = 6.0,       # box = mean ± k_box * std (per factor)
    eps_lp: float = 1e-10,    # minimal acceptable interior margin from LP
    relax_eps: float = 1e-9,  # small relaxation if LP is tight
    verbose: bool = True,
    return_mappings: bool = False,
):
    """
    Build factor-space polytope H z >= h from lattice constraints A c >= b via
        c = G0 + M z,   where M = [G_dyn^T | G_stat^T | G_sa^T].
    Then:
      1) add a data-driven bounding box in z (to ensure bounded region),
      2) find a strictly interior point z0 by LP,
      3) enumerate vertices using HalfspaceIntersection.

    Parameters
    ----------
    outs : object
        Must provide:
          - outs.G0 : (N,) base quote vector (G0)
          - outs.G_dyn, outs.G_stat, outs.G_sa : each (k_i, N) or possibly empty
          - outs.A : (R, N) constraints (dense or scipy.sparse)
          - outs.b : (R,)
        For the box in factor space:
          - Either pass `xi_builder`, or `outs` should carry Xi_*_train arrays
            consistent with (G_dyn | G_stat | G_sa) ordering.
    xi_builder : callable or None
        If provided, should return (X_train, X_test, names) with X_train shape (T, p).
        Use the same factor ordering as M (dyn | stat | sa).
    k_box : float
        Bounding-box half-width in standard deviations per factor.
    eps_lp : float
        Minimum LP margin t* considered strictly interior.
    relax_eps : float
        Relaxation applied to inequalities if LP margin is too small.
    verbose : bool
        Print small diagnostics.
    return_mappings : bool
        If True, also return (H, h, M, G0).

    Returns
    -------
    result : dict
        {
          "vertices": (M, p) ndarray of polytope vertices in factor coords,
          "z0":       (p,) interior point,
          "H":        (R, p) halfspace normals in factor space,
          "h":        (R,)   offsets in factor space,
          "box_bounds": (p,2) array of [L, U] per factor,
          "diagnostics": dict with shapes and LP margin
          [, "M", "G0"] if return_mappings=True
        }
    """
    # -----------------------------
    # 0) Factor mapping: c = G0 + M z
    # -----------------------------
    G0 = np.asarray(outs.G0, dtype=np.float64).ravel()
    N = G0.size

    def _to_np(a, fallback_rows=0):
        if a is None or (hasattr(a, "size") and a.size == 0):
            return np.zeros((fallback_rows, N), dtype=np.float64)
        return np.asarray(a, dtype=np.float64)

    Gd = _to_np(getattr(outs, "G_dyn", None))
    Gs = _to_np(getattr(outs, "G_stat", None))
    Ga = _to_np(getattr(outs, "G_sa",   None))

    # M columns follow z ordering: dyn | stat | sa
    M = np.hstack([Gd.T, Gs.T, Ga.T])
    p = M.shape[1]

    # -----------------------------
    # 1) Map A c >= b  ->  H z >= h
    # -----------------------------
    A_raw = outs.A
    b = np.asarray(outs.b, dtype=np.float64).ravel()

    if sp.issparse(A_raw):
        A = A_raw.tocsr().astype(np.float64)
        H = A @ M                     # (R, p)
        h = b - A @ G0                # (R,)
    else:
        A = np.asarray(A_raw, dtype=np.float64)
        H = A @ M
        h = b - A @ G0

    H = np.asarray(H, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    R = H.shape[0]
    if verbose:
        print(f"[map] A: {A.shape} -> H: {H.shape}, p={p}")

    # -----------------------------
    # 2) Build z-box from training Xi
    # -----------------------------
    def _fallback_xi_train(_outs):
        parts = []
        for name in ("Xi_dyn_train", "Xi_stat_train", "Xi_sa_train"):
            arr = getattr(_outs, name, None)
            if arr is not None and getattr(arr, "size", 0):
                parts.append(np.asarray(arr, dtype=np.float64))
        if not parts:
            raise ValueError("Cannot build Xi_train: provide `xi_builder` or add Xi_*_train to `outs`.")
        return np.concatenate(parts, axis=1)

    if xi_builder is not None:
        X_tr, *_ = xi_builder(outs)
    else:
        X_tr = _fallback_xi_train(outs)

    if X_tr.shape[1] != p:
        raise ValueError(f"Xi_train has p={X_tr.shape[1]} columns but M has p={p}. "
                         "Ensure factor ordering and counts match (dyn|stat|sa).")

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)   # avoid zeros

    L = mu - k_box * sd
    U = mu + k_box * sd

    # Box -> halfspaces a^T z + c <= 0
    A_box = []
    c_box = []
    for j in range(p):
        ej = np.zeros(p); ej[j] = 1.0
        A_box.append( ej.copy()); c_box.append(-U[j])   # z_j <= U_j
        A_box.append(-ej.copy()); c_box.append( L[j])   # z_j >= L_j
    A_box = np.asarray(A_box, dtype=np.float64)
    c_box = np.asarray(c_box, dtype=np.float64)

    # -----------------------------
    # 3) LP: strictly interior point
    # maximize t  s.t. H z - h >= t  and  A_box z + c_box <= 0
    # -----------------------------
    G_lp = np.hstack([-H, np.ones((R, 1))])            # -H z + 1*t <= -h
    h_lp = -h
    Gb   = np.hstack([A_box, np.zeros((A_box.shape[0], 1))])
    hb   = -c_box
    G_ub = np.vstack([G_lp, Gb])
    h_ub = np.concatenate([h_lp, hb])

    c_vec = np.zeros(p + 1); c_vec[-1] = -1.0          # maximize t -> minimize -t
    res = linprog(c_vec, A_ub=G_ub, b_ub=h_ub, method="highs")

    if (not res.success) or (res.x[-1] <= eps_lp):
        # tiny uniform relaxation
        res = linprog(c_vec, A_ub=G_ub, b_ub=h_ub - relax_eps, method="highs")
        if (not res.success) or (res.x[-1] <= 0.0):
            raise RuntimeError("Failed to find a strictly interior point for (H,h) with box.")

    z0 = res.x[:p]
    t_star = float(res.x[-1])

    # -----------------------------
    # 4) Vertex enumeration
    # HalfspaceIntersection expects a_i^T z + c_i <= 0
    # Our system: H z >= h   <=>  -H z + h <= 0
    # -----------------------------
    halfspaces = np.hstack([-H, h[:, None]])
    halfspaces = np.vstack([halfspaces, np.hstack([A_box, c_box[:, None]])])

    hs = HalfspaceIntersection(halfspaces, z0)
    vertices = np.asarray(hs.intersections)

    if verbose:
        print(f"[polytope] p={p}, R={R}, vertices={vertices.shape[0]}, LP margin t*={t_star:.3e}")

    out = {
        "vertices": vertices,
        "z0": z0,
        "H": H,
        "h": h,
        "box_bounds": np.vstack([L, U]).T,  # (p,2)
        "diagnostics": {
            "p": int(p),
            "R": int(R),
            "N": int(N),
            "num_vertices": int(vertices.shape[0]),
            "lp_margin": t_star,
            "A_is_sparse": bool(sp.issparse(A_raw)),
        },
    }
    if return_mappings:
        out["M"]  = M
        out["G0"] = G0
    return out

# Diffusion Shrinkage Profile
def h_sigma(x):
    # Properties: h(0)=0, monotone increasing, ->1 as x->∞
    x = np.maximum(x, 0.0)  # ensure nonnegativity
    return 1.0 - 1.0 / (1.0 + x)

# Torch version 
def torch_h_sigma(x: torch.Tensor) -> torch.Tensor:
    # h(x) = x / (1 + x) == 1 - 1/(1+x)
    return x / (1.0 + x).clamp_min(1e-12)

# Shrinkage 
def shrink_Q_eps_from_Hh(y: torch.Tensor, H: torch.Tensor, h: torch.Tensor, k_take: int = None):
    """
    Build (Q, eps_sel) from polytope faces for each batch state y.
    Inputs:
      y: (B,d) factor point
      H: (R,d), h: (R,)
      k_take: how many closest faces to use (default d)
    Returns:
      Q: (B,d,d) orthonormal columns
      eps_sel: (B,d) shrink coeffs per selected directions
    """
    B, d = y.shape
    R = H.shape[0]
    k = d if (k_take is None) else min(k_take, d, R)

    # distances ρ_i(y) = (H_i·y - h_i)/||H_i||, interior ⇒ ρ>=0
    Hi_norm = torch.linalg.norm(H, dim=1).clamp_min(1e-12)     # (R,)
    rho = (y @ H.T - h) / Hi_norm                              # (B,R)
    rho = rho.clamp_min(0.0)
    eps = torch_h_sigma(rho)                                   # (B,R)

    # indices of k smallest eps
    idx = torch.topk(eps, k=k, dim=1, largest=False).indices   # (B,k)

    # gather boundary normals v^(k) and normalise
    V = torch.stack([H[i] / Hi_norm[i] for i in range(R)], dim=0)  # (R,d)
    Vk = V[idx]                                                    # (B,k,d)
    eps_k = torch.gather(eps, 1, idx)                              # (B,k)

    # Gram–Schmidt (batched)
    Q = torch.zeros((B, d, d), dtype=y.dtype, device=y.device)
    # fill first k columns by GS on Vk, remaining by completing with canonical axes
    eps_sqrt = eps_k.sqrt()                                        # (B,k)
    for b in range(B):
        cols = []
        for j in range(k):
            v = Vk[b, j]
            for t in cols:
                v = v - (v @ t) * t
            n = torch.linalg.norm(v).clamp_min(1e-12)
            cols.append(v / n)
        # complete basis if needed
        e = torch.eye(d, device=y.device, dtype=y.dtype)
        a = 0
        while len(cols) < d and a < d:
            v = e[a]
            for t in cols:
                v = v - (v @ t) * t
            n = torch.linalg.norm(v)
            if n > 1e-8:
                cols.append(v / n)
            a += 1
        Q[b] = torch.stack(cols, dim=1)  # columns

    # pack eps (pad with ones for remaining directions so they are not shrunk)
    if k < d:
        pad = torch.ones((B, d - k), dtype=y.dtype, device=y.device)
        eps_all = torch.cat([eps_sqrt, pad], dim=1)  # (B,d)
    else:
        eps_all = eps_sqrt

    return Q, eps_all

def shrink_matrix_and_diag(y: torch.Tensor, H: torch.Tensor, h: torch.Tensor, mode: str = "diag"):
    """
    mode:
      - 'diag': return per-dim shrink multipliers s_i in [0,1] by taking diag(Q diag(√ε) Qᵀ)
      - 'matrix': return full P = Q diag(√ε) Qᵀ
    """
    Q, eps_all = shrink_Q_eps_from_Hh(y, H, h)          # Q:(B,d,d), eps_all:(B,d)
    if mode == "matrix":
        P = Q @ torch.diag_embed(eps_all) @ Q.transpose(1, 2)  # (B,d,d)
        return P
    # diagonal approximation for drop-in with diagonal diffusion
    P = Q @ torch.diag_embed(eps_all) @ Q.transpose(1, 2)      # (B,d,d)
    s = torch.diagonal(P, dim1=1, dim2=2)                      # (B,d) in [0,1]
    return s

def shrink_diag_covmatch(P: torch.Tensor, sig: torch.Tensor, eps: float = 1e-8):
    # P: (B,p,p); sig: (B,p)
    num = torch.sqrt(torch.sum(P.pow(2) * sig.unsqueeze(1).pow(2), dim=2))  # (B,p)
    den = sig.clamp_min(eps)
    s = (num / den).clamp(0.0, 1.0)
    return s


def assemble_Wb_for_shrinkage(poly, include_box=True):
    """
    Turn polytope result into (W,b) usable by calc_diffusion_scaling.
    We build W,b with inequalities in the form  W z >= b.
    """
    H = np.asarray(poly["H"], dtype=np.float64)
    h = np.asarray(poly["h"], dtype=np.float64).ravel()
    W_list = [H]
    b_list = [h]

    if include_box:
        box = np.asarray(poly["box_bounds"], dtype=np.float64)  # shape (p,2): [L, U]
        p = box.shape[0]
        # z_j >= L_j  ->  e_j^T z >= L_j
        # z_j <= U_j  -> -e_j^T z >= -U_j
        W_box = []
        b_box = []
        for j in range(p):
            ej = np.zeros(p); ej[j] = 1.0
            L, U = box[j, 0], box[j, 1]
            W_box.append( ej.copy()); b_box.append( L)
            W_box.append(-ej.copy()); b_box.append(-U)
        W_box = np.asarray(W_box, dtype=np.float64)
        b_box = np.asarray(b_box, dtype=np.float64)
        W_list.append(W_box); b_list.append(b_box)

    W = np.vstack(W_list)
    b = np.concatenate(b_list)

    # Row-normalise so ||W_i|| = 1 (your shrinkage code asserts this)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    ok = norms.squeeze() > 1e-14
    W[ok] /= norms[ok]
    b[ok] /= norms[ok, 0]
    return W, b

def build_factor_path(outs, xi_builder=None):
    """Concatenate factors in the same order used by M = [G_dyn^T|G_stat^T|G_sa^T]."""
    if xi_builder is not None:
        X_tr, X_te, _ = xi_builder(outs)
        return np.asarray(X_tr, dtype=np.float64)
    parts = []
    for name in ("Xi_dyn_train", "Xi_stat_train", "Xi_sa_train"):
        arr = getattr(outs, name, None)
        if arr is not None and getattr(arr, "size", 0):
            parts.append(np.asarray(arr, dtype=np.float64))
    if not parts:
        raise ValueError("No Xi_*_train found; pass xi_builder=...")
    return np.concatenate(parts, axis=1)


@staticmethod
def normalise_dist_diffusion(rho, dist_multiplier, proj_scale):
    return proj_scale * (1 - 1. / (rho * dist_multiplier + 1))

@staticmethod
def calc_diffusion_scaling(W, b, X, dX, dist_multiplier, proj_scale):
    """
    Pre-calculate diffusion shrinking transformation matrix and other
    related data.

    Parameters
    __________
    W: numpy.array, 2D, shape = (n_constraint, n_factor)
        The coefficient matrix of the static arbitrage constraints in terms
        of the factors.

    b: numpy.array, 1D, shape = (n_constraint, )
        The constant vector term of the static arbitrage constraints in
        terms of the factors.

    X: numpy.array, 2D, shape = (n_time, n_factor)
        Decoded factor data.

    dX: numpy.array, 2D, shape = (n_time, n_factor)
        First-order difference of the decoded factor data.

    dist_multiplier, proj_scale: float
        Hyper-parameters that are used to normalise distance between the
        process to the static arbitrage boundaries. The maximal normalised
        distance is proj_scale, and dist_multiplier adjusts the convergence
        rate to zero when distance is dropping to zero.

    Returns
    _______
    Omegas: numpy.array, 2D, shape = (n_time, n_factor x n_factor)
        Diffusion shrinking matrices (flattened) over time.

    det_Omega: numpy.array, 2D, shape = (n_time, 1)
        The determinants of diffusion shrinking matrices over time.

    proj_dX: numpy.array, 2D, shape = (n_time, n_factor)
        The term in the likelihood function of xi that relates the diffusion
        shrinking matrix and dX.

    """

    # normalise boundary coefficients
    norm_W = np.linalg.norm(W, axis=1)
    assert np.max(np.abs(norm_W - 1.)) < 1e-12

    # compute distances
    dist_X = np.abs(W.dot(X.T) - b[:, None]) / \
                np.linalg.norm(W, axis=1, keepdims=True)

    # compute normalised distances
    epsilons = normalise_dist_diffusion(
        dist_X, dist_multiplier, proj_scale)

    n_obs, n_dim = X.shape
    proj_dX = np.zeros((n_obs, n_dim))
    Omegas = np.zeros((n_obs, n_dim * n_dim))
    det_Omega = np.zeros((n_obs, 1))

    for idx_obs in tqdm(range(n_obs)):
        epsilon = epsilons[:, idx_obs]
        idxs_sorted_eps = np.argsort(epsilon)
        idxs_use = idxs_sorted_eps[:n_dim]

        if np.max(epsilon[idxs_use]) < 1e-8:
            raise ValueError('Some data in the sample path is on corners!')
        else:  # if the anchor point is not on the corner
            # compute new bases
            V = np.linalg.qr(W[idxs_use].T)[0].T
            Omega = np.diag(np.sqrt(epsilon[idxs_use])).dot(V)
            Omegas[idx_obs, :] = Omega.flatten()
            det_Omega[idx_obs, 0] = abs(np.linalg.det(Omega))
            try:
                proj_dX[idx_obs, :] = np.linalg.solve(Omega.T, dX[idx_obs, :])
            except:
                raise ValueError('Some data is on boundaries!')

    return Omegas, det_Omega, proj_dX

# ------------------ Drift Reflection ------------------
def _apply_boundary_correction(
    mu,                       # [B,p] raw model drift at y0
    t_idx,                    # [B] integer indices of the batch rows in the global time axis
    corr_dirs,                # [T,K,p] correction directions from calc_drift_correction
    epsmu,                    # [T,K]   normalised distances from calc_drift_correction
    bc_lambda: float = 1.0,   # strength multiplier
    bc_eps_floor: float = 1e-8,
    bc_cap: float | None = None,      # optional per-sample ||Δμ|| cap
    bc_epsmu_cutoff: float | None = None,  # only correct when epsmu <= cutoff (near boundary)
    device=None,
):
    """
    Returns μ_corrected = μ + Δμ where Δμ = sum_k w_{t,k} * dir_{t,k};  w_{t,k} = bc_lambda / (epsmu_{t,k}+floor).
    Larger correction when closer to boundary (small epsmu). If bc_epsmu_cutoff is set, only apply when within that band.
    """
    if device is None:
        device = mu.device

    # Torch-ify inputs on the right device/dtype
    if not torch.is_tensor(corr_dirs):
        corr_dirs = torch.as_tensor(corr_dirs, dtype=mu.dtype, device=device)   # [T,K,p]
    if not torch.is_tensor(epsmu):
        epsmu = torch.as_tensor(epsmu, dtype=mu.dtype, device=device)           # [T,K]

    # If directions were flattened earlier: [T, K*p] -> reshape back
    if corr_dirs.ndim == 2:
        T = epsmu.shape[0]
        K = epsmu.shape[1]
        p = mu.shape[1]
        corr_dirs = corr_dirs.view(T, K, p)

    # Gather per-batch slices
    dirs_t = corr_dirs.index_select(0, t_idx)   # [B,K,p]
    eps_t  = epsmu.index_select(0, t_idx)       # [B,K]

    # Weights: stronger when eps is small (closer to the boundary)
    w = bc_lambda / (eps_t.clamp_min(bc_eps_floor))   # [B,K]

    # Optional “within-band only” correction
    if bc_epsmu_cutoff is not None:
        mask = (eps_t <= bc_epsmu_cutoff).to(mu.dtype)   # [B,K]
        w = w * mask

    # Δμ = sum_k w_k * dir_k
    delta_mu = (w.unsqueeze(-1) * dirs_t).sum(dim=1)     # [B,p]

    # Optional norm cap per sample
    if bc_cap is not None:
        nrm = torch.linalg.vector_norm(delta_mu, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(nrm), torch.as_tensor(bc_cap, dtype=mu.dtype, device=device) / nrm)
        delta_mu = delta_mu * scale

    return mu + delta_mu

def compute_polytope_vertices(W, b, interior_offset=1e-6):
    """
    Compute vertices of polytope defined by W x >= b using SciPy only.
    Equivalent to pypoman.compute_polytope_vertices(-W, -b).
    """

    # Convert inequalities to half-space form: A x + b' <= 0
    # (HalfspaceIntersection expects a_i x + b_i <= 0)
    halfspaces = np.hstack([-W, b[:, None]])

    # Find a feasible interior point by linear programming
    # We minimize a dummy objective subject to W x >= b
    n_dim = W.shape[1]
    res = linprog(
        c=np.zeros(n_dim),
        A_ub=-W,
        b_ub=-b - interior_offset,
        bounds=[(None, None)] * n_dim,
        method="highs"
    )
    if not res.success:
        raise ValueError("Could not find feasible interior point for the polytope.")

    x0 = res.x

    # Intersect half-spaces
    hs = HalfspaceIntersection(halfspaces, x0)
    vertices = hs.intersections

    return vertices

def calc_drift_correction(W, b, X, rho_star, epsmu_star, tol_face=1e-6):
    """
    Robust: row-normalize W, relax facet tolerance, use 1-D algebra, and
    fallback to a projected centroid if a face has no detected vertices.
    """
    # Row-normalize constraints to stabilize tolerances
    W = np.asarray(W, float)
    b = np.asarray(b, float).reshape(-1)
    row_norms = np.linalg.norm(W, axis=1, keepdims=True)
    row_norms[row_norms == 0.0] = 1.0
    Wn = W / row_norms
    bn = b / row_norms.ravel()

    # Vertices of {x: Wn x >= bn}
    vertices = compute_polytope_vertices(Wn, bn)  # uses HalfspaceIntersection
    if vertices.size == 0:
        raise ValueError("Polytope has no vertices (infeasible or degenerate).")

    V = np.asarray(vertices, float)                 # (V, n_dim)
    v_centroid = V.mean(axis=0)                     # global centroid for fallback
    n_bdy, n_dim = Wn.shape

    X_interior = np.zeros((n_bdy, n_dim))
    for k in range(n_bdy):
        wk = Wn[k]           # (n_dim,)
        bk = bn[k]           # scalar

        # collect vertices on (or very near) the facet wk·x = bk
        on_face_mask = np.isclose(V @ wk, bk, rtol=1e-6, atol=tol_face)
        Vk = V[on_face_mask]
        if Vk.shape[0] == 0:
            # fallback: project centroid onto the plane wk·x = bk
            # mk = v_centroid - ((wk·v_centroid - bk)/||wk||^2) wk
            proj = (wk @ v_centroid - bk) / max(np.dot(wk, wk), 1e-12)
            mk = v_centroid - proj * wk
        else:
            mk = Vk.mean(axis=0)   # (n_dim,)

        # Move interior along +wk direction: t ∈ [max_lb, min_ub]
        W_rmv_k = np.delete(Wn, k, axis=0)         # (m-1, n_dim)
        b_rmv_k = np.delete(bn, k) + rho_star      # (m-1,)

        coeffs = W_rmv_k @ wk                      # (m-1,)
        values = b_rmv_k - (W_rmv_k @ mk)          # (m-1,)

        mask_neg = coeffs < 0
        if np.any(~mask_neg):
            max_lb = np.max(values[~mask_neg] / np.maximum(coeffs[~mask_neg], 1e-12))
        else:
            max_lb = 0.0
        if np.any(mask_neg):
            min_ub = np.min(values[mask_neg] / np.minimum(coeffs[mask_neg], -1e-12))
        else:
            # no upper-bound constraints active along +wk; take a small step
            min_ub = max_lb + 1.0

        if max_lb > min_ub:
            raise ValueError(f"Infeasible step interval on face {k}: lb={max_lb}, ub={min_ub}. "
                             f"Try a smaller rho_star or larger tol_face.")

        t_star = min_ub
        X_interior[k] = mk + t_star * wk

    # distances per face/time (use normalized Wn, bn)
    X = np.asarray(X, float)
    dist_X = np.abs(Wn @ X.T - bn[:, None])  # (m, T)

    # directions from X toward interior anchors on each face
    n_obs = X.shape[0]
    corr_dirs = (X_interior[None, :, :] - X[:, None, :])    # (T, m, n_dim)

    # normalized distances for drift scaling
    epsmu = normalise_dist_drift(dist_X, rho_star, epsmu_star).T  # (T, m)

    return X_interior, corr_dirs.reshape(n_obs, -1), epsmu

@staticmethod
def normalise_dist_drift(rho, rho_star, epsmu_star):
    c = epsmu_star / (np.exp(rho_star) - 1.)
    return c * (np.exp(rho) - 1.)

def _apply_boundary_correction(
    mu,                       # [B,p] raw model drift at y0
    t_idx,                    # [B] integer indices of the batch rows in the global time axis
    corr_dirs,                # [T,K,p] correction directions from calc_drift_correction
    epsmu,                    # [T,K]   normalised distances from calc_drift_correction
    bc_lambda: float = 1.0,   # strength multiplier
    bc_eps_floor: float = 1e-8,
    bc_cap: float | None = None,      # optional per-sample ||Δμ|| cap
    bc_epsmu_cutoff: float | None = None,  # only correct when epsmu <= cutoff (near boundary)
    device=None,
):
    """
    Returns μ_corrected = μ + Δμ where Δμ = sum_k w_{t,k} * dir_{t,k};  w_{t,k} = bc_lambda / (epsmu_{t,k}+floor).
    Larger correction when closer to boundary (small epsmu). If bc_epsmu_cutoff is set, only apply when within that band.
    """
    if device is None:
        device = mu.device

    # Torch-ify inputs on the right device/dtype
    if not torch.is_tensor(corr_dirs):
        corr_dirs = torch.as_tensor(corr_dirs, dtype=mu.dtype, device=device)   # [T,K,p]
    if not torch.is_tensor(epsmu):
        epsmu = torch.as_tensor(epsmu, dtype=mu.dtype, device=device)           # [T,K]

    # If directions were flattened earlier: [T, K*p] -> reshape back
    if corr_dirs.ndim == 2:
        T = epsmu.shape[0]
        K = epsmu.shape[1]
        p = mu.shape[1]
        corr_dirs = corr_dirs.view(T, K, p)

    # Gather per-batch slices
    dirs_t = corr_dirs.index_select(0, t_idx)   # [B,K,p]
    eps_t  = epsmu.index_select(0, t_idx)       # [B,K]

    # Weights: stronger when eps is small (closer to the boundary)
    w = bc_lambda / (eps_t.clamp_min(bc_eps_floor))   # [B,K]

    # Optional “within-band only” correction
    if bc_epsmu_cutoff is not None:
        mask = (eps_t <= bc_epsmu_cutoff).to(mu.dtype)   # [B,K]
        w = w * mask

    # Δμ = sum_k w_k * dir_k
    delta_mu = (w.unsqueeze(-1) * dirs_t).sum(dim=1)     # [B,p]

    # Optional norm cap per sample
    if bc_cap is not None:
        nrm = torch.linalg.vector_norm(delta_mu, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(nrm), torch.as_tensor(bc_cap, dtype=mu.dtype, device=device) / nrm)
        delta_mu = delta_mu * scale

    return mu + delta_mu