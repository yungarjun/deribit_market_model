from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
import pandas as pd
import cvxpy as cp
from type import NoArbConstraints

# def build_noarb_constraints(nodes: np.ndarray,
#                             tau_grid: np.ndarray,
#                             m_grid: Dict[float, np.ndarray]) -> Tuple[sp.csr_matrix, np.ndarray]:
#     """
#     Constraints on c = C/F (normalized call):
#       (i)   monotone in τ: c(τ_{i+1}, m) - c(τ_i, m) >= 0
#       (ii)  monotone in m (decreasing in strike): c(m_j) - c(m_{j+1}) >= 0
#       (iii) convex in K (use x = e^m), discrete curvature >= 0
#       (iv)  bounds: 0 <= c <= 1, and c >= (1 - e^m)_+
#     """
#     n = nodes.shape[0]
#     rows, cols, data, b = [], [], [], []
#     k = 0

#     # (i) τ‑monotonicity
#     for i in range(len(tau_grid) - 1):
#         τ0, τ1 = tau_grid[i], tau_grid[i+1]
#         common_ms = sorted(set(m_grid[τ0]).intersection(set(m_grid[τ1])))
#         for m in common_ms:
#             k0 = np.where((np.isclose(nodes[:, 0], τ0)) & (np.isclose(nodes[:, 1], m)))[0][0]
#             k1 = np.where((np.isclose(nodes[:, 0], τ1)) & (np.isclose(nodes[:, 1], m)))[0][0]
#             rows += [k, k]; cols += [k1, k0]; data += [1.0, -1.0]; b.append(0.0); k += 1

#     # (ii) m‑monotonicity
#     for τ in tau_grid:
#         ms = np.sort(m_grid[τ])
#         for j in range(len(ms) - 1):
#             m0, m1 = ms[j], ms[j+1]
#             k0 = np.where((np.isclose(nodes[:, 0], τ)) & (np.isclose(nodes[:, 1], m0)))[0][0]
#             k1 = np.where((np.isclose(nodes[:, 0], τ)) & (np.isclose(nodes[:, 1], m1)))[0][0]
#             rows += [k, k]; cols += [k0, k1]; data += [1.0, -1.0]; b.append(0.0); k += 1

#     # (iii) convexity in K via x = e^m
#     for τ in tau_grid:
#         ms = np.sort(m_grid[τ])
#         xs = np.exp(ms)
#         for j in range(1, len(ms) - 1):
#             kL = np.where((np.isclose(nodes[:, 0], τ)) & (np.isclose(nodes[:, 1], ms[j-1])))[0][0]
#             kM = np.where((np.isclose(nodes[:, 0], τ)) & (np.isclose(nodes[:, 1], ms[j  ])))[0][0]
#             kR = np.where((np.isclose(nodes[:, 0], τ)) & (np.isclose(nodes[:, 1], ms[j+1])))[0][0]
#             dxL = xs[j] - xs[j-1]
#             dxR = xs[j+1] - xs[j]
#             a = 1.0/dxL
#             b_mid = -(1.0/dxL + 1.0/dxR)
#             c = 1.0/dxR
#             rows += [k, k, k]; cols += [kL, kM, kR]; data += [a, b_mid, c]; b.append(0.0); k += 1

#     A0 = sp.csr_matrix((data, (rows, cols)), shape=(k, n))

#     # Bounds 0 <= c <= 1
#     A_ub = -sp.eye(n, format='csr'); b_ub = -np.ones(n)
#     A_lb =  sp.eye(n, format='csr'); 
#     mvals = nodes[:, 1]
#     lower = np.maximum(0.0, 1.0 - np.exp(mvals))   # intrinsic scaled by F
#     b_lb = lower

#     A = sp.vstack([A0, A_ub, A_lb], format='csr')
#     b = np.concatenate([np.zeros(k), b_ub, b_lb]).astype(float)
#     return NoArbConstraints(A=A, b=b)


def build_noarb_constraints(nodes: np.ndarray,
                            tau_grid: np.ndarray,
                            m_grid: Dict[float, np.ndarray],
                            tau_mono_mode: str = "interp",   # "interp" or "exact"
                            allow_extrapolate: bool = False, # if True, clamp at ends
                            eps: float = 1e-12
                           ) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Constraints on c = C/F (normalized call):
      (i)   monotone in τ (interpolated or exact): c(τ_{i+1}, ·) >= c(τ_i, ·)
      (ii)  monotone in m (decreasing in strike): c(m_j) - c(m_{j+1}) >= 0
      (iii) convex in K (use x = e^m), discrete curvature >= 0
      (iv)  bounds: 0 <= c <= 1, and c >= (1 - e^m)_+

    Parameters
    ----------
    tau_mono_mode : "interp" (default) uses linear interpolation on the τ_{i+1} row
                    to evaluate at m-points of τ_i; "exact" uses exact m matches only.
    allow_extrapolate : if True, when an m0 is outside [min(m_{τ1}), max(m_{τ1})],
                        we clamp to the nearest segment (α is clipped to [0,1]).
    """
    n = nodes.shape[0]
    rows, cols, data, b = [], [], [], []
    k = 0

    # Fast index map: (τ, m) -> node index
    idx_map: Dict[float, Dict[float, int]] = {}
    for idx, (τv, mv) in enumerate(nodes):
        τf, mf = float(τv), float(mv)
        d = idx_map.get(τf)
        if d is None:
            d = {}
            idx_map[τf] = d
        d[mf] = idx

    # ----------------------------
    # (i) τ-monotonicity block
    # ----------------------------
    if tau_mono_mode not in ("interp", "exact"):
        raise ValueError("tau_mono_mode must be 'interp' or 'exact'.")

    for i in range(len(tau_grid) - 1):
        τ0, τ1 = float(tau_grid[i]), float(tau_grid[i+1])
        m0s = np.sort(np.asarray(m_grid[τ0], float))
        m1s = np.sort(np.asarray(m_grid[τ1], float))

        if tau_mono_mode == "exact":
            # Only identical m between rows (usually empty if grids were built independently)
            common_ms = sorted(set(m0s).intersection(set(m1s)))
            for m0 in common_ms:
                k0 = idx_map[τ0][float(m0)]
                k1 = idx_map[τ1][float(m0)]
                # c(τ1,m) - c(τ0,m) >= 0
                rows += [k, k]; cols += [k1, k0]; data += [1.0, -1.0]; b.append(0.0); k += 1

        else:
            # Interpolated row τ1 at each m0 from row τ0:
            # α*c(τ1,mL) + (1-α)*c(τ1,mR) - c(τ0,m0) >= 0
            for m0 in m0s:
                # locate bracketing indices in m1s
                j = int(np.searchsorted(m1s, m0))
                if j == 0 or j == len(m1s):
                    if not allow_extrapolate:
                        continue  # skip out-of-range m0
                    # clamp to nearest segment
                    if j == 0:
                        j = 1
                    else:
                        j = len(m1s) - 1

                mL, mR = float(m1s[j-1]), float(m1s[j])
                denom = max(mR - mL, eps)
                α = (mR - float(m0)) / denom  # α in [0,1] inside; may go outside if extrapolating
                if allow_extrapolate:
                    α = float(np.clip(α, 0.0, 1.0))

                # node indices
                k0 = idx_map[τ0][float(m0)]
                kL = idx_map[τ1][mL]
                kR = idx_map[τ1][mR]

                # add row: α*c(τ1,mL) + (1-α)*c(τ1,mR) - c(τ0,m0) >= 0
                rows += [k, k, k]; cols += [kL, kR, k0]; data += [α, (1.0 - α), -1.0]; b.append(0.0); k += 1

    # ----------------------------
    # (ii) m-monotonicity block
    # ----------------------------
    for τ in map(float, tau_grid):
        ms = np.sort(np.asarray(m_grid[τ], float))
        for j in range(len(ms) - 1):
            m0, m1 = float(ms[j]), float(ms[j+1])
            k0 = idx_map[τ][m0]
            k1 = idx_map[τ][m1]
            # c(τ,m0) - c(τ,m1) >= 0
            rows += [k, k]; cols += [k0, k1]; data += [1.0, -1.0]; b.append(0.0); k += 1

    # ----------------------------
    # (iii) convexity in K (x = e^m)
    # ----------------------------
    for τ in map(float, tau_grid):
        ms = np.sort(np.asarray(m_grid[τ], float))
        xs = np.exp(ms)
        for j in range(1, len(ms) - 1):
            mL, mM, mR = float(ms[j-1]), float(ms[j]), float(ms[j+1])
            kL = idx_map[τ][mL]
            kM = idx_map[τ][mM]
            kR = idx_map[τ][mR]
            dxL = xs[j]   - xs[j-1]
            dxR = xs[j+1] - xs[j]
            a = 1.0 / max(dxL, eps)
            c = 1.0 / max(dxR, eps)
            b_mid = -(a + c)
            # a*c(τ,mL) + b_mid*c(τ,mM) + c*c(τ,mR) >= 0
            rows += [k, k, k]; cols += [kL, kM, kR]; data += [a, b_mid, c]; b.append(0.0); k += 1

    # Base matrix for shape constraints
    A0 = sp.csr_matrix((data, (rows, cols)), shape=(k, n)) 

    # ----------------------------
    # (iv) Bounds
    # ----------------------------
    A_ub = -sp.eye(n, format='csr'); b_ub = -np.ones(n)  # c <= 1  ->  -I c >= -1
    A_lb =  sp.eye(n, format='csr')                      # c >= lower
    lower = np.maximum(0.0, 1.0 - np.exp(nodes[:, 1]))
    b_lb  = lower

    A = sp.vstack([A0, A_ub, A_lb], format='csr')
    b = np.concatenate([np.zeros(k), b_ub, b_lb]).astype(float)
    return NoArbConstraints(A=A, b=b)



def project_noarb(C_interp: pd.DataFrame,
                  nodes: np.ndarray,
                  tau_grid: np.ndarray,
                  m_grid: Dict[float, np.ndarray]) -> pd.DataFrame:
    """
    For each time slice, project c_raw to nearest c under A c >= b, 0 <= c <= 1.
    Solve with OSQP (QP).
    """
    ab = build_noarb_constraints(nodes, tau_grid, m_grid)

    A, b = ab.A, ab.b
    n = nodes.shape[0]

    c_raw = cp.Parameter(n)
    c_var = cp.Variable(n)
    constraints = [A @ c_var >= b, c_var >= 0, c_var <= 1]
    obj = cp.Minimize(cp.sum_squares(c_var - c_raw))
    prob = cp.Problem(obj, constraints)

    out = []
    for _, row in C_interp.iterrows():
        x = row.values.astype(float)
        c_raw.value = x
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            out.append(x)
        else:
            out.append(c_var.value)
    out = np.vstack(out)
    return pd.DataFrame(out, index=C_interp.index, columns=C_interp.columns)


# Redundant linear constraint removal via LP tests
def reduce_constraints_geq(A: sp.csr_matrix, b: np.ndarray, tol=1e-8, solver="OSQP"):
    """
    Remove redundant rows from A x >= b using LP tests.

    Returns
    -------
    A_red, b_red, mask_kept  (mask_kept is boolean for rows kept)
    """
    if sp.issparse(A):
        A = A.tocsr()
    else:
        A = sp.csr_matrix(A)

    m, n = A.shape
    b = np.asarray(b, float).reshape(-1)
    assert b.shape[0] == m

    # Convert to <= form:  G x <= h
    G_full = (-A).toarray()
    h_full = -b.copy()

    x = cp.Variable(n)

    mask_kept = np.ones(m, dtype=bool)
    # pre-build problem objects (we’ll swap G,h per test)
    obj = None
    prob = None

    for i in range(m):
        # remove i-th row
        if not mask_kept[i]:
            continue

        rows = np.arange(m) != i
        G = G_full[rows, :]
        h = h_full[rows]

        # Maximize a_i^T x subject to Gx <= h
        ai = A.getrow(i).toarray().ravel()  # original (>=) row
        objective = cp.Maximize(ai @ x)
        constraints = [G @ x <= h]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=solver, verbose=False)
        except Exception:
            # fallback
            prob.solve(solver="SCS", verbose=False, eps=1e-5)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            # Couldn’t certify redundancy — keep it
            mask_kept[i] = True
            continue

        val = prob.value
        # If the best achievable a_i^T x is still <= b_i + tol, row i is implied (redundant)
        if val <= b[i] + tol:
            mask_kept[i] = False

    A_red = A[mask_kept]
    b_red = b[mask_kept]
    return A_red.tocsr(), b_red, mask_kept