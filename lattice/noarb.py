from typing import Dict, Tuple
import numpy as np
import scipy.sparse as sp
import pandas as pd
import cvxpy as cp
from type import NoArbConstraints

def build_noarb_constraints(nodes, tau_grid, m_grid):
    """
    A c >= b on c = C/F (normalized call).
    - Monotone in tau  (increasing)
    - Monotone in m    (decreasing)
    - Convex in strike: use x = exp(m), non-uniform spacing
    """
    n = nodes.shape[0]
    rows, cols, data, b = [], [], [], []
    cons_idx = 0

    # 1) monotone in tau: C(tau_{i+1}, m) - C(tau_i, m) >= 0
    for i in range(len(tau_grid) - 1):
        τ0, τ1 = tau_grid[i], tau_grid[i+1]
        common_ms = sorted(set(m_grid[τ0]) & set(m_grid[τ1]))
        for m in common_ms:
            mask0 = np.isclose(nodes[:,0], τ0) & np.isclose(nodes[:,1], m)
            mask1 = np.isclose(nodes[:,0], τ1) & np.isclose(nodes[:,1], m)
            k0 = np.where(mask0)[0][0]
            k1 = np.where(mask1)[0][0]
            rows += [cons_idx, cons_idx]; cols += [k1, k0]; data += [1.0, -1.0]
            b.append(0.0); cons_idx += 1

    # 2) monotone in m (decreasing in strike): C(m_j) - C(m_{j+1}) >= 0
    for τ in tau_grid:
        ms = sorted(m_grid[τ])
        for j in range(len(ms)-1):
            m0, m1 = ms[j], ms[j+1]
            k0 = np.where(np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m0))[0][0]
            k1 = np.where(np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], m1))[0][0]
            rows += [cons_idx, cons_idx]; cols += [k0, k1]; data += [1.0, -1.0]
            b.append(0.0); cons_idx += 1

    # 3) convex in strike: use x = exp(m), non-uniform Δx
    for τ in tau_grid:
        ms = sorted(m_grid[τ])
        xs = np.exp(ms)              # proxy for K up to F_t scale
        for j in range(1, len(ms)-1):
            k_prev = np.where(np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], ms[j-1]))[0][0]
            k_mid  = np.where(np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], ms[j  ]))[0][0]
            k_next = np.where(np.isclose(nodes[:,0], τ) & np.isclose(nodes[:,1], ms[j+1]))[0][0]

            dxL = xs[j]   - xs[j-1]
            dxR = xs[j+1] - xs[j]
            # (C_j - C_{j-1})/dxL <= (C_{j+1} - C_j)/dxR
            # => (1/dxL) C_{j-1} + (-(1/dxL+1/dxR)) C_j + (1/dxR) C_{j+1} >= 0
            a = 1.0/dxL
            b_mid = -(1.0/dxL + 1.0/dxR)
            c = 1.0/dxR

            rows += [cons_idx, cons_idx, cons_idx]
            cols += [k_prev, k_mid, k_next]
            data += [a, b_mid, c]
            b.append(0.0); cons_idx += 1

    A0 = sp.csr_matrix((data, (rows, cols)), shape=(cons_idx, n))
    b0 = np.array(b, dtype=float)
    
    N = nodes.shape[0]
    # upper bound: c <= 1  ==>  (-I) c >= -1
    A_ub = -sp.eye(N, format='csr')
    b_ub = -np.ones(N)

    # lower bound: c >= (1 - e^{m})_+  ==>  (+I) c >= l
    m = nodes[:,1]
    l = np.maximum(0.0, 1.0 - np.exp(m))
    A_lb = sp.eye(N, format='csr')
    b_lb = l

    A = sp.vstack([A0, A_ub, A_lb], format='csr')
    b = np.concatenate([b0, b_ub, b_lb])

    # return A, np.asarray(b, float)
    
    return NoArbConstraints(A=A, b=b)

# def build_noarb_constraints(nodes: np.ndarray,
#                             tau_grid: np.ndarray,
#                             m_grid: Dict[float, np.ndarray],
#                             tau_mono_mode: str = "interp",   # "interp" or "exact"
#                             allow_extrapolate: bool = False, # if True, clamp at ends
#                             eps: float = 1e-12
#                            ) -> Tuple[sp.csr_matrix, np.ndarray]:
#     """
#     Constraints on c = C/F (normalized call):
#       (i)   monotone in τ (interpolated or exact): c(τ_{i+1}, ·) >= c(τ_i, ·)
#       (ii)  monotone in m (decreasing in strike): c(m_j) - c(m_{j+1}) >= 0
#       (iii) convex in K (use x = e^m), discrete curvature >= 0
#       (iv)  bounds: 0 <= c <= 1, and c >= (1 - e^m)_+

#     Parameters
#     ----------
#     tau_mono_mode : "interp" (default) uses linear interpolation on the τ_{i+1} row
#                     to evaluate at m-points of τ_i; "exact" uses exact m matches only.
#     allow_extrapolate : if True, when an m0 is outside [min(m_{τ1}), max(m_{τ1})],
#                         we clamp to the nearest segment (α is clipped to [0,1]).
#     """
#     n = nodes.shape[0]
#     rows, cols, data, b = [], [], [], []
#     k = 0

#     # Fast index map: (τ, m) -> node index
#     idx_map: Dict[float, Dict[float, int]] = {}
#     for idx, (τv, mv) in enumerate(nodes):
#         τf, mf = float(τv), float(mv)
#         d = idx_map.get(τf)
#         if d is None:
#             d = {}
#             idx_map[τf] = d
#         d[mf] = idx

#     # ----------------------------
#     # (i) τ-monotonicity block
#     # ----------------------------
#     if tau_mono_mode not in ("interp", "exact"):
#         raise ValueError("tau_mono_mode must be 'interp' or 'exact'.")

#     for i in range(len(tau_grid) - 1):
#         τ0, τ1 = float(tau_grid[i]), float(tau_grid[i+1])
#         m0s = np.sort(np.asarray(m_grid[τ0], float))
#         m1s = np.sort(np.asarray(m_grid[τ1], float))

#         if tau_mono_mode == "exact":
#             # Only identical m between rows (usually empty if grids were built independently)
#             common_ms = sorted(set(m0s).intersection(set(m1s)))
#             for m0 in common_ms:
#                 k0 = idx_map[τ0][float(m0)]
#                 k1 = idx_map[τ1][float(m0)]
#                 # c(τ1,m) - c(τ0,m) >= 0
#                 rows += [k, k]; cols += [k1, k0]; data += [1.0, -1.0]; b.append(0.0); k += 1

#         else:
#             # Interpolated row τ1 at each m0 from row τ0:
#             # α*c(τ1,mL) + (1-α)*c(τ1,mR) - c(τ0,m0) >= 0
#             for m0 in m0s:
#                 # locate bracketing indices in m1s
#                 j = int(np.searchsorted(m1s, m0))
#                 if j == 0 or j == len(m1s):
#                     if not allow_extrapolate:
#                         continue  # skip out-of-range m0
#                     # clamp to nearest segment
#                     if j == 0:
#                         j = 1
#                     else:
#                         j = len(m1s) - 1

#                 mL, mR = float(m1s[j-1]), float(m1s[j])
#                 denom = max(mR - mL, eps)
#                 α = (mR - float(m0)) / denom  # α in [0,1] inside; may go outside if extrapolating
#                 if allow_extrapolate:
#                     α = float(np.clip(α, 0.0, 1.0))

#                 # node indices
#                 k0 = idx_map[τ0][float(m0)]
#                 kL = idx_map[τ1][mL]
#                 kR = idx_map[τ1][mR]

#                 # add row: α*c(τ1,mL) + (1-α)*c(τ1,mR) - c(τ0,m0) >= 0
#                 rows += [k, k, k]; cols += [kL, kR, k0]; data += [α, (1.0 - α), -1.0]; b.append(0.0); k += 1

#     # ----------------------------
#     # (ii) m-monotonicity block
#     # ----------------------------
#     for τ in map(float, tau_grid):
#         ms = np.sort(np.asarray(m_grid[τ], float))
#         for j in range(len(ms) - 1):
#             m0, m1 = float(ms[j]), float(ms[j+1])
#             k0 = idx_map[τ][m0]
#             k1 = idx_map[τ][m1]
#             # c(τ,m0) - c(τ,m1) >= 0
#             rows += [k, k]; cols += [k0, k1]; data += [1.0, -1.0]; b.append(0.0); k += 1

#     # ----------------------------
#     # (iii) convexity in K (x = e^m)
#     # ----------------------------
#     for τ in map(float, tau_grid):
#         ms = np.sort(np.asarray(m_grid[τ], float))
#         xs = np.exp(ms)
#         for j in range(1, len(ms) - 1):
#             mL, mM, mR = float(ms[j-1]), float(ms[j]), float(ms[j+1])
#             kL = idx_map[τ][mL]
#             kM = idx_map[τ][mM]
#             kR = idx_map[τ][mR]
#             dxL = xs[j]   - xs[j-1]
#             dxR = xs[j+1] - xs[j]
#             a = 1.0 / max(dxL, eps)
#             c = 1.0 / max(dxR, eps)
#             b_mid = -(a + c)
#             # a*c(τ,mL) + b_mid*c(τ,mM) + c*c(τ,mR) >= 0
#             rows += [k, k, k]; cols += [kL, kM, kR]; data += [a, b_mid, c]; b.append(0.0); k += 1

#     # Base matrix for shape constraints
#     A0 = sp.csr_matrix((data, (rows, cols)), shape=(k, n)) 

#     # ----------------------------
#     # (iv) Bounds
#     # ----------------------------
#     A_ub = -sp.eye(n, format='csr'); b_ub = -np.ones(n)  # c <= 1  ->  -I c >= -1
#     A_lb =  sp.eye(n, format='csr')                      # c >= lower
#     lower = np.maximum(0.0, 1.0 - np.exp(nodes[:, 1]))
#     b_lb  = lower

#     A = sp.vstack([A0, A_ub, A_lb], format='csr')
#     b = np.concatenate([np.zeros(k), b_ub, b_lb]).astype(float)
#     return NoArbConstraints(A=A, b=b)



# def project_noarb(C_interp: pd.DataFrame,
#                   nodes: np.ndarray,
#                   tau_grid: np.ndarray,
#                   m_grid: Dict[float, np.ndarray]) -> pd.DataFrame:
#     """
#     For each time slice, project c_raw to nearest c under A c >= b, 0 <= c <= 1.
#     Solve with OSQP (QP).
#     """
#     ab = build_noarb_constraints(nodes, tau_grid, m_grid)

#     A, b = ab.A, ab.b
#     n = nodes.shape[0]

#     c_raw = cp.Parameter(n)
#     c_var = cp.Variable(n)
#     constraints = [A @ c_var >= b, c_var >= 0, c_var <= 1]
#     obj = cp.Minimize(cp.sum_squares(c_var - c_raw))
#     prob = cp.Problem(obj, constraints)

#     out = []
#     for _, row in C_interp.iterrows():
#         x = row.values.astype(float)
#         c_raw.value = x
#         try:
#             prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
#         except Exception:
#             prob.solve(solver=cp.SCS, warm_start=True, eps_abs=1e-5, eps_rel=1e-5, verbose=False)
#         if prob.status not in ("optimal", "optimal_inaccurate"):
#             out.append(x)
#         else:
#             out.append(c_var.value)
#     out = np.vstack(out)
#     return pd.DataFrame(out, index=C_interp.index, columns=C_interp.columns)

def projection_fast_cvxpy(C_interp, nodes, tau_grid, m_grid, A, b):
    # 1) Build A, b once
    # Ab = build_noarb_constraints(nodes, tau_grid, m_grid)
    n = nodes.shape[0]

    # 2) Create a Parameter for the raw vector
    c_raw = cp.Parameter(n)

    # 3) Define the problem once
    c_var = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(c_var - c_raw))
    constraints = [A @ c_var >= b, c_var >= 0, c_var<=1]
    prob = cp.Problem(objective, constraints)

    # 4) Solve repeatedly with warm_start
    C_arb = []
    for row in C_interp.values:
        c_raw.value = row
        prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=False
        )
        C_arb.append(c_var.value)

    C_arb = np.vstack(C_arb)
    return pd.DataFrame(C_arb, index=C_interp.index, columns=C_interp.columns)

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

def remove_spot_jumps(df: pd.DataFrame, abs_thr=0.03, k_mad=6.0):
    """
    Drop timestamps with large spot jumps to reduce heavy tails.
    - abs_thr: absolute log-return threshold (e.g., 3% ≈ 0.03)
    - k_mad:   robust z-score threshold using MAD (≈ 6 is conservative)
    """
    work = df.copy()
    work['timestamp'] = pd.to_datetime(work['timestamp'])

    # one spot per timestamp (median across instruments)
    spot_ts = (work[['timestamp', 'underlying_price']]
               .groupby('timestamp')['underlying_price']
               .median()
               .sort_index())

    r = np.log(spot_ts).diff().dropna()                  # log-returns
    med = np.median(r.values)
    mad = np.median(np.abs(r.values - med)) + 1e-12
    sigma_rob = 1.4826 * mad                              # MAD->std

    z = np.abs((r.values - med) / sigma_rob)
    jump_times = r.index[(np.abs(r.values) > abs_thr) | (z > k_mad)]

    if len(jump_times) > 0:
        print(f"De-jump: removing {len(jump_times)} timestamps "
              f"(abs_thr={abs_thr:.3f}, k_mad={k_mad:.1f})")
    else:
        print("De-jump: no jump timestamps detected.")

    return work[~work['timestamp'].isin(jump_times)].copy()

def remove_option_price_jumps(C_df: pd.DataFrame,
                              rel_thr=0.35,      # relative jump threshold (e.g., 35%)
                              k_mad=8.0,         # robust z-score on abs diffs
                              abs_thr=None,      # optional absolute diff threshold (same units as C_df)
                              frac_nodes=0.05    # drop time if > this frac of nodes jump
                             ) -> pd.DataFrame:
    """
    Drop timestamps where option prices exhibit jumps across nodes.
    - rel_thr:    flag cell if |ΔC| / (|C_{t-1}|+eps) > rel_thr
    - k_mad:      flag cell if robust z on |ΔC| exceeds k_mad (per node)
    - abs_thr:    optional absolute |ΔC| threshold
    - frac_nodes: drop the whole time if fraction of flagged cells > frac_nodes
    """
    C = C_df.sort_index()
    if C.shape[0] < 3:
        return C_df

    # Absolute and relative one-step changes
    dC = C.diff().iloc[1:]                    # (T-1, N)
    C_prev = C.iloc[:-1].abs().values + 1e-12
    rel = (dC.abs() / C_prev)                 # (T-1, N)

    # Robust per-node scale of |ΔC|
    a = dC.abs()
    med = a.median(axis=0)
    mad = (a - med).abs().median(axis=0) + 1e-12
    sigma_rob = 1.4826 * mad
    z = (a - med.values) / sigma_rob.values

    # Cell-level flags
    jump_cell = (rel > rel_thr) | (z > k_mad)
    if abs_thr is not None:
        jump_cell |= (a > abs_thr)

    # Time rows to drop if too many nodes jump
    frac = jump_cell.mean(axis=1)
    drop_times = frac.index[frac > frac_nodes]
    if len(drop_times) > 0:
        print(f"Option de-jump: removing {len(drop_times)} timestamps "
              f"({100.0*len(drop_times)/len(C):.1f}%) "
              f"[rel_thr={rel_thr:.2f}, k_mad={k_mad:.1f}, frac_nodes>{frac_nodes:.2%}]")
    return C_df.loc[~C_df.index.isin(drop_times)].copy()