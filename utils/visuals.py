import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import scipy.sparse as sp
from itertools import combinations
from scipy.spatial import ConvexHull
import cvxpy as cp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon

def _prepare_nodes(nodes):
    """Return arrays τ, m from nodes (N,2)."""
    tau = nodes[:, 0].astype(float)
    m   = nodes[:, 1].astype(float)
    return tau, m

def _sym_clim(arr, q=0.995):
    """Symmetric color limits based on robust quantile."""
    a = np.asarray(arr, float)
    s = np.quantile(np.abs(a), q)
    s = float(s if s > 0 else np.max(np.abs(a)) + 1e-12)
    return (-s, s)

def _format_tau(x, _pos):
    # show τ in years with 2 decimals
    return f"{x:.2f}"

def _format_m(x, _pos):
    # log-moneyness ticks in plain number, or as % moneyness if you like:
    return f"{x:.2f}"

def _format_loading(x, _pos):
    return f"{x:.2f}"

def _plot_one_surface(ax, tau, m, z,
                      cmap="coolwarm",
                      elev=25, azim=230,
                      linewidth=0.1, antialiased=True,
                      edgecolor="k", alpha=1.0,
                      clim=None, title=None):
    # triangular surface over irregular lattice
    tri = ax.plot_trisurf(tau, m, z,
                          cmap=cmap, linewidth=linewidth,
                          antialiased=antialiased, edgecolor=edgecolor,
                          alpha=alpha)
    if clim is not None:
        tri.set_clim(*clim)
    ax.view_init(elev=elev, azim=azim)
    # axes labels & formatting
    ax.set_xlabel(r"$\tau$ (years)", labelpad=8)
    ax.set_ylabel(r"$m=\log(K/F)$", labelpad=10)
    ax.set_zlabel("loading", labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_tau))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_m))
    ax.zaxis.set_major_formatter(FuncFormatter(_format_loading))
    # gentle background grid feel via thin panes
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.4
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.4
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.4
    if title:
        ax.set_title(title, pad=10, fontsize=10)
    return tri

def _plot_one_heat(ax, tau, m, z, cmap="coolwarm", clim=None):
    # 2-D triangulation heatmap for the same data (top‑down view)
    t = ax.tripcolor(tau, m, z, shading="gouraud", cmap=cmap)
    if clim is not None:
        t.set_clim(*clim)
    ax.set_xlabel(r"$\tau$ (years)")
    ax.set_ylabel(r"$m=\log(K/F)$")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_tau))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_m))
    return t

def _standardize_factor_sign(Gi, tau, m):
    """
    Flip the sign of the factor so that the average loading over small-moneyness,
    medium-τ is positive (purely for visual consistency).
    """
    # pick middle 40% τ and m
    t_lo, t_hi = np.quantile(tau, [0.3, 0.7])
    m_lo, m_hi = np.quantile(m,   [0.3, 0.7])
    mask = (tau >= t_lo) & (tau <= t_hi) & (m >= m_lo) & (m <= m_hi)
    sgn = np.sign(np.nanmean(Gi[mask])) or 1.0
    return Gi * sgn

def plot_factors_figure_731_style(
        out,
        which=("dyn", "stat", "sa"),
        elev=25, azim=230,
        cmap="coolwarm",
        with_heatmaps=True,
        save_prefix=None, dpi=180
    ):
    """
    Parameters
    ----------
    out : Algo1Outputs
        Output from algorithm1_pipeline(...)
    which : tuple
        Any of ("dyn", "stat", "sa") to render dynamic, statistical, static‑arb.
    elev, azim : float
        Camera angles to match the paper figure.
    cmap : str
        Diverging colormap (paper‑like). "coolwarm" works well.
    with_heatmaps : bool
        Add a small 2‑D panel under each 3‑D surface.
    save_prefix : str or None
        If provided, saves PNGs like f"{save_prefix}_dyn1.png", etc.
    """

    tau, m = _prepare_nodes(out.nodes_sub)

    families = []
    if "dyn" in which and out.G_dyn is not None and len(out.G_dyn) > 0:
        families.append(("Dynamic factor", np.asarray(out.G_dyn)))
    if "stat" in which and out.G_stat is not None and len(out.G_stat) > 0:
        families.append(("Statistical factor", np.asarray(out.G_stat)))
    if "sa" in which and out.G_sa is not None and len(out.G_sa) > 0:
        families.append(("Static‑arb factor", np.asarray(out.G_sa)))

    for fam_name, G in families:
        # ensure shape (k, N)
        if G.ndim == 1:
            G = G[None, :]
        # consistent symmetric color limits per family
        clim = _sym_clim(G, q=0.995)

        for i, Gi in enumerate(G, start=1):
            Gi = _standardize_factor_sign(Gi, tau, m)

            # layout: surface (and optional heatmap beneath)
            if with_heatmaps:
                fig = plt.figure(figsize=(6.4, 6.8))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.35], hspace=0.15)
                ax3d = fig.add_subplot(gs[0, 0], projection="3d")
                ax2d = fig.add_subplot(gs[1, 0])
            else:
                fig = plt.figure(figsize=(6.4, 4.8))
                ax3d = fig.add_subplot(111, projection="3d")
                ax2d = None

            title = f"{fam_name} {i}"
            tri = _plot_one_surface(ax3d, tau, m, Gi, cmap=cmap,
                                    elev=elev, azim=azim, clim=clim,
                                    title=title, linewidth=0.05, edgecolor="k")

            cbar = fig.colorbar(tri, ax=ax3d, fraction=0.04, pad=0.03)
            cbar.formatter = FuncFormatter(_format_loading)
            cbar.update_ticks()

            if ax2d is not None:
                t = _plot_one_heat(ax2d, tau, m, Gi, cmap=cmap, clim=clim)
                cbar2 = fig.colorbar(t, ax=ax2d, fraction=0.04, pad=0.02, orientation="horizontal")
                cbar2.formatter = FuncFormatter(_format_loading)
                cbar2.update_ticks()

            # tight layout with small margins (paper‑ish)
            plt.tight_layout()
            if save_prefix:
                out_dir = Path("outputs/visuals")
                fname = out_dir / f"{save_prefix}_{fam_name.split()[0].lower()}{i}.png"
                plt.savefig(fname, dpi=dpi, bbox_inches="tight")
            plt.show()



def plot_G0(out):
    tau, m = out.nodes_sub[:, 0], out.nodes_sub[:, 1]
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(tau, m, out.G0, cmap="coolwarm", linewidth=0.05)
    ax.view_init(elev=25, azim=230)
    ax.set_title("G0 (baseline surface)")
    plt.show()





# ---------- Factor retrieval ----------
def get_three_factors(out):
    """Return (Gs, Ga, Gd) = (statistical, static-arb, dynamic-arb) bases."""
    def _pick(arr):
        if isinstance(arr, (list, tuple)):
            return np.asarray(arr[0], float).ravel()
        return np.asarray(arr, float).ravel()
    Gs = _pick(out.G_stat)
    Ga = _pick(out.G_sa)
    # adjust name here if your object differs (e.g., out.G_dyn, out.G_da, etc.)
    Gd = _pick(out.G_dyn)
    return Gs, Ga, Gd


# ---------- Polyhedron construction ----------
def polyhedron_vertices_3d(H, h, tol_feas=5e-3, tol_det=1e-12):
    """Enumerate vertices of {z: H z >= h} by triplet plane intersections."""
    m = H.shape[0]
    pts = []
    for i, j, k in combinations(range(m), 3):
        A = np.vstack([H[i], H[j], H[k]])
        b = np.array([h[i], h[j], h[k]])
        if abs(np.linalg.det(A)) < tol_det:
            continue
        z = np.linalg.solve(A, b)
        if np.all(H @ z >= h - tol_feas):
            pts.append(z)
    if not pts:
        return np.empty((0, 3))
    P = np.asarray(pts)
    key = np.round(P / max(tol_feas, 1e-8), 0).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def static_noarb_polyhedron_3d(out, tol_feas=5e-3):
    """Project A c >= b through c = G0 + M z with M=[Gs,Ga,Gd]."""
    A = out.A.toarray() if sp.issparse(out.A) else np.asarray(out.A, float)
    b = np.asarray(out.b, float).ravel()
    G0 = np.asarray(out.G0, float).ravel()

    Gs, Ga, Gd = get_three_factors(out)
    M = np.column_stack([Gs, Ga, Gd])     # (N,3)

    H = A @ M
    h = b - A @ G0
    keep = np.linalg.norm(H, axis=1) > 1e-12
    H, h = H[keep], h[keep]

    V = polyhedron_vertices_3d(H, h, tol_feas=tol_feas, tol_det=1e-12)
    return V, H, h, M, G0


# ---------- Robust scaling ----------
def scale_problem(M, H, h):
    """
    Scale columns of M to unit-2-norm. Then row-normalize H (and h accordingly).
    Improves conditioning; makes interior margin ε comparable across faces.
    """
    # Column scaling for M
    col_norms = np.linalg.norm(M, axis=0)
    col_norms[col_norms == 0.0] = 1.0
    M_s = M / col_norms

    # Row scaling for H (and h)
    row_norms = np.linalg.norm(H, axis=1)
    row_norms[row_norms == 0.0] = 1.0
    H_s = H / row_norms[:, None]
    h_s = h / row_norms

    return M_s, H_s, h_s, col_norms, row_norms


# ---------- Constrained decoding (no reprojection) ----------
def constrained_scores_3d_strict(C, out, base_eps=1e-3, tol_check=0.0, max_backoffs=6):
    """
    Decode factor scores xi_t strictly inside the polytope **without projection**:
        minimize ||M z - (C_t - G0)||^2   s.t.   H z >= h + eps
    We adapt eps (interior margin). If infeasible, back off eps -> eps/2 until solvable.

    Returns xi (T,3) in original (unscaled) coordinates and feasibility margins.
    """
    # Build polyhedron, then scale for numerics
    V, H, h, M, G0 = static_noarb_polyhedron_3d(out, tol_feas=base_eps)
    M_s, H_s, h_s, col_norms, _ = scale_problem(M, H, h)

    T, N = C.shape
    xi = np.zeros((T, 3))
    feas = np.zeros(T)
    # Parameters/variables for QP in scaled coordinates
    z = cp.Variable(3)
    y = cp.Parameter(N)

    for t in range(T):
        # Work in scaled coordinates
        y.value = (C[t] - G0)

        # Adaptive interior margin
        eps = base_eps
        solved = False
        for _ in range(max_backoffs + 1):
            constraints = [H_s @ z >= h_s + eps]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(M_s @ z - y)), constraints)
            # prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=2000, verbose=False)
            prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000, verbose=False, polish=True)
            if prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
            eps *= 0.5  # back off and try again

        if not solved:
            # Last resort: try zero interior margin (still no projection)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(M_s @ z - y)),
                              [H_s @ z >= h_s])
            # prob.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9, feastol=1e-9, max_iters=4000, verbose=False)
            prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000, verbose=False, polish=True)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"Decoding failed at t={t}. Try relaxing constraints or check inputs.")

        z_s = z.value
        # Map back to original coordinates (undo column scaling)
        xi[t] = z_s / col_norms
        feas[t] = np.min(H @ xi[t] - h)  # exact margin in original coords

        # Optional: enforce check (no projection), just assert
        if feas[t] < -tol_check:
            # If you want a hard stop when any violate:
            # raise RuntimeError(f"Strict feasibility violated at t={t}: margin={feas[t]:.3e}")
            pass

    return xi, V, feas, (H, h)


# ---------- Plotting ----------
def plot_threefactor_polyhedron_strict(out, use_test=False, base_eps=1e-3, tol_check=0.0,
                                       zoom_factor=0.8):
    """
    Plot scores strictly inside the static no-arb polyhedron, solved without reprojection.
    zoom_factor < 1 shrinks the axis ranges around the factor cloud.
    """
    C = (out.C_test.values if use_test else out.C_train.values)
    xi, V, feas, (H, h) = constrained_scores_3d_strict(
        C, out, base_eps=base_eps, tol_check=tol_check)

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection='3d')

    if V.shape[0] >= 4:
        hull = ConvexHull(V)
        faces = [V[s] for s in hull.simplices]
        poly = Poly3DCollection(faces, alpha=0.25,
                                facecolor="#98FB98", edgecolor="k", linewidths=0.5)
        ax.add_collection3d(poly)
    else:
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=20, c="#98FB98")

    ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], s=10, c="k", alpha=0.65, depthshade=True)

    ax.set_xlabel(r"$\xi_{\mathrm{stat}}$")
    ax.set_ylabel(r"$\xi_{\mathrm{static\_arb}}$")
    ax.set_zlabel(r"$\xi_{\mathrm{dyn\_arb}}$")
    ax.set_title("3-factor scores strictly inside static no-arb polyhedron (zoomed)")

    # --- tighter axis limits around the factor scores ---
    mins = xi.min(axis=0)
    maxs = xi.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = (maxs - mins) * zoom_factor
    pad = 0.05 * np.where(span > 0, span, 1.0)

    ax.set_xlim(center[0] - 0.5 * span[0] - pad[0], center[0] + 0.5 * span[0] + pad[0])
    ax.set_ylim(center[1] - 0.5 * span[1] - pad[1], center[1] + 0.5 * span[1] + pad[1])
    ax.set_zlim(center[2] - 0.5 * span[2] - pad[2], center[2] + 0.5 * span[2] + pad[2])

    plt.tight_layout()
    return fig, ax, (xi, V, feas)

def plot_threefactors_poly(out, use_test = True, base_eps=1e-3, tol_check=0):

    fig, ax, (xi, V, feas) = plot_threefactor_polyhedron_strict(out, use_test,
                                                                base_eps, tol_check)
    plt.show()



def polygon_factor_space(out, tol=1e-12):
    # Two factors only
    G0 = out.G0
    Gs = out.G_stat[0]        # “statistical accuracy” basis
    Ga = out.G_sa[0]          # your single “static-arb” basis

    A = out.A.toarray() if sp.issparse(out.A) else np.asarray(out.A, float)
    b = np.asarray(out.b, float).ravel()

    # Project A c ≥ b through c = G0 + ξ_s Gs + ξ_a Ga
    H = A @ np.vstack([Gs, Ga]).T          # (m,2)
    h = b - A @ G0                         # (m,)

    keep = np.linalg.norm(H, axis=1) > tol
    H, h = H[keep], h[keep]

    # Enumerate pairwise intersections of H_i·ξ = h_i, keep those satisfying all inequalities
    pts = []
    for i in range(H.shape[0]):
        A1,B1 = H[i]; C1 = h[i]
        for j in range(i+1, H.shape[0]):
            A2,B2 = H[j]; C2 = h[j]
            det = A1*B2 - A2*B1
            if abs(det) < 1e-14: 
                continue
            x = (C1*B2 - C2*B1)/det
            y = (A1*C2 - A2*C1)/det
            if np.all(H @ np.array([x,y]) >= h - 5e-3):   # use SAME tol as PSAS
                pts.append([x,y])
    return np.asarray(pts), H, h


def constrained_scores_2d(C: np.ndarray, out, tol=5e-3):
    # C: (T,N) repaired prices (use out.C_train.values or out.C_test.values)
    G0 = out.G0
    Gs = out.G_stat[0]
    Ga = out.G_sa[0]
    M  = np.column_stack([Gs, Ga])        # (N,2)

    pts, H, h = polygon_factor_space(out, tol=1e-12)

    T, N = C.shape
    xi = np.zeros((T,2))
    z  = cp.Variable(2)
    # Pre-build problem: min ||M z - y||^2  s.t. H z ≥ h
    # We'll update y per time.
    y_param = cp.Parameter(N)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(M @ z - y_param)),
                      [H @ z >= h - tol])  # consistent tolerance
    for t in range(T):
        y_param.value = C[t] - G0
        prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
        if prob.status not in ("optimal","optimal_inaccurate"):
            # tiny fallback: unconstrained LS then project to polytope
            z0 = np.linalg.lstsq(M, y_param.value, rcond=None)[0]
            # quick projection: min ||z - z0||^2 s.t. H z ≥ h
            z_var = cp.Variable(2)
            prob2 = cp.Problem(cp.Minimize(cp.sum_squares(z_var - z0)), [H @ z_var >= h - tol])
            prob2.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6, verbose=False)
            xi[t] = z_var.value
        else:
            xi[t] = z.value
    return xi, pts


def plot_twofactor_polygon(out, use_test=False):
    C = (out.C_test.values if use_test else out.C_train.values)
    xi, pts = constrained_scores_2d(C, out, tol=5e-3)   # same tol as PSAS
    fig, ax = plt.subplots(figsize=(6.6,3))
    ax.scatter(xi[:,0], xi[:,1], s=6, c="k", alpha=0.45, linewidths=0)

    if pts.shape[0] >= 3:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        ax.add_patch(Polygon(poly, closed=True, facecolor="#98FB98", edgecolor="none", alpha=0.35))
        cyc = np.vstack([poly, poly[0]])
        ax.plot(cyc[:,0], cyc[:,1], "r--", lw=2.0)
    ax.set_xlabel(r"$\xi_s$"); ax.set_ylabel(r"$\xi_a$")
    ax.set_title("2-factor scores within static no-arb polygon")
    return fig, ax

