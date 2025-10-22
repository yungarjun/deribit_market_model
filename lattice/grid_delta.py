import numpy as np
import pandas as pd
from typing import Dict, Sequence, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

@dataclass
class Lattice:
    nn: NearestNeighbors
    nodes: np.ndarray               # shape [Nnodes, 2] with columns (tau, m_rep)
    tau_grid: np.ndarray            # shape [Nτ]
    m_grid: Dict[float, np.ndarray] # τ -> array of representative m (aligned to deltas)
    deltas: np.ndarray              # target deltas, shape [NΔ]

# ---------- helpers ----------------------------------------------------------

def _bs_call_delta_from_m_iv(m: np.ndarray, iv: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Black (forward) call delta with inputs (m = ln(K/F)), σ, τ; Δ = N(d1)."""
    # Avoid divide-by-zero
    tau = np.maximum(tau, 1e-10)
    iv  = np.maximum(iv,  1e-10)
    d1  = (-m + 0.5 * (iv**2) * tau) / (iv * np.sqrt(tau))
    return norm.cdf(d1)

def _estimate_delta(df: pd.DataFrame) -> np.ndarray:
    """Return the delta from df['greeks'] if present, else compute or fallback."""
    if 'delta' in df.columns:
        return df['delta'].to_numpy(dtype=float)

    # --- NEW: handle dict-like 'greeks' column ---
    if 'greeks' in df.columns:
        try:
            # Works if greeks is a dict or stringified dict (e.g., JSON)
            def extract_delta(x):
                if isinstance(x, dict):
                    return x.get('delta', np.nan)
                if isinstance(x, str):
                    import ast
                    try:
                        d = ast.literal_eval(x)
                        return d.get('delta', np.nan) if isinstance(d, dict) else np.nan
                    except Exception:
                        return np.nan
                return np.nan
            return df['greeks'].apply(extract_delta).astype(float).to_numpy()
        except Exception:
            pass

    # fallback: derive from implied vol if available
    if {'iv','tau','m'}.issubset(df.columns):
        return _bs_call_delta_from_m_iv(df['m'].to_numpy(float),
                                        df['iv'].to_numpy(float),
                                        df['tau'].to_numpy(float))
    return np.full(len(df), np.nan, dtype=float)


def _cluster_tau_logspace(taus: np.ndarray, n_tau: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster τ in log-space; return sorted centers and per-row cluster index mapping."""
    x = np.log(np.clip(taus, 1e-10, None)).reshape(-1,1)
    x_unique = np.unique(x)
    if len(x_unique) <= n_tau:
        centers = np.unique(taus)
        order   = np.argsort(centers)
        tau_grid = centers[order]
        idx = np.searchsorted(tau_grid, taus, side='left')
        idx = np.clip(idx, 0, len(tau_grid)-1)
        return tau_grid.astype(float), idx.astype(int)
    km = KMeans(n_clusters=n_tau, random_state=random_state, n_init=8).fit(x_unique.reshape(-1,1))
    c  = np.exp(km.cluster_centers_.ravel())
    order = np.argsort(c)
    tau_grid = c[order]
    raw = KMeans(n_clusters=n_tau, random_state=random_state, n_init=8).fit_predict(x)
    # map labels to sorted order
    old2new = {int(old): int(new) for new, old in enumerate(order)}
    idx = np.vectorize(old2new.get)(raw).astype(int)
    return tau_grid.astype(float), idx

# ---------- NEW: build a fixed-Δ lattice ------------------------------------

def build_lattice_grid(
    df: pd.DataFrame,
    n_tau: int = 6,
    deltas: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    tau_targets: Optional[Sequence[float]] = None,   # if you want to force τ grid
    random_state: int = 0,
    min_obs_per_node: int = 10,
    n_m: int = None
) -> Lattice:
    """
    Build a *delta-based* lattice:
      - τ grid: fixed 'tau_targets' or learned via KMeans in log τ (n_tau).
      - Δ set: fixed deltas.
    For each (τ, Δ) we compute a *representative m* as the median m of the closest-Δ quotes
    across the whole sample. Returns nodes in (τ, m_rep) so you can keep the (τ,m) geometry.
    """
    work = df.copy()
    if 'tau' not in work or 'm' not in work:
        raise ValueError("df must have columns 'tau' and 'm' (log-moneyness).")
    work['tau'] = work['tau'].astype(float)
    work['m']   = work['m'].astype(float)

    # τ grid
    if tau_targets is not None and len(tau_targets) > 0:
        tau_grid = np.sort(np.asarray(tau_targets, float))
        # nearest τ-bucket per row
        τ_idx = np.searchsorted(tau_grid, work['tau'].to_numpy(), side='left')
        τ_idx = np.clip(τ_idx, 0, len(tau_grid)-1).astype(int)
    else:
        tau_grid, τ_idx = _cluster_tau_logspace(work['tau'].to_numpy(), n_tau, random_state)

    work['tau_bucket'] = τ_idx
    Δ_targets = np.asarray(deltas, float)

    # estimate Δ or prepare m-quantile proxy per bucket
    work['_delta_est'] = _estimate_delta(work)

    # representative m per (τ_bkt, Δ*)
    m_grid: Dict[float, np.ndarray] = {}
    nodes  = []
    for i, τ in enumerate(tau_grid):
        sub = work[work['tau_bucket'] == i]
        if len(sub) == 0:
            # empty bucket: fallback to global m range
            m_rep = np.linspace(*np.percentile(work['m'], [10,90]), len(Δ_targets))
            m_grid[float(τ)] = m_rep
            nodes.extend([[float(τ), float(m)] for m in m_rep])
            continue

        # If we have Δ estimates, choose nearest; else proxy Δ by m-quantiles
        if np.isfinite(sub['_delta_est']).any():
            # for each Δ*, take rows with minimal |Δ-Δ*| per timestamp, then aggregate median m
            sub = sub.assign(ts=pd.to_datetime(sub['timestamp']))
            reps = []
            for Δ in Δ_targets:
                s = (sub.assign(delta_err=np.abs(sub['_delta_est'] - Δ))
                        .sort_values(['ts','delta_err','stats_volume_usd'], ascending=[True, True, False])
                        .drop_duplicates(['ts']))
                if len(s) >= max(1, min_obs_per_node):
                    reps.append(np.median(s['m'].to_numpy()))
                else:
                    # fallback inside bucket by m-quantile ~ monotone mapping Δ↔m
                    q = 1.0 - float(Δ)
                    reps.append(np.quantile(sub['m'].to_numpy(), q))
            m_rep = np.asarray(reps, float)
        else:
            # Pure fallback: deterministic m-quantiles inside the bucket
            qs = 1.0 - Δ_targets  # larger Δ ≈ lower m
            m_rep = np.quantile(sub['m'].to_numpy(), qs)

        m_grid[float(τ)] = m_rep
        nodes.extend([[float(τ), float(m)] for m in m_rep])

    nodes = np.asarray(nodes, float)
    nn = NearestNeighbors(n_neighbors=1).fit(nodes)
    return Lattice(nn=nn, nodes=nodes, tau_grid=tau_grid, m_grid=m_grid, deltas=Δ_targets)

# ---------- NEW: apply the fixed-Δ lattice ----------------------------------

def apply_lattice(
    df: pd.DataFrame,
    nn: NearestNeighbors,
    nodes: np.ndarray,
    tau_grid: np.ndarray,
    m_grid: Dict[float, np.ndarray],
    deltas: Optional[Sequence[float]] = None,
    top_K: int = 50,
    fill_method: str = "linear",
):
    """
    Snap quotes to *delta* nodes:
      - map each row to nearest τ-bucket,
      - compute Δ (or proxy by m-quantiles),
      - map to nearest target-Δ within that τ-bucket,
      - pivot the most liquid quote per (timestamp, node),
      - interpolate in time and fill.

    Returns: (C_interp, nodes_sub, tau_sub, m_sub)
    """
    work = df.copy()
    work['timestamp'] = pd.to_datetime(work['timestamp'])
    work['tau'] = work['tau'].astype(float)
    work['m']   = work['m'].astype(float)

    # τ-bucket per row
    τ_idx = np.searchsorted(np.asarray(tau_grid), work['tau'].to_numpy(), side='left')
    τ_idx = np.clip(τ_idx, 0, len(tau_grid)-1).astype(int)
    work['tau_bucket'] = τ_idx

    # Δ estimate
    work['_delta_est'] = _estimate_delta(work)

    # target Δ set inferred from m_grid order per τ (kept in build step)
    if deltas is None:
        # infer count from first τ entry
        nΔ = len(next(iter(m_grid.values())))
        Δ_targets = np.linspace(0.1, 0.9, nΔ)
    else:
        Δ_targets = np.asarray(deltas, float)

    # assign node index by (τ-bucket, nearest Δ*)
    # Build lookup: for each τ-bucket, its Δ_targets index order matches m_grid[τ]
    nτ = len(tau_grid)
    nΔ = len(Δ_targets)

    # When Δ not available, proxy by ranking m within each τ-bucket snapshot
    if not np.isfinite(work['_delta_est']).any():
        # rank m within each (timestamp, τ-bucket); higher Δ ≈ lower m
        work['_rank_m'] = (work
            .groupby(['timestamp','tau_bucket'])['m']
            .rank(method='first', ascending=True))  # low m -> rank 1
        # map rank to Δ index by quantiles
        def _rank_to_delta_idx(g):
            n = len(g)
            # target cut points by Δ quantiles
            cuts = (1.0 - Δ_targets) * n
            return np.searchsorted(cuts, g['_rank_m'].to_numpy()-1, side='right').clip(0, nΔ-1)
        delta_idx = []
        for _, g in work.groupby(['timestamp','tau_bucket'], sort=False):
            delta_idx.append(_rank_to_delta_idx(g))
        work['_delta_idx'] = np.concatenate(delta_idx)
    else:
        # nearest Δ target
        work['_delta_idx'] = np.argmin(
            np.abs(work['_delta_est'].to_numpy()[:,None] - Δ_targets[None,:]),
            axis=1
        )

    work['node_idx'] = work['tau_bucket'] * nΔ + work['_delta_idx']
    # Attach representative lattice coordinates (τ, m_rep) for reference / NN
    τvals = tau_grid[work['tau_bucket'].to_numpy()]
    mrep  = np.array([ m_grid[float(t)][j] for t, j in zip(τvals, work['_delta_idx']) ], dtype=float)
    work = work.assign(lattice_tau=τvals, lattice_m=mrep)

    # choose most liquid per timestamp/node
    best = (work.sort_values(['stats_volume_usd'], ascending=False)
                 .drop_duplicates(['timestamp','node_idx']))

    # pick most-covered nodes
    top_nodes = (best['node_idx'].value_counts()
                        .nlargest(top_K)
                        .index.astype(int))
    sub = best[best['node_idx'].isin(top_nodes)]

    # pivot to sparse matrix of normalized calls
    C_sparse = sub.pivot_table(values='c_norm', index='timestamp', columns='node_idx')

    # drop never observed & rebuild sub-nodes / sub-grids
    never = C_sparse.columns[C_sparse.isna().all()]
    C_sparse = C_sparse.drop(columns=never)
    present  = C_sparse.columns.astype(int).to_numpy()
    nodes_sub = nodes[present]

    tau_sub = np.unique(nodes_sub[:,0])
    # rebuild m_sub dict (keep order aligned to Δ_targets)
    m_sub: Dict[float, np.ndarray] = {}
    for τ in tau_sub:
        mask = np.isclose(nodes_sub[:,0], τ)
        m_sub[float(τ)] = nodes_sub[mask,1]

    # interpolate in *time* only
    C_interp = (C_sparse
                  .sort_index()
                  .interpolate(method=fill_method, axis=0)
                  .ffill()
                  .bfill())

    return C_interp, nodes_sub, tau_sub, m_sub
