import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Tuple
import pandas as pd
from type import Lattice
from sklearn.preprocessing import StandardScaler

def build_lattice_grid(df: pd.DataFrame, n_tau=5, n_m=5, random_state=0) -> Lattice:
    """
    Learn a sorted τ‑grid via KMeans on τ (unique), then robust m‑grid (per τ) via percentiles.
    Build nodes (Cartesian product of each τ row’s m‑grid) and a 1‑NN snapper in (τ, m).
    """
    work = df.copy()
    # taus_unique = np.sort(work['tau'].unique()).reshape(-1, 1)
    # # km = KMeans(n_clusters=n_tau, random_state=random_state).fit(taus_unique)

    # # centers = km.cluster_centers_.flatten()
    # order = np.argsort(taus_unique)
    # tau_grid = taus_unique[order]

    # # Relabel to sorted
    # raw_lab = km.predict(work[['tau']].values)
    # old2new = {old: new for new, old in enumerate(order)}
    # work['tau_cluster'] = np.array([old2new[l] for l in raw_lab], dtype=int)

    # sorted unique τ as the grid (1-D float array)
    tau_grid = np.sort(work['tau'].unique().astype(float))

    # Assign cluster index = position in tau_grid for each row
    tau_to_idx = {float(t): i for i, t in enumerate(tau_grid)}
    work['tau_cluster'] = work['tau'].map(tau_to_idx).astype(int)

    # m‑grid per τ (robust 1–99 percentiles)
    m_grid: Dict[float, np.ndarray] = {}
    for i, τ in enumerate(tau_grid):
        sub = work[work['tau_cluster'] == i]
        if len(sub) == 0:
            lo, hi = np.percentile(work['m'], [1, 99])
        else:
            lo, hi = np.percentile(sub['m'], [1, 99])
        m_grid[τ] = np.linspace(lo, hi, n_m)

    # nodes (τ, m) and NN
    nodes = np.vstack([[τ, m] for τ in tau_grid for m in m_grid[τ]])
    nn = NearestNeighbors(n_neighbors=1).fit(nodes)
    return Lattice(nn=nn, nodes=nodes, tau_grid=tau_grid, m_grid=m_grid)

# def build_lattice_grid(df, n_tau=20, n_m=5, top_K=50):
#     """
#     Learn a *sorted* τ-grid and consistent labels, then build m-grids, nodes, and a 1-NN snapper.
#     """
#     df = df.copy()

#     # --- 1) KMeans on τ, then sort centers and RELABEL ---
#     unique_taus = np.sort(df['tau'].unique()).reshape(-1, 1)
#     # km = KMeans(n_clusters=n_tau, random_state=0).fit(unique_taus)

#     # centers = km.cluster_centers_.flatten()          # unsorted centers
#     # order   = np.argsort(centers)                    # permutation that sorts them
#     tau_grid = centers[order]                        # sorted centers, shape [n_tau]

#     # map old label -> new sorted label
#     old2new = {old: new for new, old in enumerate(order)}

#     # label every row via KMeans.predict, then remap to sorted indices
#     raw_lab = km.predict(df[['tau']].values)         # 0..n_tau-1 (arbitrary order)
#     df['tau_cluster'] = np.array([old2new[l] for l in raw_lab], dtype=int)

#     # --- 2) m-grid per sorted τ-cluster (use robust percentiles) ---
#     m_grid = {}
#     for i, τ in enumerate(tau_grid):
#         sub = df[df['tau_cluster'] == i]
#         if len(sub) == 0:
#             # fallback if cluster empty (rare): borrow global range
#             lo, hi = np.percentile(df['m'], [1, 99])
#         else:
#             lo, hi = np.percentile(sub['m'], [1, 99])
#         m_grid[τ] = np.linspace(lo, hi, n_m)

#     # --- 3) build nodes and a 1-NN snapper on the rectangular-in-index lattice ---
#     nodes = np.vstack([[τ, m] for τ in tau_grid for m in m_grid[τ]])

#     from sklearn.neighbors import NearestNeighbors
#     nn = NearestNeighbors(n_neighbors=1).fit(nodes)

#     return Lattice(nn=nn, nodes=nodes, tau_grid=tau_grid, m_grid=m_grid)

def apply_lattice(df, nn, nodes, tau_grid, m_grid, 
                  top_K=50, fill_method="linear"):
    """
    Given any df plus a prebuilt (nn, nodes, tau_grid, m_grid),
    snap-to-nodes, pick top_K by liquidity, pivot → C_sparse,
    drop never-observed, rebuild sub-nodes & sub-m_grid,
    then interpolate & fill → C_interp.
    """
    # snap each (τ,m) via nn
    pts = df[['tau','m']].values
    idx = nn.kneighbors(pts, return_distance=False)[:,0]
    df = df.assign(node_idx=idx,
                   lattice_tau=nodes[idx,0],
                   lattice_m=nodes[idx,1])
    
    # pick most liquid per timestamp/node
    best = (
      df.sort_values('stats_volume_usd', ascending=False)
        .drop_duplicates(['timestamp','node_idx'])
        # .assign(timestamp=pd.to_datetime(df['timestamp']))
    )

    best['timestamp'] = pd.to_datetime(best['timestamp'])
    
    # pivot to sparse C
    # top_nodes = df['node_idx'].value_counts().nlargest(top_K).index
    # sub       = best[best['node_idx'].isin(top_nodes)]
    top_nodes = best['node_idx'].value_counts().nlargest(top_K).index
    sub       = best[best['node_idx'].isin(top_nodes)]
    C_sparse  = sub.pivot_table('c_norm','timestamp','node_idx')
    
    # drop never observed & rebuild sub-nodes, sub grids
    never_obs = C_sparse.columns[C_sparse.isna().all()]
    C_sparse  = C_sparse.drop(columns=never_obs)
    present   = C_sparse.columns.astype(int).to_numpy()
    nodes_sub = nodes[present]
    tau_sub   = np.unique(nodes_sub[:,0])
    m_sub     = {τ: sorted(nodes_sub[nodes_sub[:,0]==τ,1]) for τ in tau_sub}
    
    # interpolate & fill
    C_interp = (C_sparse
                  .interpolate(method=fill_method,axis=0)
                  .ffill().bfill())
    
    return C_interp, nodes_sub, tau_sub, m_sub

# def build_lattice_grid(df: pd.DataFrame, n_tau=5, n_m=7, random_state=0,
#                        p_lo=1.0, p_hi=99.0) -> Lattice:
#     """
#     Shared global m-grid:
#       - tau_grid: KMeans on τ (sorted)
#       - m_global: linspace between robust percentiles of m across ALL rows
#     """
#     work = df.copy()

#     # ---- τ grid via KMeans on unique τ values ----
#     taus_unique = np.sort(work['tau'].unique()).reshape(-1, 1)
#     km = KMeans(n_clusters=n_tau, random_state=random_state).fit(taus_unique)
#     centers = km.cluster_centers_.flatten()
#     tau_grid = np.sort(centers)

#     # ---- global m grid (robust) ----
#     lo, hi = np.percentile(work['m'], [p_lo, p_hi])
#     m_global = np.linspace(lo, hi, n_m)

#     # ---- rectangular nodes: cartesian product ----
#     nodes = np.vstack([[τ, m] for τ in tau_grid for m in m_global])

#     # ---- standardize (τ,m) before NN for stable snapping ----
#     scaler_tm = StandardScaler().fit(nodes)
#     nodes_std = scaler_tm.transform(nodes)
#     nn = NearestNeighbors(n_neighbors=1).fit(nodes_std)

#     return Lattice(nn=nn, nodes=nodes, tau_grid=tau_grid, m_global=m_global, scaler_tm=scaler_tm)


# def apply_lattice(df: pd.DataFrame,
#                   lat: Lattice,
#                   top_K=50,
#                   fill_method="linear") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
#     """
#     Snap each (τ, m) to nearest node; keep the most liquid quote per (time, node);
#     choose the top_K most frequently observed nodes; interpolate in time, ffill/bfill.
#     Returns:
#         C_interp : DataFrame [time x N_sub] of normalized prices c (C/F)
#         nodes_sub, tau_sub, m_sub : the reduced lattice actually used
#     """
#     pts = df[['tau', 'm']].values
#     idx = lat.nn.kneighbors(pts, return_distance=False)[:, 0]

#     tmp = df.assign(node_idx=idx,
#                     lattice_tau=lat.nodes[idx, 0],
#                     lattice_m=lat.nodes[idx, 1])
#     # pick most liquid per (t, node)
#     best = (tmp.sort_values('stats_volume_usd', ascending=False)
#               .drop_duplicates(['timestamp', 'node_idx']))
#     best = best[['timestamp', 'node_idx', 'c_norm']].copy()

#     top_nodes = (best['node_idx'].value_counts()
#                            .nlargest(top_K)
#                            .index)
#     sub = best[best['node_idx'].isin(top_nodes)].copy()

#     C_sparse = sub.pivot_table(index='timestamp', columns='node_idx', values='c_norm')
#     # ensure strictly increasing time index
#     C_sparse = C_sparse.sort_index()

#     # drop dead columns
#     dead = C_sparse.columns[C_sparse.isna().all()]
#     C_sparse = C_sparse.drop(columns=dead)

#     present = C_sparse.columns.astype(int).to_numpy()
#     nodes_sub = lat.nodes[present]
#     tau_sub = np.unique(nodes_sub[:, 0])
#     m_sub: Dict[float, np.ndarray] = {τ: np.sort(nodes_sub[nodes_sub[:, 0] == τ, 1]) for τ in tau_sub}

#     # Interpolate in time, fill edges
#     C_interp = (C_sparse.interpolate(method=fill_method, axis=0)
#                          .ffill()
#                          .bfill())

#     return C_interp, nodes_sub, tau_sub, m_sub

# def apply_lattice_train(df: pd.DataFrame, lat: Lattice, top_K=50, fill_method="linear"):
#     """
#     Train-only version:
#       - snap to nodes
#       - keep most liquid per (t, node)
#       - choose top_K nodes by frequency (train only)
#       - return train lattice + the frozen train_keep_idx for later reuse
#     """
#     pts = df[['tau','m']].values
#     idx = lat.nn.kneighbors(pts, return_distance=False)[:,0]

#     tmp = df.assign(node_idx=idx)
#     best = (tmp.sort_values('stats_volume_usd', ascending=False)
#                .drop_duplicates(['timestamp','node_idx']))
#     top_nodes = (best['node_idx'].value_counts().nlargest(top_K).index.to_numpy())

#     C_sparse = (best[best['node_idx'].isin(top_nodes)]
#                 .pivot_table(index='timestamp', columns='node_idx', values='c_norm')
#                 .sort_index())

#     # Make sure columns are exactly the top_nodes, in that order
#     C_sparse = C_sparse.reindex(columns=top_nodes)

#     # Time-wise fill (train-only)
#     C = (C_sparse.interpolate(method=fill_method, axis=0).ffill().bfill())

#     nodes_sub = lat.nodes[top_nodes]
#     tau_sub   = np.unique(nodes_sub[:,0])
#     m_sub     = {float(t): np.sort(nodes_sub[nodes_sub[:,0]==t,1]) for t in tau_sub}

#     return C, nodes_sub, tau_sub, m_sub, top_nodes  # <-- keep_idx to reuse


# def apply_lattice_test(df: pd.DataFrame, lat: Lattice, train_keep_idx, fill_method="linear"):
#     """
#     Test-only version that REUSES the train mask (train_keep_idx).
#     Does not re-select columns. Ensures identical column order to train.
#     """
#     pts = df[['tau','m']].values
#     idx = lat.nn.kneighbors(pts, return_distance=False)[:,0]

#     tmp = df.assign(node_idx=idx)
#     best = (tmp.sort_values('stats_volume_usd', ascending=False)
#                .drop_duplicates(['timestamp','node_idx']))

#     # Pivot on ALL nodes seen, then reindex to train_keep_idx
#     C_sparse = (best.pivot_table(index='timestamp', columns='node_idx', values='c_norm')
#                      .sort_index())

#     # Force identical columns (same nodes, same order) as train
#     C_sparse = C_sparse.reindex(columns=train_keep_idx)

#     # Interpolate within TEST only (no leakage)
#     C = (C_sparse.interpolate(method=fill_method, axis=0).ffill().bfill())

#     nodes_sub = lat.nodes[train_keep_idx]
#     tau_sub   = np.unique(nodes_sub[:,0])
#     m_sub     = {float(t): np.sort(nodes_sub[nodes_sub[:,0]==t,1]) for t in tau_sub}

#     return C, nodes_sub, tau_sub, m_sub

def ensure_consistent_lattice(Ci_test, Ci_train, keep_idx, nodes_sub):
    """"
    Drops columns that are dead in the test lattice
    Because they don't appear
    """
    dead_cols = Ci_test.columns[Ci_test.isna().all(axis=0)]
    
    # shrink nodes_sub and matrices
    pos_mask = ~np.isin(keep_idx, dead_cols)
    keep_idx_2 = keep_idx[pos_mask]
    nodes_sub2 = nodes_sub[pos_mask]

    tau_sub2 = np.unique(nodes_sub2[:, 0])
    m_sub2 = {float(t): np.sort(nodes_sub2[nodes_sub2[:,0]==t, 1]) for t in tau_sub2}

    Ci_train2 = Ci_train.loc[:, keep_idx_2]
    Ci_test2  = Ci_test.loc[:,  keep_idx_2]

    return Ci_test2, Ci_train2, nodes_sub2, tau_sub2, m_sub2


def global_lattice_grid(df, n_tau, n_m):
    """ 
    A function to create a global grid of tau and moneyness. 
    Created after realising highly correlated residuals from
    the innovations of the neural SDE
    """

    all_taus = df['tau'].values
    # tau_grid = np.quantile(all_taus, np.linspace(0.05, 0.95, n_tau)) # universal grid of tau's

    # Usually taus are minimal. We can just train on all of them

    all_m = df['m'].values
    m_grid = np.linspace(np.percentile(all_m, 1), np.percentile(all_m, 99), n_m)

    # Build rectangular node
    nodes = np.array([[tau, m] for tau in all_taus for m in m_grid])

    nn = NearestNeighbors(n_neighbors=1).fit(nodes)
    idx = nn.kneighbors(df[['tau', 'm']].values, return_distance = False)[:, 0]

    return Lattice(nodes = nodes, m_grid = m_grid, tau_grid=all_taus)

