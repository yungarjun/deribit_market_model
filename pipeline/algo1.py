import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from config import Algo1Config, DecoderKind
from type import Lattice, Surfaces
from lattice.grid import build_lattice_grid, apply_lattice_train, apply_lattice_test
from lattice.noarb import build_noarb_constraints, project_noarb, reduce_constraints_geq
from surfaces.derivatives import compute_derivatives
from stage0.train import make_stage0_dataset, train_stage0, sigma_to_gamma
from stage0.nets import Stage0LogNet
from factors.zfield import build_Z
from factors.pca_blocks import pca_dyn_block, pca_stat_block, pda_from_pca, pls_block, DeepPLS
from factors.decode import reconstruct_prices, decode_static_arb_hinge
from metrics.metrics import mape, psas, vega_weighted_mape

from sklearn.decomposition import PCA


@dataclass
class Algo1Outputs:
    # grids
    nodes_sub: np.ndarray
    tau_sub: np.ndarray
    m_sub: dict
    A: object
    b: np.ndarray
    # surfaces
    C_train: pd.DataFrame
    C_test: pd.DataFrame
    # derivatives
    dCtau_train: np.ndarray
    dCm_train: np.ndarray
    d2Cm2_train: np.ndarray
    dCtau_test: np.ndarray
    dCm_test: np.ndarray
    d2Cm2_test: np.ndarray
    # stage-0
    pca_price: Optional[PCA]
    scaler0: Optional[object]
    net0: Stage0LogNet
    sigma_train: np.ndarray
    sigma_test: np.ndarray
    gamma_train: np.ndarray
    gamma_test: np.ndarray
    # factors
    G0: np.ndarray
    pca_dyn: Optional[PCA]
    G_dyn: np.ndarray
    Xi_dyn_train: np.ndarray
    Xi_dyn_test: np.ndarray
    pca_stat: Optional[PCA]
    G_stat: np.ndarray
    Xi_stat_train: np.ndarray
    Xi_stat_test: np.ndarray
    G_sa: Optional[np.ndarray]
    Xi_sa_train: Optional[np.ndarray]
    Xi_sa_test: Optional[np.ndarray]
    # reconstructions
    C_hat_train: np.ndarray
    C_hat_test: np.ndarray
    # metrics
    MAPE_train: float
    MAPE_test: float
    VW_MAPE_test: float
    VW_MAPE_train: float
    PDA_train: float
    PSAS_train: float
    PSAS_test: float


# ------------------ helpers ------------------
def _split_by_time(df: pd.DataFrame, frac=0.8):
    all_times = df['timestamp'].sort_values().unique()
    i_cut = int(frac * len(all_times))
    return all_times[:i_cut], all_times[i_cut:]


def _stage0_block(C_train, C_test, train_df, test_df, cfg: Algo1Config):
    spot_train = (train_df.drop_duplicates('timestamp').set_index('timestamp').loc[C_train.index, 'underlying_price'].values)
    spot_test  = (test_df.drop_duplicates('timestamp'). set_index('timestamp').loc[C_test.index,  'underlying_price'].values)

    sec_per_year = 365*24*3600
    tt_train = pd.to_datetime(C_train.index.values)
    tt_test  = pd.to_datetime(C_test.index.values)
    dt_train = (tt_train[1:] - tt_train[:-1]).total_seconds() / sec_per_year
    dt_test  = (tt_test[1:]  - tt_test[:-1]).total_seconds()  / sec_per_year

    pca_price = PCA(n_components=cfg.d0, random_state=cfg.seed).fit(C_train.values)
    pcs_train = pca_price.transform(C_train.values)[:, :cfg.d0]
    pcs_test  = pca_price.transform(C_test.values)[:,  :cfg.d0]

    X0, y0, dt0 = make_stage0_dataset(spot_train, pcs_train, dt_train)
    net0 = Stage0LogNet(input_dim=X0.shape[1], hidden=cfg.stage0_hidden)
    scaler0, sigma_train = train_stage0(net0, X0, y0, dt0, cfg.stage0_epochs, cfg.stage0_lr, cfg.stage0_batch, scale_X=True)

    from sklearn.preprocessing import StandardScaler
    X0t, y0t, dt0t = make_stage0_dataset(spot_test, pcs_test, dt_test)
    X0t_n = scaler0.transform(X0t) if isinstance(scaler0, StandardScaler) else X0t
    import torch
    with torch.no_grad():
        _, sigma_t = net0(torch.from_numpy(X0t_n).float())
    sigma_test = sigma_t.squeeze().cpu().numpy()

    gamma_train = sigma_to_gamma(sigma_train, S_tm1=spot_train[:-1])
    gamma_test  = sigma_to_gamma(sigma_test,  S_tm1=spot_test[:-1])
    return pca_price, net0, scaler0, sigma_train, sigma_test, gamma_train, gamma_test


# ------------------ pipelines ------------------

def run_algo1_deribit(df_raw: pd.DataFrame, cfg: Algo1Config, reference_date="2025-01-30", r=0.0, q=0.0) -> Algo1Outputs:
    from data_adapters.deribit_adapters import clean_deribit
    df = clean_deribit(df_raw, reference_date, r, q).sort_values('timestamp')

    # split by time
    train_times, test_times = _split_by_time(df)
    train_df = df[df['timestamp'].isin(train_times)].copy()
    test_df  = df[df['timestamp'].isin(test_times)].copy()

    # lattice discovery on train
    lat = build_lattice_grid(train_df, n_tau=cfg.n_tau, n_m=cfg.n_m, random_state=cfg.seed)
    # top_m = min(cfg.n_m, max(1, cfg.top_K // cfg.n_tau))
    # train
    Ci_train, nodes_sub, tau_sub, m_sub, keep_idx = apply_lattice_train(train_df, lat, top_K=cfg.top_K)
    # train
    Ci_test,  _,         _,      _      = apply_lattice_test(test_df, lat, keep_idx)


    # static projection
    ab = build_noarb_constraints(nodes_sub, tau_sub, m_sub)
    A_raw, b_raw = ab.A, ab.b
    A, b, kept_mask = reduce_constraints_geq(A_raw, b_raw, tol = 1e-9, solver = "OSQP")
    C_train = project_noarb(Ci_train, nodes_sub, tau_sub, m_sub)
    C_test  = project_noarb(Ci_test,  nodes_sub, tau_sub, m_sub)

    # derivatives
    dCtau_train, dCm_train, d2Cm2_train = compute_derivatives(C_train, nodes_sub, cfg.k_tau, cfg.k_m, cfg.s_spline)
    dCtau_test,  dCm_test,  d2Cm2_test  = compute_derivatives(C_test,  nodes_sub, cfg.k_tau, cfg.k_m, cfg.s_spline)

    # stage-0
    pca_price, net0, scaler0, sigma_train, sigma_test, gamma_train, gamma_test = _stage0_block(C_train, C_test, train_df, test_df, cfg)

    # Z field
    Z_train = build_Z(dCtau_train, dCm_train, d2Cm2_train, gamma_train, drop_last=True)
    Z_test  = build_Z(dCtau_test,  dCm_test,  d2Cm2_test,  gamma_test,  drop_last=True)

    # G0 & factor extraction (dyn, stat)
    G0 = C_train.mean(axis=0).values

    if cfg.decoder == DecoderKind.PLS:
        # PLS uses Z -> reconstruct C (or residuals). Here: dynamic from Z, then stat from residuals via PLS on (R0)
        from factors.pca_blocks import pls_block
        pls_dyn, G_dyn, k_dyn = pls_block(X=Z_train, Y=(C_train.values - G0[None, :]), max_comp=max(1, cfg.dda), n_splits=5, random_state=cfg.seed)
        Xi_dyn_train = (pls_dyn.predict(Z_train)) @ np.linalg.pinv(G_dyn)  # scores proxy
        Xi_dyn_test  = (pls_dyn.predict(Z_test))  @ np.linalg.pinv(G_dyn)
        R_dda_train = (C_train.values - G0[None, :]) - Xi_dyn_train @ G_dyn
        pls_stat, G_stat, k_stat = pls_block(X=Z_train, Y=R_dda_train, max_comp=max(1, cfg.dst), n_splits=5, random_state=cfg.seed)
        Xi_stat_train = (pls_stat.predict(Z_train)) @ np.linalg.pinv(G_stat)
        Xi_stat_test  = (pls_stat.predict(Z_test))  @ np.linalg.pinv(G_stat)
        G_sa = np.zeros((0, C_train.shape[1])); Xi_sa_train = np.zeros((C_train.shape[0]-1, 0)); Xi_sa_test = np.zeros((C_test.shape[0]-1, 0))
    elif cfg.decoder == DecoderKind.DEEP_PLS:
        k_dyn = max(1, cfg.dda)
        dp_dyn = DeepPLS(n_in=Z_train.shape[1], n_out=C_train.shape[1], k=k_dyn, hidden=128, lr=1e-3, epochs=200, seed=cfg.seed)
        dp_dyn.fit(Z_train, (C_train.values - G0[None, :]))
        G_dyn = dp_dyn.factors
        Xi_dyn_train = (dp_dyn.predict(Z_train)) @ np.linalg.pinv(G_dyn)
        Xi_dyn_test  = (dp_dyn.predict(Z_test))  @ np.linalg.pinv(G_dyn)
        R_dda_train = (C_train.values - G0[None, :]) - Xi_dyn_train @ G_dyn
        k_stat = max(1, cfg.dst)
        dp_stat = DeepPLS(n_in=Z_train.shape[1], n_out=C_train.shape[1], k=k_stat, hidden=128, lr=1e-3, epochs=200, seed=cfg.seed)
        dp_stat.fit(Z_train, R_dda_train)
        G_stat = dp_stat.factors
        Xi_stat_train = (dp_stat.predict(Z_train)) @ np.linalg.pinv(G_stat)
        Xi_stat_test  = (dp_stat.predict(Z_test))  @ np.linalg.pinv(G_stat)
        G_sa = np.zeros((0, C_train.shape[1])); Xi_sa_train = np.zeros((C_train.shape[0]-1, 0)); Xi_sa_test = np.zeros((C_test.shape[0]-1, 0))
    else:
        # PCA-based dyn/stat + hinge static-arb decoding (Wang)
        pca_dyn, G_dyn = pca_dyn_block(Z_train, cfg.dda, cfg.seed)
        R0_prices_train = C_train.values - G0[None, :]
        R0_prices_test  = C_test.values  - G0[None, :]
        Xi_dyn_train = R0_prices_train @ G_dyn.T
        Xi_dyn_test  = R0_prices_test  @ G_dyn.T
        R_dda_train = R0_prices_train - Xi_dyn_train @ G_dyn
        pca_stat, G_stat = pca_stat_block(R_dda_train, cfg.dst, cfg.seed)
        Xi_stat_train = R_dda_train @ G_stat.T
        Xi_stat_test = (R0_prices_test - Xi_dyn_test @ G_dyn) @ G_stat.T

        Recon_sofar_train = Xi_dyn_train @ G_dyn + Xi_stat_train @ G_stat
        G_sa, Xi_sa_train, _W = decode_static_arb_hinge(
            R_sa_train=R0_prices_train - Recon_sofar_train,
            G0=G0, Recon_sofar=Recon_sofar_train,
            A=A, b=b, n_sa=cfg.n_sa, n_PC=cfg.n_PC_sa,
            lam_rec=cfg.lam_rec, lam_hinge=cfg.lam_hinge, maxfun=8000, seed=cfg.seed
        )
        Xi_sa_test = (R0_prices_test - (Xi_dyn_test @ G_dyn + Xi_stat_test @ G_stat)) @ np.linalg.pinv(G_sa)

    # recon + metrics
    C_hat_train = reconstruct_prices(G0, Xi_dyn_train, G_dyn, Xi_stat_train, G_stat, Xi_sa_train, G_sa)
    C_hat_test  = reconstruct_prices(G0, Xi_dyn_test,  G_dyn, Xi_stat_test,  G_stat, Xi_sa_test,  G_sa)

    MAPE_train = mape(C_train.values, C_hat_train)
    MAPE_test  = mape(C_test.values,  C_hat_test)
    VW_MAPE_train = vega_weighted_mape(C_train.values, C_hat_train, nodes_sub, power = 1.0)
    VW_MAPE_test = vega_weighted_mape(C_test.values, C_hat_test, nodes_sub, power = 1.0)
    PDA_train  = pda_from_pca(Z_train, PCA(n_components=min(cfg.dda, Z_train.shape[1])).fit(Z_train)) if cfg.decoder!=DecoderKind.HINGE_STATIC else pda_from_pca(Z_train, PCA(n_components=min(cfg.dda, Z_train.shape[1])).fit(Z_train))
    PSAS_train = psas(C_hat_train, A, b, tol=5e-3)
    PSAS_test  = psas(C_hat_test,  A, b, tol=5e-3)

    return Algo1Outputs(
        nodes_sub=nodes_sub, tau_sub=tau_sub, m_sub=m_sub, A=A, b=b,
        C_train=C_train, C_test=C_test,
        dCtau_train=dCtau_train, dCm_train=dCm_train, d2Cm2_train=d2Cm2_train,
        dCtau_test=dCtau_test,   dCm_test=dCm_test,   d2Cm2_test=d2Cm2_test,
        pca_price=None if cfg.decoder in (DecoderKind.PLS, DecoderKind.DEEP_PLS) else PCA(n_components=cfg.d0),
        scaler0=scaler0, net0=net0,
        sigma_train=sigma_train, sigma_test=sigma_test,
        gamma_train=gamma_train, gamma_test=gamma_test,
        G0=G0, pca_dyn=None, G_dyn=G_dyn,
        Xi_dyn_train=Xi_dyn_train, Xi_dyn_test=Xi_dyn_test,
        pca_stat=None, G_stat=G_stat,
        Xi_stat_train=Xi_stat_train, Xi_stat_test=Xi_stat_test,
        G_sa=G_sa, Xi_sa_train=Xi_sa_train, Xi_sa_test=Xi_sa_test,
        C_hat_train=C_hat_train, C_hat_test=C_hat_test,
        MAPE_train=MAPE_train, MAPE_test=MAPE_test,
        VW_MAPE_test = VW_MAPE_test, VW_MAPE_train = VW_MAPE_train,
        PDA_train=PDA_train, PSAS_train=PSAS_train, PSAS_test=PSAS_test
    )


def run_algo1_synthetic(lattice: Lattice, surfaces: Surfaces, cfg: Algo1Config) -> Algo1Outputs:
    nodes_sub, tau_sub, m_sub = lattice.nodes, lattice.tau_grid, lattice.m_grid

    # (Optional) projection to no‑arb
    ab = build_noarb_constraints(nodes_sub, tau_sub, m_sub)
    A, b = ab.A, ab.b
    C_df = surfaces.C.sort_index()
    C_proj = project_noarb(C_df, nodes_sub, tau_sub, m_sub)

    # derivatives
    dCtau, dCm, d2Cm2 = compute_derivatives(C_proj, nodes_sub, cfg.k_tau, cfg.k_m, cfg.s_spline)

    # build dt and split
    sec_per_year = 365*24*3600
    tt = pd.to_datetime(C_proj.index.values)
    dt = (tt[1:] - tt[:-1]).total_seconds() / sec_per_year
    T = len(tt)
    i_cut = int(0.8 * T)
    C_train, C_test = C_proj.iloc[:i_cut], C_proj.iloc[i_cut:]
    spot_tr = surfaces.spot[:i_cut]
    spot_te = surfaces.spot[i_cut:]
    dt_tr = dt[:i_cut-1]
    dt_te = dt[i_cut-1:]

    # stage-0
    pca_price = PCA(n_components=cfg.d0, random_state=cfg.seed).fit(C_train.values)
    pcs = pca_price.transform(C_proj.values)[:, :cfg.d0]
    pcs_tr = pcs[:i_cut]; pcs_te = pcs[i_cut:]

    X0, y0, dt0 = make_stage0_dataset(spot_tr, pcs_tr, dt_tr)
    net0 = Stage0LogNet(input_dim=X0.shape[1], hidden=cfg.stage0_hidden)
    scaler0, sigma_tr = train_stage0(net0, X0, y0, dt0, cfg.stage0_epochs, cfg.stage0_lr, cfg.stage0_batch, scale_X=True)

    from sklearn.preprocessing import StandardScaler
    X0t, y0t, dt0t = make_stage0_dataset(spot_te, pcs_te, dt_te)
    X0t_n = scaler0.transform(X0t) if isinstance(scaler0, StandardScaler) else X0t
    import torch
    with torch.no_grad():
        _, sigma_t = net0(torch.from_numpy(X0t_n).float())
    sigma_te = sigma_t.squeeze().cpu().numpy()

    gamma_tr = sigma_to_gamma(sigma_tr, S_tm1=spot_tr[:-1])
    gamma_te = sigma_to_gamma(sigma_te, S_tm1=spot_te[:-1])

    # Z fields
    dCtau_tr, dCm_tr, d2Cm2_tr = dCtau[:i_cut], dCm[:i_cut], d2Cm2[:i_cut]
    dCtau_te, dCm_te, d2Cm2_te = dCtau[i_cut:], dCm[i_cut:], d2Cm2[i_cut:]
    Z_tr = build_Z(dCtau_tr, dCm_tr, d2Cm2_tr, gamma_tr, drop_last=True)
    Z_te = build_Z(dCtau_te, dCm_te, d2Cm2_te, gamma_te, drop_last=True)

    # G0 & factor extraction
    G0 = C_train.mean(axis=0).values

    if cfg.decoder == DecoderKind.PLS:
        pls_dyn, G_dyn, k_dyn = pls_block(X=Z_tr, Y=(C_train.values - G0[None, :]), max_comp=max(1, cfg.dda), n_splits=5, random_state=cfg.seed)
        Xi_dyn_tr = (pls_dyn.predict(Z_tr)) @ np.linalg.pinv(G_dyn)
        Xi_dyn_te = (pls_dyn.predict(Z_te)) @ np.linalg.pinv(G_dyn)
        R_dda_tr = (C_train.values - G0[None, :]) - Xi_dyn_tr @ G_dyn
        pls_stat, G_stat, k_stat = pls_block(X=Z_tr, Y=R_dda_tr, max_comp=max(1, cfg.dst), n_splits=5, random_state=cfg.seed)
        Xi_stat_tr = (pls_stat.predict(Z_tr)) @ np.linalg.pinv(G_stat)
        Xi_stat_te = (pls_stat.predict(Z_te)) @ np.linalg.pinv(G_stat)
        G_sa = np.zeros((0, C_train.shape[1])); Xi_sa_tr = np.zeros((C_train.shape[0]-1, 0)); Xi_sa_te = np.zeros((C_test.shape[0]-1, 0))
    elif cfg.decoder == DecoderKind.DEEP_PLS:
        k_dyn = max(1, cfg.dda)
        dp_dyn = DeepPLS(n_in=Z_tr.shape[1], n_out=C_train.shape[1], k=k_dyn, hidden=128, lr=1e-3, epochs=200, seed=cfg.seed)
        dp_dyn.fit(Z_tr, (C_train.values - G0[None, :]))
        G_dyn = dp_dyn.factors
        Xi_dyn_tr = (dp_dyn.predict(Z_tr)) @ np.linalg.pinv(G_dyn)
        Xi_dyn_te = (dp_dyn.predict(Z_te)) @ np.linalg.pinv(G_dyn)
        R_dda_tr = (C_train.values - G0[None, :]) - Xi_dyn_tr @ G_dyn
        k_stat = max(1, cfg.dst)
        dp_stat = DeepPLS(n_in=Z_tr.shape[1], n_out=C_train.shape[1], k=k_stat, hidden=128, lr=1e-3, epochs=200, seed=cfg.seed)
        dp_stat.fit(Z_tr, R_dda_tr)
        G_stat = dp_stat.factors
        Xi_stat_tr = (dp_stat.predict(Z_tr)) @ np.linalg.pinv(G_stat)
        Xi_stat_te = (dp_stat.predict(Z_te)) @ np.linalg.pinv(G_stat)
        G_sa = np.zeros((0, C_train.shape[1])); Xi_sa_tr = np.zeros((C_train.shape[0]-1, 0)); Xi_sa_te = np.zeros((C_test.shape[0]-1, 0))
    else:
        pca_dyn, G_dyn = pca_dyn_block(Z_tr, cfg.dda, cfg.seed)
        R0_tr = C_train.values - G0[None, :]
        R0_te = C_test.values  - G0[None, :]
        Xi_dyn_tr = R0_tr @ G_dyn.T
        Xi_dyn_te = R0_te @ G_dyn.T
        R_dda_tr = R0_tr - Xi_dyn_tr @ G_dyn
        pca_stat, G_stat = pca_stat_block(R_dda_tr, cfg.dst, cfg.seed)
        Xi_stat_tr = R_dda_tr @ G_stat.T
        Xi_stat_te = (R0_te - Xi_dyn_te @ G_dyn) @ G_stat.T
        Recon_sofar_tr = Xi_dyn_tr @ G_dyn + Xi_stat_tr @ G_stat
        G_sa, Xi_sa_tr, _W = decode_static_arb_hinge(
            R_sa_train=R0_tr - Recon_sofar_tr,
            G0=G0, Recon_sofar=Recon_sofar_tr,
            A=A, b=b, n_sa=cfg.n_sa, n_PC=cfg.n_PC_sa,
            lam_rec=cfg.lam_rec, lam_hinge=cfg.lam_hinge, maxfun=8000, seed=cfg.seed
        )
        Xi_sa_te = (R0_te - (Xi_dyn_te @ G_dyn + Xi_stat_te @ G_stat)) @ np.linalg.pinv(G_sa)

    C_hat_tr = reconstruct_prices(G0, Xi_dyn_tr, G_dyn, Xi_stat_tr, G_stat, Xi_sa_tr, G_sa)
    C_hat_te = reconstruct_prices(G0, Xi_dyn_te, G_dyn, Xi_stat_te, G_stat, Xi_sa_te, G_sa)

    MAPE_tr = mape(C_train.values, C_hat_tr)
    MAPE_te = mape(C_test.values,  C_hat_te)
    PDA_tr  = pda_from_pca(Z_tr, PCA(n_components=min(cfg.dda, Z_tr.shape[1])).fit(Z_tr))
    PSAS_tr = psas(C_hat_tr, A, b, tol=5e-3)
    PSAS_te = psas(C_hat_te, A, b, tol=5e-3)

    return Algo1Outputs(
        nodes_sub=nodes_sub, tau_sub=tau_sub, m_sub=m_sub, A=A, b=b,
        C_train=C_train, C_test=C_test,
        dCtau_train=dCtau_tr, dCm_train=dCm_tr, d2Cm2_train=d2Cm2_tr,
        dCtau_test=dCtau_te,   dCm_test=dCm_te,   d2Cm2_test=d2Cm2_te,
        pca_price=None, scaler0=scaler0, net0=net0,
        sigma_train=sigma_tr, sigma_test=sigma_te,
        gamma_train=gamma_tr, gamma_test=gamma_te,
        G0=G0, pca_dyn=None, G_dyn=G_dyn,
        Xi_dyn_train=Xi_dyn_tr, Xi_dyn_test=Xi_dyn_te,
        pca_stat=None, G_stat=G_stat,
        Xi_stat_train=Xi_stat_tr, Xi_stat_test=Xi_stat_te,
        G_sa=G_sa, Xi_sa_train=Xi_sa_tr, Xi_sa_test=Xi_sa_te,
        C_hat_train=C_hat_tr, C_hat_test=C_hat_te,
        MAPE_train=MAPE_tr, MAPE_test=MAPE_te,
        PDA_train=PDA_tr, PSAS_train=PSAS_tr, PSAS_test=PSAS_te
    )
