import torch.nn as nn
import torch.optim as optim
import torchsde
import torch
from neural_sde.nets import NeuralSDEJump
from neural_sde.constraints import _apply_boundary_correction, calc_drift_correction, calc_diffusion_scaling, build_factor_path, assemble_Wb_for_shrinkage, compute_factor_polytope_vertices, shrink_matrix_and_diag, shrink_diag_covmatch
import numpy as np
import math


# Build training dataset
def build_xi_training_data(out, use_dyn = True, use_sa = True, use_stat = True):
    parts = []
    names = []

    if use_dyn and out.Xi_dyn_train.size:
        parts.append(out.Xi_dyn_train); names += [f"dyn{j}" for j in range(out.Xi_dyn_train.shape[1])]
    if use_stat and out.Xi_stat_train.size:
        parts.append(out.Xi_stat_train); names += [f"stat{j}" for j in range(out.Xi_stat_train.shape[1])]
    if use_sa and out.Xi_sa_train.size:
        parts.append(out.Xi_sa_train);   names += [f"sa{j}" for j in range(out.Xi_sa_train.shape[1])]
    X_train = np.concatenate(parts, axis=1) if parts else None

    parts_t = []
    if use_dyn and out.Xi_dyn_test.size:
        parts_t.append(out.Xi_dyn_test)
    if use_stat and out.Xi_stat_test.size:
        parts_t.append(out.Xi_stat_test)
    if use_sa and out.Xi_sa_test.size:
        parts_t.append(out.Xi_sa_test)
    X_test = np.concatenate(parts_t, axis=1) if parts_t else None

    return X_train, X_test, names

# Build lattice training (for training on full lattice)
def build_lattice_training_data(out):
    """
    Use the full liquid lattice as the state:
      X_train = out.C_train.values  (T_train x K)
      X_test  = out.C_test.values   (T_test  x K)
    """
    X_train = out.C_train.values.astype(np.float64)
    X_test  = out.C_test.values.astype(np.float64)
    names = [str(c) for c in out.C_train.columns]  # optional labels
    return X_train, X_test, names


@torch.no_grad()
def boundary_kappa(t_idx, epsmu, cutoff=1e-3, cap=1.0):
    # epsmu: [T,K], smaller => closer to face
    em = epsmu.index_select(0, t_idx)      # [B,K]
    dmin = em.min(dim=1).values            # [B]
    kappa = (cutoff / (dmin.clamp_min(1e-12))).clamp(max=cap)
    return (kappa / cap).clamp_(0.0, 1.0)  # [B] in [0,1]


def _log_mvn_diag(y, m, sd, eps=1e-8):
    # y,m,sd: [B,p]; sd>0
    var = (sd**2).clamp_min(eps)
    quad = ((y - m)**2 / var).sum(-1)
    logdet = torch.log(var).sum(-1)
    p = y.shape[-1]
    return -0.5 * (p*math.log(2*math.pi) + logdet + quad)

def jump_mixture_nll_diag(
    y0, y1, dt, model,
    # drift boundary correction (your routine)
    t_idx=None, corr_dirs=None, epsmu=None,
    bc_lambda=1.0, bc_eps_floor=1e-8, bc_cap=5.0, bc_epsmu_cutoff=1e-3,
    # jump shrink near boundary
    jump_alpha_lambda=1.0, jump_alpha_cov=1.0, jump_alpha_mean=0.5,
    eps_dt=1e-12,
    # Diffusion shrinkage
    H=None, h=None, use_diffusion_shrink = True
):
    B, p = y0.shape
    dt  = dt.view(-1).clamp_min(eps_dt)             # [B]
    dts = dt.unsqueeze(-1)                          # [B,1]

    # Drift (with your boundary correction)
    mu = model.f(0.0, y0)                           # [B,p]
    if (corr_dirs is not None) and (epsmu is not None) and (t_idx is not None):
        mu = _apply_boundary_correction(
            mu, t_idx, corr_dirs, epsmu,
            bc_lambda=bc_lambda, bc_eps_floor=bc_eps_floor,
            bc_cap=bc_cap, bc_epsmu_cutoff=bc_epsmu_cutoff, device=y0.device
        )

    # Diffusion (diag)
    sig = model.g(0.0, y0)                          # [B,p]
    if use_diffusion_shrink and (H is not None) and (h is not None):
        H_t = torch.as_tensor(H, dtype=torch.float32, device=y0.device)
        h_t = torch.as_tensor(h, dtype=torch.float32, device=y0.device)
        s   = shrink_matrix_and_diag(y0, H_t, h_t, mode="diag")     # [B,p]
        s = s.clamp_min(0.4) # tune this
        # s   = shrink_diag_covmatch(P, sig)                          # [B,p]
        sig = s * sig
    md  = y0 + mu * dts                             # [B,p]
    sd_d = sig * dt.sqrt().unsqueeze(-1)            # [B,p]

    # Jump params
    lam, mJ, SJ_L = model.jump_head(y0)             # lam:[B], mJ:[B,p], SJ_L:[B,p,p]
    sJ  = torch.diagonal(SJ_L, dim1=-2, dim2=-1)    # [B,p] (diag std)

    # Boundary proximity -> shrink jump bits
    if (epsmu is not None) and (t_idx is not None):
        kappa = boundary_kappa(t_idx, epsmu, cutoff=bc_epsmu_cutoff, cap=1.0)  # [B]
    else:
        kappa = torch.zeros(B, device=y0.device)

    lam_eff = (1.0 - jump_alpha_lambda * kappa) * lam         # [B]
    lam_eff = lam_eff.clamp_min(1e-12)
    sJ_eff  = (1.0 - jump_alpha_cov  * kappa).clamp_min(0.0).unsqueeze(-1) * sJ
    mJ_eff  = (1.0 - jump_alpha_mean * kappa).clamp_min(0.0).unsqueeze(-1) * mJ

    # Mixture components (both diagonal Gaussians)
    # no-jump:
    gamma = (lam_eff * dt).clamp_min(1e-20).clamp_max(0.8)
    # logw0 = torch.log((1.0 - lam_eff * dt).clamp_min(1e-12))             # [B]
    logw0 = torch.log1p(-gamma) 
    logp0 = _log_mvn_diag(y1, md, sd_d)

    # one-jump: mean shift + variance inflate
    moj  = md + mJ_eff
    sd_oj = torch.sqrt(sd_d**2 + sJ_eff**2)
    # logw1 = torch.log((lam_eff * dt).clamp_min(1e-20))
    logw1 = torch.log(gamma)
    logp1 = _log_mvn_diag(y1, moj, sd_oj)

    # mix (log-sum-exp)
    mstack = torch.stack([logw0 + logp0, logw1 + logp1], dim=-1)          # [B,2]
    return -torch.logsumexp(mstack, dim=-1)                               # [B]


def likelihood_training_one_jump(
    out,
    # Omega_tr, det_Omega_tr, proj_dX_tr,   # kept for interface parity (not used here)
    # Omega_te, det_Omega_te, proj_dX_te,
    n_epochs, batch_size, zero_drift=False, lr=1e-3, data="xi",
    model=None,
    # geometry
    k_box=6.0, bc_cap=5.0,
    # jump shrink knobs (1.0 => full shrink at boundary; 0.0 => no shrink)
    jump_alpha_lambda=1.0, jump_alpha_cov=1.0, jump_alpha_mean=0.5,
    # toggle diffusion shrinkage in loss
    use_diffusion_shrink=True,
    # Jump regularisation
    lam_reg = 1e-4, mJ_reg = 1e-4, sJ_reg = 1e-9,
    target_ldt=0.2, ldt_reg = 1e-3
):
    """
    Train NeuralSDEJump with:
      - drift boundary correction,
      - diffusion shrinkage against static‑arb polytope (H,h),
      - boundary‑aware shrink of jump intensity (lambda) and jump stats.

    Returns: train_losses, test_losses, model, X_train_tensor
    """
    # ----- data -----
    if data == 'xi':
        X_train_np, X_test_np, _ = build_xi_training_data(out)
    else:
        X_train_np, X_test_np, _ = build_lattice_training_data(out)

    n_train, dim = X_train_np.shape
    n_test, _ = X_test_np.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = NeuralSDEJump(dim, zero_drift=zero_drift).to(device)
    else:
        model = model.to(device)

    X_train = torch.from_numpy(X_train_np).float().to(device)
    X_test  = torch.from_numpy(X_test_np).float().to(device)

    # ----- boundary geometry -----
    poly = compute_factor_polytope_vertices(out, xi_builder=None, k_box=k_box, verbose=False, return_mappings=True)
    W, b = assemble_Wb_for_shrinkage(poly, include_box=True)
    H_np, h_np = poly["H"], poly["h"]

    # Precompute drift‑correction helpers on full sequences
    _, corr_dirs_tr, epsmu_tr = calc_drift_correction(W, b, X=X_train_np, epsmu_star=10.0, rho_star=1e-5)
    _, corr_dirs_te, epsmu_te = calc_drift_correction(W, b, X=X_test_np,  epsmu_star=10.0, rho_star=1e-5)

    def to_torch_dirs(cdirs_np, dim_):
        m_faces = W.shape[0]
        if cdirs_np.ndim == 2 and cdirs_np.shape[1] == m_faces * dim_:
            cdirs_np = cdirs_np.reshape(-1, m_faces, dim_)
        return torch.from_numpy(cdirs_np).float().to(device)

    corr_dirs_tr_t = to_torch_dirs(corr_dirs_tr, dim)
    corr_dirs_te_t = to_torch_dirs(corr_dirs_te, dim)
    epsmu_tr_t     = torch.from_numpy(epsmu_tr).float().to(device)
    epsmu_te_t     = torch.from_numpy(epsmu_te).float().to(device)

    # ----- dt arrays -----
    sec_per_year = 1
    tt_tr = out.C_train.index.values
    dt_train_pairs = (np.diff(tt_tr).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_train_pairs = dt_train_pairs[:max(0, n_train - 1)]
    dt_train_t = torch.from_numpy(dt_train_pairs).float().to(device).unsqueeze(1)

    tt_te = out.C_test.index.values
    dt_test_pairs = (np.diff(tt_te).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_test_pairs = dt_test_pairs[:max(0, n_test - 1)]
    dt_test_t = torch.from_numpy(dt_test_pairs).float().to(device).unsqueeze(1)

    # ----- train -----
    opt = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        if n_train <= 1:
            break
        perm = torch.randperm(n_train - 1, device=device)

        for idx in perm.split(batch_size):
            y0 = X_train[idx]          # [B,p]
            y1 = X_train[idx + 1]      # [B,p]
            dtb = dt_train_t.index_select(0, idx)  # [B,1]

            nll = jump_mixture_nll_diag(
                y0, y1, dtb, model,
                # drift boundary correction
                t_idx=idx,
                corr_dirs=corr_dirs_tr_t,
                epsmu=epsmu_tr_t,
                bc_lambda=1.0, bc_eps_floor=1e-8, bc_cap=bc_cap, bc_epsmu_cutoff=1e-3,
                # jump shrink near boundary
                jump_alpha_lambda=jump_alpha_lambda,
                jump_alpha_cov=jump_alpha_cov,
                jump_alpha_mean=jump_alpha_mean,
                # diffusion shrinkage against H,h
                H=H_np, h=h_np, use_diffusion_shrink=use_diffusion_shrink
            )
            # Regularize jump params to avoid λΔ >> 1 and degenerate sJ→0 with huge mJ
            lam, mJ, SJ_L = model.jump_head(y0)
            sJ  = torch.diagonal(SJ_L, dim1=-2, dim2=-1)
            ldt = lam * dtb.view(-1)

            reg = 0.0
            reg = reg + lam_reg * lam.mean()
            reg = reg + ldt_reg * torch.relu(ldt - target_ldt).mean()
            reg = reg + mJ_reg * mJ.pow(2).mean()
            reg = reg + sJ_reg * sJ.pow(2).mean()
            reg += 5e-3 * (lam * dtb.view(-1)).mean()

            loss = nll.mean() + reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        # ----- eval (full pass) -----
        model.eval()
        with torch.no_grad():
            if n_train > 1:
                y0_tr = X_train[:-1]; y1_tr = X_train[1:]
                nll_tr = jump_mixture_nll_diag(
                    y0_tr, y1_tr, dt_train_t, model,
                    t_idx=torch.arange(y0_tr.size(0), device=device),
                    corr_dirs=corr_dirs_tr_t, epsmu=epsmu_tr_t,
                    bc_lambda=1.0, bc_eps_floor=1e-8, bc_cap=bc_cap, bc_epsmu_cutoff=1e-3,
                    jump_alpha_lambda=jump_alpha_lambda,
                    jump_alpha_cov=jump_alpha_cov,
                    jump_alpha_mean=jump_alpha_mean,
                    H=H_np, h=h_np, use_diffusion_shrink=use_diffusion_shrink
                )
                train_loss = nll_tr.mean().item()
            else:
                train_loss = float('nan')
            train_losses.append(train_loss)

            if n_test > 1:
                y0_te = X_test[:-1]; y1_te = X_test[1:]
                nll_te = jump_mixture_nll_diag(
                    y0_te, y1_te, dt_test_t, model,
                    t_idx=torch.arange(y0_te.size(0), device=device),
                    corr_dirs=corr_dirs_te_t, epsmu=epsmu_te_t,
                    bc_lambda=1.0, bc_eps_floor=1e-8, bc_cap=bc_cap, bc_epsmu_cutoff=1e-3,
                    jump_alpha_lambda=jump_alpha_lambda,
                    jump_alpha_cov=jump_alpha_cov,
                    jump_alpha_mean=jump_alpha_mean,
                    H=H_np, h=h_np, use_diffusion_shrink=use_diffusion_shrink
                )
                test_loss = nll_te.mean().item()
            else:
                test_loss = float('nan')
            test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{n_epochs}  Train NLL: {train_loss:.4e}  Test NLL: {test_loss:.4e}")

    return train_losses, test_losses, model, X_train