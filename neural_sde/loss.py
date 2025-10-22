import torch.nn as nn
import torch.optim as optim
import torchsde
import torch
from neural_sde.nets import NeuralSDE, NeuralSDEWithShrink
from neural_sde.constraints import _apply_boundary_correction, calc_drift_correction, calc_diffusion_scaling, build_factor_path, assemble_Wb_for_shrinkage, compute_factor_polytope_vertices
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

# # Training loop with likelihood based loss
def likelihood_training(out, Omega_tr, det_Omega_tr, proj_dX_tr,
                        Omega_te, det_Omega_te, proj_dX_te,
                        n_epochs, batch_size, zero_drift: bool = False, lr=1e-3, data: str = "xi", 
                        model=None):
    if data == 'xi':
        X_train, X_test, names = build_xi_training_data(out)
    elif data == "lattice":
        X_train, X_test, _ = build_lattice_training_data(out)
    
    # Training shapes
    n_train, dim = X_train.shape
    # Test Shapes
    n_test, _ = X_test.shape
    
    # Set device (i wish i had some cuda cores :( )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ability to load custom model
    if model is None:
        model = NeuralSDE(dim, zero_drift=zero_drift).to(device)
    else:
        model = model.to(device)

    # Convert data to torch tensors on device
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test  = torch.from_numpy(X_test).float().to(device)

    # Convert Diffusion Scaling parameters from numpy -> torch
    Omega_tr      = torch.from_numpy(Omega_tr).float().to(device).view(-1, dim, dim)
    det_Omega_tr  = torch.from_numpy(det_Omega_tr).float().to(device).view(-1, 1)
    proj_dX_tr    = torch.from_numpy(proj_dX_tr).float().to(device)

    Omega_te      = torch.from_numpy(Omega_te).float().to(device).view(-1, dim, dim)
    det_Omega_te  = torch.from_numpy(det_Omega_te).float().to(device).view(-1, 1)
    proj_dX_te    = torch.from_numpy(proj_dX_te).float().to(device)

    # Drift Scaling
    poly = compute_factor_polytope_vertices(
    out,
    xi_builder=None,      # or None if Xi_*_train are on outs
    k_box=6.0,
    verbose=False,
    return_mappings=True,
)
    W, b = assemble_Wb_for_shrinkage(poly, include_box=True)

    # Build drift correction geometry
    # X_interior, corr_dirs, epsmu = calc_drift_correction(W, b, X  = build_factor_path(out, xi_builder=None), epsmu_star = 10, rho_star = 1e-5)
    X_tr_np = X_train.detach().cpu().numpy()
    X_te_np = X_test.detach().cpu().numpy()
    _, corr_dirs_tr, epsmu_tr = calc_drift_correction(W, b, X=X_tr_np, epsmu_star=10.0, rho_star=1e-5)
    _, corr_dirs_te, epsmu_te = calc_drift_correction(W, b, X=X_te_np, epsmu_star=10.0, rho_star=1e-5)
    
    # reshape and move to device (match endogenous trainer)
    m_faces = W.shape[0]
    def to_torch_dirs(cdirs_np, dim):
        if cdirs_np.ndim == 2 and cdirs_np.shape[1] == m_faces * dim:
            cdirs_np = cdirs_np.reshape(-1, m_faces, dim)
        return torch.from_numpy(cdirs_np).float().to(device)
    corr_dirs_tr_t = to_torch_dirs(corr_dirs_tr, dim)
    corr_dirs_te_t = to_torch_dirs(corr_dirs_te, dim)
    epsmu_tr_t     = torch.from_numpy(epsmu_tr).float().to(device)
    epsmu_te_t     = torch.from_numpy(epsmu_te).float().to(device)

    # dt between consecutive rows in years (torch tensors on device)
    sec_per_year = 1
    tt = out.C_train.index.values
    dt_train_pairs = (np.diff(tt).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_train_pairs = dt_train_pairs[:max(0, n_train - 1)]
    dt_train_t = torch.from_numpy(dt_train_pairs).float().to(device).unsqueeze(1)  # (n_train-1, 1)

    tt_test = out.C_test.index.values
    dt_test_pairs = (np.diff(tt_test).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_test_pairs = dt_test_pairs[:max(0, n_test - 1)]
    dt_test_t = torch.from_numpy(dt_test_pairs).float().to(device).unsqueeze(1)    # (n_test-1, 1)


    # losses
    train_losses = []
    test_losses = []

    # optimiser
    opt = optim.Adam(model.parameters(), lr = lr)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(n_train - 1, device = device)

        for idx in perm.split(batch_size):
            y0 = X_train[idx]
            y1 = X_train[idx + 1]
            # Compute drift and diffusion
            drift = model.f(0, y0)
            # print(drift)
            diff = model.g(0, y0)
            # print(diff)
            # compute increments
            dy = y1 - y0
            # print(dy)
            dt = dt_train_t.index_select(0, idx)
            # print(dt)

            var = (diff ** 2) * dt + 1e-9

            # Get diffusion shrinkage parameters
            Omega_b = Omega_tr.index_select(0, idx)
            det_Omega_b = det_Omega_tr.index_select(0, idx)
            proj_dX_b = proj_dX_tr.index_select(0, idx)

            # print(var)

            # Negative log likelihood per coordinate
            # nll = 0.5 * ((dy - drift * dt) ** 2) / var + torch.log(2 * np.pi * var)
            # nll = ait_sahalia_quasi_nll(model, y0, y1, dt)
            nll = shrunk_gaussian_nll(y0, y1, dt, Omega_b, det_Omega_b, proj_dX_b,
                                       model = model, diagonal_diffusion=True,
                                       t_idx = idx, corr_dirs=corr_dirs_tr_t, epsmu=epsmu_tr_t, bc_cap = 5.0)

            loss = nll.mean()

            # print(loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            opt.step()

            epoch_loss += loss.mean().item() * y0.size(0)
        
        # train_loss = epoch_loss / (n_train - 1)

        # train_losses.append(train_loss)
        # Recompute full-train loss with the final params (comparable to test)
        model.eval()
        with torch.no_grad():
            y0_tr_full = X_train[:-1]
            y1_tr_full = X_train[1:]
            dt_tr_full = dt_train_t
            # nll_tr_full = shrunk_gaussian_nll(
            #     y0_tr_full, y1_tr_full, dt_tr_full,
            #     Omega_tr, det_Omega_tr, proj_dX_tr,
            #     model=model, diagonal_diffusion=True
            # )
            nll_tr_full = shrunk_gaussian_nll(
                y0_tr_full, y1_tr_full, dt_tr_full,
                Omega_tr, det_Omega_tr, proj_dX_tr,
                model=model, diagonal_diffusion=True,
                t_idx=torch.arange(y0_tr_full.size(0), device=device),
                corr_dirs=corr_dirs_tr_t, epsmu=epsmu_tr_t,
                bc_cap=5.0
            )
            train_loss = nll_tr_full.mean().item()
            train_losses.append(train_loss)

        # evaluate on test
        model.eval()
        with torch.no_grad():

            y0_test = X_test[:-1]
            y1_test = X_test[1:]
            dt = dt_test_t
            drift_t = model.f(0, y0_test)
            diff_t = model.g(0, y0_test)
            dy_t = y1_test - y0_test
            var_t = (diff_t ** 2) * dt + 1e-6

            # nll_t = 0.5 * ((dy_t - drift_t * dt)**2 / var_t + torch.log(2 * np.pi * var_t))
            # nll_t = shrunk_gaussian_nll(y0_test, y1_test, dt, Omega_te, det_Omega_te, proj_dX=proj_dX_te ,model=model, diagonal_diffusion=True)
            nll_t = shrunk_gaussian_nll(
                y0_test, y1_test, dt_test_t, Omega_te, det_Omega_te, proj_dX=proj_dX_te,
                model=model, diagonal_diffusion=True,
                t_idx=torch.arange(y0_test.size(0), device=device),
                corr_dirs=corr_dirs_te_t, epsmu=epsmu_te_t,
                bc_cap=5.0
            )
            # nll_t = ait_sahalia_quasi_nll(model, y0_test, y1_test, dt)
            test_loss = nll_t.mean().item()
            test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{n_epochs}  Train NLL: {train_loss:.4e}  Test NLL: {test_loss:.4e}")


    return train_losses, test_losses, model, X_train



def _diag_sigma_grad(model, y):
    """
    Compute diagonal dσ_i/dy_i at y via autograd.
    y: (B, d) requires_grad=True
    returns: (B, d)
    """
    sigma = model.g(0.0, y)  # (B, d)
    grads = []
    for i in range(y.shape[1]):
        gi = sigma[:, i].sum()
        (grad_y,) = torch.autograd.grad(gi, y, retain_graph=True, create_graph=True)
        grads.append(grad_y[:, i:i+1])
    return torch.cat(grads, dim=1)

def ait_sahalia_quasi_nll(model, y0, y1, dt, eps_dt=1e-10, eps_sig=1e-10):
    """
    First-order Lamperti quasi-likelihood (Aït-Sahalia style) with diagonal diffusion:
      - z = ∫ dy/σ(y)  (approximated by dz ≈ (y1 - y0)/σ(y0))
      - μ_L = f/σ - 0.5 σ'
      - log p(y1|y0) ≈ N(z1; z0 + μ_L Δ, Δ) + log |∂z/∂y1| = N(...) + log σ(y1)
    Shapes:
      y0,y1: (B,d); dt: (B,1)
    """
    with torch.enable_grad():
        # evaluate at y0
        y0_req = y0.detach().clone().requires_grad_(True)
        sigma0 = model.g(0.0, y0_req).clamp_min(eps_sig)     # (B,d)
        drift0 = model.f(0.0, y0_req)                         # (B,d)
        sigp0  = _diag_sigma_grad(model, y0_req)              # (B,d)
        mu_L   = drift0 / sigma0 - 0.5 * sigp0                # (B,d)

        # transformed increment and Jacobian at y1
        dz = (y1 - y0) / sigma0                               # (B,d)
        sigma1 = model.g(0.0, y1).clamp_min(eps_sig)          # (B,d)
        logJ = torch.log(sigma1)                              # (B,d)

        dtc = dt.clamp_min(eps_dt)                            # (B,1)
        w = (dz - mu_L * dtc) / torch.sqrt(dtc)               # (B,d)

        nll = 0.5 * (w**2 + torch.log(2 * math.pi * dtc)) + logJ
        return nll  # (B,d)


# def shrunk_gaussian_nll(
#     y0, y1, dt,
#     Omega, det_Omega, proj_dX,           # from calc_diffusion_scaling (aligned per step)
#     model,
#     diagonal_diffusion: bool = True,
#     eps: float = 1e-12,
# ):
#     """
#     y0,y1: [B,p]
#     dt:    [B] or [B,1]
#     Omega: [B,p,p]
#     det_Omega: [B,1]   (positive)
#     proj_dX:   [B,p]   (this is Ω^{-T} dX; if you don't have it, compute it with Omega)
#     model.f/.g at (t=0, y0) returning drift [B,p] and diffusion:
#         - if diagonal_diffusion: diff diag entries [B,p]  (>=0 via Softplus)
#         - else: lower-tri Cholesky L of Σ (B,p,p) with positive diag
#     """
#     B, p = y0.shape
#     dt = dt.view(-1)  # [B]

#     # Increments and model evals
#     dy    = y1 - y0                      # [B,p]
#     mu    = model.f(0.0, y0)             # [B,p]
#     g_out = model.g(0.0, y0)

#     # Project drift with Ω^{-T} (proj_dX is already Ω^{-T} dX)
#     # proj_mu = Ω^{-T} μ
#     proj_mu = torch.linalg.solve(Omega.transpose(-1, -2), mu.unsqueeze(-1)).squeeze(-1)  # [B,p]

#     # l1: log-determinant pieces
#     #   2 * sum(log det Ω) + 2 * sum(log diag(L))  (per-sample)
#     log_det_Omega = torch.log(det_Omega.clamp_min(eps)).squeeze(-1)  # [B]

#     if diagonal_diffusion:
#         # Σ = diag(diff^2) -> Cholesky L = diag(diff)
#         diff    = g_out.clamp_min(eps)                 # [B,p], assume Softplus already
#         logdetΣ = 2.0 * torch.sum(torch.log(diff), dim=-1)  # [B]
#         # Whiten: L^{-1} v  is just v / diff
#         sol_dX  = proj_dX / diff                       # [B,p]
#         sol_mu  = proj_mu / diff                       # [B,p]
#     else:
#         # g_out is lower-triangular Cholesky L of Σ (B,p,p)
#         L = g_out
#         logdetΣ = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # [B]
#         # Whiten by solving L z = v  (lower=True)
#         sol_dX = torch.linalg.solve_triangular(L, proj_dX.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]
#         sol_mu = torch.linalg.solve_triangular(L, proj_mu.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]

#     l1 = 2.0 * log_det_Omega + 2.0 * logdetΣ                         # [B]

#     # Quadratic parts (Euler–Gaussian form with shrinkage):
#     # l2 = (1/dt) * || L^{-1} Ω^{-T} dX ||^2
#     # l3 =  dt     * || L^{-1} Ω^{-T} μ  ||^2
#     # l4 = -2      * < L^{-1} Ω^{-T} μ , L^{-1} Ω^{-T} dX >
#     quad1 = (sol_dX.pow(2).sum(dim=-1)) / dt                         # [B]
#     quad2 = (sol_mu.pow(2).sum(dim=-1)) * dt                         # [B]
#     quad3 = -2.0 * (sol_mu * sol_dX).sum(dim=-1)                     # [B]

#     nll_per_step = l1 + quad1 + quad2 + quad3                        # [B]
#     # (Optional) add + p*log(dt) and + p*log(2π) constants; they don't affect training.
#     return nll_per_step


# Shrunk gaussian for drift and diffusion
def shrunk_gaussian_nll(
    y0, y1, dt,
    Omega, det_Omega, proj_dX,           # from calc_diffusion_scaling (aligned per step)
    model,
    diagonal_diffusion: bool = True,
    eps: float = 1e-12,
    # ------- BC (boundary correction) additions -------
    t_idx: torch.Tensor | None = None,   # [B] time indices mapping each row of y0 to global time
    corr_dirs=None,                      # [T,K,p] or [T, K*p]
    epsmu=None,                          # [T,K]
    bc_lambda: float = 1.0,              # correction strength
    bc_eps_floor: float = 1e-8,
    bc_cap: float | None = None,
    bc_epsmu_cutoff: float = 1e-3,       # Disable correction at a certain point inside
):
    """
    y0,y1: [B,p]
    dt:    [B] or [B,1]
    Omega: [B,p,p]
    det_Omega: [B,1]   (positive)
    proj_dX:   [B,p]   (this is Ω^{-T} dX)
    model.f/.g at (t=0, y0) returning drift [B,p] and diffusion.

    If corr_dirs/epsmu/t_idx are provided, μ is corrected near the static-arb boundary.
    """
    B, p = y0.shape
    dt = dt.view(-1)  # [B]

    # Increments and model evals
    dy    = y1 - y0                      # [B,p]
    mu    = model.f(0.0, y0)             # [B,p]

    # ----- apply boundary correction to μ if provided -----
    if (corr_dirs is not None) and (epsmu is not None) and (t_idx is not None):
        mu = _apply_boundary_correction(
            mu, t_idx, corr_dirs, epsmu,
            bc_lambda=bc_lambda, bc_eps_floor=bc_eps_floor,
            bc_cap=bc_cap, bc_epsmu_cutoff=bc_epsmu_cutoff,
            device=y0.device,
        )

    g_out = model.g(0.0, y0)

    # Project drift with Ω^{-T} (proj_dX is already Ω^{-T} dX)
    proj_mu = torch.linalg.solve(Omega.transpose(-1, -2), mu.unsqueeze(-1)).squeeze(-1)  # [B,p]

    # l1: log-determinant pieces
    log_det_Omega = torch.log(det_Omega.clamp_min(eps)).squeeze(-1)  # [B]

    if diagonal_diffusion:
        diff    = g_out.clamp_min(eps)                 # [B,p]
        logdetΣ = 2.0 * torch.sum(torch.log(diff), dim=-1)     # [B]
        sol_dX  = proj_dX / diff                               # [B,p]
        sol_mu  = proj_mu / diff                               # [B,p]
    else:
        L = g_out
        logdetΣ = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # [B]
        sol_dX  = torch.linalg.solve_triangular(L, proj_dX.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]
        sol_mu  = torch.linalg.solve_triangular(L, proj_mu.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]

    l1 = 2.0 * log_det_Omega + 2.0 * logdetΣ

    # Quadratic parts (Euler–Gaussian with shrinkage):
    quad1 = (sol_dX.pow(2).sum(dim=-1)) / dt
    quad2 = (sol_mu.pow(2).sum(dim=-1)) * dt
    quad3 = -2.0 * (sol_mu * sol_dX).sum(dim=-1)

    nll_per_step = l1 + quad1 + quad2 + quad3
    return nll_per_step



def _build_omega_proj_from_state(y0: torch.Tensor,
                                 y1: torch.Tensor,
                                 H: torch.Tensor, h: torch.Tensor,
                                 dist_multiplier: float, proj_scale: float,
                                 eps_floor: float = 1e-12):
    """
    Compute Ω(y0), |det Ω(y0)|, and proj_dX = Ω(y0)^{-T} (y1-y0) for each sample.
    Matches the external construction: pick p faces with smallest ε_t and set Ω = diag(√ε) @ V,
    where V = qr(H_sel^T).Q^T and ε = proj_scale * (k*rho)/(1 + k*rho).
    Shapes:
      y0,y1: (B,p)
      H: (R,p), h: (R,)
    Returns:
      Omega:     (B,p,p)
      det_Omega: (B,1)
      proj_dX:   (B,p)
    """
    device = y0.device
    B, p = y0.shape
    R = H.shape[0]
    assert H.shape[1] == p

    # distances ρ_t to faces (rows of H are already unit-norm from assemble_Wb_for_shrinkage)
    # ρ = |H y - h|
    rho = torch.abs(H @ y0.T - h.view(-1, 1)).T  # (B, R)

    # ε(ρ) = proj_scale * (k ρ)/(1 + k ρ)
    k = float(dist_multiplier)
    eps_all = proj_scale * (k * rho) / (1.0 + k * rho)               # (B, R)

    Omega_list = []
    det_list = []
    proj_dX_list = []

    dy = (y1 - y0)                                                  # (B,p)
    I = torch.eye(p, device=device)

    for b in range(B):
        # choose p faces with smallest ε
        eps_b = eps_all[b]                                          # (R,)
        idx = torch.topk(eps_b, k=p, largest=False).indices         # (p,)
        eps_sel = torch.clamp(eps_b.index_select(0, idx), min=eps_floor)  # (p,)

        H_sel = H.index_select(0, idx)                              # (p,p)
        # V = qr(H_sel^T).Q^T
        Q = torch.linalg.qr(H_sel.T, mode="reduced").Q              # (p,p)
        V = Q.transpose(0, 1)                                       # (p,p)

        Dsqrt = torch.sqrt(eps_sel).diag()                          # (p,p)
        Omega_b = Dsqrt @ V                                         # (p,p)
        # det may be negative via V; use absolute value (external uses abs)
        det_b = torch.det(Omega_b).abs().view(1, 1)                 # (1,1)

        # proj_dX = Ω^{-T} dX = (Ω^T)^{-1} dX
        proj_b = torch.linalg.solve(Omega_b.T, dy[b])               # (p,)

        Omega_list.append(Omega_b)
        det_list.append(det_b)
        proj_dX_list.append(proj_b)

    Omega = torch.stack(Omega_list, dim=0)                          # (B,p,p)
    det_Omega = torch.cat(det_list, dim=0)                          # (B,1)
    proj_dX = torch.stack(proj_dX_list, dim=0)                      # (B,p)
    return Omega, det_Omega, proj_dX


def shrunk_gaussian_nll_endogenous(
    y0, y1, dt,
    H, h,                        # faces (row-normalized) for shrink
    dist_multiplier: float, proj_scale: float,
    model,
    diagonal_diffusion: bool = True,
    eps: float = 1e-12,
    # ------- BC (boundary correction) additions -------
    t_idx: torch.Tensor | None = None,   # [B]
    corr_dirs=None,                      # [T,K,p] or [T,K*p]
    epsmu=None,                          # [T,K]
    bc_lambda: float = 1.0,
    bc_eps_floor: float = 1e-8,
    bc_cap: float | None = None,
    bc_epsmu_cutoff: float = 1e-3,
):
    """
    Same kernel as shrunk_gaussian_nll, but computes Ω(y0), |detΩ(y0)| and Ω^{-T} dX in-graph.
    """
    B, p = y0.shape
    dt = dt.view(-1)

    # base drift at y0
    mu = model.f(0.0, y0)          # [B,p]

    # optional boundary correction to drift
    if (corr_dirs is not None) and (epsmu is not None) and (t_idx is not None):
        mu = _apply_boundary_correction(
            mu, t_idx, corr_dirs, epsmu,
            bc_lambda=bc_lambda, bc_eps_floor=bc_eps_floor,
            bc_cap=bc_cap, bc_epsmu_cutoff=bc_epsmu_cutoff, device=y0.device
        )

    # build Ω(y0), detΩ(y0), and Ω^{-T} dX
    Omega, det_Omega, proj_dX = _build_omega_proj_from_state(
        y0, y1, H, h, dist_multiplier=dist_multiplier, proj_scale=proj_scale
    )

    # project μ: proj_mu = Ω^{-T} μ
    proj_mu = torch.linalg.solve(Omega.transpose(1, 2), mu.unsqueeze(-1)).squeeze(-1)  # [B,p]

    g_out = model.g(0.0, y0)

    log_det_Omega = torch.log(det_Omega.clamp_min(eps)).squeeze(-1)  # [B]
    if diagonal_diffusion:
        diff    = g_out.clamp_min(eps)                                # [B,p]
        logdetΣ = 2.0 * torch.sum(torch.log(diff), dim=-1)            # [B]
        sol_dX  = proj_dX / diff                                      # [B,p]
        sol_mu  = proj_mu / diff                                      # [B,p]
    else:
        L = g_out                                                     # [B,p,p] lower-tri
        logdetΣ = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        sol_dX  = torch.linalg.solve_triangular(L, proj_dX.unsqueeze(-1), upper=False).squeeze(-1)
        sol_mu  = torch.linalg.solve_triangular(L, proj_mu.unsqueeze(-1), upper=False).squeeze(-1)

    l1 = 2.0 * log_det_Omega + 2.0 * logdetΣ
    quad1 = (sol_dX.pow(2).sum(dim=-1)) / dt
    quad2 = (sol_mu.pow(2).sum(dim=-1)) * dt
    quad3 = -2.0 * (sol_mu * sol_dX).sum(dim=-1)
    return l1 + quad1 + quad2 + quad3



def likelihood_training_endogenous(out,
                                   n_epochs, batch_size, lr=1e-3,
                                   zero_drift: bool = False,
                                   data: str = "xi",
                                   model=None,
                                   dist_multiplier: float = 1.0,
                                   proj_scale: float = 0.9):
    """
    Like likelihood_training but computes Ω(y0) inside the loss.
    Calibrate dist_multiplier exactly as before.
    """
    if data == 'xi':
        X_train, X_test, _ = build_xi_training_data(out)
    elif data == "lattice":
        X_train, X_test, _ = build_lattice_training_data(out)

    n_train, dim = X_train.shape
    n_test, _ = X_test.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (NeuralSDE(dim, zero_drift=zero_drift) if model is None else model).to(device)

    # tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test  = torch.from_numpy(X_test).float().to(device)

    # dt (keep same units used across the repo: seconds as "years" here)
    sec_per_year = 1
    tt_tr = out.C_train.index.values; tt_te = out.C_test.index.values
    dt_tr = (np.diff(tt_tr).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_te = (np.diff(tt_te).astype('timedelta64[s]').astype(np.float64) / sec_per_year)
    dt_tr_t = torch.from_numpy(dt_tr[:max(0, n_train-1)].copy()).float().to(device).unsqueeze(1)
    dt_te_t = torch.from_numpy(dt_te[:max(0, n_test-1)].copy()).float().to(device).unsqueeze(1)

    # shrink geometry: faces for Ω
    poly = compute_factor_polytope_vertices(out, xi_builder=None, k_box=6.0, verbose=False, return_mappings=True)
    H_np, h_np = poly["H"], poly["h"]
    H_t = torch.from_numpy(H_np).float().to(device)
    h_t = torch.from_numpy(h_np).float().to(device)

    # drift BC geometry (optional; same as your existing trainer)
    W, b = assemble_Wb_for_shrinkage(poly, include_box=True)
    X_interior, corr_dirs_flat, epsmu = calc_drift_correction(W, b, X=build_factor_path(out, xi_builder=None),
                                                              epsmu_star=10.0, rho_star=1e-5)
    m_faces = W.shape[0]; p = dim
    corr_dirs = corr_dirs_flat.reshape(-1, m_faces, p) if corr_dirs_flat.ndim == 2 and corr_dirs_flat.shape[1] == m_faces*p else corr_dirs_flat
    corr_dirs_t = torch.from_numpy(corr_dirs).float().to(device) if corr_dirs is not None else None
    epsmu_t     = torch.from_numpy(epsmu).float().to(device) if epsmu is not None else None

    opt = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for idx in torch.randperm(n_train - 1, device=device).split(batch_size):
            y0 = X_train.index_select(0, idx)
            y1 = X_train.index_select(0, idx + 1)
            dtb = dt_tr_t.index_select(0, idx).view(-1, 1)

            nll = shrunk_gaussian_nll_endogenous(
                y0, y1, dtb,
                H=H_t, h=h_t,
                dist_multiplier=dist_multiplier, proj_scale=proj_scale,
                model=model, diagonal_diffusion=True,
                t_idx=idx, corr_dirs=corr_dirs_t, epsmu=epsmu_t,
            )
            loss = nll.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            epoch_loss += loss.item() * y0.size(0)

        # full-train/test eval for curves comparable to existing trainer
        model.eval()
        with torch.no_grad():
            y0_tr, y1_tr = X_train[:-1], X_train[1:]
            nll_tr = shrunk_gaussian_nll_endogenous(
                y0_tr, y1_tr, dt_tr_t,
                H=H_t, h=h_t, dist_multiplier=dist_multiplier, proj_scale=proj_scale,
                model=model, diagonal_diffusion=True,
                t_idx=torch.arange(y0_tr.size(0), device=device), corr_dirs=corr_dirs_t, epsmu=epsmu_t
            ).mean().item()
            y0_te, y1_te = X_test[:-1], X_test[1:]
            nll_te = shrunk_gaussian_nll_endogenous(
                y0_te, y1_te, dt_te_t,
                H=H_t, h=h_t, dist_multiplier=dist_multiplier, proj_scale=proj_scale,
                model=model, diagonal_diffusion=True
            ).mean().item()
        train_losses.append(nll_tr); test_losses.append(nll_te)
        print(f"Epoch {epoch+1}/{n_epochs}  Train NLL: {nll_tr:.4e}  Test NLL: {nll_te:.4e}")

    return train_losses, test_losses, model, X_train

