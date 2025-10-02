import torch.nn as nn
import torch.optim as optim
import torchsde
import torch
from neural_sde.nets import NeuralSDE, NeuralSDEWithShrink
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
            nll = shrunk_gaussian_nll(y0, y1, dt, Omega_b, det_Omega_b, proj_dX_b, model = model, diagonal_diffusion=True)

            loss = nll.mean()

            # print(loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2)
            opt.step()

            epoch_loss += loss.mean().item() * y0.size(0)
        
        train_loss = epoch_loss / (n_train - 1)

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
            nll_t = shrunk_gaussian_nll(y0_test, y1_test, dt, Omega_te, det_Omega_te, proj_dX=proj_dX_te ,model=model, diagonal_diffusion=True)
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


def shrunk_gaussian_nll(
    y0, y1, dt,
    Omega, det_Omega, proj_dX,           # from calc_diffusion_scaling (aligned per step)
    model,
    diagonal_diffusion: bool = True,
    eps: float = 1e-12,
):
    """
    y0,y1: [B,p]
    dt:    [B] or [B,1]
    Omega: [B,p,p]
    det_Omega: [B,1]   (positive)
    proj_dX:   [B,p]   (this is Ω^{-T} dX; if you don't have it, compute it with Omega)
    model.f/.g at (t=0, y0) returning drift [B,p] and diffusion:
        - if diagonal_diffusion: diff diag entries [B,p]  (>=0 via Softplus)
        - else: lower-tri Cholesky L of Σ (B,p,p) with positive diag
    """
    B, p = y0.shape
    dt = dt.view(-1)  # [B]

    # Increments and model evals
    dy    = y1 - y0                      # [B,p]
    mu    = model.f(0.0, y0)             # [B,p]
    g_out = model.g(0.0, y0)

    # Project drift with Ω^{-T} (proj_dX is already Ω^{-T} dX)
    # proj_mu = Ω^{-T} μ
    proj_mu = torch.linalg.solve(Omega.transpose(-1, -2), mu.unsqueeze(-1)).squeeze(-1)  # [B,p]

    # l1: log-determinant pieces
    #   2 * sum(log det Ω) + 2 * sum(log diag(L))  (per-sample)
    log_det_Omega = torch.log(det_Omega.clamp_min(eps)).squeeze(-1)  # [B]

    if diagonal_diffusion:
        # Σ = diag(diff^2) -> Cholesky L = diag(diff)
        diff    = g_out.clamp_min(eps)                 # [B,p], assume Softplus already
        logdetΣ = 2.0 * torch.sum(torch.log(diff), dim=-1)  # [B]
        # Whiten: L^{-1} v  is just v / diff
        sol_dX  = proj_dX / diff                       # [B,p]
        sol_mu  = proj_mu / diff                       # [B,p]
    else:
        # g_out is lower-triangular Cholesky L of Σ (B,p,p)
        L = g_out
        logdetΣ = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # [B]
        # Whiten by solving L z = v  (lower=True)
        sol_dX = torch.linalg.solve_triangular(L, proj_dX.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]
        sol_mu = torch.linalg.solve_triangular(L, proj_mu.unsqueeze(-1), upper=False).squeeze(-1)  # [B,p]

    l1 = 2.0 * log_det_Omega + 2.0 * logdetΣ                         # [B]

    # Quadratic parts (Euler–Gaussian form with shrinkage):
    # l2 = (1/dt) * || L^{-1} Ω^{-T} dX ||^2
    # l3 =  dt     * || L^{-1} Ω^{-T} μ  ||^2
    # l4 = -2      * < L^{-1} Ω^{-T} μ , L^{-1} Ω^{-T} dX >
    quad1 = (sol_dX.pow(2).sum(dim=-1)) / dt                         # [B]
    quad2 = (sol_mu.pow(2).sum(dim=-1)) * dt                         # [B]
    quad3 = -2.0 * (sol_mu * sol_dX).sum(dim=-1)                     # [B]

    nll_per_step = l1 + quad1 + quad2 + quad3                        # [B]
    # (Optional) add + p*log(dt) and + p*log(2π) constants; they don't affect training.
    return nll_per_step