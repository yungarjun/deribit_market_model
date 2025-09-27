import torch.nn as nn
import torch.optim as optim
import torchsde
import torch
from neural_sde.nets import NeuralSDE
import numpy as np
import math
import torch.nn.functional as F

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


def likelihood_training(out, n_epochs, batch_size, lr=1e-3,
                        mode: str = "lamperti_t"):  # {"em_gauss","lamperti_gauss","lamperti_t"}
    """
    Train a NeuralSDE on factor scores with selectable one-step likelihood:
      - "em_gauss": Euler–Maruyama Gaussian (per-dim diagonal σ)
      - "lamperti_gauss": Aït–Sahalia first-order Lamperti, Gaussian kernel
      - "lamperti_t": Aït–Sahalia first-order Lamperti, Student-t kernel (learn ν)

    Returns: train_losses, test_losses, model, X_train_np, w_train_last, w_test_last
    """
    X_train_np, X_test_np, _ = build_xi_training_data(out)

    # Standardise by train
    mu = X_train_np.mean(0, keepdims=True)
    sd = X_train_np.std(0, keepdims=True) + 1e-9
    X_train_np = (X_train_np - mu) / sd
    X_test_np  = (X_test_np  - mu) / sd

    # Shapes, tensors
    n_train, dim = X_train_np.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralSDE(dim).to(device)

    X_train = torch.from_numpy(X_train_np).float().to(device)
    X_test  = torch.from_numpy(X_test_np ).float().to(device)

    # dt in years
    sec_per_year = 365.0 * 24 * 3600
    tt_tr = out.C_train.index.values
    tt_te = out.C_test.index.values
    dt_tr_np = (np.diff(tt_tr).astype("timedelta64[s]").astype(np.float64) / sec_per_year)
    dt_te_np = (np.diff(tt_te).astype("timedelta64[s]").astype(np.float64) / sec_per_year)

    Y0_tr, Y1_tr = X_train[:-1], X_train[1:]
    Y0_te, Y1_te = X_test[:-1],  X_test[1:]
    DT_tr = torch.from_numpy(dt_tr_np[: Y0_tr.shape[0]].copy()).float().to(device).unsqueeze(1)
    DT_te = torch.from_numpy(dt_te_np[: Y0_te.shape[0]].copy()).float().to(device).unsqueeze(1)

    n_pairs_tr = Y0_tr.shape[0]

    # t degrees of freedom (only used/learned in lamperti_t)
    use_t = (mode == "lamperti_t")
    if use_t:
        t_nu = StudentTNu(init_nu=6.0, learnable=True).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(t_nu.parameters()), lr=lr)
    else: 
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    eps = 1e-9

    def loss_and_w(y0, y1, dt, train_flag=True):
        """Return (scalar_loss, w_matrix) following selected mode."""
        if mode == "em_gauss":
            drift = model.f(0.0, y0)          # (B,d)
            diff  = model.g(0.0, y0)          # (B,d) diagonal σ
            dy    = y1 - y0                   # (B,d)
            var   = (diff**2) * dt + eps      # (B,d)
            nll   = 0.5 * ((dy - drift*dt)**2 / var + torch.log(2*np.pi*var))  # (B,d)
            # standardized innovations for QQ: z = (dy - mu dt)/(σ sqrt(dt))
            w     = (dy - drift*dt) / (diff*torch.sqrt(dt) + 1e-12)
            return nll.sum(-1).mean(), w.detach()

        elif mode == "lamperti_gauss":
            nll_mat, w = ait_sahalia_quasi_nll_with_w(model, y0, y1, dt)  # (B,d), (B,d)
            return nll_mat.sum(-1).mean(), w.detach()

        elif mode == "lamperti_t":
            nu_val = t_nu() if train_flag else t_nu()  # same ν for eval view
            nll_mat, w = ait_sahalia_quasi_t_nll_with_w(model, y0, y1, dt, nu_val)  # (B,d),(B,d)
            return nll_mat.sum(-1).mean(), w.detach()

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ------------- training loop -------------
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        nsamp = 0

        for idx in torch.randperm(n_pairs_tr, device=device).split(batch_size):
            y0 = Y0_tr.index_select(0, idx)
            y1 = Y1_tr.index_select(0, idx)
            dt = DT_tr.index_select(0, idx)

            loss, _ = loss_and_w(y0, y1, dt, train_flag=True)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if use_t:
                torch.nn.utils.clip_grad_norm_(t_nu.parameters(), 2.0)
            opt.step()

            bs = y0.size(0)
            epoch_loss += loss.item() * bs
            nsamp += bs

        # evaluate on full train/test with the SAME kernel
        model.eval()
        with torch.enable_grad():
            tr_loss, w_train_last = loss_and_w(Y0_tr, Y1_tr, DT_tr, train_flag=False)
            te_loss, w_test_last  = loss_and_w(Y0_te, Y1_te, DT_te, train_flag=False)

        train_losses.append(tr_loss.item())
        test_losses.append(te_loss.item())
        print(f"Epoch {epoch+1}/{n_epochs}  Train NLL: {tr_loss:.4e}  Test NLL: {te_loss:.4e}")

    return train_losses, test_losses, model, X_train_np, w_train_last, w_test_last, mu, sd


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
    # Ensure autograd is enabled (needed for sigma' wrt inputs)
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

def ait_sahalia_quasi_nll_with_w(model, y0, y1, dt, eps_dt=1e-10, eps_sig=1e-10):
    # Gaussian Lamperti (per-dim), returns (nll_mat, w)
    with torch.enable_grad():
        y0_req = y0.detach().clone().requires_grad_(True)
        sigma0 = model.g(0.0, y0_req).clamp_min(eps_sig)
        drift0 = model.f(0.0, y0_req)
        sigp0  = _diag_sigma_grad(model, y0_req)
        mu_L   = drift0 / sigma0 - 0.5 * sigp0

        dz     = (y1 - y0) / sigma0
        sigma1 = model.g(0.0, y1).clamp_min(eps_sig)
        logJ   = torch.log(sigma1)

        dtc = dt.clamp_min(eps_dt)
        w   = (dz - mu_L * dtc) / torch.sqrt(dtc)

        nll = 0.5 * (w**2 + torch.log(2 * math.pi * dtc)) + logJ  # (B,d)
        return nll, w

def ait_sahalia_quasi_t_nll_with_w(model, y0, y1, dt, nu_val, eps_dt=1e-10, eps_sig=1e-10):
    # Student-t Lamperti (per-dim independent), returns (nll_mat, w)
    with torch.enable_grad():
        y0_req = y0.detach().clone().requires_grad_(True)
        sigma0 = model.g(0.0, y0_req).clamp_min(eps_sig)
        drift0 = model.f(0.0, y0_req)
        sigp0  = _diag_sigma_grad(model, y0_req)
        mu_L   = drift0 / sigma0 - 0.5 * sigp0

        dz     = (y1 - y0) / sigma0
        sigma1 = model.g(0.0, y1).clamp_min(eps_sig)
        logJ   = torch.log(sigma1)

        dtc = dt.clamp_min(eps_dt)
        w   = (dz - mu_L * dtc) / torch.sqrt(dtc)

        nu  = nu_val
        log_c = (torch.lgamma((nu + 1)/2) - torch.lgamma(nu/2)
                 - 0.5 * torch.log(nu * math.pi))
        log_pt = log_c - 0.5*(nu + 1) * torch.log1p((w**2)/nu)  # (B,d)
        nll = 0.5 * torch.log(dtc) + logJ - log_pt              # (B,d)
        return nll, w



# --- Student-t kernel on Lamperti innovations --------------------------------
class StudentTNu(nn.Module):
    """Keeps ν > 2 by ν = 2 + softplus(η). Set learnable=False to fix ν."""
    def __init__(self, init_nu=6.0, learnable=True):
        super().__init__()
        self.learnable = learnable
        if learnable:
            eta0 = math.log(math.exp(init_nu - 2.0) - 1.0)
            self.eta = nn.Parameter(torch.tensor(eta0, dtype=torch.float32))
        else:
            self.register_buffer("nu_fixed", torch.tensor(init_nu, dtype=torch.float32))
    def forward(self):
        return (2.0 + F.softplus(self.eta)) if self.learnable else self.nu_fixed


def ait_sahalia_quasi_t_nll(model, y0, y1, dt, nu_val, eps_dt=1e-10, eps_sig=1e-10):
    """
    First-order Lamperti quasi-likelihood with a Student-t kernel on w.
      y0,y1: (B,d), dt: (B,1), nu_val: scalar tensor (ν>2)
    Returns per-sample per-dim NLL: (B,d)
    """
    with torch.enable_grad():
        # Evaluate drift, sigma, sigma' at y0
        y0_req = y0.detach().clone().requires_grad_(True)
        sigma0 = model.g(0.0, y0_req).clamp_min(eps_sig)      # (B,d)
        drift0 = model.f(0.0, y0_req)                         # (B,d)
        sigp0  = _diag_sigma_grad(model, y0_req)              # (B,d)
        mu_L   = drift0 / sigma0 - 0.5 * sigp0                # (B,d)

        # Lamperti increment and Jacobian at y1
        dz     = (y1 - y0) / sigma0                           # (B,d)
        sigma1 = model.g(0.0, y1).clamp_min(eps_sig)          # (B,d)
        logJ   = torch.log(sigma1)                            # (B,d)

        dtc = dt.clamp_min(eps_dt)                            # (B,1)
        w   = (dz - mu_L * dtc) / torch.sqrt(dtc)             # (B,d)

        # Student-t per-dim log-pdf of w (independent dims)
        nu  = nu_val
        d   = w.shape[1]
        log_c = (torch.lgamma((nu + 1)/2) - torch.lgamma(nu/2)
                 - 0.5 * torch.log(nu * math.pi))             # scalar
        # log p_t(w) per dim
        log_pt = log_c - 0.5*(nu + 1) * torch.log1p((w**2) / nu)  # (B,d)

        # NLL: + 0.5*log Δt + log σ(y1)  - log p_t(w)
        nll = 0.5 * torch.log(dtc) + logJ - log_pt            # (B,d)
        return nll