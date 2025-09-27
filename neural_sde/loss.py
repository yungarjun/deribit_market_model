import torch.nn as nn
import torch.optim as optim
import torchsde
import torch
from neural_sde.nets import NeuralSDE
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
def likelihood_training(out, n_epochs, batch_size, lr=1e-3, data: str = "xi"):
    if data == 'xi':
        X_train, X_test, names = build_xi_training_data(out)
    elif data == "lattice":
        X_train, X_test, _ = build_lattice_training_data(out)
    
    # Training shapes
    n_train, dim = X_train.shape
    # Test Shapes
    n_test, _ = X_test.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuralSDE(dim).to(device)

    # Convert data to torch tensors on device
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test  = torch.from_numpy(X_test).float().to(device)

    # # dt between consecutive training rows from out
    # tt = out.C_train.index.values
    # dt_train_pairs = (np.diff(tt).astype('timedelta64[s]').astype(np.float64) / (365.0*24*3600))
    # dt_train_pairs = dt_train_pairs[:max(0, n_train - 1)]  # align with X_train pairs
    # dt_pairs_t = torch.from_numpy(dt_train_pairs).float().to(device)
    # # dt test
    # tt_test = out.C_test.index.values
    # dt_test_pairs = (np.diff(tt_test).astype('timedelta64[s]').astype(np.float64) / (365.0*24*3600))
    # dt_test_pairs = dt_test_pairs[:max(0, n_test - 1)]  # align with X_train pairs
    # dt_pairs_t = torch.from_numpy(dt_test_pairs).float().to(device)

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

            # print(var)

            # Negative log likelihood per coordinate
            nll = 0.5 * ((dy - drift * dt) ** 2) / var + torch.log(2 * np.pi * var)
            # nll = ait_sahalia_quasi_nll(model, y0, y1, dt)

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
            nll_t = 0.5 * ((dy_t - drift_t * dt)**2 / var_t + torch.log(2 * np.pi * var_t))
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
# ...existing code...