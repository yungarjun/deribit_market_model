import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from .nets import Stage0LogNet

import torch.optim as optim

def make_stage0_dataset(spot: np.ndarray, pcs: np.ndarray, dt_years: np.ndarray):
    spot = np.asarray(spot, float).reshape(-1)
    pcs  = np.asarray(pcs,  float)
    assert len(spot) == len(pcs)
    logS = np.log(spot)
    y = (logS[1:] - logS[:-1]).reshape(-1, 1)  # Δlog S
    X = np.hstack([logS[:-1, None], pcs[:-1]])
    dt = np.asarray(dt_years, float).reshape(-1, 1)
    return X, y, dt

def train_stage0(net: Stage0LogNet,
                 X: np.ndarray, y: np.ndarray, dt: np.ndarray,
                 epochs=200, lr=1e-3, batch=512, scale_X=True, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    scaler = None
    if scale_X:
        scaler = StandardScaler().fit(X)
        Xn = scaler.transform(X)
    else:
        Xn = X

    X_t = torch.from_numpy(Xn).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    dt_t = torch.from_numpy(dt).float().to(device)

    ds = TensorDataset(X_t, y_t, dt_t)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    opt = optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb, dtb in dl:
            mu, sigma = net(xb)
            var = sigma**2 * dtb + (1e-6*dtb + 1e-12)
            resid = yb - mu*dtb
            nll = 0.5 * (resid**2/var + torch.log(var))
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

    net.eval()
    with torch.no_grad():
        mu_all, sigma_all = net(X_t)
        sigma = sigma_all.squeeze().cpu().numpy()
    return scaler, sigma

def sigma_to_gamma(sigma: np.ndarray, S_tm1=None) -> np.ndarray:
    sigma = np.asarray(sigma, float).reshape(-1)
    # S_tm1 = np.asarray(S_tm1, float).reshape(-1)
    # L = min(len(sigma), len(S_tm1))
    # return np.maximum(sigma[:L] * S_tm1[:L], 1e-12)
    return np.maximum(sigma, 1e-12) # gamma is returns * vol, not price vol

