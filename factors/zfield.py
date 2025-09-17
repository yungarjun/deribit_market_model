import numpy as np

def build_Z(dC_dtau, dC_dm, d2C_dm2, gamma, drop_last=True):
    Dτ = np.asarray(dC_dtau, float); Dm = np.asarray(dC_dm, float); Dmm = np.asarray(d2C_dm2, float)
    T, N = Dτ.shape
    g = np.asarray(gamma, float).reshape(-1)
    if drop_last:
        L = min(T-1, len(g))
        s2 = (g[:L]**2)[:, None]
        Z = -Dτ[:L] - 0.5*s2*Dm[:L] + 0.5*s2*Dmm[:L]
    else:
        if len(g) < T:
            g_full = np.pad(g, (0, T-len(g)), mode='edge')
        else:
            g_full = g[:T]
        s2 = (g_full**2)[:, None]
        Z = -Dτ - 0.5*s2*Dm + 0.5*s2*Dmm
    return Z