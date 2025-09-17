import numpy as np
import pandas as pd
from typing import Tuple
from scipy.interpolate import SmoothBivariateSpline

def compute_derivatives(Csurf: pd.DataFrame,
                        nodes: np.ndarray,
                        k_tau=2, k_m=3, s=1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit SmoothBivariateSpline(τ, m) per time slice; evaluate ∂_τ C, ∂_m C, ∂²_m C at the nodes.
    """
    τs = nodes[:, 0]
    ms = nodes[:, 1]
    T, N = Csurf.shape
    dC_dtau = np.zeros((T, N))
    dC_dm   = np.zeros((T, N))
    d2C_dm2 = np.zeros((T, N))

    Carr = Csurf.values
    for t in range(T):
        spline = SmoothBivariateSpline(τs, ms, Carr[t], kx=k_tau, ky=k_m, s=s)
        dC_dtau[t] = spline.ev(τs, ms, dx=1, dy=0)
        dC_dm[t]   = spline.ev(τs, ms, dx=0, dy=1)
        d2C_dm2[t] = spline.ev(τs, ms, dx=0, dy=2)
    return dC_dtau, dC_dm, d2C_dm2