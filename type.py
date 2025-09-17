from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


@dataclass
class Lattice:
    nn: NearestNeighbors
    nodes: np.ndarray              # shape (N, 2) with columns [tau, m]
    tau_grid: np.ndarray           # shape (n_tau,)
    m_global: Optional[np.ndarray] = None
    m_grid: Optional[Dict[float, np.ndarray]] = None
    scaler_tm: Optional[StandardScaler] = None
    
@dataclass
class Surfaces:
    C: pd.DataFrame       # [time x N] normalized call prices (C/F) at lattice nodes
    times: np.ndarray     # datetime64[ns]
    spot: np.ndarray      # aligned spot values (len = len(times))

@dataclass
class NoArbConstraints:
    A: sp.csr_matrix
    b: np.ndarray
