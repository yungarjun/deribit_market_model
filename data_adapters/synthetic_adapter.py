import numpy as np
import pandas as pd
from typing import Dict, Tuple
from type import Lattice, Surfaces

def make_lattice_from_arrays(tau_grid: np.ndarray, m_grid: Dict[float, np.ndarray]) -> Lattice:
    nodes = np.vstack([[t, m] for t in tau_grid for m in m_grid[t]])
    return Lattice(nodes=nodes, tau_grid=tau_grid, m_grid=m_grid, nn=None)

def package_surfaces(C: np.ndarray, times, spot, nodes) -> Surfaces:
    # C shape: [T x N] with N == nodes.shape[0]
    df = pd.DataFrame(C, index=pd.to_datetime(times), columns=range(C.shape[1]))
    return Surfaces(C=df, times=df.index.values, spot=np.asarray(spot, float))