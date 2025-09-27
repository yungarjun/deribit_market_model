from dataclasses import dataclass, field
from enum import Enum


class DecoderKind(str, Enum):
    HINGE_STATIC = "hinge_static" # Wang's greedy SA with hinge QP penalty
    PLS = "pls" # simple (linear) PLS with CV factor count
    DEEP_PLS = "deep_pls" # NN-based PLS surrogate (optional)


@dataclass
class MicrostructureCfg:
    stale_max_s: float = 300.0           # drop quotes if last update > this (per instrument)
    spread_max: float = 0.1            # (ask-bid)/mid <= spread_max
    depth_q: float = 0.25               # require size >= τ-bucket quantile
    tau_min_minutes: float = 30.0       # τ ≥ this
    tau_max_days: float = 7.0           # τ ≤ this (set None to disable)
    vega_min_pct: float = 0.1          # drop vega below this τ-bucket percentile (0..1)
    mad_k: float = 5.0                  # |x - med| / MAD ≤ k in (τ_bin, m_bin)
    n_tau_buckets: int = 8              # provisional bucketing for adaptive thresholds
    n_m_buckets: int = 12
    calendar_check: bool = True         # enforce c longer τ ≥ shorter τ at same m-bin
    vertical_check: bool = True         # enforce monotone/convex in strike per τ row
    debug: bool = True
    


@dataclass
class Algo1Config:
    # lattice/selection (Deribit path only)
    n_tau: int = 5
    n_m: int = 5
    top_K: int = 50


    # spline
    k_tau: int = 2
    k_m: int = 3
    s_spline: float = 1e-8


    # Stage-0 (log-S SDE)
    d0: int = 15
    stage0_hidden: int = 64
    stage0_epochs: int = 200
    stage0_lr: float = 1e-3
    stage0_batch: int = 512


    # factors
    dda: int = 3 # dynamic-arbitrage factor count
    dst: int = 2 # statistical-accuracy factor count
    n_sa: int = 2 # static-arbitrage factors (only for HINGE_STATIC)
    n_PC_sa: int = 12 # PCA subspace size used in greedy SA decoding


    # penalties (hinge decoder)
    lam_rec: float = 1e-3
    lam_hinge: float = 1.0


    # decoder choice
    decoder: DecoderKind = DecoderKind.HINGE_STATIC


    # misc
    seed: int = 0

    # microstructure filtering config
    micro: MicrostructureCfg = field(default_factory=MicrostructureCfg)

    price_scale: float = 1000 # multiply lattice prices by this to aid training



