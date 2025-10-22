import torch.nn as nn
import torch.optim as optim
import torchsde
import torch



class NeuralSDE(torchsde.SDEIto):
    def __init__(self, dim, zero_drift: bool = False):
        super().__init__(noise_type = 'diagonal')
        self.zero_drift = zero_drift

        # Drift
        self.f_net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

        # Diffusion network (output >= 0)
        self.g_net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.Softplus()
        )

    def f(self, t, y):
        if self.zero_drift:
            return torch.zeros_like(y)
        return self.f_net(y)
    
    def g(self, t, y):
        return self.g_net(y)
    

class NeuralSDEWithShrink(torchsde.SDEIto):
    """
    Wraps NeuralSDE diffusion with boundary-aware shrinkage built from H z >= h.
    mode='diag' keeps diagonal noise_type (drop-in). mode='matrix' uses full P (general noise).
    """
    def __init__(self, dim, H: torch.Tensor, h: torch.Tensor, zero_drift: bool = False, mode: str = "diag"):
        noise = "diagonal" if mode == "diag" else "general"
        super().__init__(noise_type=noise)
        self.zero_drift = zero_drift
        self.mode = mode

        self.f_net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
        self.g_net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            nn.Softplus()
        )

        # store H,h, precompute as buffers
        self.register_buffer("H_faces", H.detach().clone())  # (R,d)
        self.register_buffer("h_offs",  h.detach().clone())  # (R,)

    def f(self, t, y):
        return torch.zeros_like(y) if self.zero_drift else self.f_net(y)

    def g(self, t, y):
        # base diagonal diffusion (B,d)
        sig = self.g_net(y)
        from neural_sde.constraints import shrink_matrix_and_diag, shrink_diag_covmatch  # lazy import to avoid cycles
        if self.mode == "diag":
            P = shrink_matrix_and_diag(y, self.H_faces, self.h_offs, mode="diag")  # (B,d)
            s = shrink_diag_covmatch(P, sig)
            return s * sig
        else:
            # full matrix P @ diag(sig)
            P = shrink_matrix_and_diag(y, self.H_faces, self.h_offs, mode="matrix")  # (B,d,d)
            G = P @ torch.diag_embed(sig)  # (B,d,d)
            return G
            
# ------------- Neural Jump SDE ------------------

class NeuralSDEJump(torchsde.SDEIto):
    def __init__(self, dim, zero_drift: bool = False, hidden=256, lam_cap_per_sec: float = 1 / 50 ):
        super().__init__(noise_type='diagonal')
        self.zero_drift = zero_drift
        self.dim = dim
        self.register_buffer('_lam_cap', torch.tensor(lam_cap_per_sec))

        # Drift
        self.f_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )

        # Diffusion (diagonal st.dev >=0)
        self.g_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Softplus()
        )

        # Jump head (diagonal Σ_J)
        self.jump_core = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU()
        )
        self.lam_head = nn.Sequential(    # λ(x) >= 0
            nn.Linear(hidden, 1), nn.Softplus()
        )
        self.mJ_head  = nn.Linear(hidden, dim)      # m_J(x)
        self.sJ_head  = nn.Sequential(              # diag std >= 0
            nn.Linear(hidden, dim), nn.Softplus()
        )

        self.register_buffer("_eps_sig", torch.tensor(1e-6))
        self.register_buffer("_eps_lam", torch.tensor(1e-8))

    # SDE parts
    def f(self, t, y):
        if self.zero_drift:
            return torch.zeros_like(y)
        return self.f_net(y)

    def g(self, t, y):
        return self.g_net(y).clamp_min(self._eps_sig)

    # Jump parts
    def jump_head(self, y):
        core = self.jump_core(y)
        lam  = self.lam_head(core).squeeze(-1).clamp_min(self._eps_lam)  # [B]
        lam = torch.minimum(lam, self._lam_cap.expand_as(lam)) #cap lambda to keep lamda_delta small
        mJ   = self.mJ_head(core)                                        # [B,dim]
        sJ   = self.sJ_head(core).clamp_min(self._eps_sig)               # [B,dim]
        SJ_L = torch.diag_embed(sJ)                                      # [B,dim,dim]
        return lam, mJ, SJ_L