"""
Simple augmentations to enhance the capability of capturing important features in the first components of the
signature.

This version uses iisignature (CPU/NumPy) instead of signatory (PyTorch/CUDA).
- Input/outputs remain torch.Tensors.
- No autograd/backprop support (intentionally).
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import numpy as np
import iisignature
import torch
#from eSig import exp_Volterra_signature_linear, exp_vsig_stream, vsig_next_from_sig_and_increment
class _IISignatureFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, depth: int):
        depth = int(depth)

        # iisignature is CPU/NumPy; use float64 for stability
        y_cpu = y.detach().to("cpu")
        y_np = np.asarray(y_cpu.numpy(), dtype=np.float64)

        sig_np = iisignature.sig(y_np, depth)  # levels 1..depth (no level-0) :contentReference[oaicite:1]{index=1}
        sig = torch.from_numpy(sig_np).to(device=y.device, dtype=y.dtype)

        ctx.depth = depth
        ctx.save_for_backward(y_cpu)  # store path for backward
        return sig

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (y_cpu,) = ctx.saved_tensors
        depth = ctx.depth

        go_np = np.asarray(grad_output.detach().to("cpu").numpy(), dtype=np.float64)
        y_np  = np.asarray(y_cpu.numpy(), dtype=np.float64)

        # gradient of scalar F wrt path points given dF/d(sig) :contentReference[oaicite:2]{index=2}
        grad_y_np = iisignature.sigbackprop(go_np, y_np, depth)

        grad_y = torch.from_numpy(grad_y_np).to(device=grad_output.device, dtype=grad_output.dtype)
        return grad_y, None

try:
    import iisignature
except ImportError as e:
    raise ImportError(
        "iisignature is not installed. Install with `pip install iisignature` "
        "(or in your environment's package manager)."
    ) from e

__all__ = ['AddLags', 'Concat', 'Cumsum', 'LeadLag', 'Scale']


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll


def lead_lag_transform_with_time(x: torch.Tensor) -> torch.Tensor:
    """
    Lead-lag transformation for a multivariate paths.
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    t_rep = torch.repeat_interleave(t, repeats=3, dim=1)
    x_rep = torch.repeat_interleave(x, repeats=3, dim=1)
    x_ll = torch.cat([
        t_rep[:, 0:-2],
        x_rep[:, 1:-1],
        x_rep[:, 2:],
    ], dim=2)
    return x_ll


def cat_lags(x: torch.Tensor, m: int) -> torch.Tensor:
    q = x.shape[1]
    assert q >= m, 'Lift cannot be performed. q < m : (%s < %s)' % (q, m)
    x_lifted = list()
    for i in range(m):
        x_lifted.append(x[:, i:i + m])
    return torch.cat(x_lifted, dim=-1)



@dataclass
class SignatureConfig:
    depth: int
    basepoint: bool = False
    dt: float = 1.0          # <-- add (needed by exp_Volterra_signature_linear)
    lambd: float = 5.0       # <-- add
    alpha: float = 1.0       # optional, you hard-coded 1.0 before
    augmentations: Tuple = ()  # optional for later


def _prepend_basepoint_zero(y: torch.Tensor) -> torch.Tensor:
    """
    signatory(basepoint=True) prepends an initial point (commonly zeros).
    Here we mimic that behaviour for iisignature.
    """
    b, _, c = y.shape
    bp = torch.zeros((b, 1, c), device=y.device, dtype=y.dtype)
    return torch.cat([bp, y], dim=1)

def iisignature_sig(y: torch.Tensor, cfg) -> torch.Tensor:
    """
    Differentiable signature via iisignature.sig + iisignature.sigbackprop.
    cfg is SignatureConfig (has .depth and .basepoint; optionally .augmentations later).
    """
    # (optional) apply augmentations here if you add them later
    # y = apply_augmentations(y, cfg.augmentations) if hasattr(cfg, "augmentations") else y

    if cfg.basepoint:
        y = _prepend_basepoint_zero(y)

    return _IISignatureFn.apply(y, int(cfg.depth))
import numpy as np
import torch


import numpy as np
import torch

def reconstruct_path_from_windows(x_past: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    x_past: (N,p,d) windows ordered in time, created with the SAME stride.
    Returns the reconstructed prefix path of length T_prefix = p + (N-1)*stride.
    """
    if x_past.ndim == 2:
        x_past = x_past[..., None]
        squeeze_out = True
    elif x_past.ndim == 3:
        squeeze_out = False
    else:
        raise ValueError(f"x_past must be (N,p) or (N,p,d). Got {x_past.shape}")

    N, p, d = x_past.shape
    if not (1 <= stride <= p):
        raise ValueError(f"stride must satisfy 1 <= stride <= p. Got stride={stride}, p={p}")

    first = x_past[0]  # (p,d)
    if N == 1:
        path = first
    else:
        tails = [x_past[i, -stride:, :] for i in range(1, N)]
        path = torch.cat([first] + tails, dim=0)  # (p + (N-1)*stride, d)

    return path.squeeze(-1) if squeeze_out else path


import numpy as np
import torch

def exp_volterra_sig_past(x_past: torch.Tensor, cfg) -> torch.Tensor:
    """
    x_past: (T, C) where C=d or d+1 (if time-augmented, time is last channel)
    Returns: (T, K) streaming VSIG aligned with original time steps.
             If basepoint=True, we prepend a zero point for computation but drop it in the output.
    """
    device, dtype = x_past.device, x_past.dtype
    T, C = x_past.shape

    
    dt = cfg.dt
    
    path = x_past
    use_bp = bool(getattr(cfg, "basepoint", False))
    if use_bp:
        bp = torch.zeros((1, C), device=device, dtype=dtype)  # (1, C)
        path = torch.cat([bp, path], dim=0)                  # (T+1, C)

    path_np = np.asarray(path.detach().cpu().numpy(), dtype=np.float64)
    stream = vsig_stream_slope_convention(
        path_np,         # (T,d) path points
        cfg.dt,             # Δt
        float(cfg.lambd),          # λ > 0
        int(cfg.depth)
    )
    #stream = exp_Volterra_signature_linear(
        #path_np,
        #dt,
        #truncation=int(cfg.depth),
        #alpha=float(cfg.alpha),
        #lambd=float(cfg.lambd),
    #)
    stream = np.asarray(stream, dtype=np.float64)
    if stream.ndim != 2:
        raise ValueError(f"Expected streaming output (T,K). Got {stream.shape}")

    # if basepoint was used, drop the first prefix (basepoint-only) so output aligns to original T
    if use_bp:
        stream = stream[1:]   # (T, K)

    return torch.from_numpy(stream).to(device=device, dtype=dtype)

def augment_window_time_hourly(x: torch.Tensor, t0: float = 0.0) -> torch.Tensor:
    # x: (N, W, d)
    N, W, _ = x.shape
    t = (t0 + torch.arange(W, device=x.device, dtype=x.dtype)).view(1, W, 1).expand(N, W, 1)
    return torch.cat([x, t], dim=-1)
    
import torch

def augment_with_time_torch(x: torch.Tensor, dt_sig: float = 1.0, t0: float = 0.0) -> torch.Tensor:
    """
    Append a time channel with fixed step dt_sig along the window axis.

    x: (..., W, d)
    returns: (..., W, d+1) with time in last channel: t0 + dt_sig * [0,1,...,W-1]
    """
    W = x.shape[-2]
    t = (t0 + dt_sig * torch.arange(W, device=x.device, dtype=x.dtype))  # (W,)

    # reshape to (..., W, 1) and expand to match x's leading dims
    t = t.view(*([1] * (x.ndim - 2)), W, 1).expand(*x.shape[:-1], 1)

    return torch.cat([x, t], dim=-1)

def lead_lag_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Lead–lag transform of a discrete path.

    x: (..., W, d)
    returns: (..., 2W-1, 2d)

    Construction:
      (x0,x0),
      for i=1..W-1: (xi, x_{i-1}), (xi, xi)
    """
    if x.ndim < 2:
        raise ValueError("x must have shape (..., W, d)")
    W, d = x.shape[-2], x.shape[-1]
    if W < 1:
        raise ValueError("Need at least one time point")

    out = torch.empty(*x.shape[:-2], 2 * W - 1, 2 * d, device=x.device, dtype=x.dtype)

    # first point (x0, x0)
    out[..., 0, :d] = x[..., 0, :]
    out[..., 0, d:] = x[..., 0, :]

    # fill pairs
    for i in range(1, W):
        out[..., 2 * i - 1, :d] = x[..., i, :]      # lead updated
        out[..., 2 * i - 1, d:] = x[..., i - 1, :]  # lag old

        out[..., 2 * i, :d] = x[..., i, :]          # lead stays
        out[..., 2 * i, d:] = x[..., i, :]          # lag catches up

    return out

import torch

def lead_lag_path_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Lead–lag transform for a single discrete path.

    x: (T, d)
    returns: (2T-1, 2d)

    Construction:
      (x0,x0),
      for i=1..T-1: (xi, x_{i-1}), (xi, xi)
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (T, d)")
    T, d = x.shape
    if T < 1:
        raise ValueError("Need at least one time point")

    out = torch.empty((2 * T - 1, 2 * d), device=x.device, dtype=x.dtype)

    # first point (x0, x0)
    out[0, :d] = x[0]
    out[0, d:] = x[0]

    for i in range(1, T):
        out[2 * i - 1, :d] = x[i]      # lead updated
        out[2 * i - 1, d:] = x[i - 1]  # lag old

        out[2 * i, :d] = x[i]          # lead stays
        out[2 * i, d:] = x[i]          # lag catches up

    return out





def lead_lag_plus_time_path_torch(
    x: torch.Tensor, dt_sig: float = 1.0, t0: float = 0.0, time_first: bool = False
) -> torch.Tensor:
    """
    Time-augment then lead–lag for a single path.

    x: (T, d)
    returns: (2T-1, 2(d+1))
    """
    x_aug = augment_with_time_path_torch(x, dt_sig=dt_sig, t0=t0)  # (T, d+1)

    if time_first:
        t = x_aug[:, -1:]     # (T,1)
        z = x_aug[:, :-1]     # (T,d)
        x_aug = torch.cat([t, z], dim=-1)

    return lead_lag_path_torch(x_aug)

def lead_lag_plus_time_torch(
    x: torch.Tensor, dt_sig: float = 1.0, t0: float = 0.0, time_first: bool = False
) -> torch.Tensor:
    """
    Do time-augmentation AND lead–lag.

    x: (..., W, d)
    returns: (..., 2W-1, 2(d+1))  where time is treated as an extra channel.

    Steps:
      1) x_aug = [x, t]
      2) lead_lag(x_aug)

    time_first:
      - False (default): x_aug = [x, t] (time last channel)
      - True:            x_aug = [t, x] (time first channel)
    """
    x_aug = augment_with_time_torch(x, dt_sig=dt_sig, t0=t0)  # (..., W, d+1)

    if time_first:
        # move time channel to the front: (..., W, 1+d)
        t = x_aug[..., :, -1:].contiguous()
        z = x_aug[..., :, :-1].contiguous()
        x_aug = torch.cat([t, z], dim=-1)

    return lead_lag_torch(x_aug)

def compute_sig_past(self, x, mode="normal"):
    if mode == "normal":
        x = augment_with_time_torch(x)          # (T, W, d+1)
        return iisignature_sig(x, self.sig_config_past)
    elif mode == "VSIG":
        return exp_volterra_sig_past(x, self.sig_config_past)
    else:
        raise ValueError("mode must be 'normal' or 'VSIG'")



def time_augment_along_T(x_path: torch.Tensor, dt_sig: float = 1.0) -> torch.Tensor:
    # x_path: (T,d)
    T = x_path.shape[0]
    t = (dt_sig * torch.arange(T, device=x_path.device, dtype=x_path.dtype)).view(T, 1)
    return torch.cat([x_path, t], dim=-1)


import torch
import math

def make_level_slices_torch(d: int, N: int):
    sizes = [d**k for k in range(N+1)]
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    slices = [slice(starts[k], starts[k] + sizes[k]) for k in range(N+1)]
    K = starts[-1] + sizes[-1]
    return slices, K

def drift_coeffs_torch_stable(lambd: float, dt: float, N: int, device, dtype):
    """
    c_k(dt) = (1 / λ^k) * P(k+1, λdt)  where P is regularized lower incomplete gamma.
    Returns: (N+1,)
    """
    lam = torch.tensor(lambd, device=device, dtype=dtype)
    x = lam * torch.tensor(dt, device=device, dtype=dtype)

    k = torch.arange(N+1, device=device, dtype=dtype)
    a = k + 1.0

    P = torch.special.gammainc(a, x)  # (N+1,)

    # denom = λ^k (with k=0 -> 1)
    denom = torch.where(k == 0, torch.ones_like(k), lam ** k)
    c = P / denom
    return c
from scipy.special import gammainc
def drift_coeffs_torch(lambd: float, dt: float, N: int, device, dtype):
    """
    Stable computation via regularized lower incomplete gamma:
      c_k(dt) = gammainc(k+1, x) / lambd^k,  x = lambd*dt
    with special-case handling near lambd=0.
    """
    x = float(lambd) * float(dt)

    c = np.empty(N+1, dtype=np.float64)

    # if lambd is extremely small, the true c_k -> 0 for all k (including k=0),
    # because gammainc(k+1, x) ~ x^{k+1}/(k+1)! and division by lambd^k leaves ~ lambd * dt^{k+1}/(k+1)! -> 0
    if abs(lambd) < 1e-12:
        c[:] = 0.0
    else:
        for k in range(N+1):
            c[k] = gammainc(k+1, x) / (lambd**k)

    return torch.tensor(c, device=device, dtype=dtype)
import torch

def drift_coeffs_torch_stable(lambd: float, dt: float, N: int, device, dtype, eps: float = 1e-4):
    """
    Stable c_k(dt) for k=0..N:
      c_k = (1/lambd^k) * (1 - e^{-x} sum_{j=0}^k x^j/j!), x = lambd*dt
          = dt^k * P(k+1,x)/x^k, where P = gammainc for integer k+1.
    We compute P(k+1,x) stably as 1 - e^{-x} poly_k(x) using expm1,
    and divide by x^k (with series for small x).
    """
    x = torch.tensor(lambd * dt, device=device, dtype=dtype)
    m = torch.expm1(-x)          # m = e^{-x} - 1  (stable for small x)
    # e^{-x} = 1 + m

    c = torch.empty(N+1, device=device, dtype=dtype)

    # Build poly_k(x) = sum_{j=0}^k x^j/j! iteratively
    poly = torch.ones_like(x)    # poly_0
    term = torch.ones_like(x)    # x^0/0!

    for k in range(N+1):
        if k == 0:
            poly = torch.ones_like(x)
        elif k == 1:
            term = x
            poly = 1.0 + term
        else:
            term = term * x / float(k)  # x^k/k!
            poly = poly + term

        # P(k+1,x) = 1 - e^{-x} poly_k(x) = 1 - (1+m)poly = -(poly-1) - m*poly
        P = -(poly - 1.0) - m * poly

        # r_k(x) = P / x^k, with stable small-x series: r_k ~ x/(k+1)!  for x->0
        if k == 0:
            r = P  # since x^0=1
        else:
            xk = x**k
            r_asym = x / float(torch.lgamma(torch.tensor(k+2.0)).exp().item())  # x/(k+1)!
            r = torch.where(x.abs() < eps, r_asym, P / xk)

        c[k] = (dt**k) * r

    return c

def tensor_exp_levels_torch(dx: torch.Tensor, N: int):
    """
    dx: (B,d)
    returns list levels s[k] shape (B, d^k) with s[k] = dx^{⊗k}/k!
    """
    B, d = dx.shape
    s = [None]*(N+1)
    s[0] = torch.ones(B, 1, device=dx.device, dtype=dx.dtype)
    prev = s[0]
    for k in range(1, N+1):
        # kron(prev, dx) / k
        prev = (prev.unsqueeze(-1) * dx.unsqueeze(1)).reshape(B, -1) / float(k)
        s[k] = prev
    return s

def tensor_powers_torch(v: torch.Tensor, N: int):
    """
    v: (B,d)
    returns list vpow[k] shape (B, d^k) with vpow[k] = v^{⊗k}
    """
    B, d = v.shape
    out = [None]*(N+1)
    out[0] = torch.ones(B, 1, device=v.device, dtype=v.dtype)
    prev = out[0]
    for k in range(1, N+1):
        prev = (prev.unsqueeze(-1) * v.unsqueeze(1)).reshape(B, -1)
        out[k] = prev
    return out

def chen_product_flat_torch(V_levels, s_levels, N: int):
    """
    V_levels, s_levels are lists of tensors levelwise:
      V_levels[k]: (B, d^k)
      s_levels[k]: (B, d^k)
    returns list W_levels[n] = sum_{k=0}^n V^k ⊗ s^{n-k}
    """
    W = [None]*(N+1)
    for n in range(N+1):
        acc = None
        for k in range(n+1):
            Vk = V_levels[k]          # (B, d^k)
            Snk = s_levels[n-k]       # (B, d^(n-k))
            term = (Vk.unsqueeze(-1) * Snk.unsqueeze(1)).reshape(Vk.size(0), -1)
            acc = term if acc is None else acc + term
        W[n] = acc
    return W
import torch, math

def ratio_coeffs_rk(lambd: float, dt: float, N: int, device, dtype, eps: float = 1e-4):
    """
    r_k(x) = gammainc(k+1, x) / x^k, with x = lambd*dt.
    For k=0: r_0 = gammainc(1,x) = 1 - e^{-x}.
    For small x: r_k ~ x/(k+1)! (so finite and small).
    """
    x = torch.tensor(lambd * dt, device=device, dtype=dtype)
    m = torch.expm1(-x)                 # e^{-x} - 1, stable for small x
    ex = 1.0 + m                        # e^{-x}

    r = torch.empty(N+1, device=device, dtype=dtype)

    poly = torch.ones_like(x)           # sum_{j=0}^k x^j/j!
    term = torch.ones_like(x)

    for k in range(N+1):
        if k == 0:
            poly = torch.ones_like(x)
        elif k == 1:
            term = x
            poly = 1.0 + term
        else:
            term = term * x / float(k)
            poly = poly + term

        P = 1.0 - ex * poly             # gammainc(k+1,x) for integer k+1

        if k == 0:
            r[k] = P
        else:
            # small-x asymptotic: P ~ x^{k+1}/(k+1)! => P/x^k ~ x/(k+1)!
            r_asym = x / float(math.factorial(k+1))
            r[k] = torch.where(x.abs() < eps, r_asym, P / (x**k))

    return r


#@torch.no_grad()
def vsig_update_flat_from_increment_torch(
    Vn_flat: torch.Tensor,
    dx_inc: torch.Tensor,
    dt: float,
    lambd: float,
    d: int,
    N: int,
    c_cached: torch.Tensor = None,   # keep this name for compatibility
    decay_cached: float = None,
):
    """
    One-step update (ratio-coeff version):
      V_{n+1} = e^{-λdt} (V_n ⊗ exp⊗(dx_inc)) + Σ_{k=0}^N r_k(λdt) (dx_inc)^{⊗k}
    """
    if Vn_flat.dim() == 1:
        Vn_flat = Vn_flat.unsqueeze(0)
    if dx_inc.dim() == 1:
        dx_inc = dx_inc.unsqueeze(0)

    device, dtype = Vn_flat.device, Vn_flat.dtype
    sl, K = make_level_slices_torch(d, N)
    if Vn_flat.size(-1) != K:
        raise ValueError(f"Vn_flat has last dim {Vn_flat.size(-1)} but expected {K} (d={d}, N={N})")
    if dx_inc.size(-1) != d:
        raise ValueError(f"dx_inc has last dim {dx_inc.size(-1)} but expected d={d}")

    decay = decay_cached if decay_cached is not None else math.exp(-lambd * dt)

    # IMPORTANT: interpret c_cached as r_cached now (same shape (N+1))
    r = c_cached
    if r is None:
        r = ratio_coeffs_rk(lambd, dt, N, device=device, dtype=dtype)
    else:
        r = r.to(device=device, dtype=dtype)

    V_levels = [Vn_flat[:, sl[k]] for k in range(N+1)]
    s_levels = tensor_exp_levels_torch(dx_inc, N)
    Vchen_levels = chen_product_flat_torch(V_levels, s_levels, N)
    dxpow = tensor_powers_torch(dx_inc, N)

    Vnext_levels = [decay * Vchen_levels[k] + r[k] * dxpow[k] for k in range(N+1)]
    return torch.cat(Vnext_levels, dim=-1)





def augment_with_time_window_unit(x: torch.Tensor, dt: float = 1.0, t0: float = 0.0) -> torch.Tensor:
    """
    Append a time channel with step dt along the window axis.

    x: (..., W, d)
    returns: (..., W, d+1) with time = t0 + dt * [0,1,...,W-1] in last channel
    """
    W = x.shape[-2]
    t = t0 + dt * torch.arange(W, device=x.device, dtype=x.dtype)  # (W,)

    # reshape to (..., W, 1) and expand across batch/leading dims
    t = t.view(*([1] * (x.ndim - 2)), W, 1).expand(*x.shape[:-1], 1)

    return torch.cat([x, t], dim=-1)






# ---------- exact transforms from your code ----------

def concat_x_cumsum(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, d)
    returns y: (B, T, 2d) = [x, cumsum(x)]
    """
    return torch.cat([x, x.cumsum(dim=1)], dim=-1)

def add_lag1(y: torch.Tensor) -> torch.Tensor:
    """
    1-lag augmentation that keeps length T.
    y: (B, T, c)
    returns z: (B, T, 2c) = [y_t, y_{t-1}] with y_{-1} := y_0
    """
    y_lag = torch.cat([y[:, :1, :], y[:, :-1, :]], dim=1)  # pad with first value
    return torch.cat([y, y_lag], dim=-1)

def lead_lag_repo(u: torch.Tensor) -> torch.Tensor:
    """
    Same lead-lag as your repo:
    u: (B, T, c)
    returns: (B, 2T-1, 2c)
    """
    u_rep = torch.repeat_interleave(u, repeats=2, dim=1)          # (B, 2T, c)
    return torch.cat([u_rep[:, :-1, :], u_rep[:, 1:, :]], dim=2)  # (B, 2T-1, 2c)

def transform_like_paper(x: torch.Tensor, scale: float = 1) -> torch.Tensor:
    """
    Implements: (optional Scale) -> concat(x, cumsum(x)) -> add 1 lag -> lead-lag
    x: (B, T, d)
    returns: (B, 2T-1, 8d)
    """
    x = scale * x
    #y = concat_x_cumsum(x**2)   # (B, T, 2d)
    #z = add_lag1(x)          # (B, T, 4d)
    w = lead_lag_repo(x)     # (B, 2T-1, 8d)
    #return z
    #return z
    return w

import torch

def concat_x_cumsum_path(x: torch.Tensor) -> torch.Tensor:
    """
    x: (T, d)
    returns: (T, 2d) = [x, cumsum(x)]
    """
    x = x *0.2
    return torch.cat([x, x.cumsum(dim=0)], dim=-1)

def add_lag1_path(y: torch.Tensor) -> torch.Tensor:
    """
    1-lag augmentation keeping length T.
    y: (T, c)
    returns: (T, 2c) = [y_t, y_{t-1}] with y_{-1} := y_0
    """
    y_lag = torch.cat([y[:1, :], y[:-1, :]], dim=0)  # pad with first value
    return torch.cat([y, y_lag], dim=-1)

def transform_like_paper_path(x: torch.Tensor, scale: float = 0.2) -> torch.Tensor:
    """
    Implements: Scale -> concat(x, cumsum(x)) -> add 1 lag
    x: (T, d)
    returns: (T, 4d)
    """
    x = scale * x
    y = concat_x_cumsum_path(x)   # (T, 2d)
    z = add_lag1_path(y)          # (T, 4d)
    return z
import math
import torch

import torch

@torch.no_grad()
def vsig_windows_cfg_exp(x_win: torch.Tensor, lambd: float, alpha: float, depth: int,
                         dt: float = 1.0, add_time: bool = False, t0: float = 0.0) -> torch.Tensor:
    T, W, d = x_win.shape
    if add_time:
        t = (t0 + dt * torch.arange(W, device=x_win.device, dtype=x_win.dtype)).view(1, W, 1).expand(T, W, 1)
        x_win = torch.cat([x_win, t], dim=-1)  # (T, W, d+1)
    return exp_vsig_stream(x_win, dt=dt, lambd=lambd, alpha=alpha, N=depth)  # (T, K)

def normalize_V_per_level(V: torch.Tensor, sl, eps: float = 1e-6) -> torch.Tensor:
        # sl[k] are slices / index ranges for level k
        outs = []
        for k in range(len(sl)):
            Vk = V[:, sl[k]]  # (B, d^k)
            rms = torch.sqrt((Vk * Vk).mean(dim=1, keepdim=True).detach() + eps)  # (B,1)
            outs.append(torch.tanh(Vk / rms))
        return torch.cat(outs, dim=-1)  # (B, K)

def leadlag_increments_from_dx(dx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    dx: (B, d)
    returns: dxA, dxB each (B, 2d)
      A: (dx, 0)
      B: (0, dx)
    """
    B, d = dx.shape
    zeros = torch.zeros_like(dx)
    dxA = torch.cat([dx, zeros], dim=-1)  # (B, 2d)
    dxB = torch.cat([zeros, dx], dim=-1)  # (B, 2d)
    return dxA, dxB

import torch
from typing import Tuple, List, Optional

def _tensor_dim(d: int, n: int) -> int:
    return d**n

def _sig_dim(d: int, L: int) -> int:
    # total dimension of truncated tensor algebra: sum_{n=0}^L d^n
    return sum(d**n for n in range(L + 1))

def _pack_levels(levels: List[torch.Tensor]) -> torch.Tensor:
    """
    levels: list of tensors [pi_0, pi_1, ..., pi_L], each (M, d^n)
    returns (M, d_sig)
    """
    return torch.cat(levels, dim=-1)

def _append_letter(v_prev: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    """
    v_prev: (M, d^{n-1})
    dx:     (M, d)
    returns: (M, d^n) corresponding to v_prev ⊗ dx (append last letter)
    """
    M, b = v_prev.shape
    d = dx.shape[1]
    # (M, b, d) -> (M, b*d)
    return torch.einsum("mb,md->mbd", v_prev, dx).reshape(M, b * d)

import torch
from typing import Tuple, Optional

import torch
from typing import Tuple

def _prepend_basepoint_zero_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (M,N,d) -> (M,N+1,d) by prepending a zero basepoint.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (M,N,d), got {tuple(x.shape)}")
    M, N, d = x.shape
    z = torch.zeros((M, 1, d), device=x.device, dtype=x.dtype)
    return torch.cat([z, x], dim=1)

def vsig_two_exp_euler(
    x: torch.Tensor,                 # (M, N, d) path values
    L: int,
    lambdas: Tuple[float, float],     # (lambda1, lambda2)
    alphas: Tuple[float, float],      # (alpha1, alpha2)
    dt: float = 1.0,                  # scalar timestep
    streamingmode: int = 2,           # 2 -> return (M,N,d_sig), else endpoint (M,d_sig)
    assume_increments: bool = False,  # if True: x is already increments of shape (M,N-1,d)
    basepoint: bool = True,          # <-- NEW
) -> torch.Tensor:
    """
    Solves up to level L:
        Z^i_k = (1 - lambda_i dt) Z^i_{k-1} + alpha_i * (V_{k-1} ⊗ dx_k)
        V_k   = 1 + Z^1_k + Z^2_k
    with pi_0(Z^i)=0, pi_0(V)=1.

    NEW: if basepoint=True (and assume_increments=False), prepend a zero basepoint to x.
    """

    # --- NEW: basepoint handling (only for path values, not increments) ---
    if basepoint and (not assume_increments):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = _prepend_basepoint_zero_batch(x)

    if x.ndim != 3:
        raise ValueError(f"x must be (M,N,d), got {tuple(x.shape)}")
    device = x.device
    dtype = x.dtype
    M, N, d = x.shape

    lam1, lam2 = float(lambdas[0]), float(lambdas[1])
    a1, a2     = float(alphas[0]),  float(alphas[1])

    # increments
    if assume_increments:
        dx = x
        N_incr = dx.shape[1]
        N_out = N_incr + 1
    else:
        if N < 2:
            raise ValueError("Need N>=2 to form increments.")
        dx = x[:, 1:, :] - x[:, :-1, :]
        N_incr = N - 1
        N_out = N

    # dt broadcast
    dt_k = torch.full((N_incr,), float(dt), device=device, dtype=dtype)

    # allocate levels for Z^1, Z^2, V
    Z1 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    Z2 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    V  = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]

    

    # level 0 fixed
    V[0].fill_(1.0)

    d_sig = _sig_dim(d, L)

    if streamingmode == 2:
        out = torch.empty((M, N_out, d_sig), device=device, dtype=dtype)
        out[:, 0, :] = _pack_levels(V)

    # time stepping
    for k in range(1, N_out):
        
        dx_k = dx[:, k-1, :]
        #fac1 = 1.0 - lam1 * dt_k[k-1]
        #fac2 = 1.0 - lam2 * dt_k[k-1]
        fac1 = torch.exp(-lam1 * dt_k[k-1])
        fac2 = torch.exp(-lam2 * dt_k[k-1])
        w1 = (1 - torch.exp(-lam1*dt_k[k-1])) / (lam1*dt_k[k-1])
        w2 = (1 - torch.exp(-lam2*dt_k[k-1])) / (lam2*dt_k[k-1])

        # V_{k-1}
        V[0].fill_(1.0)
        for n in range(1, L + 1):
            V[n] = Z1[n] + Z2[n]

        # update Z's
        for n in range(1, L + 1):
            incr_n = _append_letter(V[n-1], dx_k)
            Z1[n] = fac1 * Z1[n] + a1 *w1* incr_n
            Z2[n] = fac2 * Z2[n] + a2 *w2* incr_n

        # update V_k
        V[0].fill_(1.0)
        for n in range(1, L + 1):
            V[n] = Z1[n] + Z2[n]

        if streamingmode == 2:
            out[:, k, :] = _pack_levels(V)

    return out if streamingmode == 2 else _pack_levels(V)
    
#specific guyon vol models style signature
def vsig_four_exp_euler(
    x: torch.Tensor,                 # (M, N, d) path values
    L: int,
    lambdas: Tuple[float, float,float,float],     # (lambda1, lambda2)
    alphas: Tuple[float, float,float,float],      # (alpha1, alpha2)
    dt: float = 1.0,                  # scalar timestep
    streamingmode: int = 2,           # 2 -> return (M,N,d_sig), else endpoint (M,d_sig)
    assume_increments: bool = False,  # if True: x is already increments of shape (M,N-1,d)
    basepoint: bool = True,          # <-- NEW
) -> torch.Tensor:
    """
    Solves up to level L:
        Z^i_k = (1 - lambda_i dt) Z^i_{k-1} + alpha_i * (V_{k-1} ⊗ dx^\ell(i)_k), \ell(i)=1 or 2 for two dimensional in
        V_k   = 1 + Z^1_k + Z^2_k+ Z^3_k + Z^4_k
    with pi_0(Z^i)=0, pi_0(V)=1.

    NEW: if basepoint=True (and assume_increments=False), prepend a zero basepoint to x.
    """

    # --- NEW: basepoint handling (only for path values, not increments) ---
    if basepoint and (not assume_increments):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = _prepend_basepoint_zero_batch(x)

    if x.ndim != 3:
        raise ValueError(f"x must be (M,N,d), got {tuple(x.shape)}")
    device = x.device
    dtype = x.dtype
    M, N, d = x.shape

    lam1, lam2,lam3,lam4 = float(lambdas[0]), float(lambdas[1]),float(lambdas[2]),float(lambdas[3])
    a1, a2,a3,a4     = float(alphas[0]),  float(alphas[1]),float(alphas[2]),float(alphas[3])

    # increments
    if assume_increments:
        dx = x
        N_incr = dx.shape[1]
        N_out = N_incr + 1
    else:
        if N < 2:
            raise ValueError("Need N>=2 to form increments.")
        dx = x[:, 1:, :] - x[:, :-1, :]
        N_incr = N - 1
        N_out = N

    # dt broadcast
    dt_k = torch.full((N_incr,), float(dt), device=device, dtype=dtype)

    # allocate levels for Z^1, Z^2, V
    Z1 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    Z2 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    Z3 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    Z4 = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]
    V  = [torch.zeros((M, _tensor_dim(d, n)), device=device, dtype=dtype) for n in range(L + 1)]

    # level 0 fixed
    V[0].fill_(1.0)

    d_sig = _sig_dim(d, L)

    if streamingmode == 2:
        out = torch.empty((M, N_out, d_sig), device=device, dtype=dtype)
        out[:, 0, :] = _pack_levels(V)

    # time stepping
    for k in range(1, N_out):
        dx_k = dx[:, k-1, :]  # (M,d)
        dx_k_1 = torch.zeros_like(dx_k)
        dx_k_1[:, 0] = dx_k[:, 0]         # (dx0, 0, 0, ...)
    
        dx_k_2 = torch.zeros_like(dx_k)
        dx_k_2[:, 1] = dx_k[:, 1]         # (0, dx1, 0, ...)
        fac1 = torch.exp(-lam1 * dt_k[k-1])
        fac2 = torch.exp(-lam2 * dt_k[k-1])
        fac3 = torch.exp(-lam3 * dt_k[k-1])
        fac4 = torch.exp(-lam4 * dt_k[k-1])

        w1 = (1 - torch.exp(-lam1*dt_k[k-1])) / (lam1*dt_k[k-1])
        w2 = (1 - torch.exp(-lam2*dt_k[k-1])) / (lam2*dt_k[k-1])
        w3 = (1 - torch.exp(-lam3*dt_k[k-1])) / (lam3*dt_k[k-1])
        w4 = (1 - torch.exp(-lam4*dt_k[k-1])) / (lam4*dt_k[k-1])

        # V_{k-1}
        V[0].fill_(1.0)
        for n in range(1, L + 1):
            V[n] = Z1[n] + Z2[n] + Z3[n] + Z4[n]

        # update Z's
        for n in range(1, L + 1):
            incr_n_1 = _append_letter(V[n-1], dx_k_1)
            incr_n_2 = _append_letter(V[n-1], dx_k_2)
    
            Z1[n] = fac1 * Z1[n] + a1 *fac1* incr_n_1
            Z2[n] = fac2 * Z2[n] + a2 *fac2* incr_n_1
    
            Z3[n] = fac3 * Z3[n] + a3 *fac3* incr_n_2
            Z4[n] = fac4 * Z4[n] + a4 *fac4* incr_n_2

        # update V_k
        V[0].fill_(1.0)
        for n in range(1, L + 1):
            V[n] = Z1[n] + Z2[n] + Z3[n] + Z4[n]

        if streamingmode == 2:
            out[:, k, :] = _pack_levels(V)

    return out if streamingmode == 2 else _pack_levels(V)

import torch
from typing import List, Optional, Union

def _tensor_dim(d: int, n: int) -> int:
    return d**n

def _sig_dim(d: int, L: int) -> int:
    return sum(d**n for n in range(L + 1))

def _pack_levels(levels: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(levels, dim=-1)

def _prepend_basepoint_zero_batch(x: torch.Tensor) -> torch.Tensor:
    M, N, d = x.shape
    z = torch.zeros((M, 1, d), device=x.device, dtype=x.dtype)
    return torch.cat([z, x], dim=1)

import torch
from typing import Union, Optional, List

def last_letter_damped_sig(
    x: torch.Tensor,                              # (M,N,d) path values OR increments if assume_increments=True
    L: int,
    lambdas: Union[torch.Tensor, list, tuple],     # (d,)
    alphas: Optional[Union[torch.Tensor, list, tuple]] = None,  # (d,)
    dt: float = 1.0,
    streamingmode: int = 2,
    assume_increments: bool = False,
    basepoint: bool = False,
    exact_damping: bool = False,
) -> torch.Tensor:
    """
    Implements (Euler):
      pi_0(V_k)=1
      pi_n(V_k) = (I - dt*Lambda) pi_n(V_{k-1}) + pi_{n-1}(V_{k-1}) ⊗ (alpha ⊙ dx_k)
    where (Lambda v)^{i_1...i_n} = v^{i_1...i_n} * lambda_{i_n}.
    Flatten: last letter fastest axis => reshape (M, d^{n-1}, d).
    """

    if x.ndim != 3:
        raise ValueError(f"x must be (M,N,d), got {tuple(x.shape)}")

    # basepoint (only for path values)
    if basepoint and (not assume_increments):
        x = _prepend_basepoint_zero_batch(x)

    device, dtype = x.device, x.dtype
    M, N, d = x.shape

    lam = torch.as_tensor(lambdas, device=device, dtype=dtype).view(1, d)
    if lam.numel() != d:
        raise ValueError(f"lambdas must have length d={d}, got {lam.numel()}")

    if alphas is None:
        alpha = torch.ones((1, d), device=device, dtype=dtype)
    else:
        alpha = torch.as_tensor(alphas, device=device, dtype=dtype).view(1, d)
        if alpha.numel() != d:
            raise ValueError(f"alphas must have length d={d}, got {alpha.numel()}")

    # increments
    if assume_increments:
        dx = x
        N_incr = dx.shape[1]
        N_out = N_incr + 1
    else:
        if N < 2:
            raise ValueError("Need N>=2 to form increments.")
        dx = x[:, 1:, :] - x[:, :-1, :]
        N_incr = N - 1
        N_out = N

    # damping factors per coordinate
    if exact_damping:
        damp = torch.exp(-lam * float(dt))          # (1,d)
    else:
        damp = 1.0 - lam * float(dt)                # (1,d)

    # allocate V levels
    V: List[torch.Tensor] = [torch.zeros((M, d**n), device=device, dtype=dtype) for n in range(L + 1)]
    V[0].fill_(1.0)

    d_sig = sum(d**n for n in range(L + 1))
    if streamingmode == 2:
        out = torch.empty((M, N_out, d_sig), device=device, dtype=dtype)
        out[:, 0, :] = torch.cat(V, dim=-1)

    # time stepping
    for k in range(1, N_out):
        dx_k = dx[:, k-1, :] * alpha          # (M,d)

        # snapshot previous time step (CRUCIAL)
        V_prev = [v.clone() for v in V]

        V[0].fill_(1.0)
        for n in range(1, L + 1):
            # apply (I - dt*Lambda) to V_prev[n] by last-letter damping
            Vn = V_prev[n].view(M, -1, d) * damp.view(1, 1, d)     # (M, d^{n-1}, d)
            Vn = Vn.reshape(M, -1)

            # tensor product term: V_prev[n-1] ⊗ dx_k
            Vprev = V_prev[n-1].view(M, -1)
            incr = torch.einsum("mb,md->mbd", Vprev, dx_k).reshape(M, -1)

            V[n] = Vn + incr

        if streamingmode == 2:
            out[:, k, :] = torch.cat(V, dim=-1)

    return out if streamingmode == 2 else torch.cat(V, dim=-1)