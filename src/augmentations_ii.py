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







