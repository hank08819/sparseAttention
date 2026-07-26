"""
sparsemax.py — Sparsemax activation (Martins & Astudillo, 2016).

Sparsemax is the L₂-projection of z onto the probability simplex.
Unlike softmax, it produces exactly-zero probabilities for irrelevant
positions, enabling interpretable sparse attention.

Reference:
  A. F. T. Martins and R. F. Astudillo, "From softmax to sparsemax:
  A sparse model of attention and multi-label classification,"
  in Proc. ICML, 2016, pp. 1614–1623.
"""

import torch


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation along `dim`.

    Parameters
    ----------
    z : torch.Tensor
        Input logits, any shape.
    dim : int
        Dimension along which to apply sparsemax (default: last).

    Returns
    -------
    torch.Tensor
        Sparse probability distribution; shape identical to `z`.
        Entries are non-negative and sum to 1 along `dim`.
        Unlike softmax, entries can be exactly 0.

    Notes
    -----
    Algorithm (Martins & Astudillo, 2016, Algorithm 1):
      1. Sort z descending to get z_sorted.
      2. Find k̂ = max {k : 1 + k·z[k] > Σ_{j≤k} z[j]}.
      3. τ = (Σ_{j≤k̂} z_sorted[j] − 1) / k̂  computed via masked sum
         (avoids torch.gather with dynamic LongTensor indices, which
          serialises the GPU pipeline on MPS and CUDA for small dims).
      4. Return max(z − τ, 0).
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    dim_size = z.shape[dim]

    k = torch.arange(1, dim_size + 1, dtype=z.dtype, device=z.device)
    shape = [1] * z.dim()
    shape[dim] = dim_size
    k = k.view(shape)

    cumsum = torch.cumsum(z_sorted, dim=dim)
    # Boolean mask: position j is in the support iff 1 + k[j]*z_sorted[j] > cumsum[j]
    mask = (1.0 + k * z_sorted > cumsum).to(z.dtype)
    # k̂ and τ via masked reductions — no gather / no CPU↔MPS sync
    k_hat   = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_sum = (z_sorted * mask).sum(dim=dim, keepdim=True)
    tau = (tau_sum - 1.0) / k_hat
    return torch.clamp(z - tau, min=0.0)
