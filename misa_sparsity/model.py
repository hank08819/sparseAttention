"""
model.py — MISA: Multi-scale Inter-signal Sparse Attention over a bidirectional GRU.

Architecture
============
Three temporal scales (5-min, 15-min, 30-min) each encoded by an independent
BiGRU + sparsemax temporal attention, then fused via a sparse scale-attention
layer.  A second asset (BTC as price leader) provides cross-asset context
through multi-head cross-attention with a binary-regularised gate.

Sparse mechanisms
-----------------
1. Sparsemax temporal attention: weights can be exactly 0, forcing focus on
   informative bars.  Prevents attention spread in trending regimes.
2. Sparse scale fusion: sparsemax over 3 scales → model can completely ignore
   irrelevant granularities.
3. Binary gate regularisation: λ·𝔼[g(1−g)] pushes cross-asset gate toward
   {0, 1}, not stuck at 0.5.

Scale pooling (non-overlapping average):
  scale 1 (k=1):  T=30 (raw 5-min)
  scale 2 (k=3):  T=10 (15-min aggregate)
  scale 3 (k=6):  T=5  (30-min aggregate)

Forward signature
-----------------
pred, attn_scale1, gate_reg_loss = model(x_tgt, x_ca)

where gate_reg_loss is the auxiliary sparsity term added to MSE during training.
"""

import torch
import torch.nn as nn

from .sparsemax import sparsemax


def _avg_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping average pool along the time axis: (B, T, F) → (B, T//k, F)."""
    if k == 1:
        return x
    B, T, F = x.shape
    T2 = T // k
    return x[:, :T2 * k, :].reshape(B, T2, k, F).mean(dim=2)


class _SparseScaleEncoder(nn.Module):
    """BiGRU + sparsemax (or softmax) temporal attention for one time scale."""

    def __init__(self, input_dim: int, hidden: int, dropout: float,
                 use_sparsemax: bool = True):
        super().__init__()
        D = hidden * 2
        self.gru          = nn.GRU(input_dim, hidden, num_layers=1,
                                   bidirectional=True, batch_first=True)
        self.drop         = nn.Dropout(dropout)
        self.W_a          = nn.Linear(D, D, bias=True)
        self.v            = nn.Linear(D, 1, bias=False)
        self.use_sparsemax = use_sparsemax

    def forward(self, x: torch.Tensor):
        """x: (B, T_scale, F) → context (B, D), alpha (B, T_scale)."""
        H, _ = self.gru(x)
        H    = self.drop(H)
        e    = self.v(torch.tanh(self.W_a(H))).squeeze(-1)
        alpha = sparsemax(e, dim=-1) if self.use_sparsemax \
                else torch.softmax(e, dim=-1)
        ctx   = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return ctx, alpha


class MISABiGRU(nn.Module):
    """Multi-scale Inter-signal Sparse Attention (MISA).

    Parameters
    ----------
    feat_dim : int
        Number of input features per time step (default: 7).
    hidden : int
        Hidden size of each BiGRU (output dim = 2·hidden).
    n_heads : int
        Number of attention heads in the cross-asset MHA layer.
    dropout : float
        Dropout probability applied after each BiGRU.
    lambda_gate : float
        Weight of the binary gate regularisation loss (λ_g).
    use_multiscale : bool
        If False, only scale-1 (5-min) encoding is used (ablation B).
    use_cross_asset : bool
        If False, cross-asset attention is disabled (ablation C).
    """

    SCALES = [1, 3, 6]  # pool factors → windows [30, 10, 5]

    def __init__(self, feat_dim: int = 7, hidden: int = 64,
                 n_heads: int = 4, dropout: float = 0.25,
                 lambda_gate: float = 0.05,
                 use_multiscale: bool = True,
                 use_cross_asset: bool = True,
                 use_sparsemax: bool = True):
        super().__init__()
        D = hidden * 2
        self.lambda_gate     = lambda_gate
        self.use_multiscale  = use_multiscale
        self.use_cross_asset = use_cross_asset
        self._use_sparsemax  = use_sparsemax

        self.enc1 = _SparseScaleEncoder(feat_dim, hidden, dropout, use_sparsemax)
        self.enc2 = _SparseScaleEncoder(feat_dim, hidden, dropout, use_sparsemax)
        self.enc3 = _SparseScaleEncoder(feat_dim, hidden, dropout, use_sparsemax)

        self.scale_score = nn.Linear(D, 1, bias=False)

        self.ca_gru  = nn.GRU(feat_dim, hidden, num_layers=1,
                              bidirectional=True, batch_first=True)
        self.ca_drop = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(D, n_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.gate_fc = nn.Linear(D * 2, 1)

        self.ln       = nn.LayerNorm(D)
        self.drop_out = nn.Dropout(dropout)
        self.fc       = nn.Linear(D, 1)

    # ------------------------------------------------------------------
    def _fuse_scales(self, x_tgt: torch.Tensor):
        c1, a1 = self.enc1(x_tgt)
        if not self.use_multiscale:
            return c1, a1
        c2, _ = self.enc2(_avg_pool(x_tgt, 3))
        c3, _ = self.enc3(_avg_pool(x_tgt, 6))
        stack  = torch.stack([c1, c2, c3], dim=1)
        s      = self.scale_score(stack).squeeze(-1)
        w      = (sparsemax(s, dim=-1) if self._use_sparsemax
                  else torch.softmax(s, dim=-1)).unsqueeze(-1)
        return (stack * w).sum(dim=1), a1

    def _cross_asset(self, fused: torch.Tensor, x_ca: torch.Tensor):
        H_ca, _ = self.ca_gru(x_ca)
        H_ca    = self.ca_drop(H_ca)
        cross, _ = self.cross_attn(fused.unsqueeze(1), H_ca, H_ca)
        cross    = cross.squeeze(1)
        gate     = torch.sigmoid(self.gate_fc(torch.cat([fused, cross], dim=-1)))
        gate_reg = (gate * (1.0 - gate)).mean()
        out      = gate * fused + (1.0 - gate) * cross
        return out, gate_reg

    # ------------------------------------------------------------------
    def forward(self, x_tgt: torch.Tensor, x_ca: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x_tgt : torch.Tensor, shape (B, 30, F)
            Target-asset feature sequence (30 × 5-min bars).
        x_ca : torch.Tensor, shape (B, 30, F)
            Cross-asset feature sequence (BTC for ETH/XRP; ETH for BTC).

        Returns
        -------
        pred : (B,)
            Log-return prediction for the next `horizon` bars.
        attn_scale1 : (B, 30)
            Sparse temporal attention weights from scale-1 encoder.
        gate_loss : scalar tensor
            Binary gate regularisation term (add to MSE during training).
        """
        fused, a1 = self._fuse_scales(x_tgt)
        if self.use_cross_asset:
            out, gate_loss = self._cross_asset(fused, x_ca)
            gate_loss = gate_loss * self.lambda_gate
        else:
            out, gate_loss = fused, torch.tensor(0.0, device=x_tgt.device)
        pred = self.fc(self.drop_out(self.ln(out))).squeeze(-1)
        return pred, a1, gate_loss

# ---------------------------------------------------------------------------
# Diagnostic helper added for the offline ICLR revision. It leaves the public
# forward() path unchanged and is used only after training with model.eval().
def _forward_diagnostics(self, x_tgt: torch.Tensor, x_ca: torch.Tensor):
    """Return prediction plus interpretable internal weights for a frozen model.

    This method is deliberately not used by training. It exposes temporal
    attention at all three scales, scale-fusion weights, and the target/cross
    gate so nuisance-resampling diagnostics can distinguish weight sparsity
    from functional prediction sensitivity.
    """
    c1, a1 = self.enc1(x_tgt)
    if self.use_multiscale:
        c2, a2 = self.enc2(_avg_pool(x_tgt, 3))
        c3, a3 = self.enc3(_avg_pool(x_tgt, 6))
        stack = torch.stack([c1, c2, c3], dim=1)
        s = self.scale_score(stack).squeeze(-1)
        sw = sparsemax(s, dim=-1) if self._use_sparsemax else torch.softmax(s, dim=-1)
        fused = (stack * sw.unsqueeze(-1)).sum(dim=1)
    else:
        a2 = a3 = None
        sw = torch.ones((x_tgt.shape[0], 1), device=x_tgt.device, dtype=x_tgt.dtype)
        fused = c1

    gate = None
    if self.use_cross_asset:
        H_ca, _ = self.ca_gru(x_ca)
        H_ca = self.ca_drop(H_ca)
        cross, _ = self.cross_attn(fused.unsqueeze(1), H_ca, H_ca)
        cross = cross.squeeze(1)
        gate = torch.sigmoid(self.gate_fc(torch.cat([fused, cross], dim=-1)))
        out = gate * fused + (1.0 - gate) * cross
    else:
        out = fused
    pred = self.fc(self.drop_out(self.ln(out))).squeeze(-1)
    return {"pred": pred, "a1": a1, "a2": a2, "a3": a3,
            "scale_weights": sw, "gate_target": gate}

MISABiGRU.forward_diagnostics = _forward_diagnostics
