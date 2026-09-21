"""
train.py — Training and evaluation utilities for MISA.

Usage::

    from misa_sparsity import MISABiGRU, Trainer

    model   = MISABiGRU(feat_dim=7)
    trainer = Trainer(model, lr=5e-4, epochs=200, patience=25, device="cpu")
    trainer.fit(X_tr, Xca_tr, y_tr, X_val, Xca_val, y_val)
    metrics = trainer.evaluate(X_te, Xca_te, y_te, close_prices=pte)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from scipy import stats as scipy_stats


def get_device(preference: str = "auto") -> str:
    """Return the best device for MISA training.

    Parameters
    ----------
    preference : {"auto", "cpu", "cuda", "mps"}
        ``"auto"``  → CUDA if available, else CPU.
                      MPS (Apple Silicon) is intentionally skipped in auto
                      mode: benchmarks show MPS is ~2.5× slower than CPU
                      for MISA's small hidden=64, batch=64 workload because
                      MPS kernel dispatch overhead dominates at this scale.
                      Pass ``"mps"`` explicitly to force MPS.
        ``"cuda"``  → NVIDIA GPU (fastest for large hidden / batch sizes).
        ``"mps"``   → Apple Silicon GPU (use only if hidden ≥ 256 or batch ≥ 512).
        ``"cpu"``   → Always CPU; default for reproducibility.

    Examples
    --------
    >>> device = get_device()          # "cuda" on Linux server, "cpu" on Mac
    >>> device = get_device("mps")     # force Apple GPU (large models only)
    >>> device = get_device("cpu")     # always CPU
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        # MPS skipped: slower than CPU for hidden=64 models (dispatch overhead)
        return "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        print("[get_device] CUDA not available, falling back to CPU.")
        return "cpu"
    if preference == "mps" and not torch.backends.mps.is_available():
        print("[get_device] MPS not available, falling back to CPU.")
        return "cpu"
    return preference


class Trainer:
    """Wraps MISABiGRU training with cosine LR schedule and early stopping.

    Parameters
    ----------
    model : MISABiGRU
    lr : float
        Adam learning rate.
    epochs : int
        Maximum training epochs.
    patience : int
        Early-stopping patience (counted after `warmup` epochs).
    warmup : int
        Epochs before early stopping is activated.
    batch_size : int
    device : str
        "cpu", "cuda", or "mps".
    verbose : bool
        Print epoch progress every 30 epochs.
    """

    def __init__(self, model, lr: float = 5e-4, epochs: int = 200,
                 patience: int = 25, warmup: int = 30,
                 batch_size: int = 64, device: str = "cpu",
                 verbose: bool = True):
        self.model      = model.to(device)
        self.lr         = lr
        self.epochs     = epochs
        self.patience   = patience
        self.warmup     = warmup
        self.batch_size = batch_size
        self.device     = device
        self.verbose    = verbose

    # ------------------------------------------------------------------
    def fit(self,
            X_tr:  np.ndarray, Xca_tr:  np.ndarray, y_tr:  np.ndarray,
            X_val: np.ndarray, Xca_val: np.ndarray, y_val: np.ndarray):
        """Train until early stopping or max epochs.

        All arrays are numpy float32; shapes:
          X / Xca : (N, window, n_features)
          y       : (N,)   log-return targets
        """
        dev = self.device
        ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(Xca_tr),
                            torch.tensor(y_tr))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        Xv  = torch.tensor(X_val,  device=dev)
        Xcv = torch.tensor(Xca_val, device=dev)
        yv  = torch.tensor(y_val,  device=dev)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        fn  = nn.MSELoss()
        best_val, best_state, patience_count = float("inf"), None, 0

        for ep in range(self.epochs):
            self.model.train()
            for Xb, Xcab, yb in loader:
                Xb  = Xb.to(dev); Xcab = Xcab.to(dev); yb = yb.to(dev)
                opt.zero_grad()
                p, _, gl = self.model(Xb, Xcab)
                (fn(p, yb) + gl).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            sch.step()

            self.model.eval()
            with torch.no_grad():
                vp, _, _ = self.model(Xv, Xcv)
                val_loss = fn(vp, yv).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            elif ep >= self.warmup:
                patience_count += 1
                if patience_count >= self.patience:
                    if self.verbose:
                        print(f"  Early stop at epoch {ep + 1} | best_val={best_val:.6f}")
                    break

            if self.verbose and (ep + 1) % 30 == 0:
                print(f"  ep {ep + 1:3d} | val={val_loss:.6f} | best={best_val:.6f}")

        self.model.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, Xca: np.ndarray) -> np.ndarray:
        """Return raw log-return predictions, shape (N,)."""
        self.model.eval()
        dev = self.device
        with torch.no_grad():
            pred, _, _ = self.model(torch.tensor(X, device=dev),
                                    torch.tensor(Xca, device=dev))
        return pred.cpu().numpy()

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, Xca: np.ndarray, y: np.ndarray,
                 close_prices: np.ndarray, price_range: float = None):
        """Compute NMSE, RMSE, MAPE, R² on the test set.

        Parameters
        ----------
        X, Xca, y : test arrays
        close_prices : np.ndarray
            The price at the start of each prediction window (p_{t−1}).
        price_range : float, optional
            max(close) − min(close) over the full dataset for NMSE.
            If None, estimated from close_prices.

        Returns
        -------
        dict with keys NMSE, RMSE, MAPE, R2.
        """
        pred = self.predict(X, Xca)
        true_prices = close_prices * np.exp(y)
        pred_prices = close_prices * np.exp(pred)

        mse  = float(np.mean((true_prices - pred_prices) ** 2))
        pr   = float(price_range) if price_range is not None \
               else float(close_prices.max() - close_prices.min())
        nmse = mse / (pr ** 2 + 1e-12) * 1e3
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs((true_prices - pred_prices) / (np.abs(true_prices) + 1e-8))))
        r2   = float(r2_score(true_prices, pred_prices))
        return dict(NMSE=nmse, RMSE=rmse, MAPE=mape, R2=r2)


# ── Diebold–Mariano test ──────────────────────────────────────────────────────

def dm_test(e1: np.ndarray, e2: np.ndarray, horizon: int = 5):
    """Harvey (1997)-corrected Diebold–Mariano test.

    Parameters
    ----------
    e1 : np.ndarray
        Forecast errors of model 1 (e.g. MISA).
    e2 : np.ndarray
        Forecast errors of model 2 (e.g. LSTM baseline).
    horizon : int
        Forecast horizon h (used for the HAC correction).

    Returns
    -------
    stat : float
        Harvey-corrected DM statistic.
    pvalue : float
        Two-sided p-value from a t-distribution with n−1 degrees of freedom.

    Notes
    -----
    Negative stat ⟹ model 1 (e1) is more accurate.
    Positive stat ⟹ model 2 (e2) is more accurate.
    """
    d  = e1 ** 2 - e2 ** 2
    n  = len(d)
    db = d.mean()
    lag = max(1, int(np.ceil(n ** (1 / 3))))

    nw = np.var(d, ddof=1)
    for j in range(1, lag + 1):
        nw += 2 * (1 - j / (lag + 1)) * np.mean((d[j:] - db) * (d[:-j] - db))
    nw = max(nw, 1e-12)

    cf   = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = db / (cf * np.sqrt(nw / n))
    pval = 2 * (1 - scipy_stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(pval)
