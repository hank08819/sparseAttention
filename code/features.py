"""
features.py — Feature engineering pipeline for MSCA-SBiGRU.

Three-stage pipeline:
  1. Interpolate COH and Volatility from raw OHLCV bars.
  2. Pearson correlation filter to remove redundant features.
  3. PCA importance ranking to select the top-K predictive features.

Usage::

    from msca_sparsity.features import build_features
    close, feat_matrix = build_features("BTC_5m_P1.csv", n_features=7)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# ── OHLCV feature interpolation ───────────────────────────────────────────────

def _add_coh(df: pd.DataFrame) -> pd.DataFrame:
    """Close-off-High: position of close within intra-bar range ∈ [−1, +1]."""
    rng = df["high"] - df["low"]
    df["COH"] = np.where(rng == 0, 0.0, 2 * (df["high"] - df["close"]) / rng - 1)
    return df


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Intra-bar volatility = (high − low) / open."""
    df["Volatility"] = np.where(
        df["open"] == 0, 0.0, (df["high"] - df["low"]) / df["open"]
    )
    return df


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """14-period RSI via exponential moving average of gains and losses."""
    c = df["close"].values.astype(float)
    delta = np.diff(c, prepend=c[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    def _ema(x, n):
        out = np.full_like(x, np.nan)
        out[n - 1] = x[:n].mean()
        a = 1.0 / n
        for i in range(n, len(x)):
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out

    ag = _ema(gain, period)
    al = _ema(loss, period)
    rs = np.where(al == 0, 100.0, ag / (al + 1e-10))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period - 1] = 50.0
    df["RSI14"] = rsi
    return df


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].values.astype(float)
    lr1 = np.diff(np.log(np.where(c == 0, 1e-8, c)), prepend=np.nan)
    df["ret1"] = pd.Series(lr1).fillna(0).values
    lr5 = np.log(c / np.roll(c, 5))
    lr5[:5] = 0
    df["ret5"] = lr5
    return df


def _add_vol_ratio(df: pd.DataFrame) -> pd.DataFrame:
    v  = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
    vm = pd.Series(v).rolling(20, min_periods=1).mean().values
    df["vol_ratio"] = np.where(vm == 0, 1.0, v / vm)
    return df


def interpolate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add COH, Volatility, RSI14, ret1, ret5, vol_ratio to a raw OHLCV DataFrame."""
    df = df.copy()
    df = _add_coh(df)
    df = _add_volatility(df)
    df = _add_rsi(df)
    df = _add_returns(df)
    df = _add_vol_ratio(df)
    return df


# ── Pearson correlation filter ────────────────────────────────────────────────

def pearson_filter(df: pd.DataFrame, target: str = "close",
                   target_thresh: float = 0.95,
                   inter_thresh: float = 0.90) -> list:
    """Return feature names surviving a two-stage Pearson redundancy filter.

    Stage 1: drop features with |corr(feature, target)| ≥ target_thresh
             (essentially duplicate the target).
    Stage 2: iteratively drop one of each highly correlated pair, keeping
             the one more correlated with the target.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    corr = df[numeric_cols + [target]].corr().abs()
    keep = [c for c in numeric_cols if corr.loc[c, target] < target_thresh]

    to_remove: set = set()
    for i, ci in enumerate(keep):
        for cj in keep[i + 1:]:
            if cj not in to_remove and corr.loc[ci, cj] > inter_thresh:
                if corr.loc[ci, target] >= corr.loc[cj, target]:
                    to_remove.add(cj)
                else:
                    to_remove.add(ci)
    return [c for c in keep if c not in to_remove]


# ── PCA feature importance ranking ────────────────────────────────────────────

def pca_importance(df: pd.DataFrame, features: list,
                   var_threshold: float = 0.95) -> pd.Series:
    """PCA-based feature importance: I(j) = Σ_k π_k · v_{kj}².

    Parameters
    ----------
    df : pd.DataFrame
    features : list of str
    var_threshold : float
        Fraction of explained variance to retain (default 0.95).

    Returns
    -------
    pd.Series
        Importance scores, sorted descending.
    """
    X = MinMaxScaler().fit_transform(df[features].values)
    pca = PCA()
    pca.fit(X)
    K = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold)) + 1
    K = min(K, len(features))
    pi     = pca.explained_variance_ratio_[:K]
    pi     = pi / pi.sum()
    scores = (pi[:, None] * pca.components_[:K] ** 2).sum(axis=0)
    return pd.Series(scores, index=features).sort_values(ascending=False)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_features(path: str, n_features: int = 7):
    """Load a raw CSV, run the full feature-engineering pipeline.

    Parameters
    ----------
    path : str
        Path to a 5-minute OHLCV CSV with columns
        [timestamp, open, high, low, close, volume].
    n_features : int
        Number of features to select (top by PCA importance).

    Returns
    -------
    close : np.ndarray, shape (N,)
        Raw close prices.
    feat_matrix : np.ndarray, shape (N, n_features)
        Selected feature matrix (unscaled; scale per train split).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = interpolate_features(df)
    surviving = pearson_filter(df, target="close")

    # If Pearson filter leaves fewer than n_features, fall back to the full
    # numeric pool ranked by PCA importance (guarantees exactly n_features).
    if len(surviving) < n_features:
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        all_numeric = [c for c in all_numeric
                       if c != "close" and df[c].isna().sum() == 0]
        surviving = all_numeric

    top = pca_importance(df, surviving).head(n_features).index.tolist()
    return df["close"].values.astype(float), df[top].values.astype(float)
