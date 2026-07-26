"""
msca_sparsity — Multi-Scale Cross-Asset Sparse Attention for Cryptocurrency Price Discovery
===========================================================================================

Quick start::

    from msca_sparsity import MSCABiGRU, sparsemax, build_features, Trainer

    model = MSCABiGRU(feat_dim=7, hidden=64, n_heads=4, dropout=0.25)
    trainer = Trainer(model, lr=5e-4, epochs=200, patience=25)
    trainer.fit(X_train, X_ca_train, y_train, X_val, X_ca_val, y_val)
    preds = trainer.predict(X_test, X_ca_test)
"""

from .model import MSCABiGRU
from .sparsemax import sparsemax
from .features import build_features, pearson_filter, pca_importance
from .train import Trainer, get_device

__version__ = "1.0.0"

__all__ = [
    "MSCABiGRU",
    "sparsemax",
    "build_features",
    "pearson_filter",
    "pca_importance",
    "Trainer",
    "get_device",
]
