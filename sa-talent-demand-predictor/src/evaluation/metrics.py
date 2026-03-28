"""
Shared model evaluation utilities.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_summary(y_true, y_pred, y_prob=None, label: str = "") -> dict:
    """Return a dict of key classification metrics."""
    result: dict[str, Any] = {
        "label": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        result["auc"] = roc_auc_score(y_true, y_prob)
    return result


def print_summary(metrics: dict) -> None:
    label = metrics.get("label", "Model")
    print(f"\n{'=' * 40}")
    print(f"  {label}")
    print(f"{'=' * 40}")
    for k, v in metrics.items():
        if k in ("label", "confusion_matrix"):
            continue
        print(f"  {k:20s}: {v:.4f}")
    print()
