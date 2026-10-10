"""Metrics utilities for classification evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


class AverageMeter:
    """Track a running average and the most recent value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def compute_metrics(y_true, y_pred, average="macro"):
    """Compute accuracy/precision/recall/F1 and the confusion matrix.

    Scalars are converted to native Python floats so the result can be
    serialized to JSON.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average,
                                     zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
