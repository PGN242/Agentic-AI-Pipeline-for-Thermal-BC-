"""Shared small helpers used by every expert agent + the orchestrator."""

import numpy as np


def calibrate_threshold(y_true, p, default: float = 0.5) -> float:
    """
    Pick the probability threshold in (0, 1) that maximizes balanced
    accuracy on the given (held-out) labels/probabilities.

    Every classifier in this pipeline is trained on a severely imbalanced
    binary task (as few as ~19 minority-class subjects). Even with
    class-balanced sample weights, a fixed 0.5 cutoff on a GradientBoosting
    / softmax output is not guaranteed to be the balanced-accuracy-optimal
    split point on such small samples — so we search for it explicitly
    on data the model didn't see during weight/coefficient fitting.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    if len(set(y_true.tolist())) < 2:
        return default

    candidates = np.unique(np.concatenate([p, [0.0, 1.0]]))
    best_t, best_score = default, -1.0
    for t in candidates:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            continue
        bacc = 0.5 * (tp / n_pos + tn / n_neg)
        if bacc > best_score:
            best_score, best_t = bacc, float(t)
    return best_t


def signed_margin(p: float, threshold: float) -> float:
    """
    Distance of probability `p` from a calibrated decision `threshold`,
    normalized separately on each side so the result reaches exactly +-1 at
    p=1 / p=0. Positive => past the threshold toward the positive class.

    Using a single symmetric span (max(threshold, 1-threshold)) instead of
    this one-sided version under-scales whichever side is narrower — e.g. a
    threshold of 0.97 makes the "above threshold" band only 0.03 wide, and
    dividing it by the wide 0.97 side instead of its own 0.03 collapses
    every above-threshold probability into a near-zero margin. That bug is
    exactly why this helper exists: every "confidence" and cross-expert
    margin computed in this pipeline must stay comparable across experts
    calibrated at very different thresholds.
    """
    if p >= threshold:
        span = max(1.0 - threshold, 1e-6)
    else:
        span = max(threshold, 1e-6)
    return (p - threshold) / span


def balanced_sample_weights(y) -> np.ndarray:
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes.tolist(), counts.tolist()))
    return np.array([len(y) / (2.0 * freq[v]) for v in y])
