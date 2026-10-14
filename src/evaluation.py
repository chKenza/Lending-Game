#  metrics: AUC, log-loss, calibration
"""Part 2 - evaluation of predicted default probabilities.

Every function takes y (observed default, 0/1) and p (predicted probability
of default), as arrays of the same length.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def auc(y, p):
    """Area under the ROC curve: probability that a defaulter gets a higher
    p than a non-defaulter. Measures ranking only (0.5 = random, 1 = perfect).
    """
    return roc_auc_score(y, p)


def logloss(y, p):
    """Mean of -[y log(p) + (1 - y) log(1 - p)]. Lower is better.
    Sensitive to both ranking and calibration.
    """
    return log_loss(y, _clip(p), labels=[0, 1])


def brier(y, p):
    """Mean squared error (p - y)^2. Lower is better."""
    return brier_score_loss(y, p)


def calibration_slope_intercept(y, p):
    """Fit logit(P(default)) = a + b * logit(p) and return (b, a).

    Perfect calibration is b = 1, a = 0.
    b < 1: predictions too extreme (over-fitting); b > 1: too compressed.
    With b close to 1, a > 0 means risk is under-predicted on average and
    a < 0 over-predicted.
    """
    p = _clip(p)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    # Very large C = no penalty: plain maximum likelihood.
    fit = LogisticRegression(C=1e10, max_iter=1000).fit(x, np.asarray(y))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def compute_metrics(y, p):
    """All scalar metrics for one set of predictions, as a dict."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "n": len(y),
        "auc": auc(y, p),
        "logloss": logloss(y, p),
        "brier": brier(y, p),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "mean_pred": p.mean(),
        "default_rate": y.mean(),
    }


def compare_models(y, preds):
    """Metrics table for several prediction columns.

    preds : DataFrame, or dict name -> array, of predicted probabilities.
    Returns one row per model, sorted by log-loss (best first).
    """
    rows = {name: compute_metrics(y, preds[name]) for name in preds}
    return pd.DataFrame(rows).T.sort_values("logloss")


# ---------------------------------------------------------------------------
# Reliability curve
# ---------------------------------------------------------------------------

def reliability_curve(y, p, n_bins=10, strategy="quantile"):
    """Predicted vs observed default rate by bin of predicted probability.

    strategy="quantile": bins with equal numbers of borrowers (default).
    strategy="uniform" : bins of equal width on [0, 1].

    Returns one row per bin with n, the bin bounds, the mean prediction, the
    observed default rate and gap = observed - predicted (positive = risk is
    under-predicted, same convention as the winner's curse measure of Part 5).
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    if strategy == "quantile":
        # duplicates="drop": tied predictions can make fewer than n_bins bins.
        df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    elif strategy == "uniform":
        df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1),
                           labels=False, include_lowest=True)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    curve = df.groupby("bin").agg(
        n=("y", "size"),
        p_min=("p", "min"),
        p_max=("p", "max"),
        mean_pred=("p", "mean"),
        default_rate=("y", "mean"),
    ).reset_index()
    curve["bin"] = curve["bin"].astype(int)
    curve["gap"] = curve["default_rate"] - curve["mean_pred"]
    return curve


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_indices(y, n_boot, seed):
    """Yield resampled row positions, skipping samples with a single class."""
    rng = np.random.default_rng(seed)
    n, done = len(y), 0
    while done < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            done += 1
            yield idx


def bootstrap_ci(y, p, metric=auc, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap confidence interval of a metric.

    Borrowers are resampled with replacement n_boot times; the interval is
    given by the alpha/2 and 1 - alpha/2 percentiles of the metric.
    Returns a dict with the point estimate, lower and upper bounds.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    stats = [metric(y[i], p[i]) for i in _bootstrap_indices(y, n_boot, seed)]
    lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": metric(y, p), "lower": lower, "upper": upper}


def bootstrap_diff_ci(y, p_a, p_b, metric=auc, n_boot=1000, alpha=0.05,
                      seed=42):
    """Bootstrap confidence interval of metric(p_a) - metric(p_b).

    Paired: both models are scored on the same resampled borrowers, which is
    much more precise than comparing two separate intervals. If the interval
    excludes 0, the difference between the two models is significant.
    """
    y = np.asarray(y)
    p_a, p_b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    stats = [metric(y[i], p_a[i]) - metric(y[i], p_b[i])
             for i in _bootstrap_indices(y, n_boot, seed)]
    lower, upper = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": metric(y, p_a) - metric(y, p_b),
            "lower": lower, "upper": upper}