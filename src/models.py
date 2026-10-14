# own, competitor, full-information models
"""Part 2 - default probability models.

All functions work on an already prepared feature matrix (output of Part 1):
a pandas DataFrame with numeric columns only. The information set (public
variables + which signals) is chosen by the caller, by selecting columns.

Typical use:

    folds = make_folds(y_train)
    p_oof = oof_predict(make_boosting(), X_train[cols], y_train, folds)
    model = fit_model(make_boosting(), X_train[cols], y_train)
    p_val = predict_default(model, X_val[cols])
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

# Probabilities are clipped before taking logits, to avoid infinite values.
EPS = 1e-6


def logit(p):
    """Log-odds of a probability, log(p / (1 - p))."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Model factories
# Each factory returns a new, unfitted sklearn estimator with predict_proba.
# ---------------------------------------------------------------------------

def make_logistic(C=1.0):
    """Logistic regression on standardized features.

    Linear in the features it is given, so non-linear effects (log income,
    "signal is zero" flags, ...) must already be columns of X.
    C is the inverse of the L2 penalty strength.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=2000)),
    ])


def make_boosting(learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
                  min_samples_leaf=100, l2_regularization=1.0, seed=42):
    """Gradient boosted trees (histogram based).

    Small trees, a low learning rate and early stopping on an internal 10%
    hold-out: max_iter is only an upper bound on the number of trees.
    The loss is the log-loss, so predicted probabilities are not distorted by
    any class re-weighting.
    """
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


class _PresentColumns:
    """Column selector: keeps the requested columns that exist in X.

    Lets one spline model serve every information set (signal1 is simply
    skipped when X only contains signal3). A class and not a lambda so that
    fitted models can be pickled.
    """

    def __init__(self, columns):
        self.columns = list(columns)

    def __call__(self, X):
        return [c for c in self.columns if c in X.columns]


def make_spline_logistic(spline_cols, n_knots=6, degree=3, C=1.0):
    """Logistic regression with spline terms on the continuous features.

    Each column of `spline_cols` present in X is expanded into a cubic spline
    basis with knots at its quantiles, so the model can fit smooth non-linear
    effects (e.g. the default rate flattening for high signals). All other
    columns enter linearly. Still an additive model in log-odds:
    logit(p) = b0 + sum_j f_j(x_j) + linear terms.
    """
    splines = ColumnTransformer(
        [("spline",
          SplineTransformer(n_knots=n_knots, degree=degree, knots="quantile",
                            include_bias=False),
          _PresentColumns(spline_cols))],
        remainder="passthrough",
    )
    return Pipeline([
        ("splines", splines),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=5000)),
    ])


# ---------------------------------------------------------------------------
# Cross-validation: fixed folds and out-of-fold predictions
# ---------------------------------------------------------------------------

def make_folds(y, n_splits=5, seed=42):
    """Stratified K folds, as a list of (train_idx, test_idx) position arrays.

    Build the folds once and pass the same list to every model and every
    information set: predictions are then comparable row by row.
    """
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def oof_predict(model, X, y, folds):
    """Out-of-fold default probabilities on the training set.

    For each fold, a fresh copy of `model` is fitted on the other folds and
    predicts the held-out fold, so every row is predicted by a model that has
    never seen it. Preprocessing inside the model (scaling, spline knots) is
    re-fitted in each fold too.

    Returns a numpy array aligned with the rows of X.
    """
    y = np.asarray(y)
    p_oof = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y[train_idx])
        p_oof[test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
    if np.isnan(p_oof).any():
        raise ValueError("folds do not cover every row of X")
    return p_oof


def oof_predict_all(factories, info_sets, X, y, folds):
    """Out-of-fold predictions for every (information set, model) pair.

    factories : dict name -> function returning an unfitted model,
                e.g. {"logit": make_logistic, "gb": make_boosting}
    info_sets : dict name -> list of feature columns,
                e.g. {"own": public + ["signal3"], "full": ...}

    Returns a DataFrame indexed like X with one column "<info set>__<model>".
    """
    out = {}
    for set_name, cols in info_sets.items():
        for model_name, factory in factories.items():
            out[f"{set_name}__{model_name}"] = oof_predict(
                factory(), X[cols], y, folds)
    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------------------
# Final fit and prediction
# ---------------------------------------------------------------------------

def fit_model(model, X_train, y_train):
    """Fit a fresh copy of `model` on the whole training set."""
    fitted = clone(model)
    fitted.fit(X_train, np.asarray(y_train))
    return fitted


def predict_default(model, X, calibrator=None):
    """Default probabilities from a fitted model, optionally recalibrated.

    X must have the same columns as the data the model was fitted on.
    """
    p = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        p = calibrator.predict(p)
    return p


# ---------------------------------------------------------------------------
# Optional recalibration
# ---------------------------------------------------------------------------

class Calibrator:
    """Maps raw model probabilities to recalibrated probabilities.

    method="platt"    : logit(p_cal) = a + b * logit(p). Two parameters, keeps
                        the ranking; corrects probabilities that are globally
                        too high/low (a) or too extreme/too flat (b).
    method="isotonic" : non-parametric increasing step function. More flexible
                        but creates ties and can overfit in sparse regions.
    """

    def __init__(self, method="platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method

    def fit(self, p, y):
        p, y = np.asarray(p, dtype=float), np.asarray(y)
        if self.method == "platt":
            # Very large C = no penalty: plain maximum likelihood for (a, b).
            self.model_ = LogisticRegression(C=1e10, max_iter=1000)
            self.model_.fit(logit(p).reshape(-1, 1), y)
            self.intercept_ = float(self.model_.intercept_[0])
            self.slope_ = float(self.model_.coef_[0, 0])
        else:
            self.model_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model_.fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            return self.model_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return self.model_.predict(p)


def fit_calibrator(p_oof, y, method="platt"):
    """Fit a calibrator on out-of-fold predictions of the training set.

    Use out-of-fold predictions only: in-sample predictions are over-confident
    and the validation set must stay untouched. The calibrator is then applied
    to the predictions of the model refitted on the whole training set
    (see predict_default).

    Calibrated OOF predictions are slightly optimistic, since the calibrator
    has seen their labels: judge the benefit on the validation set.
    """
    return Calibrator(method).fit(p_oof, y)