# own, competitor, full-information models
"""Part 2 - default probability models.

Builds on Part 1 (src/data_prep.py): the data are prepared with the Part 1
prep functions and split with the Part 1 split, so every model sees the same
80,000 training and 20,000 validation loans.

Three model families, each reading the prepared data version it was built for:

    "logit"   logistic regression        logreg_prep_data
    "spline"  spline logistic regression logreg_prep_data
    "gb"      gradient boosted trees     boosting_prep_data

Five information sets (which private signals the model may use):

    "own"      public variables + signal3      (our model: we are lender 3)
    "full"     public variables + signals 1-3  (full-information benchmark)
    "lender1"  public variables + signal1      (our guess of lender 1's model)
    "lender2"  public variables + signal2      (our guess of lender 2's model)
    "public"   public variables only           (no signals)

Typical use from a notebook in analysis/:

    data = load_data()
    y_train = data["boosting"]["train"][TARGET]
    folds = make_folds(y_train)
    oof = oof_predict_all(data, folds)              # training set, out of fold
    val = fit_predict_all(data, on="validation")    # fitted on train only
    apps = fit_predict_all(data, on="applications", info_sets=["own"])
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

from src.config import DATA_PREDICTIONS, NEW_APPS_R1, OUR_LENDER, PAST_LOANS, SEED
from src.data_prep import TARGET, boosting_prep_data, logreg_prep_data, split_data

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
                  min_samples_leaf=100, l2_regularization=1.0, seed=SEED):
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
# Information sets and feature columns
# ---------------------------------------------------------------------------

# Which lenders' private signals each information set may use.
INFORMATION_SETS = {
    "own": (OUR_LENDER,),
    "full": (1, 2, 3),
    "lender1": (1,),
    "lender2": (2,),
    "public": (),
}

# Public variables used by each model family. Identifiers (`row_id`, `id`)
# and the target are never features.
PUBLIC_FEATURES = {
    # Linear in the log-odds: log income and its square instead of raw income
    # (EDA 3.1), and `employment_employed` left out as the reference level so
    # the coefficients read as differences from employed applicants.
    "logit": [
        "sex_F", "married",
        "employment_retired", "employment_student", "employment_unemployed",
        "income_log1p", "income_log1p_sq", "income_zero",
        "unemployed_x_income_zero",
    ],
    # Same, without the square: the spline on income_log1p already fits the
    # curvature, and the square would duplicate it.
    "spline": [
        "sex_F", "married",
        "employment_retired", "employment_student", "employment_unemployed",
        "income_log1p", "income_zero",
        "unemployed_x_income_zero",
    ],
    # Trees are invariant to monotone transforms and need no reference level:
    # every column of boosting_prep_data except the identifier and target.
    "gb": [
        "sex_F", "married",
        "employment_employed", "employment_retired", "employment_student",
        "employment_unemployed",
        "income", "income_zero",
    ],
}

# Continuous columns expanded into splines by the "spline" model. Signals
# outside the information set are skipped (see _PresentColumns).
SPLINE_COLUMNS = ["income_log1p", "signal1", "signal2", "signal3"]

# Prepared data version (Part 1 prep function) read by each model family.
MODEL_DATA = {"logit": "logreg", "spline": "logreg", "gb": "boosting"}

# Default factory of each model family.
MODEL_FACTORIES = {
    "logit": make_logistic,
    "spline": lambda: make_spline_logistic(SPLINE_COLUMNS),
    "gb": make_boosting,
}


def signal_features(lenders):
    """Signal columns for the given lenders: the raw signal and its flag
    `signalk_missing` (1 when the lender received no signal, EDA 2.4)."""
    return [col for k in lenders
            for col in (f"signal{k}", f"signal{k}_missing")]


def feature_columns(model_name, info_set):
    """Feature columns of one model family on one information set."""
    return PUBLIC_FEATURES[model_name] + signal_features(INFORMATION_SETS[info_set])


# ---------------------------------------------------------------------------
# Data: Part 1 preparation and split
# ---------------------------------------------------------------------------

PREP_FUNCTIONS = {"logreg": logreg_prep_data, "boosting": boosting_prep_data}


def load_data():
    """Prepared train, validation and applications for each data version.

    Rebuilt from the raw files with the Part 1 functions, exactly as section 9
    of analysis/eda.ipynb does: `row_id` numbers the past loans in file order,
    then each version is prepared and split with the same seed. Returns

        {"logreg":   {"train": ..., "validation": ..., "applications": ...},
         "boosting": {"train": ..., "validation": ..., "applications": ...}}

    Past loans are indexed by `row_id` and applications by `id`, so
    predictions from different versions can be joined on the index.
    """
    past = pd.read_csv(PAST_LOANS)
    past.insert(0, "row_id", np.arange(len(past)))
    applications = pd.read_csv(NEW_APPS_R1)

    data = {}
    for version, prep in PREP_FUNCTIONS.items():
        train, validation = split_data(prep(past))
        data[version] = {
            "train": train.set_index("row_id"),
            "validation": validation.set_index("row_id"),
            "applications": prep(applications).set_index("id"),
        }

    # The Part 1 split is deterministic, so both versions must hold the same
    # rows in the same order; folds built on one then apply to the other.
    for role in ("train", "validation", "applications"):
        if not data["logreg"][role].index.equals(data["boosting"][role].index):
            raise ValueError(f"logreg and boosting {role} rows differ")
    return data


def get_X(data, model_name, info_set, role):
    """Feature matrix of one model on one information set, for one role
    ("train", "validation" or "applications")."""
    frame = data[MODEL_DATA[model_name]][role]
    cols = feature_columns(model_name, info_set)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{role} has no column(s) {missing}: information set "
            f"'{info_set}' cannot be used on it (the applications only "
            f"carry signal{OUR_LENDER})")
    return frame[cols]


def get_y(data, role="train"):
    """Observed defaults of the past loans in one role, indexed by row_id."""
    return data["boosting"][role][TARGET]


# ---------------------------------------------------------------------------
# Cross-validation: fixed folds and out-of-fold predictions
# ---------------------------------------------------------------------------

def make_folds(y, n_splits=5, seed=SEED):
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


def oof_predict_all(data, folds, model_names=None, info_sets=None):
    """Out-of-fold predictions on the training set for every
    (information set, model) pair.

    model_names : model families to run (default: all of MODEL_FACTORIES)
    info_sets   : information sets to run (default: all of INFORMATION_SETS)

    Returns a DataFrame indexed by row_id with one column per pair, named
    "<info set>__<model>", e.g. "own__gb".
    """
    model_names = model_names or list(MODEL_FACTORIES)
    info_sets = info_sets or list(INFORMATION_SETS)
    y = get_y(data, "train")
    out = {}
    for info_set in info_sets:
        for model_name in model_names:
            X = get_X(data, model_name, info_set, "train")
            out[f"{info_set}__{model_name}"] = oof_predict(
                MODEL_FACTORIES[model_name](), X, y, folds)
    return pd.DataFrame(out, index=y.index)


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


def fit_models(data, info_set, model_names=None, fit_on="train"):
    """Fit each model family on one information set.

    fit_on : "train" (default) or "train+validation".
    Returns a dict model name -> fitted model, e.g. {"logit": ..., "gb": ...}.
    """
    if fit_on not in ("train", "train+validation"):
        raise ValueError("fit_on must be 'train' or 'train+validation'")
    model_names = model_names or list(MODEL_FACTORIES)
    fit_roles = fit_on.split("+")
    y_fit = pd.concat([get_y(data, role) for role in fit_roles])
    models = {}
    for model_name in model_names:
        X_fit = pd.concat([get_X(data, model_name, info_set, role)
                           for role in fit_roles])
        models[model_name] = fit_model(MODEL_FACTORIES[model_name](), X_fit, y_fit)
    return models


def fit_predict_all(data, on="validation", model_names=None, info_sets=None,
                    fit_on="train", calibrators=None):
    """Fit every (information set, model) pair and predict one role.

    on          : "validation" or "applications"
    fit_on      : "train" (default) or "train+validation". Validation
                  predictions must come from models fitted on train only, so
                  "train+validation" is refused when on="validation".
    calibrators : optional dict "<info set>__<model>" -> fitted Calibrator
                  (see fit_calibrator), applied to that pair's predictions.

    Returns a DataFrame indexed by row_id (validation) or id (applications)
    with one column per pair, named "<info set>__<model>".
    The applications only carry signal3, so only the "own" and "public"
    information sets can predict them.
    """
    if on == "validation" and fit_on != "train":
        raise ValueError("validation predictions must come from models "
                         "fitted on the training set only")
    model_names = model_names or list(MODEL_FACTORIES)
    info_sets = info_sets or list(INFORMATION_SETS)
    calibrators = calibrators or {}

    out, index = {}, None
    for info_set in info_sets:
        # Check the role carries this information set before fitting anything.
        for model_name in model_names:
            get_X(data, model_name, info_set, on)
        models = fit_models(data, info_set, model_names, fit_on)
        for model_name, model in models.items():
            name = f"{info_set}__{model_name}"
            X_new = get_X(data, model_name, info_set, on)
            out[name] = predict_default(model, X_new, calibrators.get(name))
            index = X_new.index
    return pd.DataFrame(out, index=index)


def save_predictions(preds, filename, y=None):
    """Write predictions to data/predictions/ as CSV, keeping the index
    (row_id or id). If y is given, it is added as a `default` column so the
    file is self-contained for the later parts."""
    DATA_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out = preds.copy()
    if y is not None:
        out.insert(0, TARGET, y.loc[out.index])
    path = DATA_PREDICTIONS / filename
    out.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# Recalibration
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