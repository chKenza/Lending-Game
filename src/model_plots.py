"""Part 2 - interpretation and plots of the default probability models.

Used by analysis/models.ipynb. Two kinds of functions:

* interpretation: logistic coefficients, grouped permutation importance and
  partial dependence, which describe what a fitted model has learned;
* plots: one function per figure type, drawing on a matplotlib Axes so the
  notebook can arrange them.

Colours follow the model, never its rank: logistic is always blue, spline
always orange, boosting always aqua, in every figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import SEED
from src.evaluation import logloss, reliability_curve
from src.models import (INFORMATION_SETS, MODEL_FACTORIES, fit_model, get_X,
                        get_y)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#9a9893"  # diagonals and reference lines

MODEL_COLORS = {"logit": "#2a78d6", "spline": "#eb6834", "gb": "#1baf7a"}
MODEL_MARKERS = {"logit": "o", "spline": "s", "gb": "^"}
MODEL_LABELS = {"logit": "Logistic", "spline": "Spline logistic",
                "gb": "Gradient boosting"}

# One colour for single-series charts that compare information sets, distinct
# from the three model colours.
SINGLE_COLOR = "#6250d6"

INFO_SET_LABELS = {
    "own": "Own (signal 3)",
    "full": "Full (signals 1-3)",
    "lender1": "Lender 1 (signal 1)",
    "lender2": "Lender 2 (signal 2)",
    "public": "Public only",
}


def set_style():
    """Matplotlib defaults for every figure of the notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def logit_coefficients(model, columns):
    """Coefficients of a fitted make_logistic() pipeline.

    std_coef   : change in log-odds for a one standard deviation increase
                 (comparable across variables, used for ranking)
    raw_coef   : change in log-odds for a one unit increase
    odds_ratio : exp(raw_coef); for a 0/1 variable, how the odds of default
                 change when it goes from 0 to 1
    """
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["logit"]
    std_coef = logistic.coef_[0]
    raw_coef = std_coef / scaler.scale_
    return pd.DataFrame({"std_coef": std_coef, "raw_coef": raw_coef,
                         "odds_ratio": np.exp(raw_coef)}, index=list(columns))


def feature_groups(columns):
    """Group the prepared columns by the raw variable they come from.

    Permuting a group jointly keeps each row internally consistent: income,
    log income, its square, the income-zero flag and the unemployed x zero
    income interaction move together, as do a signal and its missing flag.
    """
    groups = {}
    for col in columns:
        if col.startswith("signal"):
            group = col.split("_")[0]
        elif col.startswith("employment_"):
            group = "employment"
        elif col.startswith("income") or col == "unemployed_x_income_zero":
            group = "income"
        elif col == "sex_F":
            group = "sex"
        else:
            group = col
        groups.setdefault(group, []).append(col)
    return groups


def grouped_permutation_importance(model, X, y, n_repeats=5, seed=SEED):
    """How much worse the log-loss gets when one variable is shuffled.

    Each group of columns (see feature_groups) is permuted jointly across
    rows, which breaks its link with default while keeping its distribution.
    The model is not refitted. Use rows the model was not trained on.

    Returns a DataFrame indexed by group with the mean and standard deviation
    of the log-loss increase, in percent of the unshuffled log-loss.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = logloss(y, model.predict_proba(X)[:, 1])
    rows = {}
    for group, cols in feature_groups(X.columns).items():
        increases = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            increases.append(logloss(y, model.predict_proba(X_perm)[:, 1]) - base)
        rows[group] = {"increase_pct": 100 * np.mean(increases) / base,
                       "std_pct": 100 * np.std(increases) / base}
    return pd.DataFrame(rows).T.sort_values("increase_pct", ascending=False)


def set_income(X, income):
    """Copy of X with every applicant's income set to `income` (> 0 or 0),
    updating all the income columns the model uses consistently."""
    X = X.copy()
    zero = int(income == 0)
    if "income" in X:
        X["income"] = income
    if "income_log1p" in X:
        X["income_log1p"] = np.log1p(income)
    if "income_log1p_sq" in X:
        X["income_log1p_sq"] = np.log1p(income) ** 2
    if "income_zero" in X:
        X["income_zero"] = zero
    if "unemployed_x_income_zero" in X:
        X["unemployed_x_income_zero"] = X["employment_unemployed"] * zero
    return X


def set_signal(X, lender, value):
    """Copy of X with signal `lender` set to `value`; value 0 means "no
    signal received", so the missing flag is set accordingly."""
    X = X.copy()
    X[f"signal{lender}"] = value
    X[f"signal{lender}_missing"] = int(value == 0)
    return X


def partial_dependence(model, X, setter, grid):
    """Average predicted default probability when every applicant in X is
    given the value v of one variable, for each v in grid.

    setter : function (X, v) -> modified copy of X, e.g.
             lambda X, v: set_signal(X, 3, v)
    """
    return np.array([model.predict_proba(setter(X, v))[:, 1].mean()
                     for v in grid])


def holdout_importance(data, info_set, folds, fold=0, n_repeats=5):
    """Grouped permutation importance of every model family on one
    information set, measured honestly: each model is fitted on the training
    folds except `fold` and evaluated on `fold`, which it has not seen.

    Returns a dict model name -> grouped_permutation_importance table.
    """
    y = get_y(data, "train")
    fit_idx, eval_idx = folds[fold]
    out = {}
    for name, factory in MODEL_FACTORIES.items():
        X = get_X(data, name, info_set, "train")
        model = fit_model(factory(), X.iloc[fit_idx], y.iloc[fit_idx])
        out[name] = grouped_permutation_importance(
            model, X.iloc[eval_idx], y.iloc[eval_idx], n_repeats)
    return out


def dependence_curves(models, data, info_set, n_rows=3000, n_grid=30,
                      seed=SEED):
    """Partial dependence of every fitted model on income and on each signal
    of the information set, averaged over the same random training rows.

    Returns a dict variable -> {"grid", "curves", and for signals
    "no_signal"}, the input of plot_dependence_panels.
    """
    rows = get_y(data, "train").sample(n_rows, random_state=seed).index
    X = {name: get_X(data, name, info_set, "train").loc[rows] for name in models}

    # Income grid over the positive incomes actually observed (1st to 99.5th
    # percentile), so the curves are not extrapolations.
    income = data["boosting"]["train"]["income"]
    low, high = income[income > 0].quantile([0.01, 0.995])
    income_grid = np.geomspace(low, high, n_grid)
    out = {"income": {
        "grid": income_grid,
        "curves": {name: partial_dependence(model, X[name], set_income,
                                            income_grid)
                   for name, model in models.items()},
    }}
    signal_grid = np.linspace(0.02, 1.0, n_grid)
    for k in INFORMATION_SETS[info_set]:
        def setter(X_, v, k=k):
            return set_signal(X_, k, v)
        out[f"signal{k}"] = {
            "grid": signal_grid,
            "curves": {name: partial_dependence(model, X[name], setter,
                                                signal_grid)
                       for name, model in models.items()},
            "no_signal": {name: partial_dependence(model, X[name], setter,
                                                   [0.0])[0]
                          for name, model in models.items()},
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _legend(ax, **kwargs):
    ax.legend(labelcolor=INK_SECONDARY, **kwargs)


def plot_permutation_importance(importances, ax, title=None):
    """Grouped horizontal bars: one group per variable, one bar per model.

    importances : dict model name -> output of grouped_permutation_importance
    """
    models = list(importances)
    order = (pd.concat(importances, axis=1).xs("increase_pct", axis=1, level=1)
             .mean(axis=1).sort_values().index)
    height = 0.8 / len(models)
    y = np.arange(len(order))
    for i, name in enumerate(models):
        imp = importances[name].reindex(order)
        ax.barh(y + (i - (len(models) - 1) / 2) * height, imp["increase_pct"],
                height=height * 0.9, xerr=imp["std_pct"],
                color=MODEL_COLORS[name], label=MODEL_LABELS[name],
                error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1})
    ax.set_yticks(y, order)
    ax.axvline(0, color=REFERENCE, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("increase in log-loss when the variable is shuffled (%)")
    ax.set_title(title or "Which variables each model relies on")
    _legend(ax, loc="lower right")


def plot_logit_coefficients(coefs, ax, title=None,
                            exclude=("income_log1p", "income_log1p_sq")):
    """Horizontal bars of standardized logistic coefficients.
    Positive = more risk.

    `income_log1p` and its square are excluded by default: they are almost
    collinear, so their two large coefficients of opposite sign only make
    sense together (the income partial dependence curve shows their joint
    effect) and would hide every other bar.
    """
    coefs = coefs.drop(index=[c for c in exclude if c in coefs.index])
    coefs = coefs.sort_values("std_coef")
    colors = [MODEL_COLORS["logit"]] * len(coefs)
    ax.barh(coefs.index, coefs["std_coef"], color=colors, height=0.6)
    ax.axvline(0, color=REFERENCE, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in log-odds of default per standard deviation")
    ax.set_title(title or "Logistic regression: standardized coefficients")


def plot_partial_dependence(curves, grid, ax, title, xlabel, logx=False,
                            no_signal=None, base_rate=None, legend=True):
    """Partial dependence lines, one per model.

    curves    : dict model name -> array of average predicted probabilities
    no_signal : optional dict model name -> average probability with no
                signal received, drawn as a separate marker left of the axis
    base_rate : optional observed default rate, drawn as a reference line
    """
    for name, curve in curves.items():
        ax.plot(grid, 100 * curve, color=MODEL_COLORS[name],
                label=MODEL_LABELS[name])
    if no_signal:
        x0 = -0.1
        offsets = np.linspace(-0.025, 0.025, len(no_signal))
        for dx, (name, value) in zip(offsets, no_signal.items()):
            ax.plot(x0 + dx, 100 * value, MODEL_MARKERS[name], ms=8,
                    color=MODEL_COLORS[name], mec=SURFACE, mew=1.5)
        ax.axvline(-0.04, color=GRID, lw=1)
        ax.set_xlim(-0.16, 1.02)
        ax.set_xticks([x0, 0.2, 0.4, 0.6, 0.8, 1.0],
                      ["none", "0.2", "0.4", "0.6", "0.8", "1.0"])
    if base_rate is not None:
        ax.axhline(100 * base_rate, color=REFERENCE, lw=1, ls="--",
                   label="observed default rate")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("average predicted default probability (%)")
    ax.set_title(title)
    if legend:
        _legend(ax)


def plot_dependence_panels(curves, base_rate=None):
    """One partial dependence panel per variable (output of
    dependence_curves), on a shared y-axis so effect sizes compare."""
    n = len(curves)
    ncols = 1 if n == 1 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows),
                             sharey=True, squeeze=False)
    for i, (ax, (var, c)) in enumerate(zip(axes.flat, curves.items())):
        if var == "income":
            plot_partial_dependence(
                c["curves"], c["grid"], ax, "Income",
                "annual income (EUR, log scale)", logx=True,
                base_rate=base_rate, legend=(i == 0))
            ax.set_xticks([1e3, 1e4, 1e5], ["1k", "10k", "100k"])
        else:
            k = var.replace("signal", "")
            plot_partial_dependence(
                c["curves"], c["grid"], ax, f"Signal {k}",
                f"value of signal {k} (none = no signal received)",
                no_signal=c["no_signal"], base_rate=base_rate,
                legend=(i == 0))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_reliability(y, preds, ax, title=None, n_bins=10):
    """Reliability diagram: observed vs predicted default rate by decile of
    predicted probability, one line per column of preds.

    preds : dict or DataFrame, model name -> predicted probabilities. Keys
            that are not model names are drawn in SINGLE_COLOR.
    """
    top = 0
    for name in preds:
        curve = reliability_curve(y, preds[name], n_bins=n_bins)
        color = MODEL_COLORS.get(name, SINGLE_COLOR)
        ax.plot(100 * curve["mean_pred"], 100 * curve["default_rate"],
                marker=MODEL_MARKERS.get(name, "o"), color=color,
                mec=SURFACE, mew=1.2, label=MODEL_LABELS.get(name, name))
        top = max(top, curve["mean_pred"].max(), curve["default_rate"].max())
    lim = 100 * top * 1.05
    ax.plot([0, lim], [0, lim], color=REFERENCE, lw=1, ls="--",
            label="perfect calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted default probability (%)")
    ax.set_ylabel("observed default rate (%)")
    ax.set_title(title or "Calibration by decile of predicted risk")
    _legend(ax, loc="upper left")


def plot_prediction_distribution(preds, ax, title=None, bins=None):
    """Step histograms of predicted probabilities, one per model."""
    bins = bins if bins is not None else np.linspace(0, 0.5, 26)
    for name in preds:
        ax.hist(100 * np.asarray(preds[name]), bins=100 * bins,
                histtype="step", lw=2, color=MODEL_COLORS.get(name, SINGLE_COLOR),
                label=MODEL_LABELS.get(name, name))
    ax.set_xlabel("predicted default probability (%)")
    ax.set_ylabel("number of loans")
    ax.set_title(title or "Distribution of predicted probabilities")
    _legend(ax)


def plot_metric_by_info_set(table, metric, ax, title, ylabel, ylim=None):
    """Grouped vertical bars: information sets on the x-axis, one bar per
    model. table is a metrics_by_information_set output."""
    values = table[metric].unstack("model")
    info_sets = [s for s in INFO_SET_LABELS if s in values.index]
    models = [m for m in MODEL_COLORS if m in values.columns]
    width = 0.8 / len(models)
    x = np.arange(len(info_sets))
    for i, name in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width,
               values.loc[info_sets, name], width=width * 0.9,
               color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.set_xticks(x, [INFO_SET_LABELS[s] for s in info_sets], rotation=15)
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend(ax, loc="upper left", ncols=len(models))


def plot_interval(estimates, ax, title, xlabel, reference=None):
    """Point estimates with confidence intervals, one row per entry.

    estimates : DataFrame indexed by label with columns estimate, lower, upper
    """
    estimates = estimates.iloc[::-1]
    y = np.arange(len(estimates))
    ax.errorbar(estimates["estimate"], y,
                xerr=[estimates["estimate"] - estimates["lower"],
                      estimates["upper"] - estimates["estimate"]],
                fmt="o", ms=7, color=SINGLE_COLOR, ecolor=SINGLE_COLOR,
                elinewidth=2, capsize=4, mec=SURFACE)
    if reference is not None:
        ax.axvline(reference, color=REFERENCE, lw=1, ls="--")
    ax.set_yticks(y, estimates.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
