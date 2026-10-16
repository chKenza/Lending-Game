# pricing rule (step 6)
"""Parts 6-7 - pricing rules and competitor scenarios.

See report/pricing_plan.md for the plan. The module has three parts:

* data: the Part 2 predictions joined with the features the rules need;
* pricing rules (plan step 2): B0 and R1-R7, plus the no-offer cutoff (+C);
* competitor scenarios (plan step 1): what lenders 1 and 2 offer;
* tuning (plan step 3): best-response curve, equilibrium margins, tuning
  scenarios, grid search with weighted profit and max regret, evaluation.

Every rule takes the pricing data (a DataFrame, one row per applicant) and its
parameters, and returns our rate per applicant (a Series, NaN = no offer).
Rules only use our own information: `p_own`, `p_public` and our features,
never the competitors' predictions or signals.

The competitors' default probabilities are always our reconstructed lender
models from Part 2 (`p_lender1`, `p_lender2`). What we do not know is how they
turn p into a rate, so a scenario is an assumption about that step. Every
scenario follows the same game rules as Part 3 (src/fair_pricing.py):

* no offer (NaN) when p >= 0.5;
* no offer when the rate would exceed 100%;
* no rate below 0%.

Typical use from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.pricing_rules import make_scenario, TUNING_SCENARIOS

    r1, r2 = make_scenario("margin", data, margin=0.02)     # T2
    r1, r2 = make_scenario("markup", data, markup=0.15)     # T4
    r1, r2 = make_scenario("T2", data)                      # same as the first line

    data = load_pricing_data("train")
    r3 = rule_fixed_margin(data, m=0.02)                    # our rates, R1
"""

import json
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from src.config import DATA_DIR, DATA_PREDICTIONS, DATA_PROCESSED, SEED
from src.fair_pricing import (MAX_RATE, NO_OFFER_THRESHOLD, break_even_rate,
                              competitor_offers)
from src.simulator import simulate_market, winners_curse

# Lowest rate we ever offer, so very safe borrowers are not priced near 0%.
# 1% for now; 2% is under discussion (report/pricing_plan.md).
RATE_FLOOR = 0.01

# Probabilities are clipped before taking logits, to avoid infinite values.
EPS = 1e-6

# Our features used by the rules (R5-R7), from data/processed/*_general.csv.
# The competitors' signals (signal1, signal2) are deliberately not loaded.
OWN_FEATURES = ["sex_F", "married", "employment_retired", "employment_student",
                "employment_unemployed", "income", "income_zero",
                "signal3", "signal3_missing"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pricing_data(role):
    """Part 2 predictions joined with our own features, indexed by row_id.

    role : "train" (out-of-fold predictions, for tuning) or "validation".
    Columns: default, p_own, p_full, p_lender1, p_lender2, p_public and
    OWN_FEATURES. p_lender1 / p_lender2 / p_full are only for building
    competitor scenarios; no rule may use them.
    """
    files = {"train": ("train_oof_selected.csv", "train_general.csv"),
             "validation": ("validation_selected.csv", "validation_general.csv")}
    if role not in files:
        raise ValueError("role must be 'train' or 'validation'")
    preds_file, features_file = files[role]
    preds = pd.read_csv(DATA_PREDICTIONS / preds_file, index_col="row_id")
    features = pd.read_csv(DATA_PROCESSED / features_file, index_col="row_id")
    data = preds.join(features[OWN_FEATURES], how="left")
    if data[OWN_FEATURES].isna().any().any():
        raise ValueError("some predictions have no matching features (row_id)")
    return data


# ---------------------------------------------------------------------------
# Pricing rules (plan step 2)
# ---------------------------------------------------------------------------

def be(p):
    """Break-even rate p / (1 - p); NaN when p >= 0.5 (Part 3)."""
    return np.asarray(break_even_rate(np.asarray(p, dtype=float)), dtype=float)


def clip_rates(r, p, floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD, index=None):
    """Apply the rules shared by every family to raw rates r.

    * no offer (NaN) when p >= cutoff: 0.5 by default (break-even would reach
      100%), lower for the +C add-on;
    * no offer (NaN) when the rate is above 100%, even if p < cutoff: the
      game forbids such rates, and we do not cap them at 100% (same rule as
      the competitors in src/fair_pricing.py);
    * rates below `floor` are raised to `floor`.
    Returns a Series named "r3" (indexed like p if p is a Series).
    """
    if index is None and isinstance(p, pd.Series):
        index = p.index
    p = np.asarray(p, dtype=float)
    r = np.maximum(np.asarray(r, dtype=float), floor)          # NaN stays NaN
    r = np.where(r <= MAX_RATE, r, np.nan)                     # above 100% -> no offer
    r = np.where(p < cutoff, r, np.nan)                        # NaN p -> no offer
    return pd.Series(r, index=index, name="r3")


def _adjust_p(p, a):
    """logit(p_adj) = logit(p) + a: our risk estimate shifted up (a > 0)."""
    return expit(logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)) + a)


def rule_b0(data, floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """B0, naive baseline: r = be(p)."""
    p = data["p_own"]
    return clip_rates(be(p), p, floor, cutoff)


def rule_fixed_margin(data, m, floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """R1, fixed margin: r = be(p) + m (m in decimals, 0.02 = 2 points)."""
    p = data["p_own"]
    return clip_rates(be(p) + m, p, floor, cutoff)


def rule_markup(data, k, floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """R2, proportional markup: r = (1 + k) * be(p)."""
    p = data["p_own"]
    return clip_rates((1 + k) * be(p), p, floor, cutoff)


def rule_adj_p(data, a, m=0.0, floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """R3, adjusted p: logit(p_adj) = logit(p) + a, then r = be(p_adj) + m.

    No offer when p_adj reaches 0.5 (be(p_adj) would reach 100%) or when the
    rate is above 100%.
    """
    p = data["p_own"]
    r = be(_adjust_p(p, a)) + m        # be is NaN (no offer) when p_adj >= 0.5
    return clip_rates(r, p, floor, cutoff)


def decile_edges(p, n_bins=10):
    """Inner edges of n_bins quantile bins of p (compute them on train)."""
    return np.quantile(np.asarray(p, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])


def smooth_bin_margins(centres, margins, method="isotonic"):
    """Turn per-bin margins into an increasing margin curve m(p).

    centres : a representative p per bin (e.g. the bins' median p)
    margins : the margin chosen for each bin
    method  : "isotonic" (increasing step fit) or "linear" (straight line).
    Returns a function p -> margin, interpolated between the centres and flat
    beyond them.
    """
    centres = np.asarray(centres, dtype=float)
    margins = np.asarray(margins, dtype=float)
    if method == "isotonic":
        fitted = IsotonicRegression(increasing=True).fit_transform(centres, margins)
        return lambda p: np.interp(np.asarray(p, dtype=float), centres, fitted)
    if method == "linear":
        slope, intercept = np.polyfit(centres, margins, 1)
        lo, hi = centres.min(), centres.max()
        return lambda p: intercept + slope * np.clip(np.asarray(p, dtype=float), lo, hi)
    raise ValueError("method must be 'isotonic' or 'linear'")


def rule_bins(data, edges, margins=None, margin_curve=None, floor=RATE_FLOOR,
              cutoff=NO_OFFER_THRESHOLD):
    """R4, margin per risk bin: r = be(p) + m_b, or + m(p) once smoothed.

    edges        : inner bin edges of p_own (decile_edges on train)
    margins      : one margin per bin (step function), or
    margin_curve : a smoothed margin function from smooth_bin_margins.
    The per-bin margins themselves are tuned in plan step 3.
    """
    p = data["p_own"]
    if (margins is None) == (margin_curve is None):
        raise ValueError("give exactly one of margins or margin_curve")
    if margin_curve is not None:
        m = margin_curve(p)
    else:
        margins = np.asarray(margins, dtype=float)
        if len(margins) != len(edges) + 1:
            raise ValueError("need one margin per bin (len(edges) + 1)")
        m = margins[np.digitize(np.asarray(p, dtype=float), edges)]
    return clip_rates(be(p) + m, p, floor, cutoff)


def rule_uncertainty(data, m0, m1=0.0, m2=0.0, m3=0.0, low_signal=0.2,
                     floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """R5, margin that grows with uncertainty:

    r = be(p) + m0 + m1 * 1[signal3 missing] + m2 * 1[income = 0]
              + m3 * 1[0 < signal3 < low_signal]

    The last term follows the plan's note that a low signal3 may matter more
    than a missing one; m3 = 0 gives the formula of the plan's table.
    """
    p = data["p_own"]
    low = ((data["signal3"] > 0) & (data["signal3"] < low_signal)).to_numpy()
    m = (m0 + m1 * data["signal3_missing"].to_numpy()
         + m2 * data["income_zero"].to_numpy() + m3 * low)
    return clip_rates(be(p) + m, p, floor, cutoff)


def rule_info_advantage(data, m0, m1=0.0, floor=RATE_FLOOR,
                        cutoff=NO_OFFER_THRESHOLD):
    """R6, information advantage: r = be(p_own) + m0 + m1 * (p_pub - p_own).

    p_pub is `p_public` (Part 2 public-only model). m1 may be negative.
    """
    p = data["p_own"]
    gap = (data["p_public"] - p).to_numpy()
    return clip_rates(be(p) + m0 + m1 * gap, p, floor, cutoff)


# --- R7: undercut the predicted competitor rate -----------------------------

# Our own information used to predict the competitors' p (logit scale).
COMPETITOR_MODEL_FEATURES = ["logit_p_own", "logit_p_public"] + OWN_FEATURES


def _competitor_design(data):
    X = data[OWN_FEATURES].astype(float).copy()
    X["income"] = np.log1p(X["income"])
    for col in ("p_own", "p_public"):
        X[f"logit_{col}"] = logit(np.clip(data[col].to_numpy(dtype=float), EPS, 1 - EPS))
    return X[COMPETITOR_MODEL_FEATURES]


def fit_competitor_model(train):
    """Regress logit(p_lender1) and logit(p_lender2) on our own information.

    Fitted on the training set (out-of-fold predictions). The competitors'
    predictions are only the target here; the inputs are p_own, p_public and
    our features. Returns a dict with the two fitted regressions and the
    training residuals (kept in pairs, to preserve their correlation).
    """
    X = _competitor_design(train)
    models, residuals = {}, []
    for k in (1, 2):
        y = logit(np.clip(train[f"p_lender{k}"].to_numpy(dtype=float), EPS, 1 - EPS))
        models[k] = LinearRegression().fit(X, y)
        residuals.append(y - models[k].predict(X))
    return {"models": models, "residuals": np.column_stack(residuals)}


def predict_competitor_p(model, data, n_draws=0, seed=SEED):
    """Predicted competitor default probabilities from our own information.

    n_draws = 0 : point predictions, returns (p1_hat, p2_hat), two arrays (n,)
    n_draws > 0 : draws that add a random pair of training residuals to the
                  prediction, returns (P1, P2), two arrays (n, n_draws), to
                  compute quantiles of the competitors' rates.
    """
    X = _competitor_design(data)
    mu = np.column_stack([model["models"][k].predict(X) for k in (1, 2)])
    if n_draws == 0:
        return expit(mu[:, 0]), expit(mu[:, 1])
    rng = np.random.default_rng(seed)
    res = model["residuals"]
    pick = rng.integers(0, len(res), size=(len(X), n_draws))
    return expit(mu[:, [0]] + res[pick, 0]), expit(mu[:, [1]] + res[pick, 1])


def forecast_competitor_rate(comp_draws, pricing, q=0.5):
    """Predicted rate of the cheaper competitor, r_comp_hat, per applicant.

    comp_draws : (P1, P2) from predict_competitor_p(..., n_draws > 0)
    pricing    : how the competitors turn p into a rate in a scenario: one
                 function p -> rate for both, or a (lender 1, lender 2) pair,
                 from competitor_pricing
    q          : quantile of min(r1, r2) over the draws (0.5 = median)

    No offer counts as an infinite rate. The forecast depends only on the
    scenario and q, so compute it once and reuse it across the grid.
    """
    pricing1, pricing2 = pricing if isinstance(pricing, tuple) else (pricing, pricing)
    P1, P2 = comp_draws
    r1 = np.asarray(pricing1(P1.ravel()), dtype=float).reshape(P1.shape)
    r2 = np.asarray(pricing2(P2.ravel()), dtype=float).reshape(P2.shape)
    cheapest = np.fmin(np.where(np.isnan(r1), np.inf, r1), np.where(np.isnan(r2), np.inf, r2))
    return np.quantile(cheapest, q, axis=1)


def rule_undercut(data, r_comp_hat, delta, a=0.0, floor=RATE_FLOOR,
                  cutoff=NO_OFFER_THRESHOLD):
    """R7, undercut the predicted cheaper competitor:

    r = max(be(p_adj), r_comp_hat - delta)

    r_comp_hat : from forecast_competitor_rate (one value per applicant)
    delta      : how far below the predicted competitor rate we go (decimals);
                 -0.02 = up to 2 points above it, to keep the applicants who
                 prefer us
    a          : logit shift of our p for the be(p_adj) floor (as in R3)

    No offer when p_adj reaches 0.5, when the rate would be above 100%, or
    where the competitors are predicted to make no offer (r_comp_hat is then
    infinite, so there is no rate to undercut).
    """
    p = data["p_own"]
    r = np.maximum(be(_adjust_p(p, a)), np.asarray(r_comp_hat, dtype=float) - delta)
    return clip_rates(r, p, floor, cutoff)   # NaN if p_adj >= 0.5, inf -> no offer


def make_undercut_rule(forecasts):
    """R7 as a rule with the same signature as the others.

    forecasts : dict q -> r_comp_hat, from undercut_forecasts: the q-quantile
                of the cheaper competitor's predicted break-even rate.

    The rule never sees the scenario it is scored in, nor the competitors'
    real pricing (unknown, as it will be for the new applicants). Assuming
    they charge break-even + m_hat would only shift r_comp_hat by m_hat, i.e.
    the same as changing delta by -m_hat, so the assumed margin is carried by
    delta: delta = -2 points means "2 points above their break-even".
    """
    def rule_r7(data, delta, q=0.5, a=0.0, floor=RATE_FLOOR,
                cutoff=NO_OFFER_THRESHOLD):
        return rule_undercut(data, forecasts[q], delta, a, floor, cutoff)
    return rule_r7


# --- +C: no-offer cutoff -----------------------------------------------------

def with_cutoff(rule, p_max):
    """+C add-on: the same rule with no offer when p_own >= p_max.

    Returns a new rule with the same signature, e.g.
    rule_r1c = with_cutoff(rule_fixed_margin, 0.3); rule_r1c(data, m=0.02)
    """
    def rule_with_cutoff(data, **params):
        r = rule(data, **params)
        return r.where(data["p_own"].to_numpy() < p_max)
    rule_with_cutoff.__name__ = f"{rule.__name__}_cutoff_{p_max:g}"
    return rule_with_cutoff


# ---------------------------------------------------------------------------
# Competitor scenarios (plan step 1)
# ---------------------------------------------------------------------------

# Columns holding the competitors' default probabilities (Part 2 predictions).
COMPETITOR_P_COLS = ("p_lender1", "p_lender2")

# Tuning scenarios of the plan (section 3c). T6, the equilibrium scenario, is a
# margin scenario whose two margins come from equilibrium_margins (section
# 3b); pass them with make_scenario("T6", data, margin=(m1_eq, m2_eq)).
TUNING_SCENARIOS = {
    "T1": {"kind": "margin", "margin": 0.00},
    "T2": {"kind": "margin", "margin": 0.02},
    "T3": {"kind": "margin", "margin": 0.04},
    "T4": {"kind": "markup", "markup": 0.15},
    "T5": {"kind": "markup", "markup": 0.35},
    "T6": {"kind": "margin"},  # margins = equilibrium margins, passed by the caller
}


def _pair(value, name):
    """One value for both competitors, or a (lender 1, lender 2) pair."""
    if np.ndim(value) == 0:
        return value, value
    if len(value) != 2:
        raise ValueError(f"{name} must be one value or a (lender 1, lender 2) pair")
    return tuple(value)


def _markup_offers(p, markup):
    """Proportional markup: (1 + markup) * break-even, with the game rules."""
    if markup < -1:
        raise ValueError("markup must be >= -1 (a rate cannot be negative)")
    r = (1 + markup) * np.asarray(break_even_rate(p), dtype=float)  # NaN if p >= 0.5
    return np.where(r <= MAX_RATE, r, np.nan)                       # NaN <= 1 is False


def competitor_pricing(kind, **params):
    """How a competitor turns its p into a rate, as a function p -> rate.

    kind "margin" (param margin) or "markup" (param markup), one value. Used
    by make_scenario and by R7 to price the predicted competitor p.
    """
    if kind == "margin":
        margin = params["margin"]
        return lambda p: np.asarray(competitor_offers(np.asarray(p, dtype=float), margin), dtype=float)
    if kind == "markup":
        markup = params["markup"]
        return lambda p: _markup_offers(p, markup)
    raise ValueError("kind must be 'margin' or 'markup'")


def _rule_offers(data, p_col, rule, rule_params):
    """A competitor applying one of our pricing rules to its own p.

    The competitor's p replaces p_own; the rule's other inputs are left as
    they are, so this is only meaningful for rules that use p alone
    (B0, R1-R4).
    """
    r = np.asarray(rule(data.assign(p_own=data[p_col]), **rule_params), dtype=float)
    finite = r[np.isfinite(r)]
    if finite.size and (finite.min() < 0 or finite.max() > MAX_RATE):
        raise ValueError("the rule returned rates outside [0, 1]")
    return np.where(data[p_col].to_numpy(dtype=float) < 0.5, r, np.nan)


def make_scenario(name, data, p_cols=COMPETITOR_P_COLS, **params):
    """Rates offered by the two competitors, under one pricing scenario.

    name   : a scenario kind,
               "margin" : r = break-even + margin            (params: margin)
               "markup" : r = (1 + markup) * break-even      (params: markup)
               "rule"   : one of our rules applied to the competitor's
                          own p, rule(data with p_own = p_k, **rule_params)
                          (params: rule, rule_params); for rules using p only
             or a tuning scenario ID "T1" to "T6" (see TUNING_SCENARIOS);
             params given here override the preset (T6 needs `margin`).
    data   : DataFrame of predictions with the columns in p_cols, e.g.
             train_oof_selected.csv or validation_selected.csv.
    p_cols : the two columns used as lender 1's and lender 2's default
             probabilities. Default: our reconstructed models. Information
             scenarios swap them, e.g. ("p_public", "p_public") for
             competitors who know less, ("p_full", "p_full") for more.

    margin and markup accept one value for both competitors or a
    (lender 1, lender 2) pair, e.g. margin=(0.0, 0.05) for asymmetric margins.

    Returns (r1, r2): two Series indexed like data, decimals, NaN = no offer.
    """
    if name in TUNING_SCENARIOS:
        preset = dict(TUNING_SCENARIOS[name])
        kind = preset.pop("kind")
        params = {**preset, **params}
        if name == "T6" and "margin" not in params:
            raise ValueError("T6 needs the equilibrium margins: margin=(m1_eq, m2_eq)")
    else:
        kind = name

    missing = [c for c in p_cols if c not in data.columns]
    if missing:
        raise KeyError(f"data has no column(s) {missing}")
    p1, p2 = (data[c] for c in p_cols)

    if kind == "margin":
        m1, m2 = _pair(params.pop("margin"), "margin")
        r1, r2 = competitor_offers(p1, m1), competitor_offers(p2, m2)
    elif kind == "markup":
        k1, k2 = _pair(params.pop("markup"), "markup")
        r1, r2 = _markup_offers(p1, k1), _markup_offers(p2, k2)
    elif kind == "rule":
        rule = params.pop("rule")
        rule_params = params.pop("rule_params", {})
        r1, r2 = (_rule_offers(data, p_cols[0], rule, rule_params),
                  _rule_offers(data, p_cols[1], rule, rule_params))
    else:
        raise ValueError(f"unknown scenario '{name}': use 'margin', 'markup', "
                         f"'rule' or one of {list(TUNING_SCENARIOS)}")
    if params:
        raise TypeError(f"unused scenario parameter(s): {sorted(params)}")

    return (pd.Series(np.asarray(r1, dtype=float), index=data.index, name="r1"),
            pd.Series(np.asarray(r2, dtype=float), index=data.index, name="r2"))


# ---------------------------------------------------------------------------
# Tuning (plan step 3)
# ---------------------------------------------------------------------------

# Where grid-search results and tuned parameters are saved.
PRICING_DIR = DATA_DIR / "pricing"

# Rate floors tried in every family (0.5% to 2%, step 0.5 point).
FLOOR_GRID = [0.005, 0.01, 0.015, 0.02]

# Weights of the tuning scenarios (plan section 3c): T6, the equilibrium, is
# the most plausible case and counts double.
TUNING_WEIGHTS = {"T1": 1 / 7, "T2": 1 / 7, "T3": 1 / 7, "T4": 1 / 7,
                  "T5": 1 / 7, "T6": 2 / 7}


def pts(start, stop, step):
    """Inclusive grid start, start + step, ..., stop (decimals), e.g.
    pts(0, 0.08, 0.0025) for 0 to 8 points in steps of 0.25 point."""
    return np.round(np.arange(start, stop + step / 2, step), 6).tolist()


def _simulate(r1, r2, r3, defaults):
    """simulate_market in expected mode (average over the three preference
    types) on positional arrays: one call per rule x scenario."""
    offers = np.column_stack([np.asarray(r, dtype=float) for r in (r1, r2, r3)])
    return simulate_market(offers, np.asarray(defaults), mode="expected")


# --- 3a: sensitivity check ----------------------------------------------------

def best_response_curve(data, comp_margins=(0.0, 0.01, 0.02, 0.03, 0.05),
                        our_margins=None, floor=RATE_FLOOR):
    """R1's best margin against each symmetric competitor margin.

    Returns (table, curves):
      table  : one row per competitor margin: best_margin, best_profit and
               loans won at the best margin
      curves : our profit for every (our margin, competitor margin)
    """
    our_margins = pts(0, 0.08, 0.0025) if our_margins is None else list(our_margins)
    d = data["default"]
    rows, curves = [], {}
    for mc in comp_margins:
        r1, r2 = make_scenario("margin", data, margin=mc)
        profit, loans = [], []
        for m in our_margins:
            _, summary = _simulate(r1, r2, rule_fixed_margin(data, m, floor=floor), d)
            profit.append(summary.loc[3, "total_profit"])
            loans.append(summary.loc[3, "n_loans"])
        best = int(np.argmax(profit))
        curves[mc] = profit
        rows.append({"comp_margin": mc, "best_margin": our_margins[best],
                     "best_profit": profit[best], "loans_won": loans[best]})
    curves = pd.DataFrame(curves, index=pd.Index(our_margins, name="our_margin"))
    curves.columns.name = "comp_margin"
    return pd.DataFrame(rows).set_index("comp_margin"), curves


# --- 3b: equilibrium scenario -------------------------------------------------

def equilibrium_margins(data, grid=None, start=0.02, tol=0.0025, max_rounds=20,
                        burn_in=2):
    """Margins the three lenders reach when each best-responds to the others.

    Every lender prices r = be(p_k) + m_k with its own p (p_lender1,
    p_lender2, p_own) and the competitor rules of Part 3 (no offer at
    p >= 0.5 or above 100%). In turn, each lender's margin is set to the one
    in `grid` that maximises its own profit given the other two; rounds are
    repeated until no margin changes by more than `tol` (or max_rounds).

    If the margins do not converge they cycle (each lender undercuts the
    others until margins are so low that pricing high again pays). The last
    round is then an arbitrary point of the cycle, so the scenario margins
    are the average over the rounds after `burn_in`, rounded to the grid step.

    Returns a dict: margins (m1, m2, m3) of the last round, converged (bool),
    rounds, cycle_mean (average after burn_in), scenario_margins (the last
    round if converged, else cycle_mean rounded), and history (DataFrame of
    the margins after each round).
    """
    grid = pts(0, 0.08, 0.0025) if grid is None else list(grid)
    p_cols = {1: "p_lender1", 2: "p_lender2", 3: "p_own"}
    d = data["default"]
    margins = {k: start for k in p_cols}
    rates = {k: np.asarray(competitor_offers(data[c], start), dtype=float)
             for k, c in p_cols.items()}
    history = [{"round": 0, "m1": start, "m2": start, "m3": start}]
    converged = False
    for rnd in range(1, max_rounds + 1):
        changes = []
        for k, col in p_cols.items():
            best_m, best_profit = margins[k], -np.inf
            for m in grid:
                trial = dict(rates)
                trial[k] = np.asarray(competitor_offers(data[col], m), dtype=float)
                _, summary = _simulate(trial[1], trial[2], trial[3], d)
                profit = summary.loc[k, "total_profit"]
                if profit > best_profit:
                    best_m, best_profit = m, profit
            changes.append(abs(best_m - margins[k]))
            margins[k] = best_m
            rates[k] = np.asarray(competitor_offers(data[col], best_m), dtype=float)
        history.append({"round": rnd, "m1": margins[1], "m2": margins[2], "m3": margins[3]})
        if max(changes) <= tol + 1e-12:
            converged = True
            break
    history = pd.DataFrame(history).set_index("round")
    last = (margins[1], margins[2], margins[3])
    after = history.loc[history.index > burn_in] if len(history) > burn_in + 1 else history
    cycle_mean = tuple(after[f"m{k}"].mean() for k in (1, 2, 3))
    step = float(np.min(np.diff(sorted(grid)))) if len(grid) > 1 else 0.0025
    rounded = tuple(round(round(m / step) * step, 6) for m in cycle_mean)
    return {"margins": last, "converged": converged, "rounds": rnd,
            "cycle_mean": cycle_mean, "scenario_margins": last if converged else rounded,
            "history": history}


# --- 3c: tuning set -----------------------------------------------------------

def tuning_scenarios(data, eq_margins):
    """Competitor rates (r1, r2) for the tuning scenarios T1-T6.

    eq_margins : (m1_eq, m2_eq), the competitors' equilibrium margins (3b).
    Returns a dict scenario name -> (r1, r2).
    """
    return {name: make_scenario(name, data, **({"margin": tuple(eq_margins)}
                                                if name == "T6" else {}))
            for name in TUNING_SCENARIOS}


def undercut_forecasts(model, data, qs=(0.25, 0.5), n_draws=50, seed=SEED):
    """For R7: the q-quantile of the cheaper competitor's predicted
    break-even rate, for every q, computed once.

    The forecast uses the competitors' break-even rate only, the same in every
    scenario: their real pricing is unknown, and forecasting with a scenario's
    true pricing would leak information we will not have for the new
    applicants. Any margin we assume they add is carried by R7's delta.

    model : fit_competitor_model(train)
    Returns a dict q -> array, the input of make_undercut_rule.
    """
    draws = predict_competitor_p(model, data, n_draws=n_draws, seed=seed)
    return {q: forecast_competitor_rate(draws, competitor_pricing("margin", margin=0.0), q)
            for q in qs}


# --- 3d: grid search ----------------------------------------------------------

def grid_search(rule, grid, scenarios, weights, data, fixed=None, progress=True):
    """Our profit for every parameter combination, in every scenario.

    rule      : a pricing rule, e.g. rule_fixed_margin or make_undercut_rule(...)
                for R7 (a rule with scenario_aware = True would also be
                given the scenario name; none of ours is)
    grid      : dict parameter -> list of values; the full product is tried
                (include "floor": FLOOR_GRID to tune the floor)
    scenarios : dict scenario name -> (r1, r2), e.g. tuning_scenarios(...)
    weights   : dict scenario name -> weight, e.g. TUNING_WEIGHTS
    fixed     : parameters passed unchanged to every call (e.g. R4's edges)

    Returns one row per combination: the parameters, profit_<s> and
    loans_<s> per scenario, weighted (= sum_s w_s * profit_s), mean_rate
    (average rate offered), offer_share, and max_regret
    (= max_s [best profit of this grid in s - profit in s]).
    """
    fixed = fixed or {}
    names = list(scenarios)
    aware = getattr(rule, "scenario_aware", False)
    d = data["default"]
    combos = list(product(*grid.values()))
    rows, start = [], time.time()
    for i, values in enumerate(combos, start=1):
        params = dict(zip(grid, values))
        kwargs = {**fixed, **params}
        row = dict(params)
        rates_offered = []
        r3 = None if aware else rule(data, **kwargs)
        for name in names:
            r = rule(data, scenario=name, **kwargs) if aware else r3
            _, summary = _simulate(*scenarios[name], r, d)
            row[f"profit_{name}"] = summary.loc[3, "total_profit"]
            row[f"loans_{name}"] = summary.loc[3, "n_loans"]
            rates_offered.append(np.asarray(r, dtype=float))
        rates_offered = np.concatenate(rates_offered)
        row["mean_rate"] = np.nanmean(rates_offered)
        row["offer_share"] = np.mean(~np.isnan(rates_offered))
        row["weighted"] = sum(weights[n] * row[f"profit_{n}"] for n in names)
        rows.append(row)
        if progress and (i == len(combos) or i % max(1, len(combos) // 10) == 0):
            elapsed = time.time() - start
            print(f"  {i:>5}/{len(combos)} combinations, {elapsed:5.0f}s elapsed, "
                  f"~{elapsed / i * (len(combos) - i):5.0f}s left", flush=True)
    results = pd.DataFrame(rows)
    profits = results[[f"profit_{n}" for n in names]]
    results["max_regret"] = (profits.max() - profits).max(axis=1)
    return results


def select_parameters(results, param_cols):
    """The plan's selection rule (section 3d) on a grid_search result.

    Highest weighted profit; also the smallest max regret. If they differ,
    keep the one with the higher margin, measured by the average rate
    offered (a margin that is too low costs more than one that is too high).
    Returns a dict: best_weighted, min_max_regret, chosen (parameter dicts),
    agree (bool) and table (the candidate rows side by side).
    """
    best_w = results.loc[results["weighted"].idxmax()]
    best_r = results.loc[results["max_regret"].idxmin()]
    agree = all(best_w[c] == best_r[c] for c in param_cols)
    chosen = best_w if agree or best_w["mean_rate"] >= best_r["mean_rate"] else best_r
    cols = list(param_cols) + ["weighted", "max_regret", "mean_rate", "offer_share"]
    table = pd.DataFrame({"best weighted profit": best_w[cols],
                          "smallest max regret": best_r[cols],
                          "chosen": chosen[cols]}).T
    as_dict = lambda row: {c: row[c] for c in param_cols}
    return {"best_weighted": as_dict(best_w), "min_max_regret": as_dict(best_r),
            "chosen": as_dict(chosen), "agree": agree, "table": table}


# --- R4: per-bin margins ------------------------------------------------------

def fit_bin_margins(data, scenarios, weights, edges, margin_grid=None,
                    floor=RATE_FLOOR, cutoff=NO_OFFER_THRESHOLD):
    """R4: the best margin of each risk bin, on weighted profit over scenarios.

    One simulation per (margin, scenario) with that margin for everyone; our
    profit is then summed within each bin. This is exact because an
    applicant's choice depends only on the three rates offered to them, so a
    bin's profit does not depend on the margins of the other bins.

    Returns (table, profit):
      table  : one row per bin: n, p_low, p_high, centre (median p_own),
               best_margin, best_profit
      profit : weighted profit of each bin (rows) for each margin (columns)
    """
    margin_grid = pts(-0.02, 0.10, 0.0025) if margin_grid is None else list(margin_grid)
    p = data["p_own"].to_numpy(dtype=float)
    bins = np.digitize(p, edges)
    d = data["default"]
    profit = np.zeros((len(edges) + 1, len(margin_grid)))
    for j, m in enumerate(margin_grid):
        r3 = rule_fixed_margin(data, m, floor=floor, cutoff=cutoff)
        for name, (r1, r2) in scenarios.items():
            loans, _ = _simulate(r1, r2, r3, d)
            profit[:, j] += weights[name] * np.bincount(
                bins, weights=loans["profit3"].to_numpy(), minlength=len(edges) + 1)
    best = profit.argmax(axis=1)
    groups = pd.Series(p).groupby(bins)
    table = pd.DataFrame({"n": groups.size(), "p_low": groups.min(), "p_high": groups.max(),
                          "centre": groups.median(),
                          "best_margin": np.asarray(margin_grid)[best],
                          "best_profit": profit[np.arange(len(best)), best]})
    table.index.name = "bin"
    return table, pd.DataFrame(profit, index=table.index,
                               columns=pd.Index(margin_grid, name="margin"))


# --- Evaluation of a tuned rule -----------------------------------------------

def evaluate(rule, params, scenario, data, scenario_name=None, groups=5):
    """Full simulation of one rule in one scenario.

    scenario      : (r1, r2), e.g. tuning_scenarios(...)["T6"]
    scenario_name : only for a rule with scenario_aware = True
    Returns (row, curse): row = profit, n_loans, share, avg_rate,
    default_rate_won, gap (winner's curse, overall); curse = the
    winners_curse table by risk group (quintiles of p_own by default).
    """
    kwargs = dict(params)
    if getattr(rule, "scenario_aware", False):
        kwargs["scenario"] = scenario_name
    loans, summary = _simulate(*scenario, rule(data, **kwargs), data["default"])
    curse = winners_curse(loans, data["p_own"].to_numpy(), lender=3, groups=groups)
    ours = summary.loc[3]
    row = {"profit": ours["total_profit"], "n_loans": ours["n_loans"],
           "share": ours["share_of_applicants"], "avg_rate": ours["avg_rate"],
           "default_rate_won": ours["default_rate_won"],
           "gap": curse.loc["overall", "gap"]}
    return row, curse


# --- Saving -------------------------------------------------------------------

def scenario_key(scenarios):
    """Short fingerprint of a set of scenarios (their competitor rates), saved
    with each grid so that a grid computed against other scenarios (e.g. a
    different T6) is never reused by mistake."""
    import hashlib
    h = hashlib.md5()
    for name in sorted(scenarios):
        h.update(name.encode())
        for r in scenarios[name]:
            h.update(np.round(np.nan_to_num(np.asarray(r, dtype=float), nan=-1.0), 8).tobytes())
    return h.hexdigest()[:12]


def save_grid(results, family):
    """Save a grid_search result to data/pricing/grid_<family>.csv."""
    PRICING_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICING_DIR / f"grid_{family}.csv"
    results.to_csv(path, index=False)
    return path


def load_grid(family):
    """Load a saved grid_search result, or None if it was never saved."""
    path = PRICING_DIR / f"grid_{family}.csv"
    return pd.read_csv(path) if path.exists() else None


def save_tuned(tuned, filename="tuned_rules.json"):
    """Save the tuned parameters of every family (dict family -> params) as
    JSON in data/pricing/, for Parts 7 and 8. Arrays become lists."""
    def plain(v):
        if isinstance(v, (np.ndarray, list, tuple)):
            return [plain(x) for x in v]
        return v.item() if isinstance(v, np.generic) else v
    PRICING_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICING_DIR / filename
    path.write_text(json.dumps({f: {k: plain(v) for k, v in prm.items()}
                                for f, prm in tuned.items()}, indent=2))
    return path


# --- Loading the tuned rules (Parts 7 and 8) ---------------------------------

# Family name in tuned_rules.json -> rule function (the +C versions use the
# same function with a `cutoff`).
RULE_FUNCTIONS = {"B0": rule_b0, "R1": rule_fixed_margin, "R2": rule_markup,
                  "R3": rule_adj_p, "R4": rule_bins, "R5": rule_uncertainty,
                  "R6": rule_info_advantage, "R7": rule_undercut}


def load_tuned(filename="tuned_rules.json"):
    """The raw content of data/pricing/tuned_rules.json: a dict family ->
    parameters, plus an "equilibrium" entry (T6 margins, convergence)."""
    return json.loads((PRICING_DIR / filename).read_text())


def load_tuned_rules(train, filename="tuned_rules.json", n_draws=50, seed=SEED):
    """Ready-to-use versions of the rules tuned in Part 6.

    train : load_pricing_data("train"). Only R7 needs it, to refit the model
            that predicts the competitors' default probabilities from our own
            information (exactly as in Part 6).

    Returns a dict family -> function data -> our rates (Series, NaN = no
    offer), e.g. rules["R4+C"](load_pricing_data("validation")). Each
    function also has .family and .params attributes. The data passed in
    needs p_own (all rules) and, for R5-R7, our features and p_public, as
    returned by load_pricing_data.

    * R4 (and R4+C) is rebuilt from its saved decile edges and smoothed
      margins.
    * R7 (and R7+C) recomputes its forecast of the competitors' break-even
      rate on the data it prices, with the competitor model fitted on train.
    """
    tuned = load_tuned(filename)
    competitor_model = None
    rules = {}
    for family, params in tuned.items():
        if family == "equilibrium":
            continue
        base = family.replace("+C", "")
        params = dict(params)

        if base == "R4":
            edges = np.asarray(params["edges"], dtype=float)
            curve = smooth_bin_margins(params["centres"], params["margins"])
            kwargs = {k: v for k, v in params.items() if k not in ("edges", "centres", "margins")}

            def rule(data, edges=edges, curve=curve, kwargs=kwargs):
                return rule_bins(data, edges, margin_curve=curve, **kwargs)

        elif base == "R7":
            if competitor_model is None:
                competitor_model = fit_competitor_model(train)
            q = params["q"]
            kwargs = {k: v for k, v in params.items() if k != "q"}

            def rule(data, q=q, kwargs=kwargs):
                forecast = undercut_forecasts(competitor_model, data, qs=(q,),
                                              n_draws=n_draws, seed=seed)[q]
                return rule_undercut(data, forecast, **kwargs)

        else:
            function = RULE_FUNCTIONS[base]

            def rule(data, function=function, params=params):
                return function(data, **params)

        rule.family, rule.params = family, params
        rules[family] = rule
    return rules

