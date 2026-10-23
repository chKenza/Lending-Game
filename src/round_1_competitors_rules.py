"""Part 10 - inferring the competitors' rules from Round 1.

In Round 1 each competitor slot of the diagnostic file (r1 = a type-1 lender,
r2 = a type-2 lender) is, for every loan, one of four teams drawn at random
(three student teams and one LLM team per type; course results, section A).
So each slot's offers are a mixture of four rules, each applied to about a
quarter of the applicants. This module:

* labels each competitor offer by behaviour (no offer, high quote, pricing
  curve) and by fingerprint (the number of decimals the team rounded to, the
  fixed high quote, "charm" rates ending in 9), which separates the teams;
* counts each fingerprint in "teams' worth" (a quarter of the applicants of a
  band = one team) to recover the four teams of each slot;
* describes each team's rule: implied margin over the competitor's predicted
  break-even (regression and binned medians), probability of no offer, the
  risk at which the high-quote team switches to its high quote;
* builds the empirical Round 2 scenario: competitor offers resampled from the
  Round 1 applicants with the same risk and the same default outcome, with
  stress knobs (fewer no-offers or high quotes, a lower high quote, more
  aggressive curves).

The competitors' default probability is never observed (we do not see
signal1 or signal2). p_hat1 and p_hat2 are its forecast from our own
information (Part 6's fit_competitor_model, trained on the past loans).

Usage from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src import round_1_competitors_rules as crr
    data = crr.load_round1_competitors()
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.config import SEED
from src.fair_pricing import MAX_RATE
from src.model_plots import INK_SECONDARY, REFERENCE, SURFACE
from src.pricing_plots import LENDER_COLORS, SERIES_COLORS
from src.pricing_rules import EPS, be, fit_competitor_model, load_pricing_data, predict_competitor_p
from src.round_1_replay import HIGH_QUOTE, load_round1, replay, risk_band, score_replay

LENDERS = (1, 2)
N_TEAMS = 4  # teams per competitor slot: three student teams and one LLM team

# The fixed high quote of each slot in Round 1 (one team quoted it to many
# applicants, whatever their risk).
FIXED_QUOTES = {1: 0.9199, 2: 0.9789}

# Behaviour of one offer. "high quote" follows Part 9: any offer >= 90%,
# which a borrower takes only if nothing cheaper is on the table.
BEHAVIOURS = ["no offer", "high quote", "curve"]

# Bands of the competitors' predicted default probability p_hat (it rarely
# goes above 30%: it is an expectation given our information).
P_HAT_BANDS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0]
P_HAT_LABELS = ["< 2%", "2-4%", "4-6%", "6-8%", "8-10%", "10-12%", "12-15%", "15-20%", ">= 20%"]

# Fingerprint groups of each slot (see fingerprint()).
GROUPS = {
    1: ["no offer", "fixed quote 91.99%", "6-decimal curve", "4-decimal curve"],
    2: ["no offer", "fixed quote 97.89%", "charm curve (ends in 9)", "4-decimal curve",
        "full-precision curve"],
}


# ---------------------------------------------------------------------------
# Data and labels
# ---------------------------------------------------------------------------

def load_round1_competitors(train=None):
    """Part 9's Round 1 table (load_round1) plus p_hat1 and p_hat2, the
    competitors' default probabilities forecast from our own information.

    train : Part 6's training predictions (load_pricing_data("train")); loaded
            if None. The forecast model is fitted on it, as for rule R7.
    """
    data = load_round1()
    train = load_pricing_data("train") if train is None else train
    p1, p2 = predict_competitor_p(fit_competitor_model(train), data)
    data["p_hat1"], data["p_hat2"] = p1, p2
    return data


def decimals(rates):
    """Number of decimals each offer was submitted with (NaN = no offer).

    The course file stores rates as float32, so 0.0559 is read back as
    0.055900000. The answer is the fewest decimals d such that the rate
    rounded to d decimals gives the same float32. A team that rounds to 4
    decimals gives d <= 4 on every offer; a team that does not round gives 7-9.
    """
    x = np.asarray(rates, dtype=float)
    f = x.astype(np.float32)
    out = np.full(len(x), np.nan)
    for d in range(10, -1, -1):
        out[np.round(x, d).astype(np.float32) == f] = d
    return pd.Series(out, index=getattr(rates, "index", None), name="decimals")


def behaviour(data, k, high=HIGH_QUOTE):
    """Behaviour of lender k's offer to each applicant: no offer, high quote
    (>= `high`) or curve (any other offer)."""
    r = data[f"r{k}"]
    b = np.where(r.isna(), BEHAVIOURS[0], np.where(r >= high, BEHAVIOURS[1], BEHAVIOURS[2]))
    return pd.Series(pd.Categorical(b, categories=BEHAVIOURS), index=data.index,
                     name=f"behaviour{k}")


def fingerprint(data, k):
    """Fingerprint group of lender k's offer to each applicant (GROUPS[k]).

    lender 1: no offer; the fixed 91.99% quote; offers with 5-6 decimals;
              offers with at most 4 decimals.
    lender 2: no offer; the fixed 97.89% quote; 4-decimal offers whose last
              decimal is 9 (the high-quote team prices 0.0199, 0.0249, ...,
              and 0.9789); other 4-decimal offers; offers with 5 or more
              decimals (a team that does not round).
    """
    r = data[f"r{k}"]
    d = decimals(r).to_numpy()
    fixed = np.isclose(r.fillna(-1), FIXED_QUOTES[k], atol=1e-6)
    g = GROUPS[k]
    if k == 1:
        label = np.where(r.isna(), g[0], np.where(fixed, g[1], np.where(d <= 4, g[3], g[2])))
    else:
        last = np.round(r.fillna(0).to_numpy() * 1e4).astype(int) % 10
        label = np.where(r.isna(), g[0], np.where(fixed, g[1], np.where(
            d > 4, g[4], np.where(last == 9, g[2], g[3]))))
    return pd.Series(pd.Categorical(label, categories=g), index=data.index, name=f"group{k}")


def p_hat_band(data, k, bands=P_HAT_BANDS, labels=P_HAT_LABELS):
    """Band of lender k's predicted default probability p_hat_k."""
    return pd.cut(data[f"p_hat{k}"], bands, labels=labels, right=False)


# ---------------------------------------------------------------------------
# Behaviours and teams
# ---------------------------------------------------------------------------

def behaviour_shares(data, band=None):
    """Share of applicants getting each behaviour from each slot, per band
    (default: Part 9's bands of p_own). Columns (lender, behaviour)."""
    band = risk_band(data) if band is None else band
    return pd.concat({f"lender {k}": pd.crosstab(band, behaviour(data, k), normalize="index")
                      for k in LENDERS}, axis=1)


def team_units(data, k, band=None):
    """Each fingerprint group of lender k counted in teams' worth, per band.

    A team is drawn for a quarter of the applicants, so in a band of n
    applicants n/4 offers make one team. A group at 1.00 in every band is one
    team that always behaves that way; a group that falls from 1 to 0 is a
    team that stops offering (or switches behaviour) as risk rises. Each row
    sums to 4. Default band: p_hat_k.
    """
    band = p_hat_band(data, k) if band is None else band
    counts = pd.crosstab(band, fingerprint(data, k))
    return counts.div(counts.sum(axis=1) / N_TEAMS, axis=0)


def high_quote_switch(data, k, band=None):
    """How the high-quote team of lender k switches to its fixed quote.

    Per band: the share of that team's applicants quoted the fixed quote,
    i.e. the fixed quote's share of all applicants times 4 (the team sees a
    quarter of them). Default band: p_hat_k. Returns a Series.
    """
    band = p_hat_band(data, k) if band is None else band
    fixed = np.isclose(data[f"r{k}"].fillna(-1), FIXED_QUOTES[k], atol=1e-6)
    share = pd.Series(fixed, index=data.index).groupby(band, observed=False).mean() * N_TEAMS
    return share.clip(upper=1.0).rename(f"high-quote team quoting {FIXED_QUOTES[k]:.2%}")


# ---------------------------------------------------------------------------
# Rules: switch points and no offers
# ---------------------------------------------------------------------------

def _logit_p(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def fit_share_curve(p, y, floor=None, ceiling=None):
    """Maximum-likelihood fit of a share that moves with risk between a floor
    and a ceiling:

        P(y = 1 | p) = floor + (ceiling - floor) * expit(a + b * logit(p))

    This is the shape a mixture of teams produces: `floor` = the teams that
    always do it, `ceiling` - `floor` = the teams that start doing it as their
    (unobserved) risk estimate crosses a threshold. A plain logistic curve
    cannot have a floor of 25% (a team that never offers) or a ceiling of 25%
    (one team out of four quoting high).

    p              : the risk variable (p_hat_k)
    y              : 0/1 indicator (no offer, fixed quote, ...)
    floor, ceiling : fixed values, or None to estimate them
    Returns a dict: a, b, floor, ceiling and "midpoint", the p at which half
    of the switching teams have switched (expit(-a / b)).
    """
    x = _logit_p(p)
    y = np.asarray(y, dtype=bool)
    free = [floor is None, ceiling is None]

    def unpack(theta):
        a, b, rest = theta[0], theta[1], list(theta[2:])
        lo = expit(rest.pop(0)) if free[0] else floor
        hi = expit(rest.pop(0)) if free[1] else ceiling
        return a, b, lo, hi

    def neg_loglik(theta):
        a, b, lo, hi = unpack(theta)
        q = np.clip(lo + (hi - lo) * expit(a + b * x), 1e-12, 1 - 1e-12)
        return -np.sum(np.where(y, np.log(q), np.log(1 - q)))

    # start: midpoint at p = 10%, slope 3, floor 20%, ceiling 60%
    start = [-3.0 * logit(0.1), 3.0] + [logit(0.2)] * free[0] + [logit(0.6)] * free[1]
    theta = minimize(neg_loglik, x0=start, method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000}).x
    a, b, lo, hi = unpack(theta)
    return {"a": a, "b": b, "floor": lo, "ceiling": hi, "midpoint": float(expit(-a / b))}


def predict_share_curve(fit, p):
    """The fitted share at risk p (a fit_share_curve result)."""
    return fit["floor"] + (fit["ceiling"] - fit["floor"]) * expit(fit["a"] + fit["b"] * _logit_p(p))


def fit_high_quote(data, k):
    """The high-quote team's switch to its fixed quote, as a share curve in
    p_hat_k with floor 0 and ceiling 1/4 (one team of four). The midpoint is
    the competitor's risk at which the team quotes high to half its applicants."""
    fixed = np.isclose(data[f"r{k}"].fillna(-1), FIXED_QUOTES[k], atol=1e-6)
    return fit_share_curve(data[f"p_hat{k}"], fixed, floor=0.0, ceiling=1 / N_TEAMS)


def fit_no_offer(data, k):
    """Lender k's probability of making no offer as a function of its
    predicted risk p_hat_k, with floor and ceiling estimated (fit_share_curve)."""
    return fit_share_curve(data[f"p_hat{k}"], data[f"r{k}"].isna())


def share_curve_table(fits):
    """fits: dict name -> fit_share_curve result. One row per fit, with the
    floor and ceiling also in teams' worth (x 4)."""
    t = pd.DataFrame(fits).T[["floor", "ceiling", "midpoint", "a", "b"]].astype(float)
    t.insert(2, "floor (teams)", N_TEAMS * t["floor"])
    t.insert(3, "ceiling (teams)", N_TEAMS * t["ceiling"])
    return t


def observed_vs_fitted(data, k, fit, y, band=None):
    """Observed and fitted share per band (default: bands of p_hat_k), for an
    indicator y and its fit_share_curve result. Columns observed, fitted."""
    band = p_hat_band(data, k) if band is None else band
    fitted = pd.Series(predict_share_curve(fit, data[f"p_hat{k}"]), index=data.index)
    y = pd.Series(np.asarray(y, dtype=float), index=data.index)
    return pd.DataFrame({"observed": y.groupby(band, observed=False).mean(),
                         "fitted": fitted.groupby(band, observed=False).mean()})


# ---------------------------------------------------------------------------
# Rules: margins
# ---------------------------------------------------------------------------

def margin_table(data, k, band=None, stat="median"):
    """Implied margin of lender k's curve offers: r - be(p_hat_k), per
    fingerprint group (rows) and band of p_hat_k (columns).

    stat : "median", "n" (number of offers) or a quantile in (0, 1).
    Offers of 90% or more (high quotes) are left out.
    """
    band = p_hat_band(data, k) if band is None else band
    curve = behaviour(data, k) == "curve"
    margin = (data[f"r{k}"] - be(data[f"p_hat{k}"]))[curve]
    groups = [fingerprint(data, k)[curve], band[curve]]
    g = margin.groupby(groups, observed=False)
    out = g.size() if stat == "n" else g.median() if stat == "median" else g.quantile(stat)
    out = out.unstack()
    return out[out.sum(axis=1) > 0] if stat == "n" else out.dropna(how="all")


def margin_regression(data, k):
    """The guideline's regression, per fingerprint group of lender k's curve offers:

        r = a + b * be(p_hat_k) + c * be(p_own)

    by least squares. b + c is the total pass-through of risk into the rate
    (1 = a constant margin over break-even); b / (b + c) is how much of it
    goes through the competitor's own information rather than ours (p_hat_k
    is a forecast from our information, so the split is only indicative).
    The implied margin over be(p_hat_k) at a typical applicant is reported
    at the group's median p_hat_k. Returns one row per group.
    """
    curve = behaviour(data, k) == "curve"
    group = fingerprint(data, k)
    rows = {}
    for g in GROUPS[k]:
        m = curve & (group == g)
        if m.sum() < 50:
            continue
        X = np.column_stack([np.ones(m.sum()), be(data.loc[m, f"p_hat{k}"]),
                             be(data.loc[m, "p_own"])])
        y = data.loc[m, f"r{k}"].to_numpy()
        ok = np.isfinite(X).all(axis=1)          # be is NaN when p_own >= 0.5
        X, y = X[ok], y[ok]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        p_med = float(np.median(data.loc[m, f"p_hat{k}"]))
        be_med = float(be(p_med))
        rows[g] = {"offers": int(ok.sum()), "a (intercept)": coef[0], "b (be of p_hat)": coef[1],
                   "c (be of p_own)": coef[2], "b + c": coef[1] + coef[2],
                   "R2": 1 - resid.var() / y.var(), "median p_hat": p_med,
                   "margin at median p_hat": coef[0] + (coef[1] + coef[2] - 1) * be_med}
    return pd.DataFrame(rows).T


def competitor_results(data, seed=SEED):
    """How each fingerprint group of each competitor did in Round 1, against
    our submitted offers and with the real borrower types.

    One row per (lender, group): offers made, loans won, win rate, average
    rate and default rate of the loans won, profit and profit per loan.
    """
    loans, _ = replay(data["r3_submitted"], data, seed=seed)
    rows = []
    for k in LENDERS:
        group = fingerprint(data, k)
        w, prof = loans[f"w{k}"], loans[f"profit{k}"].fillna(0.0)
        for g in GROUPS[k]:
            m = (group == g).to_numpy()
            won = w[m].sum()
            rows.append({
                "lender": k, "group": g, "applicants": int(m.sum()),
                "offers": int(data.loc[m, f"r{k}"].notna().sum()), "loans won": won,
                "win rate": won / max(data.loc[m, f"r{k}"].notna().sum(), 1),
                "avg rate won": (w[m] * data.loc[m, f"r{k}"].fillna(0)).sum() / won if won else np.nan,
                "default rate won": (w[m] * data.loc[m, "default"]).sum() / won if won else np.nan,
                "profit": prof[m].sum(),
                "profit per loan": prof[m].sum() / won if won else np.nan})
    return pd.DataFrame(rows).set_index(["lender", "group"])


def uncontested_sources(data, band=None):
    """Where the uncontested applicants come from, per band (default: Part 9's
    bands of p_own), in % of applicants:

      no competitor offer        observed
        if independent           product of the two lenders' no-offer rates
        both never-offer teams   1/16: both slots drew the team that never offers
      only high quotes           observed (every offer >= 90%, at least one offer)
        if independent           the same from the two lenders' marginal rates

    The "independent" columns show how much the two slots' refusals go
    together (their signals are correlated through the borrower's true risk).
    """
    band = risk_band(data) if band is None else band
    b1, b2 = behaviour(data, 1), behaviour(data, 2)
    none1, none2 = b1 == "no offer", b2 == "no offer"
    off1, off2 = b1 != "curve", b2 != "curve"      # no offer or high quote
    frame = pd.DataFrame({"none": none1 & none2, "high": off1 & off2 & ~(none1 & none2),
                          "none1": none1, "none2": none2, "off1": off1, "off2": off2})
    m = frame.astype(float).groupby(band, observed=False).mean()
    return 100 * pd.DataFrame({
        "no competitor offer": m["none"],
        "  if independent": m["none1"] * m["none2"],
        "  both never-offer teams": 1 / N_TEAMS ** 2,
        "only high quotes": m["high"],
        "  if independent ": m["off1"] * m["off2"] - m["none1"] * m["none2"]})


# ---------------------------------------------------------------------------
# The empirical Round 2 scenario
# ---------------------------------------------------------------------------

def scenario_edges(donors, n_bins=20):
    """Inner edges of the risk bins: quantiles of the donors' p_own."""
    return np.quantile(donors["p_own"], np.linspace(0, 1, n_bins + 1)[1:-1])


def simulate_defaults(target, seed=SEED):
    """A copy of `target` with a default drawn from p_own for each applicant.

    For applicants without an outcome (Round 2): the scenario draws the
    competitors' offers given the default, so the winner's curse is kept. It
    relies on p_own being calibrated, which Part 9 confirmed in Round 1
    (9.96% predicted, 9.95% observed).
    """
    rng = np.random.default_rng(seed)
    out = target.copy()
    out["default"] = (rng.random(len(out)) < out["p_own"].to_numpy()).astype(int)
    return out


def _per_lender(value, k):
    """A stress knob given once for both lenders, or as {lender: value}."""
    return value.get(k) if isinstance(value, dict) else value


def empirical_scenario(target, donors, n_bins=20, use_default=True, seed=SEED,
                       never_offer_team_offers=False, keep_no_offer=1.0, keep_high_quote=1.0,
                       high_quote_rate=None, curve_shift=0.0):
    """Competitor offers for `target`, resampled from the Round 1 market.

    Each target applicant gets the two competitor offers (r1, r2) of one
    Round 1 applicant ("donor") drawn at random from the same cell: the same
    bin of p_own (n_bins quantile bins) and, if use_default, the same default
    outcome. Both offers come from the same donor, which keeps their
    correlation (the competitors' signals are correlated through the
    borrower's true risk). Conditioning on the default keeps the winner's
    curse: competitors make no offer or quote high more often to borrowers
    who then default.

    Stress knobs (Part 11), applied to each slot separately; each one is a
    single value for both lenders or a dict {lender: value}:
      never_offer_team_offers : the team that never offered in Round 1 now
                        offers like the curve teams: in each cell, one team's
                        worth of no-offers (1/4 of the applicants) is replaced
                        by a curve offer drawn from the same cell
      keep_no_offer   : share of the (remaining) no-offers kept; the others
                        are replaced by a curve offer drawn from the same cell
      keep_high_quote : the same for the high quotes (offers >= 90%)
      high_quote_rate : if set, every remaining high quote becomes this rate
                        (the high-quote teams undercutting)
      curve_shift     : points subtracted from every curve offer (> 0 = more
                        aggressive competitors), floored at 0

    target : DataFrame with p_own (and default if use_default; see
             simulate_defaults), indexed by applicant
    donors : the Round 1 table (load_round1 or load_round1_competitors)
    Returns a copy of target with r1, r2 and behaviour1, behaviour2.
    """
    rng = np.random.default_rng(seed)
    edges = scenario_edges(donors, n_bins)
    d_bin = np.searchsorted(edges, donors["p_own"].to_numpy(), side="right")
    t_bin = np.searchsorted(edges, target["p_own"].to_numpy(), side="right")
    if use_default:
        d_cell = 2 * d_bin + donors["default"].to_numpy().astype(int)
        t_cell = 2 * t_bin + target["default"].to_numpy().astype(int)
    else:
        d_cell, t_cell = d_bin, t_bin

    d_rates = {k: donors[f"r{k}"].to_numpy(dtype=float) for k in LENDERS}
    d_curve = {k: (behaviour(donors, k) == "curve").to_numpy() for k in LENDERS}
    pick = np.empty(len(target), dtype=int)
    for c in np.unique(t_cell):
        rows = np.flatnonzero(t_cell == c)
        pool = np.flatnonzero(d_cell == c)
        if len(pool) == 0:   # fall back to the whole risk bin
            pool = np.flatnonzero(d_bin == (c // 2 if use_default else c))
        pick[rows] = rng.choice(pool, len(rows))

    out = target.copy()
    for k in LENDERS:
        r = d_rates[k][pick].copy()
        b = np.where(np.isnan(r), 0, np.where(r >= HIGH_QUOTE, 1, 2))
        # probability that a no-offer is replaced: one team's worth of the
        # cell's no-offers if the never-offer team now offers, then 1 - keep
        p_drop = np.zeros(len(r))
        if _per_lender(never_offer_team_offers, k):
            none_share = pd.Series(np.isnan(d_rates[k])).groupby(d_cell).mean()
            share = none_share.reindex(t_cell).to_numpy()
            p_drop = np.minimum(1.0, (1 / N_TEAMS) / np.where(share > 0, share, np.inf))
        p_drop = 1 - (1 - p_drop) * _per_lender(keep_no_offer, k)
        u = rng.random(len(r))
        replace = ((b == 0) & (u < p_drop)) | \
                  ((b == 1) & (u >= _per_lender(keep_high_quote, k)))
        for c in np.unique(t_cell[replace]):
            rows = np.flatnonzero(replace & (t_cell == c))
            pool = np.flatnonzero((d_cell == c) & d_curve[k])
            if len(pool) == 0:
                pool = np.flatnonzero((d_bin == (c // 2 if use_default else c)) & d_curve[k])
            r[rows] = d_rates[k][rng.choice(pool, len(rows))]
        b = np.where(np.isnan(r), 0, np.where(r >= HIGH_QUOTE, 1, 2))
        if _per_lender(high_quote_rate, k) is not None:
            r = np.where(b == 1, _per_lender(high_quote_rate, k), r)
        if _per_lender(curve_shift, k):
            r = np.where(b == 2, np.clip(r - _per_lender(curve_shift, k), 0.0, MAX_RATE), r)
        out[f"r{k}"] = r
        out[f"behaviour{k}"] = pd.Categorical(np.asarray(BEHAVIOURS)[b], categories=BEHAVIOURS)
    return out


def score_scenarios(rates, target, donors, scenarios, seeds=range(5), **kwargs):
    """Our profit for each rule in each empirical scenario.

    rates     : dict rule -> our rates on the target applicants
    scenarios : dict name -> keyword arguments of empirical_scenario
    seeds     : one resampling per seed; the result is the mean and the
                standard deviation over seeds
    The borrower types are the target's real ones if it has a `pref` column
    (Round 1), else averaged over the three types (expected mode).
    Returns (mean, sd): DataFrames rules x scenarios (profit in EUR).
    """
    mode = "actual" if "pref" in target else "expected"
    runs = {}
    for name, params in scenarios.items():
        for s in seeds:
            market = empirical_scenario(target, donors, seed=s, **{**kwargs, **params})
            runs[(name, s)] = score_replay(rates, market, mode=mode)[0]["profit"]
    runs = pd.DataFrame(runs)
    by = runs.T.groupby(level=0, sort=False)
    return by.mean().T, by.std().T


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

BEHAVIOUR_COLORS = {"no offer": "#1baf7a", "high quote": "#eda100", "curve": "#9a9893"}


def plot_behaviour_shares(shares, axes):
    """One panel per lender: stacked bars of no offer / high quote / curve
    per band (a behaviour_shares table)."""
    for ax, k in zip(axes, LENDERS):
        t = shares[f"lender {k}"]
        bottom = np.zeros(len(t))
        for b in BEHAVIOURS:
            ax.bar(range(len(t)), 100 * t[b], bottom=bottom, color=BEHAVIOUR_COLORS[b],
                   edgecolor=SURFACE, label=b)
            bottom += 100 * t[b].to_numpy()
        ax.set_xticks(range(len(t)), t.index)
        ax.set_xlabel("our predicted default probability p_own")
        ax.set_ylabel("share of applicants (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Lender {k}: behaviour by risk band")
        ax.legend(loc="upper left", labelcolor=INK_SECONDARY, ncol=3)


def plot_team_units(units, ax, title):
    """Stacked bars of each fingerprint group in teams' worth (a team_units
    table); the dashed lines mark whole teams."""
    bottom = np.zeros(len(units))
    for color, g in zip(SERIES_COLORS, units.columns):
        ax.bar(range(len(units)), units[g], bottom=bottom, color=color, edgecolor=SURFACE,
               label=g)
        bottom += units[g].to_numpy()
    for y in range(1, N_TEAMS):
        ax.axhline(y, color=REFERENCE, lw=0.8, ls="--")
    ax.set_xticks(range(len(units)), units.index)
    ax.set_ylim(0, N_TEAMS)
    ax.set_xlabel("competitor's predicted default probability p_hat")
    ax.set_ylabel("teams' worth of applicants")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), labelcolor=INK_SECONDARY,
              ncol=3, fontsize=8)


def plot_margin_curves(medians, ax, title):
    """Median implied margin (points over be(p_hat)) of each fingerprint
    group per band of p_hat (a margin_table)."""
    x = np.arange(medians.shape[1])
    for color, (g, row) in zip(SERIES_COLORS[2:] + SERIES_COLORS, medians.iterrows()):
        ax.plot(x, 100 * row.to_numpy(dtype=float), marker="o", color=color, label=g)
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xticks(x, medians.columns)
    ax.set_xlabel("competitor's predicted default probability p_hat")
    ax.set_ylabel("median margin over break-even (pts)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY, fontsize=8)


def plot_share_curves(tables, ax, title, ylabel):
    """Observed (dots) and fitted (lines) shares per band of p_hat, one colour
    per curve. tables: dict label -> observed_vs_fitted table (same bands)."""
    for color, (label, t) in zip(SERIES_COLORS, tables.items()):
        x = np.arange(len(t))
        ax.plot(x, 100 * t["fitted"], color=color, lw=1.6, label=f"{label}, fitted")
        ax.plot(x, 100 * t["observed"], "o", color=color, label=f"{label}, observed")
    ax.axhline(100 / N_TEAMS, color=REFERENCE, lw=1, ls="--")
    ax.set_xticks(x, t.index)
    ax.set_xlabel("competitor's predicted default probability p_hat")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY, fontsize=8)
