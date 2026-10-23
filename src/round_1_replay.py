"""Part 9 - replaying the candidate rules on the real Round 1 market.

The course's diagnostic file (Round1_Diagnostic_31.csv) gives, for each of the
100,000 Round 1 applicants, the two competitors' actual offers, the
borrower's preference type and whether the borrower defaulted. With these, the
profit any rule of ours *would* have made in Round 1 is an exact
counterfactual: replace our offer, rerun the borrower's choice with the
existing simulator (src/simulator.py), and book the observed default.

The module has five parts:

* data: the diagnostic joined with our Round 1 predictions and features, the
  market situation of each applicant and how selected the uncontested ones are;
* replay: one rule's offers against the real market, every rule at once, and
  two stress tests of the Round 1 market (the uncontested applicants becoming
  contested, the high-quote competitors cutting their quote);
* new rules: R8 (high quote) and the R4/R8 hybrids, in both directions;
* hindsight tuning: the best rate or margin per risk band, in sample and
  cross-fitted (tuned on half of the applicants, scored on the other half);
* plots used by analysis/round1_replay.ipynb.

Column names follow the simulator: r1 and r2 are the competitors' offers
(lenders 1 and 2), r3 ours, pref the borrower's type (the lender they get a
2-point discount from).

Typical use from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.round_1_replay import load_round1, replay, score_replay

    data = load_round1()
    loans, summary = replay(data["r3_submitted"], data)     # what happened
    table, profits = score_replay({"R4": rates_r4}, data)   # any rule
"""

import numpy as np
import pandas as pd

from src.config import DATA_PREDICTIONS, DATA_PROCESSED, DISCOUNT, ROUND1_DIAGNOSTIC, SEED
from src.fair_pricing import MAX_RATE, NO_OFFER_THRESHOLD
from src.model_plots import INK, INK_SECONDARY, REFERENCE, SURFACE
from src.pricing_plots import LENDER_COLORS, SERIES_COLORS
from src.pricing_rules import OWN_FEATURES, clip_rates
from src.simulator import simulate_market, winners_curse

# Risk bands of p_own used to report results (Part 8's bands, split finer
# where the Round 1 profit is made).
BANDS = [0.0, 0.0275, 0.045, 0.08, 0.11, 0.15, 0.2, 0.3, 1.0]
BAND_LABELS = ["< 2.75%", "2.75-4.5%", "4.5-8%", "8-11%", "11-15%", "15-20%",
               "20-30%", ">= 30%"]

# A competitor quote at or above this level is a "high quote": in Round 1 two
# competitor teams quoted fixed rates of 91.99% and 97.89% to many applicants.
HIGH_QUOTE = 0.9

# Market situations of an applicant, from our point of view.
SITUATIONS = ["no competitor offer", "only high quotes", "contested"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def public_predictions(refresh=False):
    """p_public for the Round 1 applicants (R5-R7 need it).

    Part 2's selected public-only model (spline logistic), refitted on all
    100,000 past loans, exactly like the own model behind applications_own.csv.
    Saved once to data/predictions/applications_public.csv and read back
    afterwards (refresh=True refits). Returns a Series indexed by id.
    """
    path = DATA_PREDICTIONS / "applications_public.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="id")["p_public"]
    from src.models import fit_models, get_X, load_data, predict_default, save_predictions
    data = load_data()
    model = fit_models(data, "public", ["spline"], fit_on="train+validation")["spline"]
    X = get_X(data, "spline", "public", "applications")
    p = pd.Series(predict_default(model, X), index=X.index, name="p_public")
    save_predictions(p.to_frame(), "applications_public.csv")
    return p


def load_round1():
    """The Round 1 market, one row per applicant, indexed by id.

    Columns:
      r1, r2        the competitors' actual offers (NaN = no offer)
      r3_submitted  our actual offer (rule R4, Part 8)
      pref          the borrower's type: the lender they get 2 points off from
      accepted, default, profit   as reported by the course
      p_own, p_public and OWN_FEATURES   what our rules need
    """
    diag = pd.read_csv(ROUND1_DIAGNOSTIC, index_col="id").rename(columns={
        "own": "r3_submitted", "competing1": "r1", "competing2": "r2",
        "borrowertype": "pref"})
    own = pd.read_csv(DATA_PREDICTIONS / "applications_own.csv", index_col="id")
    features = pd.read_csv(DATA_PROCESSED / "applications_general.csv", index_col="id")
    data = diag.join(own).join(public_predictions()).join(features[OWN_FEATURES])
    needed = ["p_own", "p_public"] + OWN_FEATURES
    if data[needed].isna().any().any():
        raise ValueError("some Round 1 applicants have no prediction or features")
    return data


def cheapest_competitor(data):
    """The cheaper competitor offer per applicant (inf = no competitor offer)."""
    r1 = data["r1"].fillna(np.inf).to_numpy(dtype=float)
    r2 = data["r2"].fillna(np.inf).to_numpy(dtype=float)
    return pd.Series(np.minimum(r1, r2), index=data.index, name="cheapest")


def market_situation(data, high=HIGH_QUOTE):
    """Each applicant's situation: no competitor offer, only high quotes (every
    competitor offer >= `high`), or contested (at least one cheaper offer)."""
    cheapest = cheapest_competitor(data)
    situation = np.where(np.isinf(cheapest), SITUATIONS[0],
                         np.where(cheapest >= high, SITUATIONS[1], SITUATIONS[2]))
    return pd.Series(pd.Categorical(situation, categories=SITUATIONS),
                     index=data.index, name="situation")


def risk_band(data, bands=BANDS, labels=BAND_LABELS):
    """The report band of each applicant's p_own."""
    return pd.cut(data["p_own"], bands, labels=labels, right=False)


def selection_by_situation(data, bands=BANDS, labels=BAND_LABELS):
    """Are the applicants the competitors leave alone worse than p_own says?

    Per risk band and market situation: applicants, mean p_own, observed
    default rate and the gap (observed - predicted). A positive gap means the
    competitors' no-offers and high quotes select borrowers riskier than our
    model expects.
    Returns a DataFrame indexed by (band, situation).
    """
    groups = pd.DataFrame({"p": data["p_own"], "d": data["default"]}).groupby(
        [risk_band(data, bands, labels), market_situation(data)], observed=False)
    out = groups.agg(applicants=("d", "size"), mean_p_own=("p", "mean"),
                     default_rate=("d", "mean"))
    out["gap"] = out["default_rate"] - out["mean_p_own"]
    return out


def undercut_high_quotes(data, rate, high=HIGH_QUOTE):
    """A copy of the Round 1 market where every competitor quote >= `high` is
    replaced by `rate` (no-offers stay no-offers): the high-quote teams cut
    their fixed quote, e.g. to undercut a high quote of ours."""
    out = data.copy()
    for k in (1, 2):
        out[f"r{k}"] = out[f"r{k}"].mask(out[f"r{k}"] >= high, rate)
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(r3, data, mode="actual", seed=SEED):
    """Our offers r3 against the real Round 1 market, with simulate_market.

    mode "actual"   : each borrower's real type (the exact counterfactual;
                      exact ties, which almost never occur, are broken at random)
         "expected" : averaged over the three types, as in Parts 6-7, to see
                      how much the draw of types mattered
    Returns simulate_market's (loans, summary).
    """
    offers = pd.DataFrame({"r1": data["r1"], "r2": data["r2"],
                           "r3": np.asarray(r3, dtype=float)}, index=data.index)
    if mode == "actual":
        return simulate_market(offers, data["default"], pref=data["pref"],
                               mode="random", seed=seed)
    if mode == "expected":
        return simulate_market(offers, data["default"], mode="expected")
    raise ValueError("mode must be 'actual' or 'expected'")


def check_replay(data):
    """Replay our submitted offers and compare with the course's outcome.

    Returns a dict: share of applicants whose acceptance matches, our profit
    replayed and as reported, and the largest per-loan profit difference.
    """
    loans, summary = replay(data["r3_submitted"], data)
    return {"acceptance match": float((loans["w3"] == data["accepted"]).mean()),
            "replayed profit": float(summary.loc[3, "total_profit"]),
            "reported profit": float(data["profit"].sum()),
            "largest loan difference": float((loans["profit3"] - data["profit"]).abs().max())}


def score_replay(rates, data, mode="actual", bands=BANDS):
    """Every rule replayed on the Round 1 market.

    rates : dict rule name -> our rates on the Round 1 applicants (NaN = no offer)
    Returns (table, profits):
      table   : one row per rule: profit, n_loans, share, avg_rate,
                default_rate_won, avg_p_won, gap (winner's curse, observed -
                predicted default on the loans we win), profit_per_loan, and our
                profit in each market situation (profit_<situation>)
      profits : dict rule -> our profit per applicant (the loans' profit3)
    """
    situation = market_situation(data)
    p = data["p_own"].to_numpy()
    rows, profits = [], {}
    for name, r3 in rates.items():
        loans, summary = replay(r3, data, mode=mode)
        ours = summary.loc[3]
        curse = winners_curse(loans, p, lender=3, groups=bands)
        row = {"rule": name, "profit": ours["total_profit"], "n_loans": ours["n_loans"],
               "share": ours["share_of_applicants"], "avg_rate": ours["avg_rate"],
               "default_rate_won": ours["default_rate_won"],
               "avg_p_won": curse.loc["overall", "avg_pred_p"],
               "gap": curse.loc["overall", "gap"],
               "profit_per_loan": ours["total_profit"] / ours["n_loans"]
               if ours["n_loans"] > 0 else np.nan}
        by_situation = loans["profit3"].groupby(situation, observed=False).sum()
        for s in SITUATIONS:
            row[f"profit_{s}"] = by_situation[s]
        rows.append(row)
        profits[name] = loans["profit3"]
    return pd.DataFrame(rows).set_index("rule"), profits


def profit_by_band(profits, data, bands=BANDS, labels=BAND_LABELS):
    """Rules (rows) x risk bands of p_own (columns): our profit in each band."""
    band = risk_band(data, bands, labels)
    return pd.DataFrame({name: pr.groupby(band, observed=False).sum()
                         for name, pr in profits.items()}).T


def profit_by_situation(profits, data, bands=BANDS, labels=BAND_LABELS):
    """Our profit per risk band and market situation, for each rule.

    Returns a DataFrame indexed by (rule, band), one column per situation,
    plus `n_<situation>` columns with the number of applicants (the same for
    every rule).
    """
    band = risk_band(data, bands, labels)
    situation = market_situation(data)
    counts = pd.crosstab(band, situation).reindex(columns=SITUATIONS)
    counts.columns = [f"n_{s}" for s in SITUATIONS]
    out = {}
    for name, pr in profits.items():
        split = pr.groupby([band, situation], observed=False).sum().unstack()
        out[name] = split.reindex(columns=SITUATIONS).join(counts)
    return pd.concat(out, names=["rule", "band"])


def stress_uncontested(split, survival, by_band=False):
    """Our profit if only a share of the uncontested applicants stays uncontested.

    In Round 2 the other teams may lend where nobody did in Round 1. For each
    band, a share 1 - s of the applicants with no competitor offer or only
    high quotes is assumed to become contested: on them each rule earns its
    own Round 1 profit per contested applicant of that band. So

        profit(s) = C + s * U + (1 - s) * n_U * C / n_C

    with C, U the band's contested and uncontested profit and n_C, n_U the
    numbers of applicants.

    split    : profit_by_situation(...)
    survival : list of shares s in [0, 1]
    by_band  : False sums over bands (rules x shares); True keeps one row per
               (rule, band)
    Returns a DataFrame with one column per survival share.
    """
    contested = split["contested"]
    uncontested = split[SITUATIONS[0]] + split[SITUATIONS[1]]
    n_c = split[f"n_{SITUATIONS[2]}"]
    n_u = split[f"n_{SITUATIONS[0]}"] + split[f"n_{SITUATIONS[1]}"]
    per_contested = (contested / n_c.replace(0, np.nan)).fillna(0.0)
    out = pd.DataFrame({s: contested + s * uncontested + (1 - s) * n_u * per_contested
                        for s in survival}).rename_axis(columns="survival")
    return out if by_band else out.groupby(level="rule", sort=False).sum()


def breakeven_survival(stress, rule, reference):
    """The survival share above which `rule` beats `reference` (linear in s:
    interpolated between the two ends of a stress_uncontested table that
    includes s = 0 and s = 1). NaN if one rule is better at both ends."""
    a0, a1 = stress.loc[rule, 0.0], stress.loc[rule, 1.0]
    b0, b1 = stress.loc[reference, 0.0], stress.loc[reference, 1.0]
    d0, d1 = a0 - b0, a1 - b1
    if np.sign(d0) == np.sign(d1):
        return np.nan
    return d0 / (d0 - d1)


def calibration_by_band(data, loans, bands=BANDS, labels=BAND_LABELS):
    """Calibration of p_own on all Round 1 applicants versus on the loans we
    won (the winner's curse), per risk band.

    loans : replay(...)[0] for our submitted rule
    """
    band = risk_band(data, bands, labels)
    w = loans["w3"]
    d, p = data["default"], data["p_own"]
    groups = pd.DataFrame({"band": band, "w": w, "d": d, "p": p, "wd": w * d, "wp": w * p})
    g = groups.groupby("band", observed=False)
    out = pd.DataFrame({"applicants": g.size(), "mean p_own (all)": g["p"].mean(),
                        "default rate (all)": g["d"].mean(), "loans won": g["w"].sum()})
    out["gap (all)"] = out["default rate (all)"] - out["mean p_own (all)"]
    out["mean p_own (won)"] = g["wp"].sum() / out["loans won"]
    out["default rate (won)"] = g["wd"].sum() / out["loans won"]
    out["gap (won)"] = out["default rate (won)"] - out["mean p_own (won)"]
    return out


def oracle_rates(data, discount=DISCOUNT, step=1e-4):
    """Perfect foresight, the ceiling no rule can beat.

    Knowing the competitors' offers, the borrower's type and the default:
    no offer to borrowers who default; to the others, the highest rate that
    still wins (just under the cheapest competitor's effective rate, after
    the 2-point preference discounts), capped at 100%.
    """
    pref = data["pref"].to_numpy()
    eff1 = data["r1"].fillna(np.inf).to_numpy() - discount * (pref == 1)
    eff2 = data["r2"].fillna(np.inf).to_numpy() - discount * (pref == 2)
    best = np.minimum(eff1, eff2) + discount * (pref == 3) - step
    r = np.minimum(best, MAX_RATE)
    r = np.where((data["default"].to_numpy() == 0) & (r > 0), r, np.nan)
    return pd.Series(r, index=data.index, name="r3")


# ---------------------------------------------------------------------------
# New rules: R8 (high quote) and the R4/R8 hybrids
# ---------------------------------------------------------------------------

def rule_high_quote(data, rate, cutoff=NO_OFFER_THRESHOLD):
    """R8, high quote: the same rate for everyone, whatever their risk.

    It gives up the contested applicants and wins those whose competitors
    make no offer or quote even higher. No offer when p_own >= cutoff (0.5:
    no rate up to 100% can then pay for the risk).
    """
    p = data["p_own"]
    return clip_rates(np.full(len(p), rate), p, floor=0.0, cutoff=cutoff)


def rule_band_quotes(data, edges, quotes, cutoff=NO_OFFER_THRESHOLD):
    """R8 per band: a flat rate per risk band of p_own (NaN = no offer).

    edges  : inner band edges of p_own
    quotes : one rate per band (len(edges) + 1)
    """
    quotes = np.asarray(quotes, dtype=float)
    if len(quotes) != len(edges) + 1:
        raise ValueError("need one quote per band (len(edges) + 1)")
    p = data["p_own"]
    r = quotes[np.digitize(p.to_numpy(dtype=float), edges)]
    return clip_rates(r, p, floor=0.0, cutoff=cutoff)


def rule_hybrid(data, base_rates, edges, high_quotes, cutoff=NO_OFFER_THRESHOLD):
    """R4/R8 hybrid: a competitive rule in some bands, a high quote in others.

    base_rates  : the competitive rule's rates on `data` (e.g. R4's)
    edges       : inner band edges of p_own
    high_quotes : one value per band: NaN keeps the base rule's rate in that
                  band, a rate replaces it with that flat high quote
    """
    high_quotes = np.asarray(high_quotes, dtype=float)
    if len(high_quotes) != len(edges) + 1:
        raise ValueError("need one value per band (len(edges) + 1)")
    p = data["p_own"]
    q = high_quotes[np.digitize(p.to_numpy(dtype=float), edges)]
    r = np.where(np.isnan(q), np.asarray(base_rates, dtype=float), q)
    return clip_rates(r, p, floor=0.0, cutoff=cutoff)


def rule_switch(data, base_rates, p_switch, high_rate, high_side="above",
                cutoff=NO_OFFER_THRESHOLD):
    """The simplest hybrid, one switch point p_switch of p_own.

    high_side "above": the base rule below p_switch, a flat high quote from
                       p_switch on (competitive on safe applicants)
              "below": the flat high quote below p_switch, the base rule from
                       p_switch on (competitive on risky applicants)
    """
    if high_side == "above":
        quotes = [np.nan, high_rate]
    elif high_side == "below":
        quotes = [high_rate, np.nan]
    else:
        raise ValueError("high_side must be 'above' or 'below'")
    return rule_hybrid(data, base_rates, [p_switch], quotes, cutoff)


def switch_profits(base_band, flat_band, edges, high_side="above"):
    """Profit of every switch hybrid, with the switch at each band edge.

    Uses the per-band profits of the base rule and of every flat rate
    (candidate_band_profits without folds), so no extra replay is needed:
    for each switch point, the base rule's profit on its side plus the best
    flat rate's profit on the other side.

    base_band : Series, the base rule's profit per band (0 .. len(edges))
    flat_band : DataFrame, bands x flat rates
    Returns a DataFrame indexed by the switch point (0 and 1 included: all
    high quote / all base rule), with the best high quote, the total profit and
    the base rule's part of it.
    """
    n_bands = len(edges) + 1
    rows = {}
    for k in range(n_bands + 1):            # the high quote covers bands [k, n) or [0, k)
        high = list(range(k, n_bands)) if high_side == "above" else list(range(k))
        base = [b for b in range(n_bands) if b not in high]
        flat = flat_band.loc[high].sum() if high else pd.Series(0.0, index=flat_band.columns)
        base_part = base_band.loc[base].sum() if base else 0.0
        p_switch = ([0.0] + list(edges) + [1.0])[k]
        rows[p_switch] = {"base bands": len(base),
                          "best high quote": flat.idxmax() if high else np.nan,
                          "profit": base_part + (flat.max() if high else 0.0),
                          "of which base rule": base_part}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("switch at p_own")


# ---------------------------------------------------------------------------
# Hindsight tuning
# ---------------------------------------------------------------------------

def make_folds(data, n_folds=2, seed=SEED):
    """A random fold number (0 .. n_folds - 1) per applicant, in equal shares."""
    rng = np.random.default_rng(seed)
    folds = np.resize(np.arange(n_folds), len(data))
    rng.shuffle(folds)
    return pd.Series(folds, index=data.index, name="fold")


def candidate_band_profits(data, candidates, edges, folds=None):
    """Our profit in each risk band for each candidate set of rates.

    One replay per candidate. This is exact per band: an applicant's choice
    depends only on the offers made to them, so a band's profit does not
    depend on what we offer in the other bands (as for R4 in Part 6).

    candidates : dict label -> our rates on `data`
    edges      : inner band edges of p_own
    folds      : optional fold per applicant (make_folds), to cross-fit
    Returns a DataFrame indexed by band (or by (fold, band)), one column per
    candidate.
    """
    band = pd.Series(np.digitize(data["p_own"].to_numpy(dtype=float), edges),
                     index=data.index, name="band")
    keys = [band] if folds is None else [folds, band]
    out = {}
    for label, r3 in candidates.items():
        loans, _ = replay(r3, data)
        out[label] = loans["profit3"].groupby(keys).sum()
    return pd.DataFrame(out)


def best_per_band(band_profits):
    """The best candidate of each band (rows of candidate_band_profits without
    folds): its label and profit."""
    return pd.DataFrame({"best": band_profits.idxmax(axis=1),
                         "profit": band_profits.max(axis=1)})


def crossfit_per_band(band_profits_by_fold):
    """Honest version of best_per_band.

    For each fold, choose each band's best candidate on the other folds, and
    book that candidate's profit on this fold. Summed over folds, it is what
    the hindsight choice would earn on applicants it was not chosen on.
    Returns (total, choices): total cross-fitted profit, and the candidate
    chosen for each (fold, band).
    """
    folds = band_profits_by_fold.index.get_level_values(0).unique()
    total, choices = 0.0, {}
    for k in folds:
        others = band_profits_by_fold.drop(index=k, level=0).groupby(level=1).sum()
        chosen = others.idxmax(axis=1)
        this = band_profits_by_fold.loc[k]
        for b, label in chosen.items():
            total += this.loc[b, label] if b in this.index else 0.0
            choices[(k, b)] = label
    return total, pd.Series(choices, name="chosen")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

SITUATION_COLORS = {"no competitor offer": "#1baf7a", "only high quotes": "#eda100",
                    "contested": "#9a9893"}


def plot_situations(data, ax, title="Market situation of each applicant, by risk band"):
    """Stacked bars: share of applicants with no competitor offer, only high
    quotes, or at least one cheaper competitor offer, per band of p_own."""
    shares = pd.crosstab(risk_band(data), market_situation(data), normalize="index")
    bottom = np.zeros(len(shares))
    for s in SITUATIONS:
        ax.bar(range(len(shares)), 100 * shares[s], bottom=bottom, color=SITUATION_COLORS[s],
               edgecolor=SURFACE, label=s)
        bottom += 100 * shares[s].to_numpy()
    ax.set_xticks(range(len(shares)), shares.index)
    ax.set_xlabel("our predicted default probability p_own")
    ax.set_ylabel("share of applicants (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper left", labelcolor=INK_SECONDARY, ncol=3)


def plot_competitor_rates(data, ax, title="Competitors' offers in Round 1"):
    """Histogram of the two competitors' offered rates (offers only)."""
    bins = np.linspace(0, 1, 101)
    for k in (1, 2):
        ax.hist(100 * data[f"r{k}"].dropna(), bins=100 * bins, histtype="step", lw=1.8,
                color=LENDER_COLORS[k], label=f"lender {k}")
    ax.set_yscale("log")
    ax.set_xlabel("offered rate (%)")
    ax.set_ylabel("applicants (log scale)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)


def plot_rule_profits(table, ax, highlight=(), reference=None,
                      title="Realised Round 1 profit of each rule"):
    """Horizontal bars of profit (million EUR), split into contested and
    uncontested applicants; `highlight` rules in dark ink labels;
    `reference` = a profit to mark with a dashed line (e.g. the winner's)."""
    t = table.iloc[::-1]
    y = np.arange(len(t))
    contested = t["profit_contested"] / 1e6
    other = (t["profit"] - t["profit_contested"]) / 1e6
    ax.barh(y, contested, color=SITUATION_COLORS["contested"], label="contested applicants")
    ax.barh(y, other, left=np.where(contested > 0, contested, 0),
            color=SITUATION_COLORS["no competitor offer"],
            label="no offer or only high quotes from competitors")
    ax.set_yticks(y, t.index)
    for tick, name in zip(ax.get_yticklabels(), t.index):
        tick.set_color(INK if name in highlight else INK_SECONDARY)
        tick.set_fontweight("bold" if name in highlight else "normal")
    for yi, total in zip(y, t["profit"] / 1e6):
        ax.text(max(total, 0) + 1, yi, f"{total:+.1f}", va="center", fontsize=8, color=INK)
    if reference is not None:
        ax.axvline(reference / 1e6, color=INK, lw=1, ls="--")
    ax.axvline(0, color=REFERENCE, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("profit (million EUR)")
    ax.set_title(title)
    ax.legend(loc="lower right", labelcolor=INK_SECONDARY)


def plot_stress(stress, ax, rules=None, reference=None,
                title="Profit when competitors start lending to the uncontested applicants"):
    """Profit (million EUR) of each rule against the survival share of the
    uncontested pool (a stress_uncontested table); `reference` = a profit to
    mark with a dashed line."""
    rules = list(stress.index) if rules is None else rules
    x = 100 * np.asarray(stress.columns, dtype=float)
    for color, r in zip(SERIES_COLORS * 2, rules):
        ax.plot(x, stress.loc[r] / 1e6, marker="o", color=color, label=r)
    if reference is not None:
        ax.axhline(reference / 1e6, color=INK, lw=1, ls="--")
    ax.set_xlabel("share of the Round 1 uncontested applicants still uncontested (%)")
    ax.set_ylabel("profit (million EUR)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)


def plot_rate_curves(curves, ax, title="Profit of a flat rate, by risk band"):
    """Profit (million EUR) of a flat quote (x axis) in each band (one line
    per band). curves: DataFrame bands (rows) x rates (columns)."""
    for color, (band, row) in zip(SERIES_COLORS * 2, curves.iterrows()):
        ax.plot(100 * np.asarray(row.index, dtype=float), row.to_numpy() / 1e6, lw=1.6,
                color=color, label=band)
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xlabel("flat rate quoted to everyone in the band (%)")
    ax.set_ylabel("profit (million EUR)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY, ncol=2, fontsize=8)
