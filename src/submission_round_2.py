"""Part 12 - pricing the Round 2 applications and writing the submission file.

Same format as Round 1 (src/submission_round_1.py): one row per application,
columns id and rate (decimal, at most 1, empty when we make no offer). The
Round 2 applicants have ids 100,001 to 200,000.

The pipeline:

* predict: p_own_200k, our default model refitted on the 200k loans (past
  loans + Round 1), from data/predictions/applications_round2_own.csv
  (Part 11, r2.predict_round2);
* price: the Part 11 rule R4-step|R8 (data/pricing/final_rule_round2.json,
  rebuilt with r2.load_round2_rule): break-even + a step margin below the
  switch point, a flat quote above it, no offer where the quote is below our
  break-even;
* check: the course format (check_submission from Round 1) and that every
  rate is the one the saved rule prescribes (check_rule);
* preview: our expected profit on the Round 2 applicants in Part 10's
  empirical scenario and its main stresses (forecast_profit).

Usage from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src import submission_round_2 as s2

    apps = s2.load_round2_applications()
    rule, spec = s2.load_rule()
    rates = s2.price(apps, rule)
"""

import json

import numpy as np
import pandas as pd

from src.config import DATA_PREDICTIONS, NEW_APPS_R2, SUBMISSIONS_DIR
from src.pricing_rules import PRICING_DIR, be
from src.round_1_replay import load_round1
from src import round_1_competitors_rules as crr
from src import round_2_rule as r2
from src.submission_round_1 import check_submission, make_submission, write_submission

SUBMISSION_FILE = SUBMISSIONS_DIR / "Lender3_Round2_submission.csv"
RULE_FILE = "final_rule_round2.json"
P_COL = "p_own_200k"            # the model priced in Round 2 (Part 11, section 7)
FIRST_ID, LAST_ID = 100_001, 200_000

__all__ = ["SUBMISSION_FILE", "load_round2_applications", "load_rule", "rule_table", "price",
           "no_offer_reasons", "offers_by_band", "check_rule", "compare_models",
           "forecast_profit", "make_submission", "check_submission", "write_submission"]


def load_round2_applications(p_col=P_COL):
    """The Round 2 applicants indexed by id, with the course's columns, both
    predictions (p_own_100k, p_own_200k) and p_own = the one we price on.

    Raises if the prediction file does not cover exactly the applicants of
    the course's file, or if the ids are not 100,001 to 200,000."""
    raw = pd.read_csv(NEW_APPS_R2, index_col="id")
    preds = pd.read_csv(DATA_PREDICTIONS / "applications_round2_own.csv", index_col="id")
    if raw.index.duplicated().any() or preds.index.duplicated().any():
        raise ValueError("duplicated ids")
    if set(raw.index) != set(preds.index):
        raise ValueError("predictions and applications do not cover the same ids")
    if set(raw.index) != set(range(FIRST_ID, LAST_ID + 1)):
        raise ValueError(f"ids are not {FIRST_ID} to {LAST_ID}")
    apps = raw.join(preds)
    if apps[p_col].isna().any() or not apps[p_col].between(0, 1).all():
        raise ValueError(f"{p_col} missing or outside [0, 1]")
    apps["p_own"] = apps[p_col]
    return apps


def load_rule(filename=RULE_FILE):
    """(rule, spec): the Part 11 rule as a function data -> rates, and the
    saved JSON it was rebuilt from."""
    spec = json.loads((PRICING_DIR / filename).read_text())
    return r2.load_round2_rule(filename), spec


def rule_table(spec):
    """The saved rule as a readable table: one row per p_own band with the
    margin over break-even or the flat quote."""
    edges = list(spec["poly"]["edges"])
    margins = [c[0] for c in spec["poly"]["coefs"]]
    if spec["poly"]["degree"] != 0:
        raise ValueError("rule_table only describes step margins (degree 0)")
    hyb = spec["hybrid"]
    lo, hi = [0.0] + edges, edges + [1.0]
    rows = []
    for a, b, m in zip(lo, hi, margins):
        if hyb and hyb["high_side"] == "above" and a >= hyb["p_switch"] - 1e-12:
            continue
        rows.append({"p_own from": a, "p_own to": b, "rate": f"be + {100 * m:.2f} pts"})
    if hyb:
        cut = hyb["quote"] / (1 + hyb["quote"])
        rows.append({"p_own from": hyb["p_switch"], "p_own to": cut,
                     "rate": f"{hyb['quote']:.0%} flat"})
        rows.append({"p_own from": cut, "p_own to": 1.0, "rate": "no offer"})
    return pd.DataFrame(rows)


def price(apps, rule):
    """Our rate per applicant (NaN = no offer), indexed like apps."""
    rates = pd.Series(np.asarray(rule(apps), dtype=float), index=apps.index, name="rate")
    return rates


def no_offer_reasons(apps, rates, spec):
    """Why each applicant gets (or not) an offer, as a Series of labels."""
    p = apps["p_own"]
    hyb = spec["hybrid"]
    reason = pd.Series("offer: break-even + margin", index=apps.index)
    if hyb:
        high = p >= hyb["p_switch"] if hyb["high_side"] == "above" else p < hyb["p_switch"]
        reason[high] = f"offer: {hyb['quote']:.0%} quote"
        reason[high & rates.isna()] = "no offer: quote below our break-even"
    reason[~reason.str.startswith("no offer") & rates.isna()] = "no offer: other"
    return reason


def offers_by_band(apps, rates, edges, labels=None):
    """Applicants, offers, mean rate, mean break-even and mean margin by
    band of p_own (band edges `edges`, from 0 to 1)."""
    p = apps["p_own"]
    band = pd.cut(p, edges, labels=labels, right=False)
    t = pd.DataFrame({"band": band, "p": p, "rate": rates, "be": be(p.to_numpy())})
    out = t.groupby("band", observed=False).agg(
        applicants=("p", "size"), mean_p=("p", "mean"), offers=("rate", "count"),
        mean_rate=("rate", "mean"), min_rate=("rate", "min"), max_rate=("rate", "max"))
    offered = t[t["rate"].notna()]
    out["mean_margin"] = (offered["rate"] - offered["be"]).groupby(offered["band"],
                                                                   observed=False).mean()
    return out


def check_rule(apps, rates, spec, tol=1e-12):
    """Check that `rates` is exactly what the saved rule prescribes,
    recomputed from the JSON without the Part 11 code; raise ValueError on
    any mismatch. Also checks that no offer is below our break-even (every
    offer has a positive expected profit under p_own). Returns a dict of
    counts."""
    p = apps["p_own"].to_numpy(dtype=float)
    r = rates.reindex(apps.index).to_numpy(dtype=float)
    poly, hyb = spec["poly"], spec["hybrid"]
    if poly["degree"] != 0:
        raise ValueError("check_rule recomputes step margins (degree 0) only")

    margin = np.asarray([c[0] for c in poly["coefs"]])[np.digitize(p, poly["edges"])]
    expected = np.maximum(be(p) + margin, poly["floor"])
    expected = np.where(expected <= 1.0, expected, np.nan)
    high = np.zeros(len(p), dtype=bool)
    if hyb:
        high = p >= hyb["p_switch"] if hyb["high_side"] == "above" else p < hyb["p_switch"]
        expected = np.where(high, np.where(be(p) < hyb["quote"], hyb["quote"], np.nan), expected)

    same = (np.isnan(r) & np.isnan(expected)) | (np.abs(r - expected) <= tol)
    if not same.all():
        raise ValueError(f"{int((~same).sum())} rate(s) differ from the saved rule")
    offered = ~np.isnan(r)
    below = offered & (r < be(p) - tol)
    if below.any():
        raise ValueError(f"{int(below.sum())} offer(s) below our break-even")
    return {"applicants": len(p), "offers": int(offered.sum()),
            "margin offers": int((offered & ~high).sum()),
            "quote offers": int((offered & high).sum()),
            "no offer": int((~offered).sum())}


def compare_models(apps, rule, p_switch):
    """How the choice of model (100k vs 200k loans) changes the submission:
    correlation of the predictions, applicants on each side of the switch
    point, and the rates."""
    alt = apps.assign(p_own=apps["p_own_100k"])
    r_200, r_100 = price(apps, rule), price(alt, rule)
    side_200, side_100 = apps["p_own_200k"] >= p_switch, apps["p_own_100k"] >= p_switch
    both = r_200.notna() & r_100.notna()
    return pd.Series({
        "correlation of p_own": apps["p_own_100k"].corr(apps["p_own_200k"]),
        "mean p_own, 100k model": apps["p_own_100k"].mean(),
        "mean p_own, 200k model": apps["p_own_200k"].mean(),
        "above switch, 100k model": int(side_100.sum()),
        "above switch, 200k model": int(side_200.sum()),
        "change side of the switch": int((side_100 != side_200).sum()),
        "same rate (both offer)": int((both & np.isclose(r_200, r_100)).sum()),
        "mean |rate change| where both offer": float((r_200 - r_100)[both].abs().mean()),
    })


FORECAST_SCENARIOS = {
    "empirical (Round 1 repeats)": {},
    "never-offer teams offer": {"never_offer_team_offers": True},
    "high quotes at 85%": {"high_quote_rate": 0.85},
    "no-offers & high quotes -50%": {"keep_no_offer": 0.5, "keep_high_quote": 0.5},
    "curves -1 pt": {"curve_shift": 0.01},
    "pool gone (-100%)": {"keep_no_offer": 0.0, "keep_high_quote": 0.0},
}


def forecast_profit(apps, rules, scenarios=None, seeds=(0, 1, 2, 3, 4), donors=None):
    """Our expected profit on the Round 2 applicants if the competitors
    behave as in Round 1 (Part 10's empirical scenario) or as in its
    stresses.

    For each seed, defaults are drawn from p_own (simulate_defaults), the
    competitor offers are resampled from the Round 1 market given (risk bin,
    default), and each rule is scored in expected mode over the borrower
    types (r2.make_market / r2.market_profit). A forecast, not a guarantee:
    it assumes the Round 1 competitors and a calibrated p_own.

    rules : dict name -> function data -> rates
    Returns (table, loans): mean profit (EUR) and mean loans won, rules x
    scenarios; each also has a column "sd" of profit over seeds in the first
    scenario.
    """
    scenarios = FORECAST_SCENARIOS if scenarios is None else scenarios
    donors = load_round1() if donors is None else donors
    rates = {name: price(apps, rule).to_numpy() for name, rule in rules.items()}
    profit = {(n, s): [] for n in rules for s in scenarios}
    loans = {(n, s): [] for n in rules for s in scenarios}
    for seed in seeds:
        target = crr.simulate_defaults(apps, seed=seed)
        for s, knobs in scenarios.items():
            mk = crr.empirical_scenario(target, donors, seed=seed, **knobs)
            market = r2.make_market(mk["r1"], mk["r2"], mk["default"])
            for n in rules:
                pi, w = r2.market_profit(market, rates[n], return_weight=True)
                profit[(n, s)].append(pi.sum())
                loans[(n, s)].append(w.sum())

    def table(values):
        return pd.DataFrame({s: {n: np.mean(values[(n, s)]) for n in rules} for s in scenarios})

    out = table(profit)
    first = next(iter(scenarios))
    out["sd"] = [np.std(profit[(n, first)], ddof=1) if len(seeds) > 1 else np.nan for n in rules]
    return out, table(loans)
