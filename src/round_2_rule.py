"""Part 11 - improving our rule for Round 2.

Parts 9 and 10 showed that the Round 1 profit was decided by the uncontested
applicants (no competitor offer, or only fixed high quotes), and that each
competitor slot is a mixture of four teams drawn at random per loan. This
module re-tunes our candidate rules against that kind of market and
stress-tests them:

* populations: the past loans (train / validation, with our reconstructed
  competitor default probabilities p_lender1 / p_lender2) and the Round 1
  applicants split into two halves (A for tuning, B held out);
* a fast market evaluator (our expected profit only, same choice rule as
  src/simulator.py), checked against simulate_market;
* the four-team scenario generator: for every draw, four team rules per
  competitor slot are drawn from a library of team types (never offers,
  pricing curve, cutoff, high quote, flat quote, copies of our Round 1
  rules), and each applicant gets one team per slot at random;
* scenario sets: tuning (empirical Round 1 scenario of Part 10 and its
  stresses on half A, four-team mixtures on train) and stress tests (Part 7
  scenarios, held-out mixtures and the eight teams inferred in Part 10 on
  validation, Part 10's empirical stresses and the exact Round 1 market on
  half B). Half B and the validation loans are never used in tuning;
* rule families: R4 with a polynomial margin inside each bin (degree 0 =
  step, 1 = linear, 3 = cubic; no monotonicity constraint), flat high quote
  (R8), R4/R8 hybrids, R7 re-tuned. The competitive rules (R4 variants, R7)
  are tuned on the contested markets only (CONTESTED); hybrids are chosen
  by minimax regret across all tuning scenarios;
* scoring and the Part 7 criteria, the share/margin trade-off;
* retraining the default model on 200k loans (past loans + Round 1);
* plots used by analysis/round_2_rule.ipynb.

Every rule takes a population DataFrame (needs p_own; R7 also p_public and
our features) and returns our rate per applicant (NaN = no offer).

Profits are reported per 100,000 applicants (the size of a round), so
populations of different sizes are comparable.

Typical use from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src import round_2_rule as r2

    pops = r2.load_populations()
    tuning = r2.tuning_scenarios(pops)
    fit = r2.fit_bin_poly(pops, tuning, degree=1)
"""

import json
import time

import numpy as np
import pandas as pd

from src.config import (DATA_PREDICTIONS, DISCOUNT, LOAN_AMOUNT, NEW_APPS_R1, NEW_APPS_R2,
                        ROUND1_DIAGNOSTIC, SEED)
from src.data_prep import logreg_prep_data
from src.fair_pricing import MAX_RATE, NO_OFFER_THRESHOLD
from src.model_plots import INK, INK_SECONDARY, REFERENCE, SURFACE
from src.models import MODEL_FACTORIES, feature_columns, fit_model, load_data, predict_default
from src.pricing_plots import LENDER_COLORS, SERIES_COLORS
from src.pricing_rules import (PRICING_DIR, be, clip_rates, decision_criteria,
                               fit_competitor_model, load_pricing_data, load_tuned,
                               load_tuned_rules, make_scenario, part7_scenarios, pts,
                               regret_table, rule_undercut, smooth_bin_margins, undercut_forecasts)
from src.round_1_replay import (BAND_LABELS, BANDS, load_round1, make_folds, rule_high_quote,
                                rule_switch, switch_profits)
from src import round_1_competitors_rules as crr

PER_100K = 100_000          # profits are scaled to a round of 100,000 applicants

# Part 9's hybrids: the 89.5% high quote, switch at the 6th R4 decile edge (11.1%).
R1_HIGH_QUOTE = 0.895
R1_SWITCH_DECILE = 5        # index in R4's edges: edges[5] = 0.111

RATE_FLOOR = 0.005          # as R4


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def r4_edges():
    """R4's decile edges of p_own (Part 6, on train): the bins of every
    per-bin rule here, so that results stay comparable with R4."""
    return np.asarray(load_tuned()["R4"]["edges"], dtype=float)


def load_populations(train=None, seed=SEED):
    """The applicant populations used for tuning and testing, as a dict:

      train      : 80,000 past loans, out-of-fold predictions (Part 6); has
                   p_lender1 / p_lender2 for the four-team mixtures
      validation : 20,000 past loans (Part 7), same columns
      round1     : the 100,000 Round 1 applicants with the real competitor
                   offers, borrower types and defaults (Part 9)
      round1_A   : a random half of round1, used for tuning
      round1_B   : the other half, held out for the stress tests

    Every frame has p_own, p_public, our features and default.
    """
    train = load_pricing_data("train") if train is None else train
    round1 = load_round1()
    half = make_folds(round1, n_folds=2, seed=seed)
    return {"train": train, "validation": load_pricing_data("validation"), "round1": round1,
            "round1_A": round1[half == 0], "round1_B": round1[half == 1]}


# ---------------------------------------------------------------------------
# Fast market evaluator
# ---------------------------------------------------------------------------

def make_market(r1, r2, defaults, pref=None, discount=DISCOUNT):
    """Precompute the competitors' side of a market, to price many rules fast.

    The choice rule is src/simulator.py's: effective rate = offered rate
    minus `discount` for the preferred lender, lowest effective rate wins,
    exact ties split. pref None = expected mode (average over the three
    types, as in Parts 6-7); an array of types = those types (Round 1 exact).
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1 = np.where(np.isfinite(r1), r1, np.inf)
    r2 = np.where(np.isfinite(r2), r2, np.inf)
    if pref is None:
        types = np.array([1, 2, 3])[None, :]                   # (1, 3)
    else:
        types = np.asarray(pref, dtype=int)[:, None]           # (n, 1)
    return {"e1": r1[:, None] - discount * (types == 1),
            "e2": r2[:, None] - discount * (types == 2),
            "our_discount": discount * (types == 3),
            "default": np.asarray(defaults, dtype=float),
            "n": len(r1)}


def market_profit(market, r3, loan_amount=LOAN_AMOUNT, tol=1e-12, return_weight=False):
    """Our expected profit per applicant (and optionally the probability that
    the applicant takes our loan) for our rates r3 in a make_market market."""
    r = np.asarray(r3, dtype=float)
    ok = np.isfinite(r)
    e3 = np.where(ok, r, np.inf)[:, None] - market["our_discount"]
    e1, e2 = market["e1"], market["e2"]
    with np.errstate(invalid="ignore"):
        best = np.minimum(np.minimum(e1, e2), e3)
        win = ok[:, None] & (e3 <= best + tol)
        ties = 1 + (np.abs(e1 - e3) <= tol) + (np.abs(e2 - e3) <= tol)
    w = (win / ties).mean(axis=1)
    d = market["default"]
    profit = w * np.where(d == 1, -loan_amount, np.where(ok, r, 0.0) * loan_amount)
    return (profit, w) if return_weight else profit


# ---------------------------------------------------------------------------
# Our Round 1 rules (reference and copies)
# ---------------------------------------------------------------------------

def round1_r4():
    """R4 as submitted in Round 1 (Part 7), a function data -> rates."""
    params = load_tuned()["R4"]
    curve = smooth_bin_margins(params["centres"], params["margins"])
    edges = np.asarray(params["edges"], dtype=float)

    def rule(data):
        p = data["p_own"]
        return clip_rates(be(p) + curve(p), p, params["floor"])
    rule.edges = edges
    return rule


def round1_rules(train):
    """The Round 1 rules the guidelines ask to carry over, fixed as tuned in
    Parts 6 and 9: dict name -> function data -> rates."""
    r4 = round1_r4()
    switch = r4.edges[R1_SWITCH_DECILE]
    r7 = load_tuned_rules(train)["R7"]
    return {
        "R4 (Round 1)": r4,
        "R7 (Part 6)": r7,
        "R8 89.5% (Part 9)": lambda data: rule_high_quote(data, R1_HIGH_QUOTE),
        "R4|R8 11.1% (Part 9)": lambda data: rule_switch(data, r4(data), switch, R1_HIGH_QUOTE,
                                                         "above"),
        "R8|R4 11.1% (Part 9)": lambda data: rule_switch(data, r4(data), switch, R1_HIGH_QUOTE,
                                                         "below"),
    }


# ---------------------------------------------------------------------------
# Four-team competitor mixtures
# ---------------------------------------------------------------------------

# Team types (Part 10's eight teams, plus Round 2 copycats):
#   never       : never offers (teams 1A, 2A)
#   curve       : be(p) + m_safe below p_switch, + m_risky above (1C, 2C, 2D)
#   cutoff      : a curve that makes no offer from `cutoff` on (1D)
#   high quote  : a curve below p_high, a fixed quote q above (1B, 2B)
#   flat quote  : q for everyone (a team copying R8)
#   copy R4     : our Round 1 R4 on its own p
#   copy R4|R8  : R4 below 11.1%, q above (a team copying Part 9's hybrid)
TEAM_TYPES = ["never", "curve", "cutoff", "high quote", "flat quote", "copy R4", "copy R4|R8"]

# Priors over team types (the probability that each of a slot's four teams is
# of a type) and over their parameters.
#   "round 1"    : Round 1's structure (Part 10): one never-offer, one
#                  high-quote and two curve / cutoff teams per slot
#   "round 2"    : our central guess: teams have seen their diagnostic file,
#                  fewer never offer, some copy R4, R8 or R4|R8, high quotes
#                  may be undercut down to 80%
#   "gap closed" : nobody refuses, high quotes at 80-90%, curves cheaper: the
#                  uncontested pool is (almost) gone
PRIORS = {
    "round 1": {"teams": {"never": 0.25, "high quote": 0.25, "curve": 0.25, "cutoff": 0.25},
                "m_safe": (0.014, 0.03), "q_high": (0.90, 0.98), "q_copy": (0.85, 0.90)},
    "round 2": {"teams": {"never": 0.08, "high quote": 0.17, "curve": 0.30, "cutoff": 0.15,
                          "flat quote": 0.10, "copy R4": 0.10, "copy R4|R8": 0.10},
                "m_safe": (0.014, 0.035), "q_high": (0.80, 0.98), "q_copy": (0.80, 0.90)},
    "gap closed": {"teams": {"curve": 0.45, "high quote": 0.10, "copy R4": 0.25,
                             "copy R4|R8": 0.20},
                   "m_safe": (0.01, 0.03), "q_high": (0.80, 0.90), "q_copy": (0.80, 0.90)},
}


def draw_team(kind, rng, prior):
    """One team of type `kind` with parameters drawn from `prior`: a dict."""
    team = {"kind": kind}
    if kind in ("curve", "cutoff", "high quote"):
        team["m_safe"] = rng.uniform(*prior["m_safe"])
    if kind in ("curve", "cutoff"):
        team["p_switch"] = rng.choice([0.08, 0.10, 0.125])
        team["m_risky"] = rng.choice([team["m_safe"], 0.05, 0.10, 0.20, 0.35])
    if kind == "cutoff":
        team["cutoff"] = rng.choice([0.10, 0.125, 0.15, 0.20])
    if kind == "high quote":
        team["p_high"] = rng.choice([0.05, 0.07, 0.10, 0.125])
        team["q"] = rng.uniform(*prior["q_high"])
    if kind in ("flat quote", "copy R4|R8"):
        team["q"] = rng.uniform(*prior["q_copy"])
    return team


def team_rates(team, p, r4=None):
    """The rates of one team for its own default probabilities p (array),
    with the game rules: no offer when p >= 0.5 or above 100%."""
    p = np.asarray(p, dtype=float)
    kind = team["kind"]
    if kind == "never":
        r = np.full(len(p), np.nan)
    elif kind in ("curve", "cutoff"):
        r = be(p) + np.where(p < team["p_switch"], team["m_safe"], team["m_risky"])
        if kind == "cutoff":
            r = np.where(p < team["cutoff"], r, np.nan)
    elif kind == "high quote":
        r = np.where(p < team["p_high"], be(p) + team["m_safe"], team["q"])
    elif kind == "flat quote":
        r = np.full(len(p), team["q"])
    elif kind in ("copy R4", "copy R4|R8"):
        r4 = round1_r4() if r4 is None else r4
        r = np.asarray(r4(pd.DataFrame({"p_own": p})), dtype=float)
        if kind == "copy R4|R8":
            r = np.where(p < r4.edges[R1_SWITCH_DECILE], r, team["q"])
    else:
        raise ValueError(f"unknown team type '{kind}'")
    r = np.where(p < NO_OFFER_THRESHOLD, r, np.nan)
    return np.where(r <= MAX_RATE, np.maximum(r, 0.0), np.nan)


def draw_mixture(data, prior, seed, p_cols=("p_lender1", "p_lender2"), r4=None):
    """One four-team market for `data`: per slot, four teams drawn from
    `prior`, and each applicant served by one of them at random (each slot
    independently, as in the game).

    Returns (r1, r2, teams): the offers and the dict slot -> list of teams.
    """
    prior = PRIORS[prior] if isinstance(prior, str) else prior
    rng = np.random.default_rng(seed)
    kinds = list(prior["teams"])
    probs = np.asarray([prior["teams"][k] for k in kinds], dtype=float)
    teams = {slot: [draw_team(k, rng, prior) for k in rng.choice(kinds, 4, p=probs / probs.sum())]
             for slot in (1, 2)}
    r1, r2 = teams_market(data, teams, rng, p_cols, r4)
    return r1, r2, teams


def teams_market(data, teams, rng, p_cols=("p_lender1", "p_lender2"), r4=None):
    """The offers of given teams (dict slot -> list of four teams): each
    applicant is served by one team per slot, drawn at random.
    rng : a numpy Generator or a seed. Returns (r1, r2)."""
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    r4 = round1_r4() if r4 is None else r4
    n = len(data)
    rates = []
    for slot, col in zip((1, 2), p_cols):
        who = rng.integers(0, len(teams[slot]), n)
        p = data[col].to_numpy(dtype=float)
        r = np.full(n, np.nan)
        for j, team in enumerate(teams[slot]):
            m = who == j
            r[m] = team_rates(team, p[m], r4)
        rates.append(r)
    return rates[0], rates[1]


# The eight Round 1 teams inferred in Part 10 (section 3.3), in terms of the
# team types above. Margins rising with risk are approximated by a step at
# p_switch (Part 10's binned medians on either side of it).
PART10_TEAMS = {
    1: [{"kind": "never"},                                                       # 1A
        {"kind": "high quote", "m_safe": 0.02, "p_high": 0.07, "q": 0.9199},     # 1B
        {"kind": "cutoff", "m_safe": 0.02, "p_switch": 0.08, "m_risky": 0.08,
         "cutoff": 0.175},                                                        # 1C
        {"kind": "cutoff", "m_safe": 0.02, "p_switch": 0.06, "m_risky": 0.045,
         "cutoff": 0.20}],                                                        # 1D
    2: [{"kind": "never"},                                                       # 2A
        {"kind": "high quote", "m_safe": 0.025, "p_high": 0.10, "q": 0.9789},    # 2B
        {"kind": "curve", "m_safe": 0.03, "p_switch": 0.10, "m_risky": 0.08},   # 2C
        {"kind": "curve", "m_safe": 0.024, "p_switch": 0.10, "m_risky": 0.36}],  # 2D
}


def part10_variants():
    """The Part 10 teams and three ways they could adapt for Round 2:
    dict name -> teams (dict slot -> list of four teams)."""
    def replace(teams, kind, new):
        return {k: [dict(new) if t["kind"] == kind else dict(t) for t in v] for k, v in teams.items()}

    def quotes_at(teams, q):
        return {k: [{**t, "q": q} if "q" in t else dict(t) for t in v] for k, v in teams.items()}

    average_curve = {"kind": "curve", "m_safe": 0.024, "p_switch": 0.10, "m_risky": 0.10}
    return {
        "Part 10 teams": PART10_TEAMS,
        "Part 10, never-offer teams offer": replace(PART10_TEAMS, "never", average_curve),
        "Part 10, high quotes at 85%": quotes_at(PART10_TEAMS, 0.85),
        "Part 10, never-offer teams copy R4|R8": replace(
            PART10_TEAMS, "never", {"kind": "copy R4|R8", "q": 0.85}),
    }


def describe_teams(teams):
    """A readable one-line summary per team of a draw_mixture result."""
    def fmt(t):
        parts = [t["kind"]]
        for key, unit in (("m_safe", "pts"), ("m_risky", "pts risky"), ("cutoff", "cutoff"),
                          ("p_high", "from"), ("q", "quote")):
            if key in t:
                v = 100 * t[key]
                parts.append(f"{v:.1f} {unit}" if unit == "pts" or unit == "pts risky"
                             else f"{unit} {v:.1f}%")
        return ", ".join(parts)
    return pd.DataFrame({f"slot {k}": [fmt(t) for t in v] for k, v in teams.items()},
                        index=[f"team {j + 1}" for j in range(4)])


# ---------------------------------------------------------------------------
# Scenario sets
# ---------------------------------------------------------------------------

def _scenario(name, group, population, market, weight=1.0):
    return {"name": name, "group": group, "population": population, "market": market,
            "weight": weight, "scale": PER_100K / market["n"]}


# Part 10's empirical stresses (keyword arguments of empirical_scenario).
TUNING_STRESSES = {
    "never-offer teams offer": {"never_offer_team_offers": True},
    "no-offers & high quotes -50%": {"keep_no_offer": 0.5, "keep_high_quote": 0.5},
    "high quotes at 85%": {"high_quote_rate": 0.85},
    "curves -1 pt": {"curve_shift": 0.01},
    "no-offers & high quotes -100%": {"keep_no_offer": 0.0, "keep_high_quote": 0.0},
}

TEST_STRESSES = {
    "Round 1 empirical": {},
    "no-offers & high quotes -25%": {"keep_no_offer": 0.75, "keep_high_quote": 0.75},
    "no-offers & high quotes -50%": {"keep_no_offer": 0.5, "keep_high_quote": 0.5},
    "no-offers & high quotes -100%": {"keep_no_offer": 0.0, "keep_high_quote": 0.0},
    "high quotes at 90%": {"high_quote_rate": 0.90},
    "high quotes at 88%": {"high_quote_rate": 0.88},
    "high quotes at 85%": {"high_quote_rate": 0.85},
    "curves -1 pt": {"curve_shift": 0.01},
    "curves -2 pts": {"curve_shift": 0.02},
    "curves -3 pts": {"curve_shift": 0.03},
    "never-offer teams offer": {"never_offer_team_offers": True},
    "never-offer teams offer + HQ 85%": {"never_offer_team_offers": True,
                                          "high_quote_rate": 0.85},
}

# Weight of each group of the tuning objective (weighted profit). The
# empirical scenario and its stresses carry most of the weight (0.6).
TUNING_GROUP_WEIGHTS = {"empirical": 0.30, "empirical stress": 0.30,
                        "mixture round 2": 0.25, "mixture gap closed": 0.15}

# The contested markets: no uncontested pool left. The competitive rules (R4
# variants, R7) are tuned on these only, so that they answer "how to price
# when there is competition"; on the full objective every bin's margin runs
# to the top of the grid, i.e. a disguised high quote (R8).
CONTESTED = {"no-offers & high quotes -100%": 0.5, "mixture gap closed": 0.5}


def tuning_scenarios(pops, n_mixtures=(30, 20), empirical_seeds=(0, 1, 2), stress_seeds=(0, 1),
                     seed=SEED):
    """The scenarios every rule is tuned on, with their weights (summing to 1):

    * empirical (30%): Part 10's empirical Round 1 scenario on half A of the
      Round 1 applicants (offers resampled within risk bin x default);
    * empirical stress (30%): TUNING_STRESSES on half A, equal weights;
    * mixture round 2 (25%) and mixture gap closed (15%): n_mixtures
      four-team markets drawn from each prior on the 80,000 training loans
      (their competitors' p are our reconstructed lender models, so the
      winner's curse is kept).
    Returns a list of scenario dicts (name, group, population, market,
    weight, scale).
    """
    a = pops["round1_A"]
    out = []
    for s in empirical_seeds:
        m = crr.empirical_scenario(a, a, seed=s)
        out.append(_scenario("Round 1 empirical", "empirical", "round1_A",
                             make_market(m["r1"], m["r2"], a["default"]),
                             TUNING_GROUP_WEIGHTS["empirical"] / len(empirical_seeds)))
    for name, params in TUNING_STRESSES.items():
        for s in stress_seeds:
            m = crr.empirical_scenario(a, a, seed=s, **params)
            out.append(_scenario(name, "empirical stress", "round1_A",
                                 make_market(m["r1"], m["r2"], a["default"]),
                                 TUNING_GROUP_WEIGHTS["empirical stress"]
                                 / (len(TUNING_STRESSES) * len(stress_seeds))))
    train = pops["train"]
    r4 = round1_r4()
    for prior, n in zip(("round 2", "gap closed"), n_mixtures):
        for j in range(n):
            r1, r2, _ = draw_mixture(train, prior, seed + j, r4=r4)
            out.append(_scenario(f"mixture {prior}", "mixture", "train",
                                 make_market(r1, r2, train["default"]),
                                 TUNING_GROUP_WEIGHTS[f"mixture {prior}"] / n))
    return out


def reweight(scenarios, weights):
    """The scenarios named in `weights` (name -> total weight), each market
    getting an equal part of its name's weight; the others are dropped."""
    counts = pd.Series([s["name"] for s in scenarios]).value_counts()
    return [{**s, "weight": weights[s["name"]] / counts[s["name"]]}
            for s in scenarios if s["name"] in weights]


def stress_scenarios(pops, n_mixtures=30, seeds=(0, 1, 2), n_part10=10,
                     mixture_seed=SEED + 10_000):
    """The held-out scenarios of the stress tests (never used in tuning):

    * "Round 1 exact (B)": the real Round 1 offers and borrower types on half B;
    * empirical (B): TEST_STRESSES on half B (offers resampled from half B),
      one market per seed;
    * mixtures: n_mixtures four-team draws per prior on the validation loans,
      with seeds different from the tuning draws;
    * Part 10 teams: the eight inferred Round 1 teams (PART10_TEAMS) and
      their adapted variants on the validation loans, n_part10 random
      assignments of teams to applicants each;
    * Part 7: its 16 scenarios on validation (one rule per slot), with
      S_R4 = both competitors use our Round 1 R4, plus S_R4|R8 = both use
      Part 9's R4|R8 11.1% (competitors quoting 89.5% to risky applicants).
    Returns a list of scenario dicts; scenarios sharing a name are averaged
    (seeds) or summarised (mixtures) by score_rules.
    """
    b, val = pops["round1_B"], pops["validation"]
    out = [_scenario("Round 1 exact (B)", "round 1", "round1_B",
                     make_market(b["r1"], b["r2"], b["default"], pref=b["pref"]))]
    for name, params in TEST_STRESSES.items():
        for s in seeds:
            m = crr.empirical_scenario(b, b, seed=s, **params)
            out.append(_scenario(name, "empirical", "round1_B",
                                 make_market(m["r1"], m["r2"], b["default"])))
    r4 = round1_r4()
    for prior in PRIORS:
        for j in range(n_mixtures):
            r1, r2, _ = draw_mixture(val, prior, mixture_seed + j, r4=r4)
            out.append(_scenario(f"mix: {prior}", "mixture", "validation",
                                 make_market(r1, r2, val["default"])))
    for name, teams in part10_variants().items():
        for j in range(n_part10):
            r1, r2 = teams_market(val, teams, mixture_seed + j, r4=r4)
            out.append(_scenario(name, "part 10 teams", "validation",
                                 make_market(r1, r2, val["default"])))
    t6 =load_tuned()["equilibrium"]["T6_margins"]
    part7, _ = part7_scenarios(val, t6, strategic_rule=lambda d: r4(d), strategic_name="S_R4")
    switch = r4.edges[R1_SWITCH_DECILE]
    part7["S_R4|R8"] = make_scenario(
        "rule", val, rule=lambda d: rule_switch(d, r4(d), switch, R1_HIGH_QUOTE, "above"))
    for name, (r1, r2) in part7.items():
        out.append(_scenario(name, "part 7", "validation", make_market(r1, r2, val["default"])))
    return out


# Weights of the stress-test scenarios for the weighted-profit criterion
# (normalised by decision_criteria). The Round 1 market and the empirical
# scenario carry most of the weight, as the guidelines ask; Part 7's
# one-rule-per-slot scenarios, which missed the Round 1 market, the least.
def stress_weights(columns):
    w = {}
    for c in columns:
        if c in ("Round 1 exact (B)", "Round 1 empirical"):
            w[c] = 3.0
        elif c.startswith("mix: "):
            w[c] = {"mix: round 2": 3.0, "mix: gap closed": 2.0, "mix: round 1": 1.0}.get(c, 0.0)
        elif c in TEST_STRESSES:
            w[c] = 1.0
        elif c.startswith("Part 10"):
            w[c] = 2.0 if c == "Part 10 teams" else 1.0
        else:
            w[c] = 0.25
    return w


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def population_rates(rule, pops, populations):
    """A rule's rates on each population it is scored on: dict pop -> array."""
    return {name: np.asarray(rule(pops[name]), dtype=float) for name in populations}


def score_rules(rules, scenarios, pops):
    """Every rule in every scenario.

    rules : dict name -> function data -> rates
    Returns a long DataFrame, one row per (rule, scenario market): rule,
    scenario, group, draw (position among markets of the same name), profit
    (per 100k applicants), loans and share (of applicants), profit per loan.
    """
    populations = sorted({s["population"] for s in scenarios})
    rows = []
    for rule_name, rule in rules.items():
        rates = population_rates(rule, pops, populations)
        seen = {}
        for s in scenarios:
            profit, w = market_profit(s["market"], rates[s["population"]], return_weight=True)
            draw = seen[s["name"]] = seen.get(s["name"], -1) + 1
            rows.append({"rule": rule_name, "scenario": s["name"], "group": s["group"],
                         "draw": draw, "profit": s["scale"] * profit.sum(),
                         "loans": s["scale"] * w.sum(), "share": w.mean(),
                         "profit per loan": profit.sum() / w.sum() if w.sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def profit_table(scores, stat="mean", value="profit"):
    """Rules x scenarios, averaging (stat="mean") or taking a quantile (a
    float) over the markets of each scenario (seeds, mixture draws)."""
    g = scores.groupby(["rule", "scenario"], sort=False)[value]
    out = g.mean() if stat == "mean" else g.quantile(stat)
    order = list(dict.fromkeys(scores["scenario"]))
    rules = list(dict.fromkeys(scores["rule"]))
    return out.unstack().loc[rules, order]


def criteria(table, candidates, reference="R4 (Round 1)",
             gap_scenarios=("no-offers & high quotes -100%", "never-offer teams offer",
                            "mix: gap closed")):
    """The Part 7 criteria for the candidate rules, plus how each rule does
    against R4 when the uncontested pool disappears.

    Regret is measured against the best candidate of each scenario, so a
    rule left out of `candidates` (R8, a sanity check only) does not set it.
    Returns (criteria, regret).
    """
    t = table.loc[candidates]
    crit = decision_criteria(t, stress_weights(t.columns))
    for c in gap_scenarios:
        if c in t.columns:
            crit[f"vs R4: {c}"] = t[c] - t.loc[reference, c]
    return crit.sort_values("max regret"), regret_table(t)


# ---------------------------------------------------------------------------
# Tuning: per-bin objective
# ---------------------------------------------------------------------------

def _bins(pops, populations, edges):
    return {name: np.digitize(pops[name]["p_own"].to_numpy(dtype=float), edges)
            for name in populations}


def bin_objective(rates, scenarios, bins, n_bins):
    """The tuning objective split by risk bin: sum over scenarios of
    weight x our profit per 100k applicants, in each bin of p_own.

    Exact per bin: an applicant's choice depends only on the offers made to
    them, so a bin's profit does not depend on our rates in other bins.
    rates : dict population -> our rates; bins: dict population -> bin index
    """
    out = np.zeros(n_bins)
    for s in scenarios:
        profit = market_profit(s["market"], rates[s["population"]])
        out += s["weight"] * s["scale"] * np.bincount(bins[s["population"]], weights=profit,
                                                      minlength=n_bins)
    return out


def objective(rule, scenarios, pops):
    """The tuning objective of one rule (weighted profit per 100k)."""
    populations = sorted({s["population"] for s in scenarios})
    rates = population_rates(rule, pops, populations)
    return float(sum(s["weight"] * s["scale"] * market_profit(s["market"], rates[s["population"]]).sum()
                     for s in scenarios))


# ---------------------------------------------------------------------------
# R4 with a polynomial margin in each bin
# ---------------------------------------------------------------------------

# Margin values tried for the level of each bin's curve (decimals): fine
# steps where the contested market is, coarse ones up to 40 points (Part 9:
# 25-32 points paid even on contested risky applicants).
LEVEL_GRID = pts(-0.01, 0.06, 0.0025) + pts(0.065, 0.15, 0.005) + pts(0.16, 0.40, 0.01)
# Values tried for the slope / curvature coefficients (margin change at the
# bin's ends, decimals).
SHAPE_GRID = [-0.08, -0.04, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.04, 0.08]


def bin_bounds(p, edges, q=(0.005, 0.995)):
    """Per bin, the range of p_own mapped to [-1, 1] inside the bin (inner
    quantiles of the tuning population, so one outlier does not squash it)."""
    p = np.asarray(p, dtype=float)
    b = np.digitize(p, edges)
    return np.array([np.quantile(p[b == k], q) for k in range(len(edges) + 1)])


def bin_position(p, edges, bounds):
    """(bin index, position x in [-1, 1] inside the bin) of each p."""
    p = np.asarray(p, dtype=float)
    b = np.digitize(p, edges)
    lo, hi = bounds[b, 0], bounds[b, 1]
    return b, np.clip(2 * (p - lo) / (hi - lo) - 1, -1, 1)


def poly_margin(p, edges, bounds, coefs):
    """m(p) = sum_j coefs[bin, j] * x^j, x = position inside the bin."""
    b, x = bin_position(p, edges, bounds)
    coefs = np.asarray(coefs, dtype=float)
    return sum(coefs[b, j] * x ** j for j in range(coefs.shape[1]))


def rule_bin_poly(data, edges, bounds, coefs, floor=RATE_FLOOR):
    """R4 with a polynomial margin per bin: r = be(p) + m(p), m a polynomial
    of degree coefs.shape[1] - 1 inside each bin (0 = R4's step, 1 = linear,
    3 = cubic). Bins are independent: m need not increase with p, nor be
    continuous at the edges."""
    p = data["p_own"]
    return clip_rates(be(p) + poly_margin(p, edges, bounds, coefs), p, floor)


def make_poly_rule(fit):
    """A fit_bin_poly result as a function data -> rates."""
    def rule(data):
        return rule_bin_poly(data, fit["edges"], fit["bounds"], fit["coefs"])
    rule.fit = fit
    return rule


def fit_bin_poly(pops, scenarios, degree, edges=None, start=None, level_grid=None,
                 shape_grid=None, sweeps=3, progress=True):
    """Tune R4's polynomial margins, bin by bin, on the tuning objective.

    Coordinate descent: for each coefficient j in turn, every value of its
    grid is tried in all bins at once (the others fixed), and each bin keeps
    its best value; repeated `sweeps` times. One evaluation prices every bin,
    since bins are independent (bin_objective). Starting from the step
    solution (or `start`), the objective can only go up.

    Returns a dict: edges, bounds, coefs (bins x degree + 1), objective per
    bin, total objective and its history.
    """
    edges = r4_edges() if edges is None else np.asarray(edges, dtype=float)
    level_grid = LEVEL_GRID if level_grid is None else list(level_grid)
    shape_grid = SHAPE_GRID if shape_grid is None else list(shape_grid)
    n_bins = len(edges) + 1
    populations = sorted({s["population"] for s in scenarios})
    bins = _bins(pops, populations, edges)
    bounds = bin_bounds(pops["train"]["p_own"], edges)
    pos = {name: bin_position(pops[name]["p_own"], edges, bounds) for name in populations}
    be_p = {name: be(pops[name]["p_own"]) for name in populations}
    p_own = {name: pops[name]["p_own"].to_numpy(dtype=float) for name in populations}

    def evaluate(coefs):
        rates = {}
        for name in populations:
            b, x = pos[name]
            m = sum(coefs[b, j] * x ** j for j in range(coefs.shape[1]))
            rates[name] = clip_rates(be_p[name] + m, p_own[name], RATE_FLOOR).to_numpy()
        return bin_objective(rates, scenarios, bins, n_bins)

    coefs = np.zeros((n_bins, degree + 1))
    if start is not None:
        start = np.asarray(start, dtype=float)
        coefs[:, :start.shape[1]] = start[:, :degree + 1]
    best = evaluate(coefs)
    history, t0 = [best.sum()], time.time()
    order = list(range(degree + 1))
    for sweep in range(sweeps):
        for j in order:
            grid = level_grid if j == 0 else shape_grid
            for v in grid:
                trial = coefs.copy()
                trial[:, j] = v
                obj = evaluate(trial)
                better = obj > best + 1e-9
                coefs[better, j] = v
                best = np.where(better, obj, best)
        history.append(best.sum())
        if progress:
            print(f"  degree {degree}, sweep {sweep + 1}: objective {best.sum():,.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if degree == 0 or history[-1] - history[-2] < 1.0:
            break
    return {"degree": degree, "edges": edges, "bounds": bounds, "coefs": coefs,
            "bin_objective": best, "objective": best.sum(), "history": history}


def poly_table(fit, p_grid_points=5):
    """The tuned margin of each bin at a few positions inside it (points)."""
    xs = np.linspace(-1, 1, p_grid_points)
    rows = {}
    for b in range(len(fit["edges"]) + 1):
        lo, hi = fit["bounds"][b]
        ps = lo + (xs + 1) / 2 * (hi - lo)
        m = sum(fit["coefs"][b, j] * xs ** j for j in range(fit["coefs"].shape[1]))
        rows[f"bin {b + 1} ({100 * lo:.1f}-{100 * hi:.1f}%)"] = 100 * m
    return pd.DataFrame(rows, index=[f"x = {x:+.1f}" for x in xs]).T


# ---------------------------------------------------------------------------
# R8 (flat quote) and R4/R8 hybrids
# ---------------------------------------------------------------------------

QUOTE_GRID = pts(0.50, 1.0, 0.005)


def flat_quote_objective(pops, scenarios, edges=None, quotes=None):
    """The tuning objective of a flat quote q in each bin: DataFrame bins x
    quotes. Its column sums give R8; with a base rule's per-bin objective it
    gives every switch hybrid (switch_profits)."""
    edges = r4_edges() if edges is None else np.asarray(edges, dtype=float)
    quotes = QUOTE_GRID if quotes is None else list(quotes)
    populations = sorted({s["population"] for s in scenarios})
    bins = _bins(pops, populations, edges)
    out = {}
    for q in quotes:
        rates = {name: np.asarray(rule_high_quote(pops[name], q), dtype=float)
                 for name in populations}
        out[q] = bin_objective(rates, scenarios, bins, len(edges) + 1)
    return pd.DataFrame(out).rename_axis(index="bin", columns="quote")


def rule_objective_by_bin(rule, pops, scenarios, edges=None):
    """A rule's tuning objective in each bin (array)."""
    edges = r4_edges() if edges is None else np.asarray(edges, dtype=float)
    populations = sorted({s["population"] for s in scenarios})
    rates = population_rates(rule, pops, populations)
    return bin_objective(rates, scenarios, _bins(pops, populations, edges), len(edges) + 1)


def tune_hybrid(base_bins, flat_bins, edges=None, high_side="above"):
    """Best switch point and high quote for a hybrid of a base rule and a
    flat quote, on the tuning objective (round_1_replay.switch_profits).
    The ends (all base, all high quote) are left out: they are not hybrids.
    Returns (best row as a dict, the full table)."""
    edges = r4_edges() if edges is None else np.asarray(edges, dtype=float)
    table = switch_profits(pd.Series(base_bins), flat_bins, edges, high_side)
    inner = table.iloc[1:-1]
    best = inner["profit"].idxmax()
    return {"p_switch": float(best), "quote": float(inner.loc[best, "best high quote"]),
            "objective": float(inner.loc[best, "profit"])}, table


def make_hybrid_rule(base_rule, p_switch, quote, high_side):
    """A switch hybrid as a function data -> rates.

    No offer where the flat quote is below our own break-even (be(p) >= quote,
    i.e. p >= quote / (1 + quote), about 45% for an 82% quote): the expected
    profit (1 - p) * quote - p would be negative."""
    def rule(data):
        r = rule_switch(data, base_rule(data), p_switch, quote, high_side)
        p = data["p_own"]
        high = p >= p_switch if high_side == "above" else p < p_switch
        return r.where(~(high & (be(p) >= quote)).to_numpy())
    rule.params = {"p_switch": p_switch, "quote": quote, "high_side": high_side}
    return rule


def scenario_means(rates, scenarios):
    """Our profit per 100k in each tuning scenario name (mean over its
    markets), for rates = dict population -> our rates."""
    out = {}
    for s in scenarios:
        out.setdefault(s["name"], []).append(
            s["scale"] * market_profit(s["market"], rates[s["population"]]).sum())
    return pd.Series({k: np.mean(v) for k, v in out.items()})


def hybrid_grid(base_rule, pops, scenarios, switches=None, quotes=None,
                sides=("above", "below")):
    """Every switch hybrid of base_rule with a flat quote, in every tuning
    scenario: one row per (side, switch, quote), one column per scenario
    name, plus the base rule alone (side "none")."""
    switches = r4_edges() if switches is None else list(switches)
    quotes = pts(0.60, 0.95, 0.01) if quotes is None else list(quotes)
    populations = sorted({s["population"] for s in scenarios})
    base = population_rates(base_rule, pops, populations)
    p = {name: pops[name]["p_own"].to_numpy(dtype=float) for name in populations}
    rows = {("none", np.nan, np.nan): scenario_means(base, scenarios)}
    for side in sides:
        for sw in switches:
            for q in quotes:
                high = {n: (p[n] >= sw) if side == "above" else (p[n] < sw) for n in populations}
                # the flat quote, except where it is below our break-even (as make_hybrid_rule)
                rates = {n: np.where(high[n], np.where(be(p[n]) < q, q, np.nan), base[n])
                         for n in populations}
                rows[(side, sw, q)] = scenario_means(rates, scenarios)
    table = pd.DataFrame(rows).T
    table.index.names = ["side", "switch", "quote"]
    return table


def select_hybrid(grid, weights, side=None):
    """The Part 7 choice on a hybrid_grid: smallest max regret across the
    tuning scenarios (regret against the best grid row of each scenario),
    ties broken by weighted profit. `side` restricts to one direction.
    Returns (chosen row as a dict, the grid with max regret and weighted)."""
    w = pd.Series(weights, dtype=float).reindex(grid.columns).fillna(0.0)
    out = grid.copy()
    out["max regret"] = (grid.max() - grid).max(axis=1)
    out["weighted"] = grid.mul(w / w.sum(), axis=1).sum(axis=1)
    cand = out if side is None else out.loc[[side]]
    best = cand.sort_values(["max regret", "weighted"], ascending=[True, False]).index[0]
    return {"high_side": best[0], "p_switch": float(best[1]), "quote": float(best[2]),
            "max regret": float(out.loc[best, "max regret"]),
            "weighted": float(out.loc[best, "weighted"])}, out


# ---------------------------------------------------------------------------
# R7 re-tuned
# ---------------------------------------------------------------------------

def tune_r7(pops, scenarios, train, deltas=None, adjustments=(0.0, 0.2, 0.4, 0.6),
            qs=(0.25, 0.5), n_draws=50, seed=SEED):
    """R7 (undercut the predicted cheaper competitor) on the new objective:
    a grid over delta, a and q. The forecast of the competitors' break-even
    rate is computed once per population (model fitted on train, Part 6).
    Returns (best parameters, table of every combination)."""
    deltas = pts(-0.10, 0.02, 0.005) if deltas is None else list(deltas)
    model = fit_competitor_model(train)
    populations = sorted({s["population"] for s in scenarios})
    forecasts = {name: undercut_forecasts(model, pops[name], qs=qs, n_draws=n_draws, seed=seed)
                 for name in populations}
    rows = []
    for q in qs:
        for a in adjustments:
            for delta in deltas:
                rates = {name: rule_undercut(pops[name], forecasts[name][q], delta, a,
                                             RATE_FLOOR).to_numpy() for name in populations}
                total = sum(s["weight"] * s["scale"]
                            * market_profit(s["market"], rates[s["population"]]).sum()
                            for s in scenarios)
                rows.append({"q": q, "a": a, "delta": delta, "objective": total})
    table = pd.DataFrame(rows)
    best = table.loc[table["objective"].idxmax()]
    return {"q": best["q"], "a": best["a"], "delta": best["delta"]}, table


def make_r7_rule(train, q, a, delta, n_draws=50, seed=SEED):
    """R7 with given parameters, as a function data -> rates."""
    model = fit_competitor_model(train)

    def rule(data):
        forecast = undercut_forecasts(model, data, qs=(q,), n_draws=n_draws, seed=seed)[q]
        return rule_undercut(data, forecast, delta, a, RATE_FLOOR)
    rule.params = {"q": q, "a": a, "delta": delta}
    return rule


# ---------------------------------------------------------------------------
# Share / margin trade-off
# ---------------------------------------------------------------------------

def shift_safe_margins(rule, shift, p_max=0.11):
    """A rule with `shift` added to its rates where p_own < p_max (the safe,
    contested applicants), to trace market share against margin."""
    def shifted(data):
        r = rule(data)
        return r.where(data["p_own"] >= p_max, clip_rates(r + shift, data["p_own"], RATE_FLOOR))
    return shifted


def margin_tradeoff(rule, scenarios, pops, shifts, p_max=0.11):
    """Profit and share of the safe applicants (p_own < p_max) for each
    margin shift, averaged within each scenario name.
    Returns a DataFrame indexed by (scenario, shift)."""
    rows = []
    populations = sorted({s["population"] for s in scenarios})
    for shift in shifts:
        rates = population_rates(shift_safe_margins(rule, shift, p_max), pops, populations)
        for s in scenarios:
            safe = pops[s["population"]]["p_own"].to_numpy() < p_max
            profit, w = market_profit(s["market"], rates[s["population"]], return_weight=True)
            rows.append({"scenario": s["name"], "shift": shift,
                         "profit": s["scale"] * profit.sum(),
                         "safe profit": s["scale"] * profit[safe].sum(),
                         "safe share": w[safe].mean()})
    return pd.DataFrame(rows).groupby(["scenario", "shift"], sort=False).mean()


# ---------------------------------------------------------------------------
# Round 1 replay of the final candidates
# ---------------------------------------------------------------------------

def round1_exact(rules, pops):
    """Every rule on the real Round 1 market, with the real borrower types:
    profit on all 100,000 applicants and on each half (A was used in tuning,
    B was not). Returns a DataFrame, one row per rule."""
    r1 = pops["round1"]
    market = make_market(r1["r1"], r1["r2"], r1["default"], pref=r1["pref"])
    in_a = r1.index.isin(pops["round1_A"].index)
    rows = {}
    for name, rule in rules.items():
        profit, w = market_profit(market, np.asarray(rule(r1), dtype=float), return_weight=True)
        rows[name] = {"profit": profit.sum(), "loans": w.sum(), "share": w.mean(),
                      "profit per loan": profit.sum() / w.sum(),
                      "half A (tuning) x2": 2 * profit[in_a].sum(),
                      "half B (held out) x2": 2 * profit[~in_a].sum()}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Retraining the default model on 200k loans
# ---------------------------------------------------------------------------

def round1_loans():
    """The Round 1 applicants as loans: Part 1's logreg preparation of their
    features, with the default observed in the diagnostic file (all 100,000
    applicants, whether or not they borrowed from us)."""
    apps = logreg_prep_data(pd.read_csv(NEW_APPS_R1)).set_index("id")
    diag = pd.read_csv(ROUND1_DIAGNOSTIC, index_col="id")
    return apps.join(diag["default"])


def own_training_sets():
    """X, y of our own model (public + signal3, spline logistic) for the
    100,000 past loans and the 100,000 Round 1 loans, and the source of
    each row ("past" / "round 1")."""
    data = load_data()["logreg"]
    cols = feature_columns("spline", "own")
    past = pd.concat([data["train"], data["validation"]])
    r1 = round1_loans()
    X = pd.concat([past[cols], r1[cols]], ignore_index=True)
    y = pd.concat([past["default"], r1["default"]], ignore_index=True)
    source = np.array(["past"] * len(past) + ["round 1"] * len(r1))
    return X, y, source, r1.index


def crossfit_round1(n_folds=5, seed=SEED):
    """Does adding the Round 1 loans improve our predictions of Round 1?

    Each fold of the Round 1 loans is predicted by a model fitted on the
    100,000 past loans plus the other folds of Round 1 (so up to 180k loans),
    to compare with p_own (fitted on the past loans only), on the same
    applicants. Returns a DataFrame indexed by Round 1 id: default, p_own
    (100k model) and p_200k (cross-fitted)."""
    X, y, source, ids = own_training_sets()
    is_r1 = source == "round 1"
    r1_rows = np.flatnonzero(is_r1)
    rng = np.random.default_rng(seed)
    folds = np.resize(np.arange(n_folds), len(r1_rows))
    rng.shuffle(folds)
    p = np.full(len(r1_rows), np.nan)
    for k in range(n_folds):
        test = r1_rows[folds == k]
        fit_rows = np.concatenate([np.flatnonzero(~is_r1), r1_rows[folds != k]])
        model = fit_model(MODEL_FACTORIES["spline"](), X.iloc[fit_rows], y.iloc[fit_rows])
        p[folds == k] = predict_default(model, X.iloc[test])
    own = pd.read_csv(DATA_PREDICTIONS / "applications_own.csv", index_col="id")["p_own"]
    return pd.DataFrame({"default": y[is_r1].to_numpy(), "p_own": own.loc[ids].to_numpy(),
                         "p_200k": p}, index=ids)


def calibration_compare(preds, bands=BANDS, labels=BAND_LABELS):
    """Calibration by band of p_own for the 100k and 200k models: mean
    prediction, default rate and gap (observed - predicted), plus log-loss
    and AUC overall."""
    from sklearn.metrics import log_loss, roc_auc_score
    band = pd.cut(preds["p_own"], bands, labels=labels, right=False)
    g = preds.groupby(band, observed=False)
    out = pd.DataFrame({"applicants": g.size(), "default rate": g["default"].mean(),
                        "mean p (100k)": g["p_own"].mean(), "mean p (200k)": g["p_200k"].mean()})
    out["gap (100k)"] = out["default rate"] - out["mean p (100k)"]
    out["gap (200k)"] = out["default rate"] - out["mean p (200k)"]
    overall = {f"{m} ({k})": f(preds["default"], preds[c])
               for m, f in (("log-loss", log_loss), ("AUC", roc_auc_score))
               for k, c in (("100k", "p_own"), ("200k", "p_200k"))}
    return out, pd.Series(overall)


def predict_round2(save=True):
    """p_own for the Round 2 applicants from the model fitted on all 200k
    loans (p_own_200k) and on the 100k past loans only (p_own_100k, as in
    Round 1). Saved to data/predictions/applications_round2_own.csv."""
    X, y, source, _ = own_training_sets()
    cols = feature_columns("spline", "own")
    apps = logreg_prep_data(pd.read_csv(NEW_APPS_R2)).set_index("id")
    past = source == "past"
    out = pd.DataFrame(index=apps.index)
    out["p_own_100k"] = predict_default(fit_model(MODEL_FACTORIES["spline"](), X[past], y[past]),
                                        apps[cols])
    out["p_own_200k"] = predict_default(fit_model(MODEL_FACTORIES["spline"](), X, y), apps[cols])
    if save:
        out.to_csv(DATA_PREDICTIONS / "applications_round2_own.csv")
    return out


# ---------------------------------------------------------------------------
# Saving the chosen rule (Part 12)
# ---------------------------------------------------------------------------

def _plain(v):
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_plain(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


def save_round2_rule(name, poly_fit, hybrid=None, notes="", filename="final_rule_round2.json"):
    """Save the Round 2 rule: a polynomial R4 (poly_fit), optionally switched
    to a high quote (hybrid = {"p_switch", "quote", "high_side"})."""
    path = PRICING_DIR / filename
    path.write_text(json.dumps({
        "rule": name,
        "poly": {"degree": poly_fit["degree"], "edges": _plain(poly_fit["edges"]),
                 "bounds": _plain(poly_fit["bounds"]), "coefs": _plain(poly_fit["coefs"]),
                 "floor": RATE_FLOOR},
        "hybrid": {k: _plain(v) for k, v in (hybrid or {}).items()} or None,
        "notes": notes}, indent=2))
    return path


def load_round2_rule(filename="final_rule_round2.json"):
    """The saved Round 2 rule as a function data -> rates (needs p_own)."""
    spec = json.loads((PRICING_DIR / filename).read_text())
    poly = spec["poly"]
    fit = {"degree": poly["degree"], "edges": np.asarray(poly["edges"]),
           "bounds": np.asarray(poly["bounds"]), "coefs": np.asarray(poly["coefs"])}
    rule = make_poly_rule(fit)
    if spec["hybrid"]:
        rule = make_hybrid_rule(rule, **spec["hybrid"])
    rule.name = spec["rule"]
    return rule


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_margin_curves(curves, ax, p_max=0.45, title="Margin over break-even by risk"):
    """Margin (points) against p_own for several rules: curves = dict
    label -> function data -> rates (margin = rate - be(p))."""
    p = np.linspace(0.001, p_max, 900)
    frame = pd.DataFrame({"p_own": p})
    for color, (label, rule) in zip([REFERENCE] + SERIES_COLORS, curves.items()):
        r = np.asarray(rule(frame), dtype=float)
        ax.plot(100 * p, 100 * (r - be(p)), color=color, lw=1.6 if color != REFERENCE else 2.4,
                label=label)
    ax.axvline(11.1, color=REFERENCE, lw=0.8, ls=":")
    ax.set_xlabel("our predicted default probability p_own (%)")
    ax.set_ylabel("margin over break-even (points)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY, fontsize=8)


def plot_hybrid_tuning(flat_bins, base_bins, edges, ax, title="Hybrids: objective by switch point"):
    """Objective (million EUR per 100k) of R4|R8 and R8|R4 for each switch
    point, at the best quote of each."""
    for color, side, label in ((SERIES_COLORS[0], "above", "R4|R8 (high quote above)"),
                               (SERIES_COLORS[1], "below", "R8|R4 (high quote below)")):
        t = switch_profits(pd.Series(base_bins), flat_bins, edges, side)
        ax.plot(100 * t.index, t["profit"] / 1e6, "o-", color=color, label=label)
    ax.axhline(base_bins.sum() / 1e6, color=REFERENCE, lw=1, ls="--", label="base rule alone")
    ax.set_xlabel("switch point p_own (%)")
    ax.set_ylabel("tuning objective (million EUR / 100k)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)


def plot_regret_heatmap(regret, ax, title="Regret (million EUR per 100k applicants)"):
    """Rules x scenarios heatmap of regret."""
    v = regret.to_numpy(dtype=float) / 1e6
    im = ax.imshow(v, cmap="Purples", aspect="auto")
    ax.set_xticks(range(regret.shape[1]), regret.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(regret.shape[0]), regret.index, fontsize=8)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color=SURFACE if v[i, j] > 0.6 * np.nanmax(v) else INK)
    ax.grid(False)
    ax.set_title(title)
    return im


def plot_mixture_spread(scores, rules, ax, title="Profit across four-team mixture draws"):
    """Box plot of each rule's profit over the mixture draws, one panel group
    per prior (colour)."""
    mix = scores[scores["group"] == "mixture"]
    priors = list(dict.fromkeys(mix["scenario"]))
    width = 0.8 / len(priors)
    for j, (color, prior) in enumerate(zip(SERIES_COLORS, priors)):
        data = [mix[(mix["rule"] == r) & (mix["scenario"] == prior)]["profit"] / 1e6 for r in rules]
        pos = np.arange(len(rules)) + (j - (len(priors) - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops={"color": INK})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.plot([], [], "s", color=color, label=prior)
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xticks(range(len(rules)), rules, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("profit (million EUR per 100k)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)


def plot_tradeoff(tradeoff, scenarios, ax, title="Safe applicants: market share against margin"):
    """Profit on safe applicants against their market share, one line per
    scenario, one point per margin shift (labelled in points)."""
    for color, s in zip(SERIES_COLORS, scenarios):
        t = tradeoff.loc[s]
        ax.plot(100 * t["safe share"], t["safe profit"] / 1e6, "o-", color=color, label=s)
        for shift, row in t.iterrows():
            ax.annotate(f"{100 * shift:+g}", (100 * row["safe share"], row["safe profit"] / 1e6),
                        fontsize=7, color=INK_SECONDARY, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("share of safe applicants (p_own < 11%) who take our loan (%)")
    ax.set_ylabel("profit on safe applicants (million EUR / 100k)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY, fontsize=8)


def plot_calibration(table, ax, title="Calibration on Round 1: 100k vs 200k model"):
    """Gap (observed - predicted default) by band for both models."""
    x = np.arange(len(table))
    ax.bar(x - 0.2, 100 * table["gap (100k)"], 0.4, color=LENDER_COLORS[3], label="past loans only (100k)")
    ax.bar(x + 0.2, 100 * table["gap (200k)"], 0.4, color=SERIES_COLORS[2],
           label="past + Round 1 (200k, cross-fitted)")
    ax.axhline(0, color=REFERENCE, lw=1)
    ax.set_xticks(x, table.index)
    ax.set_xlabel("our predicted default probability p_own")
    ax.set_ylabel("observed - predicted default (points)")
    ax.set_title(title)
    ax.legend(labelcolor=INK_SECONDARY)
