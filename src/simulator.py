# market simulator + winner's curse
import numpy as np
import pandas as pd

# Notebooks put the project root on sys.path before importing this module, so `src` resolves.
from src.config import LOAN_AMOUNT, DISCOUNT


RATE_COLS = ["r1", "r2", "r3"]   # offers: decimals (0.12 = 12%), NaN = no offer

# Highest rate accepted, as a guard against rates passed in percent (12 instead of 0.12).
# Not 1.0: break-even p/(1-p) tends to 1 as p tends to 0.5 (competitors make no offer at
# p >= 0.5), so break-even + margin can legitimately exceed 100%: p = 0.48 gives a break-even of
# 0.923, plus an 8 pp margin = 1.003.
# Any margin below 100 pp stays under 2.0.
MAX_RATE = 2.0


# Preferences
def assign_preferences(applicants, seed=42, col="pref"):
    """Copy of `applicants` with a column in {1,2,3}: exact 1/3 split, seeded shuffle."""
    n = len(applicants)
    rng = np.random.default_rng(seed)
    prefs = np.resize(np.array([1, 2, 3]), n)   # 1,2,3,1,2,3,...
    rng.shuffle(prefs)
    out = applicants.copy()
    out[col] = prefs
    return out


# Choice rule
def rates_matrix(offers):
    """DataFrame with r1,r2,r3 (or (n,3) array) -> validated float array.

    Rates are decimals; NaN = no offer. Raises ValueError if a rate is negative or above
    MAX_RATE (most likely a rate given in percent).
    """
    if isinstance(offers, pd.DataFrame):
        r = offers[RATE_COLS].to_numpy(dtype=float)
    else:
        r = np.asarray(offers, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("offers must have shape (n, 3)")
    finite = r[np.isfinite(r)]
    if finite.size and finite.max() > MAX_RATE:
        raise ValueError(f"Rate above {MAX_RATE}: use decimals, e.g. 0.12 for 12%.")
    if finite.size and finite.min() < 0:
        raise ValueError("Negative rate found.")
    return r


def choice_weights(rates, pref, discount=DISCOUNT, tol=1e-12):
    """(n,3) probabilities that each applicant picks each lender, given their type.

    Effective rate = offered rate, minus `discount` for the preferred lender.
    Lowest effective rate wins; exact ties are split equally; NaN/inf never win;
    a row of all zeros means no loan.
    """
    pref = np.asarray(pref, dtype=int)
    if not np.isin(pref, [1, 2, 3]).all():
        raise ValueError("pref must contain only 1, 2 or 3")
    eff = np.where(np.isfinite(rates), rates, np.inf).astype(float)
    eff[np.arange(len(eff)), pref - 1] -= discount      # inf - 0.02 stays inf
    best = eff.min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        is_min = (np.abs(eff - best) <= tol) & np.isfinite(eff)
    counts = is_min.sum(axis=1, keepdims=True)
    return np.where(counts > 0, is_min / np.maximum(counts, 1), 0.0)


def choose_lender(rates, pref, discount=DISCOUNT, rng=None):
    """Chosen lender (1,2,3) per applicant, 0 = no loan. Ties broken at random."""
    rng = np.random.default_rng() if rng is None else rng
    w = choice_weights(rates, pref, discount)
    noise = rng.random(w.shape) * (w > 0)
    choice = noise.argmax(axis=1) + 1
    choice[w.sum(axis=1) == 0] = 0
    return choice


#  Full simulation
def simulate_market(offers, defaults, pref=None, mode="expected",
                    loan_amount=LOAN_AMOUNT, discount=DISCOUNT, seed=42):
    """Returns (loans, summary).

    offers   : DataFrame with r1, r2, r3 (indexed by applicant id) or (n,3) array
    defaults : 0/1 per applicant, same order
    mode     : "expected" -> average over the 3 types (weights 1/3, ties split)
               "random"   -> one seeded type per applicant, random tie-break
    pref     : only used in "random" mode; drawn from `seed` if None
    loan_amount : one amount for every loan, or one per applicant (n,), same order

    loans (one row per applicant): r1, r2, r3 (offered rates), default, w1..w3 (prob. of taking lender k's loan;
        feed these to Part 5), profit1..3, exp_rate_taken, loan_made_prob,
        and in random mode: pref, chosen_lender.
    summary (one row per lender): n_loans, share_of_applicants, share_of_loans,
        avg_rate, default_rate_won, total_profit, profit_per_loan.
    """
    rates = rates_matrix(offers)
    n = len(rates)
    d = np.asarray(defaults, dtype=float)
    if len(d) != n:
        raise ValueError("offers and defaults must have the same length")
    if not np.isin(d, [0, 1]).all():
        raise ValueError("defaults must be 0/1")
    amount = np.asarray(loan_amount, dtype=float)
    if amount.ndim > 1 or (amount.ndim == 1 and len(amount) != n):
        raise ValueError("loan_amount must be a scalar or have one value per applicant")
    amount = np.broadcast_to(amount, (n,))   # a scalar becomes the same amount for everyone
    index = offers.index if isinstance(offers, pd.DataFrame) else np.arange(n)

    chosen = None
    if mode == "expected":
        w = sum(choice_weights(rates, np.full(n, t), discount)
                for t in (1, 2, 3)) / 3.0
    elif mode == "random":
        if pref is None:
            pref = assign_preferences(pd.DataFrame(index=index), seed)["pref"]
        pref = np.asarray(pref, dtype=int)
        rng = np.random.default_rng(seed + 1)
        chosen = choose_lender(rates, pref, discount, rng)
        w = np.zeros((n, 3))
        won = chosen > 0
        w[np.where(won)[0], chosen[won] - 1] = 1.0
    else:
        raise ValueError("mode must be 'expected' or 'random'")

    # Profit uses the OFFERED rate, never the discounted one.
    r0 = np.nan_to_num(rates, nan=0.0, posinf=0.0)       # w = 0 where no offer
    # amount[:, None] lines each applicant's amount up with their 3 offers.
    profit_if_taken = np.where(d[:, None] == 1, -amount[:, None], r0 * amount[:, None])
    profit = w * profit_if_taken

    loans = pd.DataFrame(rates, index=index, columns=RATE_COLS)
    loans["default"] = d.astype(int)
    for k in (1, 2, 3):
        loans[f"w{k}"] = w[:, k - 1]
    for k in (1, 2, 3):
        loans[f"profit{k}"] = profit[:, k - 1]
    loans["exp_rate_taken"] = (w * r0).sum(axis=1)
    loans["loan_made_prob"] = w.sum(axis=1)
    if chosen is not None:
        loans["pref"] = pref
        loans["chosen_lender"] = chosen

    n_loans = w.sum(axis=0)
    total_profit = profit.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        summary = pd.DataFrame({
            "n_loans": n_loans,
            "share_of_applicants": n_loans / n,
            "share_of_loans": n_loans / n_loans.sum(),
            "avg_rate": (w * r0).sum(axis=0) / n_loans,
            "default_rate_won": (w * d[:, None]).sum(axis=0) / n_loans,
            "total_profit": total_profit,
            "profit_per_loan": total_profit / n_loans,
        }, index=pd.Index([1, 2, 3], name="lender"))
    return loans, summary


# Winner's curse
def winners_curse(loans, p_pred, defaults=None, lender=3, groups=5):
    """Gap between observed and predicted default risk on the loans WE win.

    loans    : loan-level DataFrame returned by simulate_market (needs w{lender}, default)
    p_pred   : our predicted default probabilities, same row order as `loans`
               (must be out-of-sample predictions)
    defaults : optional observed defaults; taken from loans["default"] if None
    lender   : which lender we are (3)
    groups   : int -> that many quantile groups of p_pred (5 = quintiles),
               or a list of bin edges, e.g. [0, .02, .05, .1, .2, 1]
    Groups are defined on ALL applicants (not only won loans), so they stay
    identical across pricing rules. G1 = lowest predicted risk.

    Returns a DataFrame (one row per group + 'overall'):
      group_size        applicants in the group
      n_loans_won       sum of our weights (fractional in expected mode)
      win_rate          n_loans_won / group_size
      avg_pred_p        weighted mean predicted default probability on won loans
      obs_default_rate  weighted observed default rate on won loans
      gap               obs_default_rate - avg_pred_p   (>0: we UNDER-predict risk)
      se_obs            approx. standard error of obs_default_rate (to judge noise)
      p_low, p_high     range of p_pred in the group (all applicants)
    """
    n = len(loans)
    p = np.asarray(p_pred, dtype=float)
    if len(p) != n:
        raise ValueError("p_pred must have one value per row of `loans`, same order")
    d = np.asarray(loans["default"] if defaults is None else defaults, dtype=float)
    if len(d) != n:
        raise ValueError("defaults must have one value per row of `loans`")
    w = loans[f"w{lender}"].to_numpy(dtype=float)

    # risk groups on all applicants
    if isinstance(groups, int):
        g = pd.qcut(p, groups, labels=False, duplicates="drop")
    else:
        g = pd.cut(p, bins=groups, labels=False, include_lowest=True)
    g = np.asarray(pd.Series(g, dtype=float))

    def stats(mask):
        ww = w[mask]
        size = int(mask.sum())
        n_won = ww.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pred = (ww * p[mask]).sum() / n_won
            obs = (ww * d[mask]).sum() / n_won
            n_eff = n_won ** 2 / (ww ** 2).sum()          # effective sample size
            se = np.sqrt(obs * (1 - obs) / n_eff)
        return {
            "group_size": size,
            "n_loans_won": n_won,
            "win_rate": n_won / size if size else np.nan,
            "avg_pred_p": pred,
            "obs_default_rate": obs,
            "gap": obs - pred,
            "se_obs": se,
            "p_low": p[mask].min() if size else np.nan,
            "p_high": p[mask].max() if size else np.nan,
        }

    rows = {}
    for k in np.unique(g[~np.isnan(g)]):
        rows[f"G{int(k) + 1}"] = stats(g == k)
    rows["overall"] = stats(np.ones(n, dtype=bool))
    return pd.DataFrame.from_dict(rows, orient="index")