# break-even rates helper functions, competitor_offers(p_comp, margin) function
"""Part 3 - break-even (fair) pricing.

Expected profit per euro lent, with default probability p and rate r:
    (1 - p) * r - p
Setting it to zero gives the break-even rate:
    r = p / (1 - p)

Break-even is a floor, the minimum rate we could offer. 
Applicants with p >= 0.5 get
no offer (break-even would be >= 100%, and the game forbids rates above 100%).
"No offer" is NaN everywhere, which is what the submission file (missing rate)
and src/simulator.py (NaN = no offer) both expect.

All functions accept a float, a numpy array (what models.predict_default
returns) or a pandas Series.

Typical use from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.models import fit_model, predict_default
    from src.fair_pricing import break_even_rate, add_break_even_rates, \
        competitor_offers, market_offers

    p_own = predict_default(model_own, X_val[own_cols])
    r_floor = break_even_rate(p_own)

    # or on a DataFrame of predictions, e.g. the output of oof_predict_all
    # (columns "own__gb", "full__logit", ...)
    preds = add_break_even_rates(preds)

    # competitor scenario for Parts 4, 6 and 7
    offers = market_offers(p_lender1, p_lender2, r_ours, margin1=0.02)
    loans, summary = simulate_market(offers, y_val)
"""

import numpy as np
import pandas as pd

NO_OFFER_THRESHOLD = 0.5  # p >= 0.5 -> break-even rate >= 100%, no offer
MAX_RATE = 1.0  # rates above 100% are not allowed in the game

# Prefixes of the columns the add_... functions create. Their values can also
# lie in [0, 1], so they are never mistaken for probabilities.
GENERATED_PREFIXES = ("offer_", "be_", "comp_")


def _check_probabilities(p_arr):
    """Raise if a (non-missing) probability lies outside [0, 1]: such a value
    is a bug upstream and would otherwise produce a meaningless rate."""
    finite = p_arr[np.isfinite(p_arr)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        raise ValueError("default probabilities must lie in [0, 1]")


def offer(p):
    """
    p : float, numpy array or pandas Series of default probabilities.

    Return if the bank makes an offer to the applicant:
    1 = offer (p < 0.5), 0 = no offer (p >= 0.5, or p missing).
    Returns the same type as p (a Series keeps p's index).
    """
    p_arr = np.asarray(p, dtype=float)
    _check_probabilities(p_arr)
    flag = (p_arr < NO_OFFER_THRESHOLD).astype(int)  # NaN < 0.5 is False -> 0

    if isinstance(p, pd.Series):
        return pd.Series(flag, index=p.index, name="offer")
    if flag.ndim == 0:
        return int(flag)
    return flag


def break_even_rate(p):
    """Break-even interest rate r = p / (1 - p), NaN (no offer) where p >= 0.5.

    p : float, numpy array or pandas Series of default probabilities.
    Returns the same type as p (a Series keeps p's index).
    """
    p_arr = np.asarray(p, dtype=float)
    _check_probabilities(p_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p_arr < NO_OFFER_THRESHOLD, p_arr / (1 - p_arr), np.nan)

    if isinstance(p, pd.Series):
        return pd.Series(r, index=p.index, name="break_even_rate")
    if r.ndim == 0:
        return float(r)
    return r

def _prob_columns(dataset, prob_cols):
    """Columns to convert: `prob_cols` if given (a name or a list), otherwise
    every float column whose values all lie in [0, 1], except the columns
    created by the add_... functions (offer_, be_, comp_ prefixes)."""
    if prob_cols is None:
        return [c for c in dataset.columns
                if not str(c).startswith(GENERATED_PREFIXES)
                and pd.api.types.is_float_dtype(dataset[c])
                and dataset[c].between(0, 1).all()]
    if isinstance(prob_cols, str):
        return [prob_cols]
    return list(prob_cols)


def add_offer_flags(dataset, prob_cols=None, prefix="offer_"):
    """Copy of `dataset` with one offer flag column (1 = offer, 0 = no offer)
    per probability column. The flag for "own__gb" is "offer_own__gb".

    prob_cols : column name or list of names. Default: every float column
                whose values all lie in [0, 1] (see _prob_columns).
    """
    out = dataset.copy()
    prob_cols = _prob_columns(out, prob_cols)

    for col in prob_cols:
        out[prefix + col] = offer(out[col])
    return out


def add_break_even_rates(dataset, prob_cols=None, prefix="be_"):
    """Copy of `dataset` with one break-even rate column per probability column.

    dataset   : DataFrame of predicted default probabilities, e.g. the output
                of models.oof_predict_all (columns "<info set>__<model>"), or
                a predictions file with id / row_id + probability columns.
    prob_cols : column name or list of names to convert. Default: every float
                column whose values all lie in [0, 1] (see _prob_columns).
    prefix    : the rate column for "own__gb" is "be_own__gb".
    """
    out = dataset.copy()
    prob_cols = _prob_columns(out, prob_cols)

    for col in prob_cols:
        out[prefix + col] = break_even_rate(out[col])
    return out


def competitor_offers(p_comp, margin):
    """Rate a competitor offers: break-even rate plus an assumed margin.

    p_comp : a competitor's predicted default probabilities (float, array or
             Series), e.g. our Lender 1 or Lender 2 model's predictions.
    margin : assumed margin over break-even, a scenario parameter for
             Parts 6 and 7 (0.02 = +2 percentage points).

    No offer (NaN) when p_comp >= 0.5, or when break-even + margin would
    exceed 100% (not allowed in the game). A negative margin (a competitor
    pricing below break-even) is allowed as a scenario, but the rate never
    goes below 0%.
    """
    r = np.asarray(break_even_rate(p_comp), dtype=float) + margin  # NaN stays NaN
    r = np.where(np.isnan(r), np.nan, np.maximum(r, 0.0))  # no negative rate
    r = np.where(r <= MAX_RATE, r, np.nan)  # NaN <= 1 is False -> stays NaN

    if isinstance(p_comp, pd.Series):
        return pd.Series(r, index=p_comp.index, name="competitor_rate")
    if r.ndim == 0:
        return float(r)
    return r


def market_offers(p_lender1, p_lender2, r_ours, margin1, margin2=None,
                  index=None):
    """Offers of all three lenders, in the format src/simulator.py expects.

    p_lender1, p_lender2 : default probabilities from our Lender 1 / Lender 2
                           models (same row order as r_ours)
    r_ours               : our offered rates (lender 3), NaN = no offer
    margin1, margin2     : competitors' assumed margins (margin2 defaults to
                           margin1)
    index                : optional index (e.g. row_id); taken from the
                           inputs if they are Series (which must then share
                           the same index)

    Returns a DataFrame with columns r1, r2, r3 (decimals, NaN = no offer).
    """
    if margin2 is None:
        margin2 = margin1
    # Rows are matched by position, so Series inputs must share one index.
    series = [x for x in (p_lender1, p_lender2, r_ours) if isinstance(x, pd.Series)]
    if any(not x.index.equals(series[0].index) for x in series[1:]):
        raise ValueError("p_lender1, p_lender2 and r_ours must have the same "
                         "index, in the same order")
    if index is None and series:
        index = series[0].index

    return pd.DataFrame({
        "r1": np.asarray(competitor_offers(p_lender1, margin1), dtype=float),
        "r2": np.asarray(competitor_offers(p_lender2, margin2), dtype=float),
        "r3": np.asarray(r_ours, dtype=float),
    }, index=index)


def add_competitor_offers(dataset, prob_cols, margin, prefix="comp_"):
    """Copy of `dataset` with one competitor offer column per probability column.

    dataset   : DataFrame of predicted default probabilities
    prob_cols : column name or list of names holding the competitors'
                probabilities, e.g. ["lender1__logit", "lender2__logit"].
                Required (no default), so our own model's column is never
                priced as a competitor by accident.
    margin    : assumed margin over break-even (0.02 = +2 percentage points)
    prefix    : the offer column for "lender1__logit" is "comp_lender1__logit".

    Offer = break-even + margin, NaN (no offer) when p >= 0.5 or when the
    rate would exceed 100% (see competitor_offers).
    """
    out = dataset.copy()
    if isinstance(prob_cols, str):
        prob_cols = [prob_cols]

    for col in prob_cols:
        out[prefix + col] = competitor_offers(out[col], margin)
    return out
