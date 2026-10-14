# break-even rates helper functions, competitor_offers(p_comp, margin) function

# fair-pricing formula
# profit per euro lent:
#  (1−p)*r−p

# break-even rate (setting profit to 0) --> r = p/(1-p)

# use the break-even rate for all probabilities computed in models.py/competitor_scenarios.ipynb

def break_even_rate(dataset):
    BE_rates = dataset.copy()
    rates = []
    p = dataset['probability column']

    # for instance in the probability column, find the break-even rate
    # if the probability in the probability column is greater than or equal to 0.5
    # leave it blank, we will not make an offer to this person

    for i in BE_rates['probability column']: #each instance in the probability column
        if p >= 0.5:
            return
        
        else:
            r = p/(1-p) # get the rate

        rates.append(r)  # append the rate 

    # add the rates to the dataset here
    # TODO

    return dataset.csv


def competitor_offers(p_comp, margin):
    """
    p_comp : probability of default from competitors' models
    margin : an assumed margin

    Notes:
    - no offer when p>= 0.5

    """

    r = (p_comp/(1-p_comp)) + margin


# ---------------------------------------------------------------------------
# Vectorised versions (work on a single float, a numpy array or a pandas Series)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

NO_OFFER_THRESHOLD = 0.5  # p >= 0.5 -> break-even rate would be >= 100%, no offer


def break_even(p):
    """
    Break-even interest rate r = p / (1 - p) for probability(ies) of default p.

    p : float, numpy array or pandas Series of default probabilities

    Returns the same shape as p, with NaN wherever p >= 0.5 (no offer).
    """
    p_arr = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p_arr < NO_OFFER_THRESHOLD, p_arr / (1 - p_arr), np.nan)

    if isinstance(p, pd.Series):
        return pd.Series(r, index=p.index, name="break_even_rate")
    if np.ndim(r) == 0:
        return float(r)
    return r


def add_break_even_rates(dataset, prob_col="p_default", rate_col="break_even_rate"):
    """
    Return a copy of `dataset` with a break-even rate column added.

    dataset  : DataFrame holding the Part 2 predictions
    prob_col : name of the column with predicted default probabilities
    rate_col : name of the new column (NaN = no offer, p >= 0.5)
    """
    out = dataset.copy()
    out[rate_col] = break_even(out[prob_col])
    return out


def competitor_offers(p_comp, margin):
    """
    Rate a competitor offers: break-even rate plus an assumed margin.

    p_comp : float, numpy array or pandas Series of the competitor's predicted
             default probabilities (e.g. Lender 1 / Lender 2 model output)
    margin : assumed margin added on top of break-even (scenario parameter,
             e.g. 0.02 for +2 percentage points)

    Returns the offered rate(s), NaN where p_comp >= 0.5 (no offer).
    """
    r = break_even(p_comp) + margin  # NaN + margin stays NaN -> still no offer
    if isinstance(r, pd.Series):
        r.name = "competitor_rate"
    return r
