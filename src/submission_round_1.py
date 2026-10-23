"""Part 8 - pricing the new applications and writing the submission file.

The submission is a CSV with one row per loan application and two columns
(course instructions, Stage 1):

* id   : the applicant identifier of LoanApplications (1 to 100,000)
* rate : the interest rate we offer, as a decimal (0.12 = 12%), at most 1;
         left empty when we make no offer.

Usage from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.submission_round_1 import load_final_rule, make_submission, check_submission, write_submission
"""

import json

import numpy as np
import pandas as pd

from src.config import SUBMISSIONS_DIR
from src.fair_pricing import MAX_RATE
from src.pricing_rules import PRICING_DIR, load_tuned_rules

SUBMISSION_FILE = SUBMISSIONS_DIR / "Lender3_Round1_submission.csv"


def load_final_rule(train, filename="final_rule.json"):
    """The rule chosen in Part 7 (data/pricing/final_rule.json), rebuilt as a
    function data -> our rates, with its name and parameters attached
    (.family, .params)."""
    name = json.loads((PRICING_DIR / filename).read_text())["rule"]
    return load_tuned_rules(train)[name]


def make_submission(rates):
    """Submission table from our rates indexed by applicant id: columns id and
    rate, sorted by id, NaN where we make no offer."""
    sub = pd.DataFrame({"id": np.asarray(rates.index, dtype=int),
                        "rate": np.asarray(rates, dtype=float)})
    return sub.sort_values("id").reset_index(drop=True)


def check_submission(sub, expected_ids):
    """Check the submission against the course rules; raise ValueError on any
    problem. Returns a dict summarising the offers.

    * exactly the columns id and rate;
    * every expected id exactly once, nothing else;
    * every rate either missing (no offer) or a decimal in (0, 1].
    """
    if list(sub.columns) != ["id", "rate"]:
        raise ValueError(f"columns must be ['id', 'rate'], got {list(sub.columns)}")
    ids = pd.Series(sub["id"])
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} duplicated id(s)")
    expected = set(pd.Series(expected_ids).astype(int))
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra:
        raise ValueError(f"{len(missing)} id(s) missing, {len(extra)} unexpected id(s)")
    rates = pd.to_numeric(sub["rate"], errors="raise")
    offered = rates.dropna()
    if (offered <= 0).any():
        raise ValueError(f"{int((offered <= 0).sum())} rate(s) not positive")
    if (offered > MAX_RATE).any():
        raise ValueError(f"{int((offered > MAX_RATE).sum())} rate(s) above 100%")
    if (offered > 1.5).any() or (offered > 0).sum() == 0:
        raise ValueError("rates look like percents, not decimals")
    return {"applications": len(sub), "offers": int(offered.size),
            "no offer": int(rates.isna().sum()), "min rate": float(offered.min()),
            "median rate": float(offered.median()), "max rate": float(offered.max())}


def write_submission(sub, path=SUBMISSION_FILE, decimals=6):
    """Write the submission CSV: rates with `decimals` decimals, empty cell
    for no offer. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False, float_format=f"%.{decimals}f", na_rep="")
    return path
