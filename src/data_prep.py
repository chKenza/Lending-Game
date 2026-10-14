"""Data preparation and train/validation split for the Lending Game (Part 1).

Three prep functions share one stateless core:

    general_prep_data   encodings every model uses
    logreg_prep_data    general + features a linear model needs
    boosting_prep_data  general + (for now) nothing extra

`split_data` makes the stratified 80/20 train/validation split.

**Everything here is stateless**: no mean, median, category list or scale is learned from the
data that is passed in, so the same function applied to any file returns the same columns
computed the same way. That is what lets us prepare `PastLoans.csv` and the new applications
with the same code and get features that mean the same thing in both. The one consequence is
that the four `employment` categories are hardcoded below rather than read off the data.

Every prep function works on both shapes of input file:

* `PastLoans.csv`  -- all three signals, `default` present, identified by `row_id`
* `NewApplications_Lender3_RoundN.csv` -- `signal3` only, no `default`, identified by `id`

Columns that are absent are simply skipped, so the applications come out with the same columns
as the past loans minus `default`, `signal1`, `signal2` and the two signal flags.

Usage from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.data_prep import general_prep_data, logreg_prep_data, boosting_prep_data, split_data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED

# --- fixed vocabularies ----------------------------------------------------
# Hardcoded on purpose (see the module docstring): one-hot encoding the four categories from a
# fixed list guarantees that a file which happens to contain no retired applicants still gets an
# `employment_retired` column of zeros, so every file produces identical columns in the same order.
EMPLOYMENT_CATEGORIES = ("employed", "retired", "student", "unemployed")

# All signals that can appear. Only those actually present in the input are encoded; we are
# lender 3, so the new applications carry signal3 alone.
SIGNAL_COLUMNS = ("signal1", "signal2", "signal3")

# Row identifiers: `row_id` is the one we add to PastLoans, `id` is the one the applications ship
# with. Whichever is present is carried through untouched.
ID_COLUMNS = ("row_id", "id")

TARGET = "default"

# Value of `sex` encoded as 1 in the `sex_F` column.
SEX_POSITIVE = "F"


def general_prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the transformations every model shares.

    Stateless. Produces, in this order:

    ==========================  ==========================================================
    column                      definition
    ==========================  ==========================================================
    `row_id` or `id`            the identifier the input came with, unchanged
    `sex_F`                     1 if `sex == "F"`, 0 if `sex == "M"`
    `married`                   unchanged 0/1
    `employment_<category>`     one-hot, one column per category in EMPLOYMENT_CATEGORIES
    `income`                    raw income, unchanged
    `income_zero`               1 if `income == 0`
    `signalX`                   raw signal value, unchanged (one pair per signal present)
    `signalX_missing`           1 if `signalX` is exactly 0
    `default`                   unchanged, if the file has it
    ==========================  ==========================================================

    `income == 0` and `signal == 0` get flags because the EDA found both to be structural
    (`report/eda_report.md`, 2.2 and 2.4): zero income occurs only among students and the
    unemployed and carries its own risk, and a zero signal means "no signal received" rather
    than a low score. The raw column is kept alongside each flag so a model can use both.

    Parameters
    ----------
    data : pd.DataFrame
        Raw loans or applications, as read from CSV.

    Returns
    -------
    pd.DataFrame
        A new frame with the encoded columns and the original index. The input is not modified.

    Raises
    ------
    KeyError
        If one of the always-expected columns (`sex`, `employment`, `married`, `income`) is absent.
    ValueError
        If `sex` or `employment` contains a value outside the expected vocabulary, which would
        otherwise be silently encoded as all-zeros.
    """
    required = ["sex", "employment", "married", "income"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"input is missing expected column(s): {missing}")

    # Fail loudly on unexpected category values: with hardcoded vocabularies an unknown value
    # would quietly become a row of zeros instead of an error.
    unknown_sex = set(data["sex"].unique()) - {"M", "F"}
    if unknown_sex:
        raise ValueError(f"unexpected sex value(s): {sorted(unknown_sex)}")
    unknown_emp = set(data["employment"].unique()) - set(EMPLOYMENT_CATEGORIES)
    if unknown_emp:
        raise ValueError(f"unexpected employment value(s): {sorted(unknown_emp)}")

    out = {}

    # --- identifier ---------------------------------------------------------
    # Carried through unchanged so a prepared row can always be traced back to its raw row.
    for col in ID_COLUMNS:
        if col in data.columns:
            out[col] = data[col]

    # --- sex: one 0/1 column -----------------------------------------------
    # Two levels need only one column. F is the 1 so that a positive coefficient reads as extra
    # risk: the EDA has women defaulting slightly more often (10.67% vs 9.41%).
    out["sex_F"] = (data["sex"] == SEX_POSITIVE).astype(int)

    # --- married: already 0/1 ----------------------------------------------
    out["married"] = data["married"].astype(int)

    # --- employment: one-hot over the fixed category list -------------------
    # One-hot rather than ordinal: the EDA found the risk ordering reverses once income-zero rows
    # are set aside (rank correlation -0.60), so there is no stable ranking to encode.
    # All four columns are kept; models that need a reference level can drop one themselves.
    for category in EMPLOYMENT_CATEGORIES:
        out[f"employment_{category}"] = (data["employment"] == category).astype(int)

    # --- income: raw value plus a structural-zero flag ----------------------
    out["income"] = data["income"]
    out["income_zero"] = (data["income"] == 0).astype(int)

    # --- signals: raw value plus a "no signal received" flag ----------------
    # Only the signals the file actually contains, so this works unchanged on the applications.
    for signal in SIGNAL_COLUMNS:
        if signal in data.columns:
            out[signal] = data[signal]
            out[f"{signal}_missing"] = (data[signal] == 0).astype(int)

    # --- target -------------------------------------------------------------
    if TARGET in data.columns:
        out[TARGET] = data[TARGET]

    return pd.DataFrame(out, index=data.index)


def logreg_prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the logistic regression: the general features plus two more.

    Stateless, like `general_prep_data`, which it calls first.

    Adds:

    ==============================  ======================================================
    column                          definition
    ==============================  ======================================================
    `income_log1p`                  `log1p(income)`
    `unemployed_x_income_zero`      `employment_unemployed * income_zero`
    ==============================  ======================================================

    `log1p` rather than `log` because 17% of incomes are exactly 0; it also leaves the raw
    `income` column in place, so the choice of scale stays with the model. The EDA found the log
    scale is what makes income linear in the log-odds (R^2 0.713 -> 0.965), which matters for a
    linear model and not for a tree.

    Only the one interaction is created. `income_zero` occurs only among students and the
    unemployed, so `student_x_income_zero` would equal `income_zero` minus this column and make
    `income_zero` redundant; one of the two interactions is enough to separate the two groups.

    Parameters
    ----------
    data : pd.DataFrame
        Raw loans or applications, as read from CSV.

    Returns
    -------
    pd.DataFrame
        The general output with the two extra columns appended. The input is not modified.
    """
    preprocessed_logreg = general_prep_data(data)

    # log1p(income): defined at income == 0, where log would not be.
    preprocessed_logreg["income_log1p"] = np.log1p(preprocessed_logreg["income"])

    # The one interaction worth having (see above and report/eda_report.md, 3.4 and 3.6).
    preprocessed_logreg["unemployed_x_income_zero"] = (
        preprocessed_logreg["employment_unemployed"] * preprocessed_logreg["income_zero"]
    )

    # Keep `default` last so the prepared frames stay readable.
    if TARGET in preprocessed_logreg.columns:
        preprocessed_logreg = preprocessed_logreg[
            [c for c in preprocessed_logreg.columns if c != TARGET] + [TARGET]
        ]

    return preprocessed_logreg


def boosting_prep_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the boosted-tree model: the general features, unchanged for now.

    Stateless, like `general_prep_data`, which it calls first.

    A gradient-boosted tree is invariant to monotone transforms of its inputs, so the log scale
    and the interaction that the logistic regression needs would add nothing here: a tree can
    split on raw `income` and combine `employment_unemployed` with `income_zero` by itself.

    Boosting-specific features can be added here later (the counterpart of the two columns
    `logreg_prep_data` appends) without touching the general or logreg paths.

    Parameters
    ----------
    data : pd.DataFrame
        Raw loans or applications, as read from CSV.

    Returns
    -------
    pd.DataFrame
        The general output. The input is not modified.
    """
    preprocessed_boost = general_prep_data(data)

    # No boosting-specific features yet; add them here when the model asks for them.

    return preprocessed_boost


def split_data(
    preprocessed_data: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = SEED,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prepared data 80/20 into train and validation, stratified on the target.

    Stratified because defaults are rare (9.95%): a plain random split would let the two halves
    drift apart on the very quantity being modelled.

    The split depends only on the row order, `test_size` and `seed`, none of which differ between
    the general, logreg and boosting versions of a file. Called with the same defaults on all
    three, it therefore puts the same rows in train and the same rows in validation -- the
    notebook asserts this on `row_id` rather than relying on it.

    Parameters
    ----------
    preprocessed_data : pd.DataFrame
        Output of one of the prep functions above. Must contain `target`.
    test_size : float
        Share of rows held out for validation. Default 0.2, i.e. an 80/20 split.
    seed : int
        Random seed; defaults to `SEED` from src/config.py so every part of the project splits
        the same way.
    target : str
        Column to stratify on. Default `default`.

    Returns
    -------
    (train, validation) : tuple of pd.DataFrame
        Two new frames; the input is not modified. Row order is the shuffled order produced by
        the split, and each row keeps its original index.

    Raises
    ------
    KeyError
        If `target` is absent -- as it is in the applications, which are predicted on, not split.
    """
    if target not in preprocessed_data.columns:
        raise KeyError(
            f"cannot stratify on '{target}': column absent. "
            "The new applications have no outcome and are not meant to be split."
        )

    train, validation = train_test_split(
        preprocessed_data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=preprocessed_data[target],
    )
    return train, validation
