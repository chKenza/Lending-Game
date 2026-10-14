"""Shared configuration for the Lending Game project.

Central place for file paths and the random seed so that every notebook
(Parts 1-8) reads the same data from the same locations and reproduces the
same results.

Usage from a notebook in analysis/:

    import sys; sys.path.append("..")
    from src.config import PAST_LOANS, NEW_APPS_R1, SEED
"""

from pathlib import Path

# --- directories -----------------------------------------------------------
# config.py lives in src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"  # read-only inputs provided by the course
DATA_PROCESSED = DATA_DIR / "processed"  # train / validation sets (Part 1)
DATA_PREDICTIONS = DATA_DIR / "predictions"  # default probabilities (Part 2)

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"  # PNGs embedded in the reports

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"  # final id + rate CSV

# --- input files -----------------------------------------------------------
PAST_LOANS = DATA_RAW / "PastLoans.csv"  # historical loans, outcome known
NEW_APPS_R1 = DATA_RAW / "NewApplications_Lender3_Round1.csv"  # round 1 applicants

# --- reproducibility -------------------------------------------------------
SEED = 42

# --- game constants --------------------------------------------------------
LOAN_AMOUNT = 10_000  # every applicant asks for exactly this amount (euros)
INCOME_TOPCODE = 1_000_000  # income is top-coded at this value in the raw data
OUR_LENDER = 3  # we are lender 3; signal3 is our private signal

DISCOUNT = 0.02