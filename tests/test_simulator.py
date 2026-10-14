"""Tests for the market simulator and winner's curse in src/simulator.py (Parts 4 and 5).

Small hand-made markets whose outcome can be worked out on paper. Run from the project root:

    python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `src` importable however pytest is launched: the project root is one level above tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulator import simulate_market, winners_curse  # noqa: E402

nan = np.nan


def test_cheapest_offer_wins_and_discount_flips_choice():
    """Lowest effective rate wins; the preferred lender's 2 pp discount can flip the choice.

    All three borrowers prefer lender 3 (effective rate = its offer - 0.02):
    0: .06 vs .09 vs .10-.02=.08 -> lender 1, the discount is not enough.
    1: .07 vs .10 vs .08-.02=.06 -> lender 3, although lender 1 is cheaper at face value.
    2: .10 vs .05 vs .08-.02=.06 -> lender 2.
    """
    offers = np.array([[0.06, 0.09, 0.10], [0.07, 0.10, 0.08], [0.10, 0.05, 0.08]])
    loans, _ = simulate_market(offers, [0, 0, 0], pref=[3, 3, 3], mode="random")
    assert loans["chosen_lender"].tolist() == [1, 3, 2]


def test_tie_split_in_expected_mode_and_reproducible_in_random_mode():
    """Lender 1 at 5% preferred vs lender 2 at 3%: both are 3% effective, a tie.

    0.05 - 0.02 is 0.030000000000000002 in floating point, so this also checks the tolerance.
    Expected mode: type 1 splits 1/2-1/2, types 2 and 3 go to lender 2 -> weights 1/6, 5/6, 0.
    Random mode: the tie is broken at random, identically for the same seed.
    """
    one = np.array([[0.05, 0.03, nan]])
    loans, _ = simulate_market(one, [0], mode="expected")
    np.testing.assert_allclose(loans[["w1", "w2", "w3"]].to_numpy(), [[1 / 6, 5 / 6, 0]])

    many = np.tile(one, (1_000, 1))
    pref, defaults = np.ones(1_000, dtype=int), np.zeros(1_000, dtype=int)
    first, _ = simulate_market(many, defaults, pref=pref, mode="random", seed=1)
    again, _ = simulate_market(many, defaults, pref=pref, mode="random", seed=1)
    other, _ = simulate_market(many, defaults, pref=pref, mode="random", seed=2)
    assert first["chosen_lender"].equals(again["chosen_lender"])
    assert set(first["chosen_lender"]) == {1, 2}  # both tied lenders win sometimes
    assert not first["chosen_lender"].equals(other["chosen_lender"])


@pytest.mark.parametrize("mode", ["expected", "random"])
def test_no_offer_means_no_loan(mode):
    """A borrower with three NaN offers takes no loan: zero weights, zero profit."""
    loans, summary = simulate_market(np.array([[nan, nan, nan]]), [1], mode=mode)
    assert loans["loan_made_prob"].iloc[0] == 0
    assert (loans[["w1", "w2", "w3", "profit1", "profit2", "profit3"]].to_numpy() == 0).all()
    assert (summary["n_loans"] == 0).all()
    if mode == "random":
        assert loans["chosen_lender"].iloc[0] == 0


def test_profit_arithmetic_at_the_nominal_rate():
    """Profit = amount * offered rate if repaid, -amount if defaulted.

    Both borrowers prefer lender 3 and take its 8% offer (8% - 2% = 6% beats lender 1's 7%).
    The repaid loan earns 8% of 10,000 = 800 -- the nominal rate, not the 6% used to compare.
    The defaulted loan loses the full 10,000.
    """
    offers = np.array([[0.07, 0.10, 0.08], [0.07, 0.10, 0.08]])
    loans, summary = simulate_market(
        offers, [0, 1], pref=[3, 3], mode="random", loan_amount=10_000
    )
    np.testing.assert_allclose(loans["profit3"], [800, -10_000])
    assert summary.loc[3, "total_profit"] == pytest.approx(-9_200)
    assert summary.loc[3, "avg_rate"] == pytest.approx(0.08)
    assert summary.loc[3, "default_rate_won"] == pytest.approx(0.5)


def test_loan_amount_per_applicant():
    """loan_amount may differ by applicant: 10% of 10,000 earned, 5,000 lost."""
    offers = np.array([[0.10, nan, nan], [0.10, nan, nan]])
    loans, _ = simulate_market(offers, [0, 1], loan_amount=np.array([10_000, 5_000]))
    np.testing.assert_allclose(loans["profit1"], [1_000, -5_000])


def test_rates_above_100_percent_accepted_but_percent_inputs_rejected():
    """Break-even + margin can pass 100% (p = 0.48: 0.48/0.52 + 0.08 = 1.003); 12 for 12% cannot."""
    near_half = np.array([[0.48 / 0.52 + 0.08, nan, nan]])
    loans, _ = simulate_market(near_half, [0])
    assert loans["w1"].iloc[0] == 1

    with pytest.raises(ValueError, match="decimals"):
        simulate_market(np.array([[12.0, nan, nan]]), [0])


def test_winners_curse_hand_computed():
    """Gap between observed and predicted default rates on the loans lender 3 wins.

    Borrowers 0-3 get an offer from lender 3 only, so lender 3 wins them. Borrowers 4-5 also get
    5% from lender 1, which beats lender 3's 10% whatever the preference, so lender 3 loses them.

    borrower  0    1    2    3    4    5
    p_hat     .05  .15  .05  .15  .05  .15
    default   1    0    0    1    1    0
    won       yes  yes  yes  yes  no   no

    Groups by p_hat: G1 = {0, 2, 4}, G2 = {1, 3, 5}; only won loans enter the averages.
    G1: won 2 of 3, predicted .05, observed (1+0)/2 = .5, gap .45
    G2: won 2 of 3, predicted .15, observed (0+1)/2 = .5, gap .35
    overall: won 4 of 6, predicted .10, observed 2/4 = .5, gap .40
    Borrower 4 defaults but is lender 1's loan, so it does not count against us.
    """
    offers = np.array([[nan, nan, 0.10]] * 4 + [[0.05, nan, 0.10]] * 2)
    defaults = [1, 0, 0, 1, 1, 0]
    p_hat = [0.05, 0.15, 0.05, 0.15, 0.05, 0.15]
    loans, _ = simulate_market(offers, defaults, mode="expected")

    curse = winners_curse(loans, p_hat, lender=3, groups=[0, 0.10, 1])

    assert curse.index.tolist() == ["G1", "G2", "overall"]
    np.testing.assert_allclose(curse["group_size"], [3, 3, 6])
    np.testing.assert_allclose(curse["n_loans_won"], [2, 2, 4])
    np.testing.assert_allclose(curse["avg_pred_p"], [0.05, 0.15, 0.10])
    np.testing.assert_allclose(curse["obs_default_rate"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(curse["gap"], [0.45, 0.35, 0.40])
