# Lending Game — Part 8: Pricing the new applications and the Stage 1 submission

How the 100,000 new applicants were priced and how the submission file was built and checked.
Everything comes from `analysis/submission_round_1.ipynb`; the code is in `src/submission_round_1.py`.

## Summary

- **Submission file:** `submissions/Lender3_Round1_submission.csv`, with 100,000 rows (ids 1 to
  100,000) and columns `id` and `rate`.
  - **99,946 offers**, at rates from 1.76% to 99.9%;
  - **54 empty rates** (no offer).
- **Pipeline:** Part 2's model predicts $p_{own}$, Part 7's rule **R4** turns it into a rate.
  - Safer applicants ($p_{own} < 11\%$, 60% of the pool) get 1.8–3.6 points over break-even.
  - Riskier applicants get deliberately high quotes, 16–19 points over break-even.
- **Every course rule is checked**, on the table and on the file read back from disk.

---

## 1. Predict

Our default probability $p_{own}$ for each new applicant comes from Part 2
(`data/predictions/applications_own.csv`). The model is the spline logistic regression on the
public variables and our signal `signal3`, refitted on all 100,000 past loans.

| | new applicants | validation set (Part 2) |
|---|---|---|
| mean $p_{own}$ | 9.96% | 9.96% |
| median | 8.67% | 8.71% |
| 90th percentile | 20.5% | 20.4% |
| 99th percentile | 33.7% | 32.9% |
| maximum | 50.2% | 46.5% |

The new applicants look like the validation set, as the course states (same population). The rule
tuned on these probabilities therefore applies without adjustment.

## 2. Price

**Rule R4** (chosen in Part 7, saved in `data/pricing/final_rule.json`):

$$r = \frac{p_{own}}{1 - p_{own}} + m(p_{own})$$

| $p_{own}$ | margin $m$ over break-even |
|---|---|
| below 2.75% | 1.75 points |
| 2.75% – 4.5% | 2 points |
| 4.5% – 11% | 3.2 points |
| 11% – 15.4% | 16.9 points |
| above 15.4% | 19.4 points |

The margins are the plateaus of a smooth increasing curve, which is interpolated between the decile
centres. So the margin climbs gradually between plateaus: most steeply from 3.2 to 16.9 points as
$p_{own}$ goes from about 10% to 12%. The rate floor is 0.5%. There is no offer when
$p_{own} \ge 50\%$ or when the rate would exceed 100%.

R4 deliberately **quotes high rates to risky applicants instead of refusing them**. If the
competitors lend to them more cheaply, these quotes rarely win. If the competitors refuse them, we
are their only lender and the high rate pays for the risk (Part 7).

**Resulting offers:**

| $p_{own}$ band | applicants | offers | average rate | average margin |
|---|---|---|---|---|
| below 2.75% | 19,615 | 19,615 | 3.1% | 1.8 pts |
| 2.75% – 4.5% | 10,482 | 10,482 | 5.8% | 2.1 pts |
| 4.5% – 11% | 29,661 | 29,661 | 11.9% | 3.6 pts |
| 11% – 15% | 18,745 | 18,745 | 30.8% | 15.9 pts |
| 15% – 50% | 21,496 | 21,443 | 47.3% | 18.9 pts |
| 50% and above | 1 | 0 | — | — |

- **54 applicants get no offer:**
  - 1 because $p_{own} \ge 50\%$ (id 23772, $p_{own} = 50.2\%$);
  - 53 with $p_{own}$ between about 45% and 50%, because their rate would exceed 100%.

  Part 7 showed that capping these 53 at 100% instead would change nothing: competitors always quote
  them less.
- The 0.5% floor never binds; the lowest rate offered is 1.76%.