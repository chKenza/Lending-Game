# fair_pricing.py — Part 3: break-even pricing

Expected profit per euro lent, with default probability `p` and rate `r`, is `(1 − p)·r − p`.
Setting it to zero gives the **break-even rate** `r = p / (1 − p)`. This is a floor, not the price we charge.

Rules used everywhere:
- Applicants with `p ≥ 0.5` get **no offer** (break-even would be ≥ 100%).
- Rates above 100% are not allowed in the game.
- "No offer" is always `NaN`, which is what the submission file and `simulator.py` expect.

All functions accept a single number, a numpy array (what `models.predict_default` returns) or a pandas Series.

## Core functions (work on probabilities directly)

| Function | Returns |
|---|---|
| `offer(p)` | Offer flag: `1` if `p < 0.5` (we make an offer), `0` if `p ≥ 0.5` or `p` is missing. |
| `break_even_rate(p)` | `p / (1 − p)`, or `NaN` if `p ≥ 0.5`. |
| `competitor_offers(p_comp, margin)` | A competitor's assumed rate: break-even + `margin` (e.g. `0.02` = +2 points). `NaN` if `p ≥ 0.5` or the rate would exceed 100%. `margin` is the scenario parameter for Parts 6 and 7. |

## `add_...` functions (add columns to a dataset)

Each one takes a DataFrame of predicted probabilities (e.g. the output of `models.oof_predict_all`, with columns like `own__logit`, `lender1__gb`), and returns a **copy** with one new column per probability column. The original DataFrame is not changed.

| Function | New column for `own__logit` | Content |
|---|---|---|
| `add_offer_flags(dataset, prob_cols=None)` | `offer_own__logit` | `offer()` flag (1/0) |
| `add_break_even_rates(dataset, prob_cols=None)` | `be_own__logit` | `break_even_rate()` |
| `add_competitor_offers(dataset, prob_cols, margin)` | `comp_lender1__logit` | `competitor_offers()` |

- `prob_cols`: one column name or a list. For the first two, if omitted, every float column with values in [0, 1] is used.
- For `add_competitor_offers`, `prob_cols` is **required**, so our own model's column is never priced as a competitor by accident.
- The column prefix can be changed with `prefix=`.

## `market_offers(p_lender1, p_lender2, r_ours, margin1, margin2=None, index=None)`

Builds the offers table of all three lenders in the format `simulate_market` expects: columns `r1`, `r2`, `r3`, as decimals, `NaN` = no offer.

- `r1`, `r2`: competitor offers from our Lender 1 / Lender 2 models' probabilities, with margins `margin1` / `margin2` (`margin2` defaults to `margin1`).
- `r3`: our own rate, passed in as `r_ours`. It comes from the pricing rule (Part 6), not computed here.
- `index`: row labels (e.g. `row_id`); taken from `r_ours` if it is a Series.

## Example

```python
import sys; sys.path.append("..")
from src.fair_pricing import add_offer_flags, add_break_even_rates, \
    add_competitor_offers, competitor_offers, market_offers

preds = add_offer_flags(preds, "own__logit")
preds = add_break_even_rates(preds, "own__logit")
preds = add_competitor_offers(preds, ["lender1__logit", "lender2__logit"], margin=0.02)

offers = market_offers(preds["lender1__logit"], preds["lender2__logit"],
                       r_ours=competitor_offers(preds["own__logit"], 0.03),  # placeholder until Part 6
                       margin1=0.02)
loans, summary = simulate_market(offers, y_val)
```
