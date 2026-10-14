# Lending Game — Part 3: Break-even rates

Findings from applying the break-even rate to the Part 2 predictions. Everything comes from
`analysis/fair_pricing.ipynb`; the functions are in `src/fair_pricing.py`.

A loan of 1 euro at rate $r$ to a borrower with default probability $p$:

* with probability $1 - p$ the borrower repays and we earn $r$;
* with probability $p$ the borrower defaults and we lose the euro (recovery is zero).

Expected profit per euro lent:

$$\mathbb{E}[\pi] = (1 - p)\,r - p .$$

Setting it to zero gives the **break-even rate**

$$r^{*}(p) = \frac{p}{1 - p}.$$

At $r^*$ we expect to make exactly zero; any rate above it is profitable *if $p$ is right*.
$r^*$ is the **benchmark for every pricing decision**, a floor and not the price we will charge
(Part 6 decides the price). Two rules apply everywhere:

* **No offer when $p \ge 0.5$**: then $r^* \ge 100\%$, and the game forbids rates above 100%.
* "No offer" is a missing rate (`NaN`), as in the submission file and the simulator.

## Summary

- **Break-even rate** $r^* = p/(1-p)$ sets expected profit $(1-p)\,r - p$ to zero. No offer when $p \ge 0.5$.
- **Our break-even rates** (validation): median 9.5%, mean 12.0%, maximum 86.8%. The "no offer" rule
  almost never binds.
- **Without competition, break-even pricing breaks even** (+0.0002 per euro): our probabilities are
  well calibrated.
- **With competition, the same rates lose money** (−2.4 cents per euro on the loans we win): the
  winner's curse. Break-even is a floor, not a price.

## 1. The break-even curve

| p | 1% | 5% | 10% | 20% | 30% | 40% | 49% | ≥ 50% |
|---|---|---|---|---|---|---|---|---|
| $r^*$ | 1.01% | 5.26% | 11.1% | 25.0% | 42.9% | 66.7% | 96.1% | no offer |

Close to $p$ for safe borrowers, **convex** for risky ones: its slope $1/(1-p)^2$ means a 1-point error
in $p$ costs about 1 point of rate at low risk but about 2 points at $p = 30\%$. Half of our borrowers
have $p < 8.7\%$ and 99% have $p < 32.9\%$, so most pricing happens on the flat part of the curve.

## 2. Break-even rates of the Part 2 models (validation)

| information set | median | mean | 90th pct | 99th pct | max | no offer |
|---|---|---|---|---|---|---|
| own | 9.5% | 12.0% | 25.6% | 48.9% | 86.8% | 0 |
| full | 8.6% | 12.1% | 27.3% | 62.2% | 99.2% | 52 |
| lender 1 | 9.6% | 12.0% | 25.3% | 49.4% | 79.6% | 0 |
| lender 2 | 9.7% | 11.9% | 25.7% | 47.3% | 78.8% | 0 |
| public | 10.2% | 11.8% | 21.2% | 34.8% | 41.0% | 0 |

- The mean rate (12%) exceeds the mean default probability (10%) because the curve is convex.
- More information spreads the rates out: the full model has the most extreme rates (52 no-offers),
  the public model the most compressed (max 41%).
- Lenders 1 and 2 have almost the same distribution as ours: they differ on *which* borrowers are risky,
  not on the overall level.
- Out-of-fold training rates give the same picture.

## 3. Does break-even pricing break even?

**Without competition, yes.** Lending to all 20,000 validation borrowers at our break-even rate yields
**+0.0002 per euro** (+€36k on €200M lent). Every decile of predicted risk is within its 95% confidence
interval of zero, and all five information sets are within ±0.0004 per euro.

**With competition, no.** If all three lenders price at break-even and borrowers take the cheapest offer
(preferences ignored):

| | value |
|---|---|
| loans we win | 6,657 (33.3% of applicants) |
| our average predicted p | 8.0% |
| observed default rate | 10.1% |
| our average rate | 9.2% |
| profit per euro lent | −0.024 |
| total profit | −€1.6M |

The borrowers who pick us are the ones whose risk we under-estimated. **The price must add a margin over
break-even** to cover this adverse selection (measured in Part 5, priced in Part 6).

## 4. Competitor offers

`competitor_offers(p, margin)` = competitor's break-even rate + margin, never below 0% or above 100%.
Share of validation applicants for whom **our break-even rate is below both competitors' offers**:

| competitors' margin | 0 | +1 pt | +2 pts | +3 pts | +5 pts | +10 pts |
|---|---|---|---|---|---|---|
| share | 33% | 59% | 70% | 76% | 84% | 93% |

The larger the competitors' margin, the more of the market we can undercut while covering expected
losses. The margin is unknown: it is the scenario parameter of Parts 6 and 7.

## 5. New applications

Break-even rates for the 100,000 applicants have the same distribution as on validation (median 9.5%,
99th percentile 50.8%). **One applicant gets no offer** (id 23772, $p = 0.502$).

## 6. Deliverables (`data/predictions/`)

| file | content |
|---|---|
| `validation_break_even.csv` | `default`, `p_*`, break-even rates `be_p_*` and offer flags `offer_p_*` for the five information sets |
| `train_oof_break_even.csv` | the same for the 80,000 training loans (out of fold) |
| `applications_break_even.csv` | `p_own`, `be_p_own`, `offer_p_own` for the 100,000 new applicants |
