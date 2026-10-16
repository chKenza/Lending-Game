# Lending Game — Part 6: Tuning the pricing rule

How we turned our default probabilities into rates that maximise profit in the simulated market, and
which rule we recommend. Everything comes from `analysis/pricing_rules.ipynb`; the code is in
`src/pricing_rules.py` and `src/pricing_plots.py`.

## Summary

- **Recommended rule: R4+C.** Break-even plus a margin that rises with risk (about 2 points for
  $p_{own} < 4.5\%$, about 3 points up to 11%), and **no offer from $p_{own} = 0.125$**.
  - Weighted profit **+€2.97M** on the 80,000 training loans, the best of all rules.
  - Smallest max regret (€0.10M).
  - Profitable in **all six** competitor scenarios.
- **R7+C is a close challenger** (+€2.94M): it prices 2 points above the competitors' predicted
  break-even and makes no offer from $p = 0.10$. It needs more machinery.
- **Break-even pricing loses money in every scenario** (−€3.40M weighted): the winner's curse.
- **What makes money:** a margin that grows with risk, and **no lending to the riskiest third of
  applicants**, where the borrowers we would win default 8–18 points above our prediction.
- **Competitors don't settle on an equilibrium.** Best-responding lenders cycle between about 2.25
  and 5.25 points over break-even, averaging 3–4.5 points.

---

## 1. Method

**Data.** The 80,000 training loans with their out-of-fold Part 2 predictions. The validation set
is kept for Part 7.

**Market.** Each rule's rates are simulated with Part 4's `simulate_market` in expected mode (each
borrower's choice averaged over the three preference types, so there is no random noise). Profit:
$+r \cdot 10{,}000$ if the borrower repays, $-10{,}000$ if they default.

**Competitors.** Their default probabilities come from our reconstructed lender 1 and 2 models. Their
pricing is unknown, so every rule is scored against six scenarios:

| scenario | competitors' rate | weight |
|---|---|---|
| T1, T2, T3 | break-even + 0, 2, 4 points | 1/7 each |
| T4, T5 | break-even × 1.15, × 1.35 | 1/7 each |
| T6 | break-even + the average margins of the best-response cycle (4.25 and 3.75 points, section 2) | 2/7 |

**Rule families.** All rules start from the break-even rate $be(p) = p/(1-p)$ and use only our own
information.

| ID | rule | formula |
|---|---|---|
| B0 | break-even | $r = be(p)$ |
| R1 | fixed margin | $r = be(p) + m$ |
| R2 | proportional markup | $r = (1+k)\,be(p)$ |
| R3 | adjusted p | $\text{logit}(p_{adj}) = \text{logit}(p) + a$, $r = be(p_{adj}) + m$ |
| R4 | margin per risk decile | $r = be(p) + m(p)$, an increasing curve fitted decile by decile |
| R5 | uncertainty premiums | $r = be(p) + m_0 + m_1\mathbb 1[\text{signal3 missing}] + m_2\mathbb 1[\text{income}=0]$ |
| R6 | information advantage | $r = be(p) + m_0 + m_1 (p_{public} - p)$ |
| R7 | price against the predicted competitor | $r = \max(be(p_{adj}),\ \hat r_{comp} - \delta)$, $\hat r_{comp}$ = predicted cheaper competitor break-even |
| +C | cutoff (added to the best two) | no offer when $p \ge \bar p$ |

Shared rules: no offer when $p \ge 0.5$ or when the rate would exceed 100%. Rates below the floor are
raised to it, and the floor (0.5–2%) is tuned in every family.

**Selection.** For each family, about 10,800 parameter combinations in total, each simulated in the
six scenarios. We keep the combination with the highest **weighted profit** $\sum_s w_s \Pi_s$. We
also check the smallest **max regret** (worst-case shortfall versus the family's best in each
scenario); when the two disagree, we keep the one with the higher margin.

**Two corrections made along the way.**

- **R7 leaked information.** The first version forecast the competitors with each scenario's true
  pricing, which we will not know. It now uses their predicted break-even rate only, the same in
  every scenario, and an assumed margin is carried by δ. The correction cost R7 €0.65M.
- **T6 was an arbitrary point of a cycle.** The best responses never converge (section 2), so T6 now
  uses the cycle's average margins rather than its last round.

## 2. The competitors matter, and there is no equilibrium

**Best response (R1 against symmetric competitors):**

| competitors' margin | our best margin | our profit | loans |
|---|---|---|---|
| 0 / +1 pt | 15 pts (= stay out) | −€0.39M / −€0.43M | < 340 |
| +2 pts | 3.5 pts | +€0.23M | 18,006 |
| +3 pts | 4.25 pts | +€1.96M | 19,646 |
| +5 pts | 2.75 pts | +€7.91M | 50,923 |

- **Against aggressive competitors, no margin makes money**: the best response is to stay out.
- **Against expensive competitors, undercutting them by about 2 points pays.**

**Best-response dynamics.** Each lender in turn picks its best margin (0–8 points) given the others.
After 20 rounds the margins never converge; they repeat a five-round **Edgeworth cycle**: each lender
undercuts the others until margins are low enough that jumping back up pays. On average over the
cycle: lender 1 at 4.32 points, lender 2 at 3.82, us at 3.35. This average defines T6.

## 3. Results

Weighted profit on the 80,000 training loans; regret is measured across rules.

| rank | rule | tuned parameters | weighted profit | max regret | T1 profit | T6 profit | T6 share |
|---|---|---|---|---|---|---|---|
| 1 | **R4+C** | 1.75–2 pts (p < 4.5%), 3.2 pts (4.5–11%); cutoff 0.125; floor 0.5% | **+€2.97M** | **€0.10M** | +€0.44M | **+€5.10M** | 37.2% |
| 2 | **R7+C** | δ = −2 pts, q = 0.5, a = 0.4; cutoff 0.10; floor 0.5% | +€2.94M | €0.12M | +€0.34M | +€4.98M | 33.1% |
| 3 | R4 | as R4+C, no cutoff | +€2.71M | €0.39M | +€0.14M | +€4.83M | 37.5% |
| 4 | R7 | as R7+C, no cutoff | +€2.16M | €1.14M | −€0.56M | +€4.23M | 37.6% |
| 5 | R3 | a = 0.35, m = 1 pt; floor 2% | +€2.05M | €1.30M | −€0.43M | +€4.14M | 37.0% |
| 6 | R2 | k = 0.70; floor 2% | +€1.99M | €1.31M | −€0.00M | +€3.79M | 32.4% |
| 7 | R5 | m0 = 3.5, m1 = 2, m2 = 8 pts; floor 0.5% | +€1.83M | €2.21M | −€1.40M | +€4.39M | 29.8% |
| 8 | R1 | m = 4 pts; floor 0.5% | +€1.18M | €3.18M | −€2.45M | +€4.06M | 33.6% |
| 9 | R6 | m0 = 4.75 pts, m1 = −0.7; floor 0.5% | +€1.04M | €2.97M | −€2.37M | +€3.85M | 34.8% |
| 10 | B0 | floor 2% | −€3.40M | €7.23M | −€5.04M | −€2.13M | 77.2% |

T6 share = share of applicants who take our loan in T6. The real round has 100,000 applicants (about
×1.25 these profits).

## 4. What we learned

1. **The margin must grow with risk.** Rules whose margin rises with $p$ (R2, R3, R4) beat a flat
   margin by €0.8–1.5M. R4 shows the shape: about 2 points below 4.5%, about 3 points up to 11%, then
   margins of 14–20 points that win almost nobody.
2. **Don't lend to the riskiest borrowers.**
   - Above $p_{own} \approx 11$–12%, lending does not pay at any margin.
   - The few borrowers we win there are those the competitors, who know more, priced even higher.
     They default 17–18 points above our prediction.
   - A cutoff adds €0.26M to R4 and €0.78M to R7, and makes both profitable in all six scenarios.
   - The cutoff costs almost no loans: R4+C wins 29,733 loans in T6 against 30,011 without it, because
     R4 already priced those borrowers out.
3. **The winner's curse is concentrated in the risky quintiles.** With the cutoff, the overall gap
   between observed and predicted default on the loans we win falls to about +0.5 point. It stays
   between +0.1 and +1.7 points in the three safest quintiles.
4. **Uncertainty flags and information-advantage pricing add little.**
   - Zero-income applicants are best priced out (+8 points).
   - A missing signal is worth about 2 points.
   - Charging more where our signal is more optimistic than public data (R6) does not help; the
     fitted slope even has the opposite sign.
5. **The rate floor barely matters**: a 2% floor helps only rules that can price safe borrowers near
   0%.
6. **Without its leak, the competitor-based rule (R7) is only competitive with a cutoff.** It then
   reaches the same policy as R4+C by another route: lend to the safer half of the market, about 2
   points over the competitors' break-even.

## 5. Recommendation and next steps

**R4+C is the recommended rule:**

- best weighted profit and smallest regret;
- profitable in every scenario;
- simple and transparent: it uses only `p_own`;
- directly applicable to the new applicants.

R7+C is effectively tied (€24k, under 1% of profit) and stays as the challenger.

**Rules carried to Part 7:** R4+C, R7+C, R3, R2, and R1 and B0 as references. Part 7 scores them on
the validation set against held-out scenarios: asymmetric margins, +1/+3/+5 points, competitors using
our rule, competitors knowing more or less. **Its results, not these training-set profits, decide
the submission.**

**Ready to use.** The tuned parameters are in `data/pricing/tuned_rules.json` (with the T6 margins),
and the grids in `data/pricing/grid_*.csv`. `load_tuned_rules(train)` in `src/pricing_rules.py`
rebuilds every rule as a function of the data (example in `analysis/competitor_scenarios.ipynb`):

```python
rules = load_tuned_rules(load_pricing_data("train"))
rates = rules["R4+C"](load_pricing_data("validation"))   # our rate per applicant, NaN = no offer
```

The loaded rules reproduce the notebook's profits exactly. For Part 8, R7 also needs `p_public` for
the new applicants.

## 6. Limitations

- **In-sample.** The rules were tuned and scored on the same training loans, so these profits are
  optimistic.
- **Judgement calls.** The scenario weights (T6 = 2/7) are a choice, and T6 is a cycle average, not an
  equilibrium.
- **One grid edge left.** R7+C's best cutoff (0.10) is the lowest value tried. The other edge values
  (R4's top-decile margins, R5's zero-income premium) only mean "price these borrowers out".
- **Competitors are our reconstruction.** Lenders 1 and 2 may use other models or pricing; Part 7 tests
  that.
