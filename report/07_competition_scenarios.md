# Lending Game — Part 7: Testing the pricing rules against competitor scenarios

How the rules tuned in Part 6 hold up on data and competitor behaviour they were not tuned on, and
which rule we submit. Everything comes from `analysis/competitor_scenarios.ipynb`; the code is in
`src/pricing_rules.py` and `src/pricing_plots.py`.

## Summary

- **We submit R4 :** It is break-even plus a margin that rises with risk (about 2 points for
  $p_{own} < 4.5\%$, about 3 points up to 11%, 17–19 points above), **without a cutoff**.
- **R4 wins the main criterion.** It has the smallest worst-case regret (€0.99M, next best €1.30M)
  and the highest weighted profit (+€1.35M on 20,000 validation loans). The bootstrap confirms its
  lead.
- **R4 and Part 6's recommendation R4+C differ only on the riskiest third of applicants** (above
  $p_{own} = 12.5\%$): R4 quotes them very high rates, R4+C refuses them.
  - When competitors lend to these borrowers, the difference is small: 3–5% in favour of R4+C, not
    significant.
  - When competitors stay away from them, R4 earns €1.5–6.6M more.
- **R4+C never loses money in any scenario, but it has the largest regret** of the good rules, because
  it gives up that market.

---

## 1. What we did

**Data.** The 20,000 validation loans, which no rule saw during tuning. The parameters of the ten
rules tuned in Part 6 are fixed; there is no re-tuning. The rules are loaded with
`load_tuned_rules(train)`, which reproduces the Part 6 profits exactly.

**Scenarios.** Sixteen ways the competitors (lenders 1 and 2) could price. All of them are break-even
on their own default probabilities plus some pricing:

| group | scenarios | competitors |
|---|---|---|
| tuning (as in Part 6) | T1–T3 | break-even + 0, 2, 4 points |
| | T4–T5 | break-even × 1.15, × 1.35 |
| | T6 | the average margins of the best-response cycle (4.25 / 3.75 points) |
| held-out: margins | H1, H3, H5 | break-even + 1, 3, 5 points |
| held-out: asymmetric | A05, A13 | lender 1 / lender 2 at (+0, +5) and (−1, +3) points |
| held-out: markup | K50 | break-even × 1.5 |
| held-out: information | IPUB | both use public information only (`p_public`), +2 points: they know less |
| | IFULL | both see all three signals (`p_full`), +2 points: they know more |
| held-out: strategic | S_R4C | both apply our rule R4+C to their own $p$ |
| | S_R4 | both apply our rule R4 to their own $p$ |

**IFULL is a stress test, not a realistic case.** In the game, each lender sees only its own signal,
so no competitor can reach `p_full`. It bounds how much better-informed competitors could hurt us.

**The strategic scenarios ask "what if the competitors price like us?"** S_R4C is the striking one:
competitors who refuse applicants above 12.5% risk (by their own estimate) leave **27.8% of
applicants with no competitor offer at all**. Under the game rules, those applicants take our offer at
any rate up to 100%.

**Scoring and decision** (as planned in [`06_pricing_plan.md`](06_pricing_plan.md)):

- Every rule against every scenario, with the simulator in expected mode. We record profit, loans,
  market share, profit per loan and the winner's-curse gap.
- **Three criteria:**
  - **minimax regret** (the main criterion): the smallest worst-case shortfall versus the best rule of
    each scenario;
  - **maximin**: the best worst-case profit;
  - **weighted profit**, with weights:
    - 1.5 for each strategic scenario;
    - 2 for H1, T2, H3 and T6 (margins of 1–4 points, where best-responding competitors end up);
    - 1 for every other scenario.
- **A paired bootstrap** (500 resamples of the applicants) to check whether the winner's lead is real.
- **Random-mode simulations** (20 draws of the preference types) to see how much luck matters.

## 2. Results

Profit on the 20,000 validation loans (the real round has 100,000 applicants, about ×5):

| rule | max regret (main) | worst case | weighted profit | scenarios where it loses money |
|---|---|---|---|---|
| **R4** | **€0.99M** | −€0.40M | **+€1.35M** | IFULL only |
| R2 | €1.30M | −€0.63M | +€1.14M | IFULL, A13 |
| R7 | €2.54M | −€0.93M | +€1.10M | T1, IFULL, A13 |
| R3 | €2.71M | −€0.98M | +€1.07M | T1, IFULL, A13 |
| R4+C | €6.62M | **+€0.06M** | +€0.84M | **none** |
| R7+C | €7.14M | +€0.03M | +€0.74M | none |
| B0 | €8.43M | −€2.91M | −€0.95M | all but IPUB |

R5, R6 and R1 come between R3 and R4+C on max regret. R4 also ranks first on the held-out scenarios
alone.

**What the scenarios show:**

- **Ordinary scenarios** (all tuning scenarios, other margins, asymmetric competitors): the Part 6
  ranking holds. R4+C and R7+C are best or near-best, and R4 is almost identical to R4+C (e.g. in T6:
  7,440 vs 7,382 loans, €175 vs €183 per loan).
- **Competitors who know more (IFULL):** every rule without a cutoff loses money, from −€0.40M (R4) to
  −€2.91M (B0). The risky borrowers we win are the ones the better-informed competitors rightly priced
  high: gap +3.7 points for R4. Only R4+C (+€274k) and R7+C (+€320k) stay profitable.
- **Competitors who know less (IPUB):** every rule does well, even B0 (+€0.37M).
- **Strategic competitors (S_R4C, S_R4):** the rules that still quote risky applicants make large
  profits.
  - In S_R4C, R4 wins 10,000 loans at an average rate of 26%. They default at 13.3%, only 0.9 point
    above our prediction, since no competitor selected them: **+€7.9M**, against +€1.3M for R4+C.
  - In S_R4, R4 makes +€2.8M against +€1.3M for R4+C.

**Bootstrap.**

- **R4 vs R2 (runner-up):** R4 is significantly better in 12 of the 16 scenarios and on weighted profit,
  +€206k (95% interval [+€79k, +€318k]).
- **R4 vs R7:** R4 is significantly better on weighted profit, +€244k ([+€116k, +€383k]).
- **R4 vs R4+C, the real decision:**
  - **Not significant in 13 of the 16 scenarios.** R4+C is ahead by €37–75k (3–5% of profit) in 11 of
    them, R4 is ahead in K50, and they tie in IPUB.
  - **Significant in three:** IFULL (R4 −€673k), S_R4C (R4 +€6.6M) and S_R4 (R4 +€1.5M).
  - **Weighted profit:** R4 ahead by +€510k ([+€386k, +€642k]).

**Random mode.**

- One draw of the preference types moves our profit by about ±€50–130k, the same size as the
  R4 / R4+C difference in ordinary scenarios.
- R4's 5th percentile is negative only in A13 (−€40k) and IFULL (−€620k).
- With enough draws, random mode averages to expected mode, as it should: 300 draws gave €1.296M vs
  €1.305M for R4 in T6.

## 3. Decision: R4

**Why R4 rather than R4+C:**

- It has the smallest worst-case regret, with a significant
  lead on the bootstrap.
- **It keeps a cheap option.** Quoting the riskiest applicants a very high rate, rather than refusing
  them, costs 3–5% (not significant) when competitors lend to them, because those quotes rarely win.
  It pays €1.5–6.6M if competitors stay away from them. Careful competitors may well do so, since
  break-even pricing loses money and the winner's curse lives in that segment.
- **R4 loses money only in IFULL**, which cannot happen in the game.
- **It is the simplest strong rule**: it uses only `p_own` and a fixed margin curve.

**R4+C would be better if** the competitors will lend to everyone at
moderate rates and we want a rule that never lost money in any scenario tested.

**Final rule (R4):**

- $r = be(p_{own}) + m(p_{own})$, with $be(p) = p/(1-p)$;
- $m(p)$ is about 1.75 points for $p_{own} < 2.75\%$, 2 points up to 4.5%, 3.2 points up to 11%,
  and 17–19 points above (a smooth increasing curve through the decile margins);
- rate floor 0.5%;
- no offer when $p_{own} \ge 0.5$ or when the rate would exceed 100%.


**For Part 8.** The rule is saved in `data/pricing/final_rule.json`, and rebuilt as a function with
`load_tuned_rules(load_pricing_data("train"))["R4"]`. It needs only `p_own` for the new applicants
(`data/predictions/applications_own.csv`).

## 4. Limitations

- **The decision rests on the strategic scenarios.** Without them, R4+C would be marginally ahead.
Their weights, like all the scenario weights, are a judgement call.
- **The competitors are our reconstruction** (our models of their default probabilities, plus assumed
  pricing). The real teams, and the LLM competitor, may behave differently from all sixteen scenarios.
