# Lending Game — Parts 6–7: Pricing rule plan

How we design, tune and choose our pricing rule, then test it against competitor scenarios.
Code goes in `src/pricing_rules.py`; Part 6 in `analysis/pricing_rules.ipynb`, Part 7 in
`analysis/competitor_scenarios.ipynb`.

**Data.** Prediction files: `data/predictions/train_oof_selected.csv` (80k loans, out of fold), used
for tuning, and `data/predictions/validation_selected.csv` (20k loans), used for scoring. Both have
`default`, `p_own`, `p_lender1`, `p_lender2`, `p_full` and `p_public`. Rules R5–R7 also need
`signal3_missing` and `income_zero`; join them from `data/processed/*_general.csv` on `row_id`.

**Notation.** $be(p) = p/(1-p)$ is the break-even rate. All rates are decimals.

**Rules that apply to every family:**

- Rates are clipped to [floor, 100%], with a floor of 1%, so very safe borrowers are not priced near 0. maybe 2% ?? to be discussed; no offer above 100%
- No offer (`NaN`) when $p \ge 0.5$.
- No rule ever uses competitor information; it only takes `p_own` and our own features.

**Assumption to confirm:** the game is scored on total profit.

---

## Part 6: Design the pricing rule (due Friday)

### 1. Write the scenario builder first

Tuning needs competitor rates, so `make_scenario` goes in `src/pricing_rules.py` before anything
else. It returns `(r1, r2)` built from `p_lender1` and `p_lender2`.

Competitors' $p$ always comes from our reconstructed lender models (`p_lender1`, `p_lender2`, out of
fold). What we don't know is how they turn $p$ into a rate, so we tune against a **set** of pricing
scenarios rather than a single guess (step 3).

### 2. Rule families (few parameters each)

| ID | Rule | Formula | Parameter grid |
|---|---|---|---|
| **B0** | Naive baseline | $r = be(p)$ | none |
| **R1** | Fixed margin | $r = be(p) + m$ | m ∈ 0–8 pts, step 0.25 |
| **R2** | Proportional markup | $r = (1+k)\,be(p)$ | k ∈ 0–1, step 0.05 |
| **R3** | Adjusted p | $\text{logit}(p_{adj}) = \text{logit}(p) + a$, then $r = be(p_{adj}) + m$ | a ∈ 0–0.6, step 0.05; m ∈ {0, 1, 2} pts |
| **R4** | Best rate per risk bin | decile bins of `p_own`. Per bin, $r = be(p) + m_b$, with $m_b$ maximising $\Pi_b = \sum_{i\in b} w_{3,i}\,[(1-d_i)\,r_i - d_i]$, then smoothed into an increasing curve $m(p)$ | $m_b$ ∈ −2 to +20 pts, step 0.25 |
| **R5** | Margin that grows with uncertainty | $r = be(p) + m_0 + m_1\,\mathbb 1[\text{signal3 missing}] + m_2\,\mathbb 1[\text{income}=0]$ | $m_0$, $m_1$ ∈ 0–4 pts, $m_2$ ∈ 0–8 pts, step 0.5 |
| **R6** | Information advantage | $r = be(p_{own}) + m_0 + m_1\,(p_{pub} - p_{own})$ | $m_0$ ∈ 0–8 pts, step 0.25; $m_1$ ∈ −1 to +1, step 0.1 |
| **R7** | Undercut the predicted competitor rate | $r = \max\big(be(p_{adj}),\ \hat r_{comp} - \delta\big)$, with $\hat r_{comp}$ the predicted cheaper competitor rate (see notes) | $\delta$ ∈ −8 to +3 pts, step 0.25; quantile q ∈ {0.25, 0.5}; $a$ (for $p_{adj}$) ∈ {0, 0.2, …, 1} |
| **+C** | No-offer cutoff (add-on to the best 1–2 rules) | no offer if $p > \bar p$ | $\bar p$ ∈ {0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5} |

Ideas to be explored by Karin (maybe): exp(), p**1.5
Note: for R5, more important to look at low signal 3 than to look at signal 3 missing

**Notes on R4:**

- Optimising each bin separately is valid. A bin's rates only affect the applicants in that bin,
  so its profit doesn't depend on the other bins.
- It has the most parameters, so it is the most exposed to overfitting. Smooth it with an isotonic
  fit or a straight line in $p$, and judge it only on validation.

**Notes on R6:**

- $p_{pub}$ is `p_public`, already in `train_oof_selected.csv` and `validation_selected.csv`. For
  the new applicants it still has to be computed (the public model run on the applications) before
  Part 8.
- Logic: competitors don't see `signal3`, so on what our signal reveals their estimates fall back
  toward $p_{pub}$. Where $p_{own} < p_{pub}$ their rates are likely above ours, leaving room for
  margin.
- $m_1$ may be negative: the grid lets the data decide whether the effect is real.
- Overlaps with R5 (missing `signal3` gives $p_{own} \approx p_{pub}$). If time is short, R6 can
  replace R5.

**Notes on R7:**

- Predict the competitors' **default probability**, not their rate, because the rate depends on
  the unknown pricing scenario: on `train_oof`, regress `p_lender1` and `p_lender2` on our own
  information (`p_own`, `p_public`, public features, `signal3`). Turn the predicted $p$ into the
  competitors' **break-even** rates, the same in every scenario: their true pricing is unknown, and
  using each scenario's true pricing would leak information we will not have. An assumed margin
  $\hat m$ on top would only shift $\hat r_{comp}$ like changing $\delta$ by $-\hat m$, so it is
  carried by $\delta$ (hence $\delta$'s wide range).
- Target the **cheaper** competitor: $\hat r_{comp}$ is the q-quantile of $\min(\hat r_1, \hat r_2)$,
  using the regression residuals. The expected minimum is lower than the minimum of the
  expectations.
- The $be(p_{adj})$ floor is essential: we win exactly where competitors priced higher than
  predicted, i.e. where they saw more risk, so the winner's curse is still there.
- $\delta = -2$ pts (just under) is the variant that targets our loyal third: applicants who prefer
  us stay if our rate is less than 2 pts above the cheapest competitor.
- Overlaps with R4, which learns "what it takes to win" from `p_own` alone. If R7 doesn't beat R4,
  the extra information isn't worth the extra parameters.
- The most work of all families; do it last.

### 3. Tuning, on `train_oof` only

**Why several scenarios.** Real competitor offers are unknown, and the best parameter depends on
them. Varying only the margin is not enough: an additive margin makes competitors relatively
expensive on safe borrowers, while a proportional markup makes them cheap on safe borrowers and
expensive on risky ones. Tuning on one shape fits our rule to that shape.

**3a. Sensitivity check (do first, minutes).** For R1, find our best margin against each competitor
margin in {0, 1, 2, 3, 5} pts and plot the best-response curve $m^*(m_{comp})$. If it is flat, the
choice of scenarios matters little; if it moves a lot, the tuning set below is essential.

**3b. Equilibrium scenario.** Competitors will
not price at break-even (Part 3 shows it loses money). Compute the margins the three lenders end up
with when each best-responds to the others:

1. Start all three lenders at $be(p_k) + 2$ pts, each with its own $p$ (`p_lender1`, `p_lender2`,
   `p_own`).
2. In turn, set each lender's margin to the one (grid 0–8 pts, step 0.25) that maximises its own
   profit given the other two.
3. Repeat until no margin changes by more than 0.25 pt (cap at 20 rounds).

The resulting competitor margins $(m_1^{eq}, m_2^{eq})$ define the equilibrium scenario. If the
margins cycle instead of converging, T6 uses their average over rounds 3–20 (rounds 1–2 are a
warm-up), rounded to the 0.25-point grid.

**3c. Tuning set.**

| Scenario | Competitor rates | Weight | Why |
|---|---|---|---|
| T1–T3 | $be(p_k) + m$, m ∈ {0, 2, 4} pts | 1/7 each | low, central, high margin |
| T4–T5 | $(1+k)\,be(p_k)$, k ∈ {0.15, 0.35} | 1/7 each | the other shape |
| T6 | equilibrium margins from 3b | 2/7 | the most plausible case |

Everything else is held out for Part 7, so that Part 7 tests scenarios the rules were not tuned on.

**3d. Grid search.** For each family and each parameter value:

1. Run `simulate_market(mode="expected")` against each of T1–T6 and record our profit.
2. **Weighted profit** = $\sum_s w_s\,\Pi_s$ with the weights above.
3. **Max regret** = $\max_s\,[\max_\theta \Pi_s(\theta) - \Pi_s]$, where $\max_\theta \Pi_s(\theta)$
   is the best profit this family reaches in scenario $s$.

Pick the parameters with the highest weighted profit. Check the smallest-max-regret choice too: if
the two agree, keep it; if not, report both and take the one with the higher margin, since a margin
that is too low is the costlier mistake (losses on won loans) while one that is too high only loses
volume.

**R4 with several scenarios.** Each bin is still optimised separately, using the weighted profit
across T1–T6 for that bin.

**3e. Plots.** For each family, plot profit against the parameter, one line per scenario plus the
weighted line. If the weighted curve is flat around the optimum, prefer the simpler or more cautious
value.

**Output:** one tuned rule per family (B0, R1–R7, plus a cutoff version of the best one or two), and
the equilibrium margins.

### 4. Part 6 deliverable

- The tuned parameters and the equilibrium margins.
- The best-response curve from 3a.
- Profit curves for each family on `train_oof` (per scenario and weighted).
- `winners_curse` table (by quintile) for the tuned rules.

---

## Part 7: Test against competitor scenarios (Friday/Saturday)

### 1. Scenario grid (competitor rates computed on validation)

The six tuning scenarios T1–T6, plus held-out scenarios never used in tuning:

| Group | Held-out scenarios | What it tests |
|---|---|---|
| **Margin** | symmetric m ∈ {1, 3, 5} pts | values between and beyond the tuning grid |
| **Asymmetric margin** | (0, 5) and (−1, 3) pts | competitors behaving differently from each other |
| **Proportional markup** | k = 0.5 | a markup beyond the tuning grid |
| **Strategic** | both competitors apply **our** final tuned rule to **their own** $p$ | our rule against itself (needs the rule first, so it can't be in tuning) |
| **Information** | competitors use `p_public` (they know less) or `p_full` (they know more), +2 pt margin | competitors knowing less or more than assumed |

Report the tuning and held-out scenarios separately: a rule that holds up on the held-out ones
generalises.

### 2. Scoring

Run every tuned rule (fixed parameters, no re-tuning) against every scenario on validation, using
`simulate_market(mode="expected")`.

For each cell, record:

- total profit;
- profit per loan;
- number of loans and market share;
- winner's-curse gap (from `winners_curse`).

### 3. Decision

**Tables:**

- **Profit table:** rules × scenarios.
- **Regret table:** regret(rule, scenario) = best profit in that scenario − this rule's profit.

**Three criteria:**

- **Maximin:** best worst-case profit.
- **Minimax regret (main criterion):** smallest worst-case regret.
- **Weighted average:** highest weighted profit. Put more weight on the strategic and +1–3 pt
  scenarios, and write the weights down.

**Is the winner really better?**

- **Paired bootstrap:** resample the validation rows 500 times. Each time, compute the profit
  difference between the top 2–3 rules in each scenario (same resampled rows for each rule).
  Report the 95% interval.
- **Random mode:** run about 20 seeds and report the 5th percentile of profit as well as the mean.

**Tie-break.** If the bootstrap interval for the difference contains 0, take the simpler rule
(fewer parameters).

### 4. Part 7 deliverable

- Heatmaps of profit and regret.
- Bootstrap intervals for the top rules.
- One paragraph justifying the choice.
- Final rule and parameters, handed to Part 8 (submission).

---

## Code layout (`src/pricing_rules.py`)

```text
be(p) · clip_rates(r, floor, cutoff)
rule_b0, rule_fixed_margin, rule_markup, rule_adj_p, rule_bins, rule_uncertainty,
rule_info_advantage, rule_undercut · predict_competitor_p(data, features) -> (p1_hat, p2_hat)
make_scenario(name, data, **params) -> (r1, r2)
evaluate(rule, params, scenario, data) -> one row: profit, n_loans, share, gap
equilibrium_margins(data, grid, start=0.02, tol=0.0025, max_rounds=10) -> (m1, m2, m3)
grid_search(rule, grid, scenarios, weights, data) -> DataFrame: profit per scenario, weighted, max regret
bootstrap_diff(rule_a, rule_b, scenario, data, n=500)
```

The notebooks only call these functions.

## Order of work

1. `make_scenario`, `evaluate`, B0, R1 and R2, the sensitivity check (3a) and `equilibrium_margins` (3b).
2. R3, R4, R6, R5 and the cutoff, then R7 last; tuning on T1–T6 and the Part 6
write-up. If time runs out, drop R7 or R5 and keep R6.
3. scenario grid, regret, bootstrap, final choice.

## Choices to confirm

- **Tuning set** T1–T6 and its weights (equilibrium 2/7, the others 1/7 each). These are choices,
  not facts.
- **Weights** for the weighted-average criterion in Part 7.
- **Rate floor** of 1%.
