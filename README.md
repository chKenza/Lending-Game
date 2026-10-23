# Lending Game — Ponzi & Co

As part of a finance challenge,w e run a consumer-lending platform and compete with two
other lenders for the same 100,000 loan applicants. For each applicant we submit an interest rate (or
no offer); each borrower takes the cheapest offer, adjusted for a hidden preference for one lender, and
may then default. The goal is to **maximise profit**, not market share.

All lenders see the same public variables. Each also has a
private credit signal. We are **lender 3**, so we see only `signal3`. We train on 100,000 past loans
that include all three signals and the observed default.

## Approach

The game is played in two rounds. Each step has its own notebook in [analysis/](analysis/), reusable
code in [src/](src/), and a write-up in [report/](report/).

**Round 1:** build a model, simulate the market, choose a rule.

| Part | Step | Report |
|---|---|---|
| 1 | EDA and feature engineering | [01_eda_report.md](report/01_eda_report.md) |
| 2 | Default-probability models for five information sets: ours, each competitor's, full and public | [03_models_report.md](report/03_models_report.md) |
| 3 | Break-even ("fair") rates $r^* = p/(1-p)$ | [02_fair_pricing_report.md](report/02_fair_pricing_report.md) |
| 4–5 | Market simulator and winner's-curse measurement | [04_05_market_simulator.md](report/04_05_market_simulator.md) |
| 6 | Tune 10 pricing-rule families against 6 competitor scenarios | [06_pricing_rule.md](report/06_pricing_rule.md) |
| 7 | Stress-test the rules on held-out data and 16 scenarios | [07_competition_scenarios.md](report/07_competition_scenarios.md) |
| 8 | Price the new applicants and submit | [08_submission_round_1.md](report/08_submission_round_1.md) |

**Round 2:** learn from what the competitors actually did in Round 1.

| Part | Step | Report |
|---|---|---|
| 9 | Exact replay of Round 1 using the course's diagnostic file | [09_round_1_replay.md](report/09_round_1_replay.md) |
| 10 | Infer the competitors' pricing rules from their offers | [10_competitors_rules.md](report/10_competitors_rules.md) |
| 11 | Re-tune and stress-test our rule (36 held-out scenarios) | [11_round_2_rule.md](report/11_round_2_rule.md) |
| 12 | Price the Round 2 applicants and submit | [12_submission_round_2.md](report/12_submission_round_2.md) |

## Key results

### Model
- **Spline logistic regression** on public variables plus `signal3`: validation AUC **0.736**,
  log-loss 0.291, well calibrated (slope 0.99). Gradient boosting brought no gain.
- **Income drives most of the prediction.** Zero income is a separate risk group (20% default vs 8%).
  `signal = 0` means "no signal", not a low score.
- Our signal adds +0.021 AUC over public data. The competitors' signals are worth as much as ours, and
  combining all three would add another +0.024.

### Market simulator
A rebuilt version of the game's market ([src/simulator.py](src/simulator.py)), used to score any
pricing rule on held-out past loans.
- **Offers:** competitors price at their break-even rate (from our reconstruction of their models)
  plus an assumed margin, the **scenario**.
- **Choice:** each borrower takes the lowest rate, with a 2-point discount for their preferred lender.
  Profit per €10,000 loan is $10{,}000\,[(1-d)\,r - d]$, using the **observed** default $d$.
- Profit is averaged over the three preference types, so results have no random noise. Given the
  competitors' real offers, it reproduced our Round 1 profit to the euro.

### Winner's curse
- Our model is calibrated overall, but **on the borrowers we win we under-predict default by about
  2 points**: competitors with other signals undercut us on the safe borrowers.
- Pricing at break-even therefore **loses money in every scenario**. The margin must rise with risk.

### Round 1: rule R4
$r = \dfrac{p}{1-p} + m(p)$: break-even plus a margin $m$ that depends on our default
probability $p$. The margin was tuned for each **decile of $p$** in the simulator, then smoothed into
an increasing curve. No offer if $p \ge 50\%$.
- **Result of Round 2: €35.0M profit, 55.6% market share, 4th of 12 teams** (winner: €73.3M).

### What the Round 1 replay showed
- Our replay reproduces the course's result **to the euro**. R4 was the best of our ten candidate
  rules. Quoting risky applicants instead of refusing them was worth +€25.7M.
- **20% of applicants were uncontested**: some competitor teams never made an offer, and others
  only quoted fixed rates of about 92–98%. A flat 89.5% quote to everyone would have earned €118.9M.
- **Market share was the wrong target**: €630 per loan for R4 vs about €5,850 for the high-quote rule.

### Round 2: rule R4-step|R8
- **Below 8.7% risk** (about half the applicants): break-even plus 2.25–3.25 points.
  **From 8.7%:** a flat **82%** quote. No offer above 45% risk. The model was refitted on 200k loans.
- **Smallest max regret** over 36 held-out scenarios: €15.7M, against €119M for Round 1's R4.
  It would have made **€82.3M on the real Round 1 market**.
- **Expected profit:** about €82M if competitors behave as in Round 1, €25–26M if half the
  uncontested pool disappears, and about €4M if it disappears entirely.

## Repository structure

```text
├── analysis/      # notebooks, one per part (eda, models, fair_pricing, ..., submission_round_2)
├── src/           # reusable code: data prep, models, simulator, pricing rules, submissions
├── report/        # write-up for each part + figures/
├── data/
│   ├── raw/          # PastLoans, new applications (Rounds 1–2), Round 1 diagnostic file
│   ├── processed/    # encoded train / validation / applications sets
│   ├── predictions/  # default probabilities and break-even rates
│   └── pricing/      # tuning grids and saved rules (final_rule.json, final_rule_round2.json)
└── submissions/   # Lender3_Round1_submission.csv, Lender3_Round2_submission.csv
```

## Reproducing

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
```

Run the notebooks in [analysis/](analysis/) in part order. Each one writes the files the next one
reads.
