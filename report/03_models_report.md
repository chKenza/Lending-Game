# Lending Game — Part 2: Default probability models

What we learned from predicting default under five information sets. Everything comes from
`analysis/models.ipynb`; the code is in `src/models.py`, `src/evaluation.py` and `src/model_plots.py`.

## Summary

- **Spline logistic regression is selected for all five information sets.** Logistic regression,
  spline logistic and gradient boosting are within 0.001 log-loss of each other; the spline is best
  or tied-best everywhere, well calibrated, and keeps an explicit formula.
- **Our model (public + `signal3`)**: validation AUC 0.736, log-loss 0.2909, calibration slope 0.99,
  mean prediction = default rate (9.95%). No recalibration needed.
- **Income drives most of the prediction**; our signal adds a real but modest +0.021 AUC.
- **Lenders 1 and 2 are as well informed as we are**, with different signals. The full-information
  model is clearly better than ours (+0.024 AUC).
- **The winner's curse is already visible**: where our model is the most optimistic of the three
  lenders, we predict 8.0% default and 10.1% actually default. Our predictions are an input to
  pricing, not the price.

## 1. Setup

| information set | variables | role |
|---|---|---|
| `own` | public + `signal3` | our model, the only one usable on new applicants |
| `full` | public + signals 1–3 | benchmark, best estimate of true risk |
| `lender1` / `lender2` | public + `signal1` / `signal2` | our reconstruction of the competitors' models |
| `public` | public only | baseline, measures what a signal is worth |

Part 1's prep functions and 80/20 stratified split (seed 42). Models are compared on 5-fold
out-of-fold predictions of the 80,000 training loans, selected on **log-loss** (we price on the level
of p, not just the ranking), then refitted on train and checked once on the 20,000 validation loans.

## 2. Results

Validation set, selected model (spline logistic):

| information set | AUC | log-loss | log-loss skill | calibration slope |
|---|---|---|---|---|
| own | 0.736 | 0.2909 | 0.102 | 0.99 |
| full | 0.760 | 0.2824 | 0.128 | 1.01 |
| lender1 | 0.733 | 0.2912 | 0.101 | 0.99 |
| lender2 | 0.733 | 0.2915 | 0.100 | 0.99 |
| public | 0.715 | 0.2961 | 0.086 | 0.97 |

**The full-information model is clearly the best, the three single-signal models (ours,
lender 1's and lender 2's) are more or less equivalent, and the public-only model comes last.**

Log-loss skill = share of log-loss removed compared with predicting the average default rate for
everyone. Out-of-fold results are the same within noise.

**Model choice barely matters.** The spline beats plain logistic regression significantly for `own`
and `full` (paired bootstrap on validation), thanks to the bend in the signal effect and a more
sensible income curve at low incomes. It is statistically tied with gradient boosting. The data are
close to additive in the log-odds, so trees bring nothing extra.

Selected model:

$$\text{logit}\,P(\text{default}) = \beta_0 + f_{inc}(\log(1+\text{income})) + f_3(\text{signal}_3)
+ \gamma_0\,\mathbb{1}[\text{income}=0] + \gamma_3\,\mathbb{1}[\text{signal}_3\ \text{missing}]
+ \boldsymbol{\delta}'\,\text{employment} + \delta_u\,\mathbb{1}[\text{unemployed}, \text{income}=0]
+ \alpha_F\,\mathbb{1}[F] + \alpha_m\,\text{married}$$

$f_{inc}$, $f_3$: cubic splines, 6 knots at quantiles; mild L2 penalty ($C=1$, standardized inputs).

## 3. What predicts default

- **Income, by far.** Shuffling it worsens log-loss by 28–35%, ten times more than any other
  variable. Predicted risk falls from ~20% around €3k a year to ~10% at €9k and below 1% above €60k.
  Having **no income** roughly triples the odds of default.
- **The private signal comes second** (~3% log-loss increase when shuffled). Higher = safer: risk
  falls from ~18% at low values to ~6% at high values, steeply at first and flatter above ~0.5.
- **A missing signal means average risk** (~10%), not high risk: the model treats it like a signal
  of about 0.47. This confirms the EDA's "not scored" reading.
- **Employment** matters given income: retired applicants and students are *less* risky than
  employed ones with the same income. Their high raw default rates come from low or zero income.
- **Sex and marital status carry almost nothing** once income and employment are known.
- **The three signals behave identically** and are **complementary**: each keeps its importance in
  the full model, which is why combining them helps so much.

## 4. Value of information (validation, paired bootstrap)

| comparison | AUC difference | 95% CI |
|---|---|---|
| own vs public | +0.021 | [0.016, 0.026] |
| full vs own | +0.024 | [0.019, 0.029] |
| own vs lender 1 | +0.003 | [−0.004, 0.009] |
| own vs lender 2 | +0.003 | [−0.003, 0.010] |

The competitors' two signals together add about as much again as our own signal adds over public
data. We see a lot less than the market as a whole.

## 5. Implications for pricing (Parts 3–7)

- **The lenders disagree on individual applicants.** Our predictions correlate at only 0.82–0.83 with
  lender 1's and lender 2's, and 0.87 with the full-information model's.
- **Winner's curse.** For the third of validation borrowers where we are the most optimistic lender,
  we predict 8.0% but 10.1% default, an under-prediction of about 2.1 points. The model is calibrated
  overall; the bias comes from **selection**: the borrowers we would win are those whose risk we
  under-estimated. Pricing at our break-even rate $p/(1-p)$ would lose money on the loans we win.
  Part 5 measures this with the simulator; Part 6 must correct it.
- **Very high incomes get p ≈ 0**, based on few observations (no defaults above €125k). Pricing should
  keep a floor rather than go to zero.

## 6. Deliverables (`data/predictions/`)

| file | content |
|---|---|
| `validation_selected.csv` | `default`, `p_own`, `p_full`, `p_lender1`, `p_lender2`, `p_public`; models fitted on train only. **Main input for Parts 3–7** |
| `train_oof_selected.csv` | the same for the 80,000 training loans, out of fold (for grid searches without touching validation) |
| `validation.csv`, `train_oof.csv` | all 15 models (`<info set>__<model>`) |
| `applications_own.csv` | `p_own` for the 100,000 new applicants; mean 9.96%, same distribution as validation |

The application predictions come from the own model refitted on train + validation (all 100,000 past
loans). Its predictions correlate at 0.9996 with the train-only model's.

## 7. Limitations

The lender 1 and 2 models are our reconstruction; the real competitors may use other models or
shade their estimates (Part 7 scenarios).

