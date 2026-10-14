# Lending Game — Part 1: EDA findings

What the data looks like, and what it implies for encoding and transforming the variables.
Everything comes from `analysis/eda.ipynb`. Section 3 lists the recommendations; section 5 records what the team chose and implemented. Fuller reasoning: [`eda_report_detailed.md`](eda_report_detailed.md).

We are **lender 3**, so `signal3` is our private signal; `signal1` and `signal2` appear only in the
historical file.

## Summary

- **Income dominates** (AUC 0.710). Logging adds no information but makes the relationship linear
  in the log-odds (R² 0.713 → 0.965). → **log-transform**.
- **`income == 0` is a structural zero**: 17.0% of applicants, students and unemployed only,
  20.19% vs 7.84% default, and it survives inside both groups. → **add a flag**.
- **The employment ranking reverses** once income-zero rows are removed (rank correlation −0.60).
  → **one-hot, not ordinal**.
- **~21% of every signal is exactly `0`, and it means "no signal received"**, not a low score:
  within every employment category the zero group defaults at that category's base rate.
  → **don't encode it as the number `0`**; exact coding is the team's call.
- **The three signals are interchangeable and not redundant** (AUC 0.5429 / 0.5425 / 0.5446;
  mutual correlation ≈ 0.18) **but all three track employment strongly** (η ≈ 0.40). Their
  apparent −0.16 correlation with `log1p(income)` is a composition effect of the zero-income
  block, not an income relationship.
- **The new applicants are the same population** on every input we can test.

---

## 1. Data overview

| | PastLoans | NewApplications_Lender3_Round1 |
|---|---|---|
| Rows × columns | 100,000 × 8 | 100,000 × 6 |
| `id` | absent | present, 1…100,000, unique |
| Public variables | `sex`, `employment`, `married`, `income` | identical |
| Signals | `signal1`, `signal2`, `signal3` | `signal3` only |
| `default` | present | absent |
| Missing values | none | none |

- **Default rate 9.95%** (9,947 / 100,000), ≈ 1:9.
- **Categorical coding is identical in both files** — same four employment categories, same
  spelling, no stray levels. An encoder fitted on PastLoans transfers unchanged.
- **PastLoans has no id**, so past loans cannot be linked to new applicants.
- **68 duplicated rows**, all inside the all-signals-zero subset — coincidental collisions, not
  repeated records (see 3.7).

**Same population:** category shares match within 0.24 pp; KS = 0.0027 (income) and 0.0036
(`signal3`); no test rejects at 5%. This covers the *inputs* only — the new file has no outcomes,
so the default mechanism itself cannot be checked.

![Past vs new](figures/fig17_past_vs_new.png)

---

## 2. Key findings

### 2.1 Income dominates, and the log scale is what makes it linear

AUC 0.710, far ahead of everything else. `AUC(income) = AUC(log1p(income)) = 0.7095` exactly — AUC
is invariant to monotone transforms, so logging adds nothing. What it changes is shape:

| x-axis | R² of a linear fit to the bin log-odds |
|---|---|
| raw `income` | 0.713 |
| `log(income)` | **0.965** |

*(20 quantile bins on the 82,965 rows with `income > 0`; the zeros are excluded because `log(0)`
is undefined.)*

![Univariate AUC](figures/fig14_univariate_auc.png)
![Income vs default](figures/fig12_default_by_income.png)

### 2.2 `income == 0` is structural and carries large independent risk

17.0% of applicants, occurring **only** among students (62.8% of them) and the unemployed (48.0%),
**never** for employed or retired. Default rate **20.19%** vs **7.84%**. Held within employment:

| group | `income == 0` | `income > 0` | difference |
|---|---|---|---|
| student | 16.96% (n=10,169) | 5.90% (n=6,014) | **+11.06 pp** (z = 23.0) |
| unemployed | 24.98% (n=6,866) | 6.98% (n=7,433) | **+18.00 pp** (z = 30.0) |

It is also worse than the lowest *positive* income quartile of the same group (16.96% vs 12.83%
for students; 24.98% vs 14.79% for the unemployed) — a discontinuous jump, not low income.

![income == 0 within employment](figures/fig11_income_zero_within_employment.png)

### 2.3 The employment risk ordering reverses once the income-zero rows are removed

| | safest → riskiest |
|---|---|
| all 100,000 | employed 7.93% → retired 8.95% → student 12.85% → unemployed 15.62% |
| the 82,965 with `income > 0` | **student 5.90% → unemployed 6.98% → employed 7.93% → retired 8.95%** |

Rank correlation between the two orderings: **−0.60**.

This is **confounding**: all 17,035 income-zero rows sit in just two of the four categories, so the
aggregate ranking partly measures "has no income" under an employment label. Employed and retired
rates are identical in both rows because neither category contains an income-zero applicant.

![Employment ordering](figures/fig10_employment_ordering.png)

### 2.4 `signal = 0` means "no signal received", not a low score

| check | result |
|---|---|
| Separate atom? | ~21,000 at exactly 0 vs ~325 per 0.005 bin above (~60×); positive values run down to 1e-5, so no gap |
| Lenders' zeros coincide? | Barely — pairwise φ 0.022–0.034 |
| Random? | No — P(zero) is 36.5% retired, 23.0% unemployed, 21.2% employed, 7.9% student |
| **Risk of the zero group?** | **Population average, within every employment category** |

| employment | group base rate | `signal3 = 0` rate (95% CI) | lowest positive decile |
|---|---|---|---|
| employed | 7.93% | 8.28% [7.80, 8.78] | 14.08% |
| retired | 8.95% | 9.04% [8.21, 9.94] | 17.39% |
| student | 12.85% | 13.30% [11.54, 15.27] | 21.65% |
| unemployed | 15.62% | 14.28% [13.12, 15.52] | 26.50% |

Every base rate falls inside its zero group's interval, while the lowest positive decile runs
14–27%. A clipped low score would put the zero group at the *risky* end; it sits at the average.

![Signal zero spike](figures/fig05_signal_zero_spike.png)
![Default by signal bins](figures/fig08_default_by_signal_bins.png)

### 2.5 The signals are interchangeable, non-redundant, and tied to employment

- Near-identical: AUC 0.5429 / 0.5425 / **0.5446**; zero shares 20.8% / 20.9% / 21.1%.
- Mutual correlation 0.174–0.179 (all rows) / 0.197–0.200 (the 50,500 rows where all three are
  present) — far from redundant.
- Unrelated to sex (η ≈ 0.01); weakly related to `married` (≈ −0.12); **strongly related to
  employment (η ≈ 0.40**, rising to 0.434 on positive values only, so the zeros do not drive it).
  Mean positive signal: 0.210 retired, 0.524 employed/unemployed, 0.687 student.

So employment and the signals carry overlapping information — worth knowing before treating them
as independent inputs.

**Income: the figure depends on the scale** — Pearson −0.02 against raw income, Spearman −0.075,
but −0.16 against `log1p(income)`. Among the 82,965 rows with `income > 0` the correlation flips
sign to about +0.06, so the negative `log1p` figure comes entirely from the zero-income block,
which has a much higher mean signal (0.538 vs 0.390) and is 59.7% students against 16.2% of all
rows. **Interpretation (ours):** that is a composition effect rather than an income–signal
relationship, and a further reason to pair any log transform with an `income == 0` flag
(3.1, 3.2); full decomposition in [the detailed report](eda_report_detailed.md), §7.2.

![Correlations](figures/fig13_correlations.png)

### 2.6 Smaller points

- **`sex` and `married` are weak but real**: female 10.67% vs male 9.41%; unmarried 10.79% vs
  married 9.41% (AUC 0.517 and 0.518). The marriage gap is near-identical for both sexes.
- **The signal→default curve bends**: steep from the lowest positive decile (16.96%) to about the
  median, then flat; the wiggles above have overlapping intervals.
- **The upper income tail is near-separable**: the highest-earning defaulter made €123,814; above
  €125k there are 1,646 rows and zero defaults.

---

## 3. Recommendations

### 3.1 Log-transform income — recommended
Raw income has a skew of 11.94; a linear fit to the bin log-odds improves from R² 0.713 to 0.965. AUC is
unchanged either way, so this is about shape, not information — it matters for linear models and
not for tree-based ones, which are invariant to monotone transforms. Pair with 3.2: `log1p` alone
represents the zeros badly, parking 17,035 rows at exactly 0.

### 3.2 Add an `income == 0` flag — strongly recommended
17.0% of rows; 20.19% vs 7.84%; survives within students (+11.06 pp) and the unemployed
(+18.00 pp); worse than the lowest positive income quartile; AUC 0.5974 standalone — second only
to income itself.

### 3.3 Do not encode the signal zeros as the number `0` — recommended, coding left open
Like-for-like on the **same 100,000 rows**, only the zero cells recoded:

| signal | zeros as `0` | zeros as median present | gain | 95% bootstrap CI |
|---|---|---|---|---|
| signal1 | 0.5429 | 0.5670 | +0.0240 | [+0.0195, +0.0288] |
| signal2 | 0.5425 | 0.5631 | +0.0206 | [+0.0160, +0.0253] |
| signal3 | 0.5446 | 0.5663 | +0.0217 | [+0.0173, +0.0264] |

All intervals exclude zero. The median is an arbitrary stand-in, so this measures the cost of the
**worst** coding, not the gain from the best. Options to compare:
1. a `signal_missing` indicator plus the signal imputed to a neutral value,
2. imputation alone,
3. leaving the `0` in place for a tree model to split off by itself,
4. a binned signal with "missing" as its own level.

Whatever is chosen must be applied identically to the new applications, where `signal3` is 0 for
20.94% of applicants.

### 3.4 One-hot encode employment, not ordinal — recommended
The ordering reverses (rank correlation −0.60), so no stable ranking exists for an ordinal code to
represent. Four levels, three columns.
**Open question:** does an explicit `employment × income==0` interaction add anything beyond
one-hot employment plus the flag from 3.2? Income-zero occurs in only two of the four categories,
so those two features may already span the pattern.

### 3.5 Income top-code flag — recommended against
Only 71 rows sit at the €1M cap (0.071%) and **all 71 repaid**, so the flag is perfectly separating
on a very thin slice. With n = 71 the 95% CI on their default rate still reaches **5.13%**, so
"these never default" is not established. The log transform already places them at the top of the
scale. A broader `income ≥ €100k` flag (2,284 rows, 2 defaults) is less degenerate if the team
wants one.

### 3.6 Other interactions — low priority
Dividing each `signal3 × employment` curve by its group's base rate makes the four nearly coincide
(s1/s5 ratios 1.91–2.69), so the signal shifts risk by roughly a constant factor rather than a
constant number of points. `sex × married` gaps are parallel. **Only `employment × income==0`
looks like a real interaction.**

### 3.7 Do not de-duplicate — recommended
All duplicates sit in the all-signals-zero subset, where 1,258 rows occupy only 1,185 distinct
`(sex, employment, married, income)` profiles, making collisions unavoidable; there are zero
duplicates among rows with any positive signal. Dropping them would discard real borrowers and
bias the sample towards profiles that happen not to collide.

---

## 4. Open questions

1. **What does `signal = 0` actually mean?** The evidence strongly indicates "no signal received",
   but the brief does not say. Worth confirming with the lecturer, since it determines the encoding.
2. **Does an explicit `employment × income==0` interaction beat one-hot employment plus an
   income-zero flag?** (3.4)
3. **Does the near-separation above €125,000 generalise**, or is it a small-sample artefact?
4. **Are the duplicate rows acceptable to keep?** We think yes (3.7), but it is a team call.
5. **Is the default mechanism unchanged** in the new file? Unverifiable without outcomes.

---

## 5. Transformations

What was implemented from section 3, by which function and where. The functions are in
[`../src/data_prep.py`](../src/data_prep.py) and are run by section 9 of `analysis/eda.ipynb`.
Everything here is **stateless** — nothing is learned from the data — so the past loans and the
new applications go through the same functions and come out with columns that mean the same thing.
The four `employment` categories are hardcoded for that reason.

### Columns

`general_prep_data` — the encodings every model uses. Replaces `sex` and `employment`; keeps
`row_id`/`id`, the raw `income`, the raw signals and `default` as they are.

| column | what it is |
|---|---|
| `sex_F` | 1 if `sex == "F"`, 0 if `"M"` |
| `married` | unchanged 0/1 |
| `employment_employed`, `employment_retired`, `employment_student`, `employment_unemployed` | one-hot over the four fixed categories (3.4). All four kept; a model can drop a reference level itself |
| `income_zero` | 1 if `income == 0` (3.2) |
| `signal1_missing`, `signal2_missing`, `signal3_missing` | 1 if that signal is exactly 0, i.e. no signal received (3.3). One per signal present in the file |

`logreg_prep_data` — calls `general_prep_data`, then adds:

| column | what it is |
|---|---|
| `income_log1p` | `log1p(income)`, defined at the 17% of rows where income is 0 (3.1) |
| `income_log1p_sq` | `log1p(income)²`. The log-odds are close to linear in `log1p(income)` but still bend (risk falls faster at high incomes); the square lets a linear model follow that. Cross-validated log-loss of a logistic model on public variables + `signal3`: 0.29218 without it, 0.29137 with it. A squared or kinked `signal3` term gained much less and was not added |
| `unemployed_x_income_zero` | `employment_unemployed × income_zero`. The only interaction created: zero income occurs only among students and the unemployed, so adding `student × income_zero` as well would make `income_zero` redundant (3.4) |

`boosting_prep_data` — calls `general_prep_data` and adds nothing for now. A tree is invariant to
monotone transforms and can combine the one-hot columns with `income_zero` by itself.

Not implemented: no income top-code flag (3.5), no de-duplication (3.7).

### The split

`split_data` splits the 100,000 past loans 80/20, stratified on `default`, with
`random_state = SEED` (42, from `src/config.py`). It is called once on each of the three versions;
since the input rows, order and seed are identical, all three put the same 80,000 `row_id`s in
train and the same 20,000 in validation, which the notebook asserts. Default rate: 9.947% overall,
9.948% in train, 9.945% in validation.

`row_id` is added to `PastLoans` first, numbered `0…99,999` in original CSV row order — the file
has no `id` of its own. The new applications keep their own `id` and are not split: they have no
outcome.

### Output files

In `data/processed/`, written by section 9 of the notebook:

| | general | logreg | boosting |
|---|---|---|---|
| train (80,000 rows) | `train_general.csv` | `train_logreg.csv` | `train_boosting.csv` |
| validation (20,000 rows) | `validation_general.csv` | `validation_logreg.csv` | `validation_boosting.csv` |
| applications (100,000 rows) | `applications_general.csv` | `applications_logreg.csv` | `applications_boosting.csv` |

16 columns per past-loans file (19 for logreg) and 11 per applications file (14 for logreg) — the
applications lack `default`, `signal1`, `signal2` and the two signal flags. The files in
`data/raw/` are read-only and were only ever read.
