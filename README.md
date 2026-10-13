# Lending Game
Introduction to finance project

The repository has the following structure:

```text
Lending-Game/
├── README.md
├── requirements.txt
├── .gitignore
│
├── analysis/                           # All actual deliverables go here: simulations, figures, explanations
│   ├── eda.ipynb                       # Part 1: EDA + Feature Engineering
│   ├── models.ipynb                    # Part 2: models: Our Model + Full-Information Model + Competitor Models
│   ├── fair_pricing.ipynb              # Part 3: Break Even Rates
│   ├── market_simulator.ipynb          # Part 4 + 5: Market Simulator + winner's curse
│   ├── pricing_rules.ipynb             # Part 6: Design Pricing Rule
│   ├── competitor_scenarios.ipynb      # Part 7: Test Our Pricing Rules Against Competitor Scenarios
│   └── submission.ipynb                # Part 8: make submission
│
├── src/                                # reusable functions only
│   ├── __init__.py
│   ├── config.py                       # paths, seed, ect
│   ├── data_prep.py                    # Part 1: prep function, stratified split
│   ├── models.py                       # Part 2: the four information-set models, out-of-fold predictions
│   ├── evaluation.py                   # Part 2: AUC, log-loss, calibration
│   ├── fair_pricing.py                 # Part 3: break-even rate, competitor_offers
│   ├── simulator.py                    # Part 4 + 5: Market Simulator + winner's curse
│   └── pricing_rules.py                # Parts 6–7: baseline, candidate rules, grid search
│
├── data/
│   ├── raw/                            # PastLoans.csv, NewApplications_Lender3_Round1.csv
│   ├── processed/                      # train / validation sets from Part 1
│   └── predictions/                    # default probabilities from Part 2
│
└── submissions/                        # final id + rate CSV
```
