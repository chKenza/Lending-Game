# Lending Game - Guidelines and Pipeline

## Project
We run a lending platform making consumer loans to individuals. We receive loan applications
from a large number of individuals and must decide whether and at which interest rate you
offer to lend money to each loan applicant. We are in competition with two other lenders who also make loan offers to these individuals. We are lender 3.

Borrowers may default
on their loan, so it is important to choose the interest rate at the right level to compensate for
the risk of default.

We have information about loan applicants to estimate their risk of default. Some of the
information about loan applicants is available to all lenders. Each lender also has its own source
of private information about loan applicants extracted from proprietary data. Because different
lenders use different proprietary data, they obtain different private signals capturing different
dimensions of loan applicants’ creditworthiness (signal1, signal 2 and signal 3). All lenders also have access to data on past
loans to train a credit scoring model.

## Work Pipeline

### Round 1
**Part 1**: Exploratory Data Analysis and Feature Engineering

**Part 2**: Predict Default Probability: Our Model + Full-Information Model + Competitor Models + Public Model

**Part 3**: Break-Even Rates: “Fair Pricing”

**Part 4**: Build Market Simulator on the Validation Set

**Part 5**: Measure Winner’s Curse

**Part 6**: Design Pricing Rule

**Part 7**: Test Our Pricing Rules Against Competitor Scenarios

**Part 8**: Score New Applications: Predict, Price, Submit

#### Round 1 results:

After round 1, we made 35M profit with 55.6% market share. We ranked 4th out of 12 groups. For round 2, we will work on making our rule better.

### Round 2

**Part 9**: Round 1 replay: Simulate the Performance of all Candidate Rules for Round 1

**Part 10**: Infer the Rules of Competitors from Round 1

**Part 11**: Improve our Rule for Round 2

**Part 12**: Score New Applications for Round 2: Predict, Price, Submit