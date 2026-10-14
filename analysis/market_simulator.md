# Part 4 – Market simulator (summary)

**Goal.** Given the offers of the three lenders, simulate which lender each applicant chooses, then compute each lender's number of loans, average rate and profit, using the observed defaults of the validation set.

## 1. Borrower choice rule
Applicant $i$ has a hidden preference type $t_i \in \{1,2,3\}$. Let $r_{ik}$ be lender $k$'s offered rate ($r_{ik}=\infty$ if no offer) and $\delta = 0.02$. The **effective rate** is

$$\tilde r_{ik}(t) = r_{ik} - \delta \cdot \mathbf 1\{k = t\}$$

and the applicant takes the lender with the lowest effective rate, $\arg\min_k \tilde r_{ik}(t)$.

- The discount is only used to *compare* offers. The lender is paid the rate it actually offered.
- A competitor wins only if it is strictly more than 2 points cheaper than the preferred lender (exactly 2 points is a tie).
- If nobody makes an offer, no loan is made.
- Ties are split equally (expected mode) or broken at random (random mode).

Choice weight: $w_{ik}(t) = \dfrac{\mathbf 1\{k \in \arg\min_j \tilde r_{ij}(t)\}}{|\arg\min_j \tilde r_{ij}(t)|}$ (a row of zeros if there are no offers).

## 2. Profit
With $L = 10{,}000$ and $d_i \in \{0,1\}$ the observed default:

$$\pi_{ik} = L\,\big[(1-d_i)\,r_{ik} - d_i\big], \qquad \Pi_k = \sum_i w_{ik}\,\pi_{ik}$$

## 3. Two strategies for the unknown preference type

**A. Random draw (`mode="random"`).** Each applicant is given ONE type $\tau_i$, with an exact one-third split (the list 1,2,3,1,2,3,... is shuffled with a fixed seed). The choice rule is run once: $w_{ik} = w_{ik}(\tau_i) \in \{0,1\}$.
- Realistic: it is one possible "world", and each applicant clearly goes to one lender.
- But noisy: results change with the seed, so when comparing two pricing rules some of the difference is luck.

**B. Expected outcome over types (`mode="expected"`).** Nothing is drawn. The choice rule is run three times per applicant, once for each type, and averaged with weight 1/3:

$$\bar w_{ik} = \frac{1}{3}\sum_{t=1}^{3} w_{ik}(t)$$

Example: rates 5%, 6%, 8% give lender 1 for type 1, lender 2 for type 2, lender 1 for type 3, so $\bar w = (2/3,\ 1/3,\ 0)$.
Since profit is linear in $w$, this gives the expected profit over the type draw: $E[\Pi_k] = \sum_i \bar w_{ik}\,\pi_{ik}$.
- Deterministic and noise-free, which is ideal for the grid search in Part 6.
- Loan counts can be fractional.
- Only the type is averaged out. The defaults are still the observed ones.

We use B for comparing pricing rules, and A to see how much profit varies by chance.

## 4. Outputs (per lender $k$, using weights $w_{ik}$)
- Loans won: $n_k = \sum_i w_{ik}$; market share $n_k / N$ (of applicants) and $n_k / \sum_j n_j$ (of loans)
- Average rate: $\bar r_k = \sum_i w_{ik} r_{ik} / n_k$
- Default rate on loans won: $\bar d_k = \sum_i w_{ik} d_i / n_k$
- Total profit $\Pi_k$ and profit per loan $\Pi_k / n_k$
- Loan-level table: rates `r1, r2, r3`, `default`, weights `w1..w3`, profits `profit1..3` (plus `pref`, `chosen_lender` in random mode)


# Part 5 – Measuring the winner's curse

**The problem (adverse selection).** Our predicted default probability $\hat p_i$ uses the public variables and signal 3 only. Competitors see other signals. The applicants who end up with us are not a random sample: they are the ones where the competitors' offers were not low enough to beat ours. If a competitor's signal says an applicant is safe, it undercuts us. If its signal says the applicant is risky, it prices high and we win. So we tend to win applicants whose risk we **under-estimated**. This is the winner's curse.

**The measure.** Let $w_i$ be the probability that applicant $i$ takes our loan (the `w3` column of the simulator; 0/1 in random mode, fractional in expected mode), and $d_i$ the observed default. On the loans we win:

$$n_{won} = \sum_i w_i,\qquad \bar p = \frac{\sum_i w_i\,\hat p_i}{n_{won}},\qquad \bar d = \frac{\sum_i w_i\,d_i}{n_{won}},\qquad \text{gap} = \bar d - \bar p$$

A positive gap means we under-predict the risk of the loans we win. The same quantities are computed within risk groups $g$ (quintiles of $\hat p_i$ over all applicants), summing over $i \in g$, together with the group sizes $|g|$.

**Notes.**
- Groups are defined on all applicants, so they are identical across pricing rules and results are comparable.
- If $\hat p$ were calibrated, the gap would be about 0 on a random sample of applicants. Any gap on the loans we win is therefore due to selection.
- $\hat p$ must be out-of-sample (validation predictions from a model trained on the training set only). In-sample predictions would make the gap look artificially small.
- Small groups are noisy, so we report an approximate standard error `se_obs` to judge whether a gap is signal or noise.
- In Part 6 the function is called after each simulated pricing rule, to see how selection changes with the rule.

Why use groups? A single overall gap tells you that you under-predict risk on the loans you win, but not where. The problem may be concentrated in one kind of applicant. Groups split the applicants into risk buckets, so you can see which buckets cause it.

How they are built. The function looks at our predicted default probability p_pred for all applicants and sorts them into buckets:

groups=5 creates quintiles: five buckets with the same number of applicants (20,000 each out of 100,000). G1 holds the 20% of applicants we think are safest, and G5 the 20% we think are riskiest.
groups=[0, .02, .05, .1, .2, 1] instead uses fixed risk bands (below 2%, 2 to 5%, and so on). Bands can be more readable, but the group sizes will be uneven.

Why groups are defined on all applicants and not only the ones we win. If the groups were built from the winners only, the buckets would change every time a pricing rule changes who wins, and you couldn’t compare rules. With the groups fixed, the same G1 means the same applicants under every rule. It also lets you compute the win rate, “what share of this bucket did we win?”.

Where the winners come in. Inside each group, the statistics are computed only over the loans we win. This uses the weights w3, so an applicant we win with probability 0.5 counts as half an applicant.