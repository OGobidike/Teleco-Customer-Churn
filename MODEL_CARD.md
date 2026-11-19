# Model card: telco churn, v2.0.0

## What it is

A logistic regression that estimates each telecom customer's chance of leaving, paired with a
rule that turns that chance into a decision: send a retention offer when the expected gain is
larger than the offer's cost. It is a scikit-learn pipeline (feature preparation and model in
one object), saved with skops and served by a FastAPI app.

## What it is for

- Ranking customers by churn risk so a retention team can decide whom to contact.
- Teaching: the repository shows a leak-proof pipeline, an honest evaluation, and an audit
  that measures what common mistakes do to a score.

## What it is not for

- Decisions about real customers of a real company. It was trained on one public IBM sample
  with no dates, so it has never been tested on *future* customers, which is what a real
  deployment needs.
- Explaining *why* people leave. The drivers it leans on are associations in this dataset,
  not causes. A month-to-month contract goes with churn; that alone does not show that
  changing the contract would keep anyone.
- Any decision that harms a customer (pricing up, refusing service). The cost rule assumes
  the only action is a benefit: an offer.

## Data

- [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (Apache-2.0),
  IBM's sample dataset of 7,043 customers, 21 columns, 26.5% churned. Pinned by SHA-256
  `16320c9c…3e055e91` and downloaded at run time, never committed.
- Cleaning changes two values: the 11 blank `TotalCharges` (all brand-new customers, tenure 0)
  become $0, and the target becomes 1/0. `SeniorCitizen` is relabeled from 0/1 to No/Yes.
- Split once: 5,634 training rows, 1,409 test rows, stratified, seed 42.

## Inputs

Tenure, monthly and total charges, two derived features (number of internet add-ons, average
monthly spend), and the account's services, contract, billing and payment method, plus
whether the customer is a senior citizen, has a partner, or has dependents.

**Gender is not an input.** Measured on the training rows, adding it changed cross-validated
PR-AUC by −0.0002, well inside the fold-to-fold spread of 0.0014 (recorded as
`gender_check` in [`results/metrics.json`](results/metrics.json)), and who gets an offer should
not depend on it. It is still used to check the model (below).

## How it was chosen

Four candidates (a baseline that guesses the churn rate, logistic regression, gradient boosting,
and a small TensorFlow network) were compared by 5-fold cross-validation on the training rows,
with tuning nested inside each fold. The rule, fixed before any results: the best mean PR-AUC
wins, unless a simpler model is within one standard error of it. Logistic regression won by that
rule. The rule is blunt: gradient boosting was ahead in 5 of 5 folds (by 0.006 on average) and by
0.028 on the test rows. A rule on the fold-by-fold difference would have chosen it. The full comparison is in [`results/metrics.json`](results/metrics.json).

One footnote, for honesty: the procedure was run twice. After the first run, gender was removed
from the inputs (decided on training rows, as above) and everything was run again. Both runs
scored the test rows at the end, so the test rows have been looked at twice, not once.

## How it does (test rows, scored after the choice)

| Score | Value | 95% bootstrap interval |
|---|---|---|
| PR-AUC (main score) | 0.634 | 0.577–0.686 |
| ROC-AUC | 0.842 | 0.819–0.864 |
| Brier score (lower is better; guessing scores 0.195) | 0.138 | 0.128–0.149 |
| Offer rule: churners reached | 301 of 374 (80%) | |
| Offer rule: net value (illustrative assumptions) | $53.2k | $45.2k to $61.5k |

The offer assumptions ($50 per offer, 30% of would-be churners kept, 12 more months of revenue) are
illustrative, not the telecom's. They live in [`evaluate.py`](src/churn/evaluate.py).

## Results by group

The model does not see gender; the gender rows below check that it works as well for women and men.
Groups with fewer than 30 churners are marked as too small to score.

| Group | Customers | Churn rate | Ranking (ROC-AUC) | Churners reached | Sent an offer |
|---|---:|---:|---:|---:|---:|
| Gender: Female | 687 | 28% | 0.838 | 79% | 46% |
| Gender: Male | 722 | 25% | 0.847 | 82% | 45% |
| Senior citizen: No | 1,187 | 23% | 0.843 | 76% | 40% |
| Senior citizen: Yes | 222 | 44% | 0.784 | 94% | 76% |
| Has a partner: No | 736 | 34% | 0.815 | 81% | 54% |
| Has a partner: Yes | 673 | 19% | 0.856 | 79% | 36% |
| Has dependents: No | 978 | 31% | 0.827 | 84% | 54% |
| Has dependents: Yes | 431 | 16% | 0.855 | 66% | 25% |
| Contract: Month-to-month | 773 | 43% | 0.744 | 85% | 71% |
| Contract: One year | 300 | 12% | 0.741 | 58% | 28% |
| Contract: Two year | 336 | 3% | too few churners | 0 of 9 | 1% |
| Months as a customer: 0–12 | 449 | 48% | 0.747 | 82% | 67% |
| Months as a customer: 13–24 | 206 | 29% | 0.814 | 92% | 62% |
| Months as a customer: 25–48 | 304 | 21% | 0.800 | 73% | 42% |
| Months as a customer: 49–72 | 450 | 8% | 0.833 | 63% | 18% |

- Women and men: offered at 46% and 45%; ranking gap −0.052 to +0.034 (95%
  interval), no evidence of a difference.
- Senior citizens: churn rate 44%, offered at 76%; ranked less sharply
  (0.784 vs 0.843), gap interval −0.122 to +0.004 with 98 senior churners.
  Unsettled; the first group to monitor.

## Known limits

- No time column: the split is random, not "past months → next month".
- Permutation importance shows what the model leans on (tenure first), not what causes churn.
- Two-year-contract customers: only 9 churned in the test rows, too few to judge.
- The score depends on the split: the audit found the 2025 notebook's single split read about
  1.5 points of accuracy higher than the average of ten others, which is why the main scores here
  carry an interval.
