# Estimating Expected Returns in Options Derivatives Trading
### Statistical Inference on the bendgame Options Market Trades Dataset

A modular Python analysis pipeline for estimating expected returns in options derivatives
trading using likelihood-based and Bayesian inference methods. Built for STAT 401 —
Probability and Statistical Inference, University of British Columbia Okanagan.

---

## Overview

This project applies classical and Bayesian statistical inference to 62,795 real options
trades from the bendgame dataset (Kaggle, 2017–2019). The return variable is Signed
Moneyness (%), defined as the percentage difference between stock price and strike price
at trade time, directionally adjusted for contract type.

**Research question:** Can the expected return of an options trading strategy be reliably
estimated and quantified using likelihood-based and Bayesian inference methods, and do
expected returns differ significantly across contract types (calls vs. puts) and moneyness
categories (ITM vs. OTM)?

**Key findings:**
- Mean Signed Moneyness: -4.841% (95% CI [-4.902%, -4.780%], p < 0.001)
- No statistically significant difference between calls and puts (p = 0.175)
- ITM contracts (+3.036%) significantly outperform OTM contracts (-6.422%) (p < 0.001)
- Bootstrap 95% CVaR: 28.8%, reflecting substantial tail risk in options flow trading

---

## Project Structure

```
.
├── config.py
├── data
│   ├── bendgame
│       ├── options-nohead.csv
│       └── options.csv
│   
├── eda.py
├── graphs
│   ├── plot_01_eda.png
│   ├── plot_02_mle.png
│   ├── plot_03_bootstrap.png
│   ├── plot_04_bayesian.png
│   ├── plot_05_simulation.png
│   ├── plot_06_ttest.png
│   ├── plot_07_subgroups.png
│   ├── plot_08_var.png
│   └── plot_09_sectors.png
├── Options Paper.pdf
├── inference.py
├── requirements.txt
├── risk.py
├── run_all.py
├── simulation.py
└── subgroups.py
```

---

## Modules

### `config.py`
Shared configuration imported by all other modules. Contains colour constants, matplotlib
style settings, the Bayesian prior parameters, and the `load_data()` and `print_summary()`
functions. The return variable (Signed Moneyness) is computed here and outliers at the
1st and 99th percentile are trimmed.

### `eda.py`
Exploratory data analysis. Produces a four-panel figure showing the full return
distribution with KDE, call vs. put boxplots, ITM/ATM/OTM boxplots, and monthly
trade volume across the dataset period.

Output: `plot_01_eda.png`

### `inference.py`
Core statistical inference pipeline:
- Maximum Likelihood Estimation (MLE) for mu and sigma
- Fisher Information and Cramér-Rao Lower Bound (CRLB)
- Bootstrap resampling (B = 10,000) with percentile confidence intervals
- Bayesian posterior inference under a normal-normal conjugate model with prior N(0, 100)
- One-sample t-test against H0: mu = 0

Returns a results dictionary consumed by downstream modules.

Outputs: `plot_02_mle.png`, `plot_03_bootstrap.png`, `plot_04_bayesian.png`, `plot_06_ttest.png`

### `simulation.py`
Simulation study validating estimator performance under controlled conditions. Simulates
returns from N(mu, sigma) with parameters matched to the real dataset, across six sample
sizes (n = 50, 100, 200, 500, 1000, 2000) with 1,000 replications each. Compares MLE,
bootstrap, and Bayesian RMSE against the CRLB.

Output: `plot_05_simulation.png`

### `subgroups.py`
Subgroup inference pipeline. Runs the full MLE, bootstrap CI, Bayesian CI, and t-test
pipeline separately on calls, puts, ITM, and OTM contracts. Applies Welch two-sample
t-tests to compare call vs. put and ITM vs. OTM groups. Produces a forest plot of all
group means and posterior comparison plots.

Output: `plot_07_subgroups.png`

### `risk.py`
Risk estimation and sector analysis:
- Bootstrap Value at Risk (VaR) at 90%, 95%, and 99% confidence levels
- Conditional VaR / Expected Shortfall (CVaR) with bootstrap confidence intervals
- Sector-level mean return estimation with 95% bootstrap CIs for the eight most-traded sectors

Outputs: `plot_08_var.png`, `plot_09_sectors.png`

---

## Output Plots

| File | Description |
|---|---|
| `plot_01_eda.png` | Exploratory data analysis: distribution, boxplots, trade volume |
| `plot_02_mle.png` | Log-likelihood surface and MLE normal fit |
| `plot_03_bootstrap.png` | Bootstrap sampling distribution and Q-Q plot |
| `plot_04_bayesian.png` | Prior vs. posterior and sensitivity to prior variance |
| `plot_05_simulation.png` | Simulation RMSE vs. sample size for all three estimators |
| `plot_06_ttest.png` | One-sample t-test visualisation |
| `plot_07_subgroups.png` | Subgroup posteriors and forest plot: Call/Put, ITM/OTM |
| `plot_08_var.png` | VaR and CVaR at 90%, 95%, 99% with bootstrap CIs |
| `plot_09_sectors.png` | Sector-level mean return with 95% bootstrap CIs |

---

## Data

This project uses the **bendgame Options Market Trades** dataset, available on Kaggle:

https://www.kaggle.com/datasets/bendgame/options-market-trades

Download the CSV file, rename it `options.csv.csv`, and place it in the same
directory as the Python scripts before running.

The dataset contains 62,795 options trades from November 2017 to August 2019 across
2,346 unique underlying symbols and 418 trading days. After trimming the 1st and 99th
percentile outliers of Signed Moneyness, the cleaned sample is n = 61,539.

---

## Setup

**Requirements:**

```
numpy
pandas
scipy
matplotlib
```

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

Or using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

pip install numpy pandas scipy matplotlib
```

---

## Usage

Run the full analysis pipeline:

```bash
python run_all.py
```

This executes all modules in order and saves all nine plot files to the working directory.
Key numerical results are printed to stdout at each stage.

To run a single module in isolation:

```python
import config
import inference

df_raw, df_clean, returns = config.load_data("options.csv")
results = inference.run(returns)
```

Each module exposes a `run()` function that accepts the cleaned dataframe and/or returns
array and returns a results dictionary.

---

## Methods Summary

| Method | Purpose |
|---|---|
| MLE | Point estimate of mean return and variance |
| Fisher Information / CRLB | Theoretical precision bound for the MLE |
| Bootstrap (B = 10,000) | Non-parametric confidence intervals and bias estimation |
| Bayesian inference | Posterior distribution under prior N(0, 100) |
| One-sample t-test | Test whether mean return differs from zero |
| Welch two-sample t-test | Compare mean returns across subgroups |
| Bootstrap VaR / CVaR | Tail risk estimation at 90%, 95%, 99% |
| Sector analysis | Bootstrap CIs for mean return by sector |

---

## References

1. Casella, G. & Berger, R. L. (2002). Statistical Inference (2nd ed.). Duxbury Press.
2. Bendgame. (2019). Options market trades [Dataset]. Kaggle. https://www.kaggle.
com/datasets/bendgame/options-market-trades
3. Csizmazia, C. (2023). Applications of reinforcement learning in derivatives hedging:
A literature survey. Unpublished manuscript prepared for Ernst & Young, Rotman
School of Management, University of Toronto.
4. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of
Statistics, 7(1), 1–26.
5. Hull, J. C. (2018). Options, Futures, and Other Derivatives (10th ed.). Pearson.
6. Harper, D. R. (2025, November 25). Understanding Value at Risk (VaR): Explanation
and calculation methods. Investopedia. https://www.investopedia.com/articles/
04/092904.asp
7. Quant StackExchange. (n.d.). Why is infimum chosen to define Value at Risk as
opposed to the minimum? https://quant.stackexchange.com/questions/65694/
why-is-infimum-chosen-to-define-value-at-risk-as-opposed-to-the-minimum
8. Horowitz, J. L. (2018). Bootstrap Methods in Econometrics. arXiv preprint arXiv:1809.04016.
9. Robert, C. P., Marin, J. M., & Rousseau, J. (2010). Bayesian Inference. arXiv preprint
arXiv:1002.2080.
10. Efron, B. (2012). Bayesian inference and the parametric bootstrap. The Annals of
Applied Statistics, 6(4), 1971–1997.
11. Bandyopadhyay, P. S., & Forster, M. R. (2011). Philosophy of Statistics. Elsevier.
12. Jacquier, E., Johannes, M., & Polson, N. (2000). Bayesian analysis of contingent claim
model error. Journal of Econometrics.
13. Ludkovski, M. (2022). Statistical Machine Learning for Quantitative Finance.

---

## Author

Sebastian Csizmazia <br>
University of British Columbia Okanagan
