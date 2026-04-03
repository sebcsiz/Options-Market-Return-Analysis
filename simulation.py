import numpy as np
import matplotlib.pyplot as plt
from config import BLUE, ORANGE, GREEN, PRIOR_MU, PRIOR_TAU2

def run(mu_mle, sigma_mle, n_reps=1000, b_sim=300):
    TRUE_MU = mu_mle
    TRUE_SIGMA = sigma_mle
    SIZES = [50, 100, 200, 500, 1000, 2000]

    print("\n" + "=" * 65)
    print(f"SIMULATION  ({n_reps} reps, true μ={TRUE_MU:.2f}, σ={TRUE_SIGMA:.2f})")
    print("=" * 65)
    print(f"  {'n':>5}  {'MLE bias':>10}  {'MLE RMSE':>10}  "
            f"{'Boot RMSE':>10}  {'Bayes RMSE':>10}")

    sim_rows = []
    for sn in SIZES:
        mle_e, boot_e, bayes_e = [], [], []
        for _ in range(n_reps):
            samp = np.random.normal(TRUE_MU, TRUE_SIGMA, sn)
            mu_s = np.mean(samp)
            s2 = np.var(samp, ddof=0)
            mle_e.append(mu_s)

            boot_e.append(np.mean([
                np.mean(np.random.choice(samp, sn, replace=True))
                for _ in range(b_sim)
            ]))

            sp = 1 / (1 / PRIOR_TAU2 + sn / s2) if s2 > 0 else PRIOR_TAU2
            mp = sp * (PRIOR_MU / PRIOR_TAU2 + sn * mu_s / s2) if s2 > 0 else PRIOR_MU
            bayes_e.append(mp)

        mle_a, boot_a, bayes_a = map(np.array, [mle_e, boot_e, bayes_e])
        row = dict(
            n = sn,
            mle_bias = float(mle_a.mean() - TRUE_MU),
            mle_rmse = float(np.sqrt(((mle_a - TRUE_MU) ** 2).mean())),
            boot_rmse = float(np.sqrt(((boot_a - TRUE_MU) ** 2).mean())),
            bayes_rmse = float(np.sqrt(((bayes_a - TRUE_MU) ** 2).mean())),
        )
        sim_rows.append(row)
        print(f"  {sn:>5}  {row['mle_bias']:>+10.4f}  {row['mle_rmse']:>10.4f}  "
                f"{row['boot_rmse']:>10.4f}  {row['bayes_rmse']:>10.4f}")

    _plot(sim_rows, TRUE_SIGMA)
    return sim_rows


def _plot(sim_rows, true_sigma):
    ns = [r['n'] for r in sim_rows]
    mle_rmse = [r['mle_rmse'] for r in sim_rows]
    boot_rmse = [r['boot_rmse'] for r in sim_rows]
    bayes_rmse = [r['bayes_rmse'] for r in sim_rows]
    crlb_curve = [true_sigma / np.sqrt(sn) for sn in ns]

    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    ax = axes[0]
    ax.plot(ns, mle_rmse,   "o-",  color=BLUE,   lw=2, label="MLE")
    ax.plot(ns, boot_rmse,  "s--", color=ORANGE, lw=2, label="Bootstrap")
    ax.plot(ns, bayes_rmse, "^:",  color=GREEN,  lw=2, label="Bayesian")
    ax.plot(ns, crlb_curve, "k-.", lw=1.5, alpha=0.6,  label="CRLB  σ/√n")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Sample Size")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.4)
    ax.plot(ns, [r['mle_bias'] for r in sim_rows],
            "o-", color=BLUE, lw=2, label="MLE bias")
    ax.plot(ns, [r['bayes_rmse'] - r['mle_rmse'] for r in sim_rows],
            "^:", color=GREEN, lw=2, label="Bayesian RMSE − MLE RMSE")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Value")
    ax.set_title("MLE Bias & Bayesian Advantage over MLE")
    ax.legend(fontsize=9)

    plt.tight_layout()

    # Save figure
    # plt.savefig("plot_05_simulation.png", bbox_inches="tight")
    # plt.close()