import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import BLUE, ORANGE, GREEN, RED, GRAY, PRIOR_MU, PRIOR_TAU2

def group_inference(data, label, B=5000):
    data = np.asarray(data)
    ng = len(data)
    mu_g = np.mean(data)
    sig_g = np.std(data, ddof=0)
    se_g = sig_g / np.sqrt(ng)

    bm = np.array([np.mean(np.random.choice(data, ng, replace=True)) for _ in range(B)])
    bci = np.percentile(bm, [2.5, 97.5])

    sp = 1 / (1 / PRIOR_TAU2 + ng / sig_g ** 2)
    mp = sp * (PRIOR_MU / PRIOR_TAU2 + ng * mu_g / sig_g ** 2)
    sdp = np.sqrt(sp)
    cred = (mp - 1.96 * sdp, mp + 1.96 * sdp)

    t_g, p_g = stats.ttest_1samp(data, 0)

    return dict(label=label, n=ng, mean=mu_g, std=sig_g, se=se_g,
                boot_ci=bci, post_mean=mp, post_sd=sdp, cred_int=cred,
                t_stat=t_g, p_val=p_g)

def run(df_clean, returns, mu_mle):
    calls = df_clean[df_clean['OptionType'] == 'call']['SignedMoneyness'].values
    puts  = df_clean[df_clean['OptionType'] == 'put'] ['SignedMoneyness'].values
    itm = df_clean[df_clean['ChainLocation'] == 'ITM'] ['SignedMoneyness'].values
    otm = df_clean[df_clean['ChainLocation'] == 'OTM'] ['SignedMoneyness'].values

    res_call = group_inference(calls, 'Call')
    res_put = group_inference(puts, 'Put')
    res_itm = group_inference(itm, 'ITM')
    res_otm = group_inference(otm, 'OTM')

    t_cp, p_cp = stats.ttest_ind(calls, puts, equal_var=False)
    t_io, p_io = stats.ttest_ind(itm, otm, equal_var=False)

    _print(res_call, res_put, res_itm, res_otm, t_cp, p_cp, t_io, p_io)
    _plot(calls, puts, returns, mu_mle,
            res_call, res_put, res_itm, res_otm,
            t_cp, p_cp, t_io, p_io)

    return dict(
        res_call=res_call, res_put=res_put,
        res_itm=res_itm,   res_otm=res_otm,
        t_cp=t_cp, p_cp=p_cp,
        t_io=t_io, p_io=p_io,
    )

def _print(res_call, res_put, res_itm, res_otm, t_cp, p_cp, t_io, p_io):
    print("\n" + "=" * 65)
    print("SUBGROUP ANALYSIS")
    print("=" * 65)
    for r in [res_call, res_put, res_itm, res_otm]:
        print(f"\n  [{r['label']}]  n={r['n']:,}  mean={r['mean']:.4f}%  "
                f"SE={r['se']:.4f}%")
        print(f"    Bootstrap 95% CI : [{r['boot_ci'][0]:.4f}%, {r['boot_ci'][1]:.4f}%]")
        print(f"    Bayesian  95% CI : [{r['cred_int'][0]:.4f}%, {r['cred_int'][1]:.4f}%]")
        print(f"    t vs 0           : t={r['t_stat']:.3f},  p={r['p_val']:.2e}")
    print(f"\n  Welch t-test Call vs Put : t={t_cp:.3f},  p={p_cp:.4f}")
    print(f"  Welch t-test ITM vs OTM  : t={t_io:.3f},  p={p_io:.2e}")

def _plot(calls, puts, returns, mu_mle,
            res_call, res_put, res_itm, res_otm,
            t_cp, p_cp, t_io, p_io):
    
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for res, c in [(res_call, BLUE), (res_put, ORANGE)]:
        x_r = np.linspace(res['post_mean'] - 5 * res['post_sd'],
                          res['post_mean'] + 5 * res['post_sd'], 400)
        py  = stats.norm.pdf(x_r, res['post_mean'], res['post_sd'])
        ax.plot(x_r, py, color=c, lw=2.5,
                label=f"{res['label']}  μ={res['post_mean']:.3f}%")
        ax.fill_between(x_r, py,
                        where=(x_r >= res['cred_int'][0]) & (x_r <= res['cred_int'][1]),
                        alpha=0.15, color=c)
    ax.axvline(0, color=GRAY, lw=1, ls="--", alpha=0.6, label="Zero")
    ax.set_xlabel("μ (Mean Signed Moneyness %)")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior: Call vs Put\n(Welch t={t_cp:.2f}, p={p_cp:.4f})")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for res, c in [(res_itm, GREEN), (res_otm, RED)]:
        x_r = np.linspace(res['post_mean'] - 5 * res['post_sd'],
                          res['post_mean'] + 5 * res['post_sd'], 400)
        py  = stats.norm.pdf(x_r, res['post_mean'], res['post_sd'])
        ax.plot(x_r, py, color=c, lw=2.5,
                label=f"{res['label']}  μ={res['post_mean']:.3f}%")
        ax.fill_between(x_r, py,
                        where=(x_r >= res['cred_int'][0]) & (x_r <= res['cred_int'][1]),
                        alpha=0.15, color=c)
    ax.axvline(0, color=GRAY, lw=1, ls="--", alpha=0.6, label="Zero")
    ax.set_xlabel("μ (Mean Signed Moneyness %)")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior: ITM vs OTM\n(Welch t={t_io:.2f}, p={p_io:.2e})")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    res_full = group_inference(returns, 'Full sample', B=3000)
    all_res  = [res_full, res_call, res_put, res_itm, res_otm]
    colors_f = [GRAY, BLUE, ORANGE, GREEN, RED]
    for i, (r, c) in enumerate(zip(all_res, colors_f)):
        ax.plot(r['mean'], i, 'o', color=c, ms=7, zorder=3)
        ax.plot([r['boot_ci'][0], r['boot_ci'][1]], [i, i],
                '-', color=c, lw=3, alpha=0.7)
    ax.axvline(0, color=GRAY, lw=1, ls="--", alpha=0.5)
    ax.set_yticks(range(len(all_res)))
    ax.set_yticklabels([r['label'] for r in all_res])
    ax.set_xlabel("Mean Signed Moneyness (%)  —  point + 95% bootstrap CI")
    ax.set_title("Forest Plot: Mean Return with 95% Bootstrap CIs")

    ax = axes[1, 1]
    for data, c, lbl in [(calls, BLUE, f"Call (n={len(calls):,})"),
                            (puts,  ORANGE, f"Put (n={len(puts):,})")]:
        ax.hist(data, bins=60, density=True, alpha=0.4,
                color=c, label=lbl, edgecolor="none")
    ax.axvline(calls.mean(), color=BLUE,   lw=2, ls="--", alpha=0.8)
    ax.axvline(puts.mean(),  color=ORANGE, lw=2, ls="--", alpha=0.8)
    ax.axvline(0, color=GRAY, lw=1, alpha=0.5)
    ax.set_xlabel("Signed Moneyness (%)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution Overlay: Call vs Put")
    ax.legend(fontsize=9)

    plt.tight_layout()

    # Save figure
    # plt.savefig("plot_07_subgroups.png", bbox_inches="tight")
    # plt.close()