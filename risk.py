import numpy as np
import matplotlib.pyplot as plt
from config import BLUE, ORANGE, GREEN, RED, GRAY, PURPLE

def _var_cvar(data, alpha):
    var = np.percentile(data, alpha * 100)
    cvar = data[data >= var].mean()
    return var, cvar

def compute_var(returns, B=5000, alphas=(0.90, 0.95, 0.99)):
    n = len(returns)
    losses = -np.asarray(returns)

    var_boot = {a: [] for a in alphas}
    cvar_boot = {a: [] for a in alphas}

    for _ in range(B):
        samp = np.random.choice(losses, size=n, replace=True)
        for a in alphas:
            v, c = _var_cvar(samp, a)
            var_boot[a].append(v)
            cvar_boot[a].append(c)

    results = {}
    for a in alphas:
        var_pt, cvar_pt = _var_cvar(losses, a)
        results[a] = dict(
            var = var_pt,
            cvar = cvar_pt,
            var_ci = np.percentile(var_boot[a],  [2.5, 97.5]),
            cvar_ci = np.percentile(cvar_boot[a], [2.5, 97.5]),
        )
    return results, losses

def print_var(var_results):
    print("\n" + "=" * 65)
    print("VALUE AT RISK & CONDITIONAL VaR  (bootstrap, B=5,000)")
    print("=" * 65)
    for a, vr in var_results.items():
        print(f"  α={int(a*100)}%  VaR  = {vr['var']:>7.3f}%  "
                f"95% CI [{vr['var_ci'][0]:.3f}, {vr['var_ci'][1]:.3f}]")
        print(f"         CVaR = {vr['cvar']:>7.3f}%  "
                f"95% CI [{vr['cvar_ci'][0]:.3f}, {vr['cvar_ci'][1]:.3f}]")

def plot_var(var_results, losses):
    alphas   = list(var_results.keys())
    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    ax = axes[0]
    ax.hist(losses, bins=80, density=True, color=BLUE, alpha=0.6,
            edgecolor="white", label="Loss distribution (−SignedMoneyness)")
    for a, c in zip(alphas, [ORANGE, RED, PURPLE]):
        vr = var_results[a]['var']
        ax.axvline(vr, color=c, lw=2, ls="--",
                    label=f"VaR {int(a*100)}% = {vr:.2f}%")
    ax.set_xlabel("Loss (%) = −Signed Moneyness")
    ax.set_ylabel("Density")
    ax.set_title("Loss Distribution with VaR Thresholds")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels_v = ([f"VaR {int(a*100)}%" for a in alphas] +
                [f"CVaR {int(a*100)}%" for a in alphas])
    vals_v = ([var_results[a]['var'] for a in alphas] +
                [var_results[a]['cvar'] for a in alphas])
    ci_lo_v = ([var_results[a]['var_ci'][0] for a in alphas] +
                [var_results[a]['cvar_ci'][0] for a in alphas])
    ci_hi_v = ([var_results[a]['var_ci'][1] for a in alphas] +
                [var_results[a]['cvar_ci'][1] for a in alphas])
    colors_v = [ORANGE, RED, PURPLE] * 2

    for i, (v, lo, hi, c) in enumerate(zip(vals_v, ci_lo_v, ci_hi_v, colors_v)):
        ax.plot(v, i, 'o', color=c, ms=8, zorder=3)
        ax.plot([lo, hi], [i, i], '-', color=c, lw=3, alpha=0.7)
    ax.set_yticks(range(len(labels_v)))
    ax.set_yticklabels(labels_v)
    ax.set_xlabel("Loss (%)  —  point estimate + 95% bootstrap CI")
    ax.set_title("VaR & CVaR Forest Plot")

    # plt.tight_layout()
    # plt.savefig("plot_08_var.png", bbox_inches="tight")
    # plt.close()

def compute_sectors(df_clean, top_n=8, B=3000):
    top_sectors = (df_clean['Sector'].value_counts().head(top_n).index.tolist())
    sector_res  = []
    for sec in top_sectors:
        data = df_clean[df_clean['Sector'] == sec]['SignedMoneyness'].values
        bm = np.array([np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(B)])
        bci = np.percentile(bm, [2.5, 97.5])
        sector_res.append(dict(sector=sec, n=len(data), mean=data.mean(), ci=bci))
    sector_res.sort(key=lambda x: x['mean'])
    return sector_res

def print_sectors(sector_res):
    print("\n" + "=" * 65)
    print("SECTOR-LEVEL MEAN RETURN")
    print("=" * 65)
    for r in sector_res:
        print(f"  {r['sector']:<35} n={r['n']:>5,}  "
                f"μ={r['mean']:>7.3f}%  "
                f"95% CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

def plot_sectors(sector_res, mu_mle):
    _, ax = plt.subplots(figsize=(11, 5))
    
    colors_s = [RED if r['mean'] < 0 else GREEN for r in sector_res]
    for i, (r, c) in enumerate(zip(sector_res, colors_s)):
        ax.plot(r['mean'], i, 'o', color=c, ms=8, zorder=3)
        ax.plot([r['ci'][0], r['ci'][1]], [i, i], '-', color=c, lw=3, alpha=0.7)
        ax.text(r['ci'][1] + 0.1, i, f"n={r['n']:,}", va='center', fontsize=8)
    ax.axvline(0, color=GRAY, lw=1, ls="--", alpha=0.6)
    ax.axvline(mu_mle, color=BLUE, lw=1.5, ls=":", alpha=0.8,
                label=f"Overall mean = {mu_mle:.2f}%")
    ax.set_yticks(range(len(sector_res)))
    ax.set_yticklabels([r['sector'] for r in sector_res])
    ax.set_xlabel("Mean Signed Moneyness (%)  —  point + 95% bootstrap CI")
    ax.set_title("Sector-Level Mean Return with 95% Bootstrap CIs")
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    # plt.savefig("plot_09_sectors.png", bbox_inches="tight")
    # plt.close()

def run(df_clean, returns, mu_mle):
    var_results, losses = compute_var(returns)
    print_var(var_results)
    plot_var(var_results, losses)

    sector_res = compute_sectors(df_clean)
    print_sectors(sector_res)
    plot_sectors(sector_res, mu_mle)

    return dict(var_results=var_results, sector_res=sector_res)