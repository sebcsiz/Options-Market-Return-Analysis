import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from config import BLUE, ORANGE, GREEN, RED, GRAY

def run(df_clean, returns):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(returns, bins=80, color=BLUE, edgecolor="white", alpha=0.8, density=True)
    x_kde = np.linspace(returns.min(), returns.max(), 400)
    kde   = stats.gaussian_kde(returns, bw_method=0.15)
    ax.plot(x_kde, kde(x_kde), color=ORANGE, lw=2, label="KDE")
    ax.axvline(returns.mean(),     color=RED,   lw=2, ls="--",
                label=f"Mean = {returns.mean():.2f}%")
    ax.axvline(np.median(returns), color=GREEN, lw=2, ls=":",
                label=f"Median = {np.median(returns):.2f}%")
    ax.axvline(0, color=GRAY, lw=1, ls="-", alpha=0.5, label="Zero")
    ax.set_xlabel("Signed Moneyness (%)")
    ax.set_ylabel("Density")
    ax.set_title("Full Distribution")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    grp_cp = [
        df_clean[df_clean['OptionType'] == 'call']['SignedMoneyness'].values,
        df_clean[df_clean['OptionType'] == 'put'] ['SignedMoneyness'].values,
    ]
    bp = ax.boxplot(grp_cp, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2))
    for patch, color in zip(bp['boxes'], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set(color=GRAY, alpha=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Call', 'Put'])
    ax.set_ylabel("Signed Moneyness (%)")
    ax.set_title("Call vs Put")
    for i, (g, _) in enumerate(zip(grp_cp, ['Call', 'Put']), 1):
        ax.text(i, ax.get_ylim()[1] * 0.88,
                f"n={len(g):,}\nμ={g.mean():.2f}%", ha='center', fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    locs    = ['ITM', 'ATM', 'OTM']
    grp_loc = [df_clean[df_clean['ChainLocation'] == l]['SignedMoneyness'].values
                for l in locs]
    bp = ax.boxplot(grp_loc, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp['boxes'], [GREEN, GRAY, RED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'fliers']:
        for item in bp[element]:
            item.set(color=GRAY, alpha=0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(locs)
    ax.set_ylabel("Signed Moneyness (%)")
    ax.set_title("Chain Location")
    for i, g in enumerate(grp_loc, 1):
        ax.text(i, ax.get_ylim()[1] * 0.88,
                f"n={len(g):,}\nμ={g.mean():.2f}%", ha='center', fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    monthly = df_clean.groupby(df_clean['Date'].dt.to_period('M')).size()
    ax.bar(range(len(monthly)), monthly.values,
            color=BLUE, alpha=0.7, edgecolor="white")
    ax.set_xlabel("Month (Nov 2017 – Aug 2019)")
    ax.set_ylabel("Trade Count")
    ax.set_title("Trades per Month")
    tick_pos    = [0, 6, 12, 18, len(monthly) - 1]
    tick_labels = [str(monthly.index[i]) for i in tick_pos if i < len(monthly)]
    ax.set_xticks([p for p in tick_pos if p < len(monthly)])
    ax.set_xticklabels(tick_labels, rotation=25, fontsize=8)

    plt.tight_layout()
    
    # Save figure
    # plt.savefig("plot_01_eda.png", bbox_inches="tight")
    # plt.close()