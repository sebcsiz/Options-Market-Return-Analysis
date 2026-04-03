import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, PRIOR_MU, PRIOR_TAU2

def compute_mle(returns):
    mu_mle = np.mean(returns)
    sigma_mle = np.std(returns, ddof=0)
    sigma_s = np.std(returns, ddof=1)
    se_mle = sigma_mle / np.sqrt(len(returns))
    return mu_mle, sigma_mle, sigma_s, se_mle

def plot_mle(returns, mu_mle, sigma_mle, se_mle):
    n = len(returns)
    mu_grid = np.linspace(mu_mle - 1.0, mu_mle + 1.0, 400)
    log_lik = np.array([
        np.sum(stats.norm.logpdf(returns, loc=m, scale=sigma_mle))
        for m in mu_grid
    ])

    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(mu_grid, log_lik, color=BLUE, lw=2)
    ax.axvline(mu_mle, color=RED, lw=2, ls="--", label=f"μ̂ = {mu_mle:.4f}%")
    ax.fill_betweenx(
        [log_lik.min(), log_lik.max()],
        mu_mle - se_mle, mu_mle + se_mle,
        alpha=0.15, color=RED, label="±1 SE",
    )
    ax.set_xlabel("μ (Mean Signed Moneyness %)")
    ax.set_ylabel("Log-Likelihood l(mu)")
    ax.set_title("Log-Likelihood Surface (σ fixed at MLE)")
    ax.legend(fontsize=9)

    ax = axes[1]
    clip  = np.percentile(np.abs(returns), 95)
    r_clp = returns[np.abs(returns) <= clip]
    x_fit = np.linspace(r_clp.min(), r_clp.max(), 400)
    ax.hist(r_clp, bins=70, density=True, color=BLUE,
            edgecolor="white", alpha=0.6, label="Observed (clipped at 95th |r|)")
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu_mle, sigma_mle),
            color=RED, lw=2.5, label=f"N({mu_mle:.2f}, {sigma_mle:.2f}²)")
    ax.set_xlabel("Signed Moneyness (%) — clipped for display")
    ax.set_ylabel("Density")
    ax.set_title("MLE Normal Fit vs Observed Data")
    ax.legend(fontsize=9)

    plt.tight_layout()
    
    # Save Figure
    # plt.savefig("plot_02_mle.png", bbox_inches="tight")
    # plt.close()

def print_mle(mu_mle, sigma_mle, sigma_s, se_mle, fisher_info, crlb, crlb_se, n):
    print("\n" + "=" * 65)
    print("MLE")
    print("=" * 65)
    print(f"  μ̂  (MLE mean)          : {mu_mle:.6f}%")
    print(f"  σ̂  (MLE std, ddof=0)   : {sigma_mle:.6f}%")
    print(f"  s  (unbiased std)      : {sigma_s:.6f}%")
    print(f"  SE(μ̂) = σ̂/√n           : {se_mle:.6f}%")
    print("\n" + "=" * 65)
    print("FISHER INFORMATION & CRLB")
    print("=" * 65)
    print(f"  I(μ) = n/σ²            : {fisher_info:.4f}")
    print(f"  CRLB Var(μ̂) ≥ σ²/n    : {crlb:.6f}")
    print(f"  CRLB SE floor          : {crlb_se:.6f}%")
    print(f"  Actual SE(μ̂) = s/√n    : {sigma_s/np.sqrt(n)+1:.6f}%")
    print(f"  MLE is efficient — SE ≈ CRLB SE")

def compute_bootstrap(returns, B=10_000):
    n          = len(returns)
    boot_means = np.array([
        np.mean(np.random.choice(returns, size=n, replace=True))
        for _ in range(B)
    ])
    boot_mean = np.mean(boot_means)
    boot_se   = np.std(boot_means, ddof=1)
    boot_ci   = np.percentile(boot_means, [2.5, 97.5])
    boot_bias = boot_mean - np.mean(returns)
    return boot_means, boot_mean, boot_se, boot_ci, boot_bias


def plot_bootstrap(boot_means, boot_mean, boot_se, boot_ci, mu_mle):
    x_norm = np.linspace(boot_means.min(), boot_means.max(), 300)

    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.hist(boot_means, bins=60, color=BLUE,
            edgecolor="white", alpha=0.85, density=True)
    ax.plot(x_norm, stats.norm.pdf(x_norm, boot_mean, boot_se),
            color=ORANGE, lw=2, label="Normal approx.")
    ax.axvline(mu_mle,     color=RED,   lw=2,   ls="--",
                label=f"MLE  μ̂ = {mu_mle:.3f}%")
    ax.axvline(boot_ci[0], color=GREEN, lw=1.8, ls=":",
                label=f"95% CI [{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]")
    ax.axvline(boot_ci[1], color=GREEN, lw=1.8, ls=":")
    ax.axvline(0, color=GRAY, lw=1, alpha=0.5, label="Zero")
    ax.fill_between(
        x_norm,
        stats.norm.pdf(x_norm, boot_mean, boot_se),
        where=(x_norm >= boot_ci[0]) & (x_norm <= boot_ci[1]),
        alpha=0.15, color=GREEN,
    )
    ax.set_xlabel("Bootstrap Mean Estimate (%)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap Distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(boot_means, dist="norm")
    ax.scatter(osm, osr, color=BLUE, s=6, alpha=0.5, label="Bootstrap means")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color=RED, lw=2, label="Normal reference")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Q-Q Plot: Bootstrap Means vs Normal")
    ax.legend(fontsize=9)

    plt.tight_layout()

    # Save Figure
    # plt.savefig("plot_03_bootstrap.png", bbox_inches="tight")
    # plt.close()

def print_bootstrap(boot_mean, boot_se, boot_ci, boot_bias, B):
    print("\n" + "=" * 65)
    print(f"BOOTSTRAP  (B = {B:,} resamples)")
    print("=" * 65)
    print(f"  Bootstrap mean         : {boot_mean:.6f}%")
    print(f"  Bootstrap SE           : {boot_se:.6f}%")
    print(f"  Bootstrap bias         : {boot_bias:.6f}%")
    print(f"  95% Percentile CI      : [{boot_ci[0]:.4f}%, {boot_ci[1]:.4f}%]")

def compute_bayesian(returns, mu_mle, sigma_mle):
    n = len(returns)
    sig2 = sigma_mle ** 2
    sig2_post = 1.0 / (1.0 / PRIOR_TAU2 + n / sig2)
    mu_post = sig2_post * (PRIOR_MU / PRIOR_TAU2 + n * mu_mle / sig2)
    sd_post = np.sqrt(sig2_post)
    cred_int = (mu_post - 1.96 * sd_post, mu_post + 1.96 * sd_post)
    return mu_post, sd_post, cred_int, sig2_post


def plot_bayesian(mu_mle, sigma_mle, mu_post, sd_post, cred_int, sig2_post):
    import numpy as _np
    from scipy import stats as _st
    
    x_b = np.linspace(mu_post - 5 * sd_post, mu_post + 5 * sd_post, 500)
    sig2 = sigma_mle ** 2
    n_eff = len(x_b)

    _, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    
    prior_y = _st.norm.pdf(x_b, PRIOR_MU, _np.sqrt(PRIOR_TAU2))
    post_y  = _st.norm.pdf(x_b, mu_post, sd_post)
    ax.plot(x_b, prior_y / prior_y.max() * post_y.max(),
            color=GRAY, lw=1.8, ls="--", label="Prior (scaled)")
    ax.plot(x_b, post_y, color=BLUE, lw=2.5,
            label=f"Posterior N({mu_post:.2f}, {sd_post:.3f}²)")
    ax.fill_between(
        x_b, post_y,
        where=(x_b >= cred_int[0]) & (x_b <= cred_int[1]),
        alpha=0.2, color=BLUE,
        label=f"95% CI [{cred_int[0]:.2f}%, {cred_int[1]:.2f}%]",
    )
    ax.axvline(mu_post, color=RED, lw=2, ls="--",
                label=f"Posterior mean = {mu_post:.3f}%")
    ax.axvline(0, color=GRAY, lw=1, alpha=0.5, label="Zero")
    ax.set_xlabel("μ (Mean Signed Moneyness %)")
    ax.set_ylabel("Density")
    ax.set_title("Prior vs Posterior")
    ax.legend(fontsize=8)

    ax = axes[1]
    n_approx = round((1 / sig2_post - 1 / PRIOR_TAU2) * sig2)
    for t2, c, lbl in [
        (1, GREEN,  "τ²=1 (strong prior)"),
        (10, ORANGE, "τ²=10"),
        (100, BLUE,   "τ²=100 (used)"),
        (1000, PURPLE, "τ²=1000 (diffuse)"),
    ]:
        sp  = 1 / (1 / t2 + n_approx / sig2)
        mp  = sp * (PRIOR_MU / t2 + n_approx * mu_mle / sig2)
        sdp = _np.sqrt(sp)
        ax.plot(x_b, _st.norm.pdf(x_b, mp, sdp), color=c, lw=2,
                label=f"{lbl} → μ={mp:.3f}%")
    ax.axvline(mu_mle, color="black", lw=1.5, ls=":",
                label=f"MLE = {mu_mle:.3f}%")
    ax.set_xlabel("μ (Mean Signed Moneyness %)")
    ax.set_ylabel("Density")
    ax.set_title("Sensitivity to Prior τ²")
    ax.legend(fontsize=8)

    plt.tight_layout()

    # Save figure
    # plt.savefig("plot_04_bayesian.png", bbox_inches="tight")
    # plt.close()

def print_bayesian(mu_post, sd_post, cred_int, sig2_post):
    print("\n" + "=" * 65)
    print("BAYESIAN INFERENCE  (Normal–Normal conjugate)")
    print("=" * 65)
    print(f"  Prior: N(0, 100)  ->  prior SD = 10%")
    print(f"  Posterior mean             : {mu_post:.6f}%")
    print(f"  Posterior SD               : {sd_post:.6f}%")
    print(f"  95% Credible interval      : [{cred_int[0]:.4f}%, {cred_int[1]:.4f}%]")
    print(f"  Prior weight               : {sig2_post/PRIOR_TAU2:.5f}")

def compute_ttest(returns):
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    df_t   = len(returns) - 1
    ci_95  = stats.t.interval(0.95, df_t,
                                loc=np.mean(returns),
                                scale=np.std(returns, ddof=1) / np.sqrt(len(returns)))
    return t_stat, p_value, df_t, ci_95


def plot_ttest(t_stat, p_value, df_t):
    x_t   = np.linspace(-5, 5, 500)
    pdf_t = stats.t.pdf(x_t, df_t)

    _, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x_t, pdf_t, color=BLUE, lw=2.5, label=f"t({df_t:,})")
    ax.fill_between(x_t, pdf_t, where=(x_t <= -1.96),
                    color=RED, alpha=0.4, label="Rejection region (α=0.05)")
    ax.fill_between(x_t, pdf_t, where=(x_t >=  1.96),
                    color=RED, alpha=0.4)
    t_disp = max(min(t_stat, 4.7), -4.7)
    ax.axvline(t_disp, color=ORANGE, lw=2.5, ls="--",
                label=f"Observed t = {t_stat:.2f}  (p = {p_value:.1e})")
    ax.annotate(
        f"t = {t_stat:.2f}\n(off-scale →)",
        xy=(t_disp, stats.t.pdf(t_disp, df_t)),
        xytext=(2.8, 0.22),
        arrowprops=dict(arrowstyle="->", color=ORANGE),
        fontsize=9, color=ORANGE,
    )
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    decision = "Reject" if p_value < 0.05 else "Fail to reject"
    ax.set_title(f"t({df_t:,})  |  t = {t_stat:.2f},  p = {p_value:.1e}"
                    f"  ->  {decision} H₀")
    ax.legend(fontsize=9)
    plt.tight_layout()
    
    # Save figure
    # plt.savefig("plot_06_ttest.png", bbox_inches="tight")
    # plt.close()

def print_ttest(t_stat, p_value, df_t, ci_95):
    print("\n" + "=" * 65)
    print("HYPOTHESIS TEST — one-sample t-test  H₀: μ = 0")
    print("=" * 65)
    print(f"  t-statistic            : {t_stat:.4f}")
    print(f"  Degrees of freedom     : {df_t:,}")
    print(f"  p-value (two-sided)    : {p_value:.2e}")
    print(f"  95% t-CI for μ         : [{ci_95[0]:.4f}%, {ci_95[1]:.4f}%]")
    decision = "Reject H₀" if p_value < 0.05 else "Fail to reject H₀"
    print(f"  Decision               : {decision} at α = 0.05")

def run(returns):
    n = len(returns)

    mu_mle, sigma_mle, sigma_s, se_mle = compute_mle(returns)
    fisher_info = n / sigma_mle ** 2
    crlb = sigma_mle ** 2 / n
    crlb_se = np.sqrt(crlb)
    print_mle(mu_mle, sigma_mle, sigma_s, se_mle, fisher_info, crlb, crlb_se, n)
    plot_mle(returns, mu_mle, sigma_mle, se_mle)

    B = 10_000
    boot_means, boot_mean, boot_se, boot_ci, boot_bias = compute_bootstrap(returns, B)
    print_bootstrap(boot_mean, boot_se, boot_ci, boot_bias, B)
    plot_bootstrap(boot_means, boot_mean, boot_se, boot_ci, mu_mle)

    mu_post, sd_post, cred_int, sig2_post = compute_bayesian(returns, mu_mle, sigma_mle)
    print_bayesian(mu_post, sd_post, cred_int, sig2_post)
    plot_bayesian(mu_mle, sigma_mle, mu_post, sd_post, cred_int, sig2_post)

    t_stat, p_value, df_t, ci_95 = compute_ttest(returns)
    print_ttest(t_stat, p_value, df_t, ci_95)
    plot_ttest(t_stat, p_value, df_t)

    return dict(
        mu_mle=mu_mle, sigma_mle=sigma_mle, sigma_s=sigma_s,
        se_mle=se_mle, fisher_info=fisher_info, crlb=crlb, crlb_se=crlb_se,
        boot_means=boot_means, boot_mean=boot_mean, boot_se=boot_se,
        boot_ci=boot_ci, boot_bias=boot_bias, B=B,
        mu_post=mu_post, sd_post=sd_post, cred_int=cred_int,
        t_stat=t_stat, p_value=p_value, df_t=df_t, ci_95=ci_95,
    )