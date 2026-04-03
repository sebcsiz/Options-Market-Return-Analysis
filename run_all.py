import config
import eda
import inference
import simulation
import subgroups
import risk

df_raw, df_clean, returns = config.load_data("data/bendgame/options.csv")
config.print_summary(df_raw, df_clean, returns)

print("\nSection 1: EDA")
eda.run(df_clean, returns)

print("\nSections 2–6: Core Inference")
inf_results = inference.run(returns)

mu_mle = inf_results['mu_mle']
sigma_mle = inf_results['sigma_mle']
boot_ci = inf_results['boot_ci']
cred_int = inf_results['cred_int']
crlb_se = inf_results['crlb_se']
t_stat = inf_results['t_stat']
p_value = inf_results['p_value']

print("\nSection 3: Simulation Study")
simulation.run(mu_mle, sigma_mle)

print("\nSection 4: Subgroup Analysis")
sub_results = subgroups.run(df_clean, returns, mu_mle)

res_call = sub_results['res_call']
res_put = sub_results['res_put']
res_itm = sub_results['res_itm']
res_otm = sub_results['res_otm']
t_cp, p_cp = sub_results['t_cp'], sub_results['p_cp']
t_io, p_io = sub_results['t_io'], sub_results['p_io']

print("\nSection 4: Risk Metrics & Sectors")
risk_results = risk.run(df_clean, returns, mu_mle)

var_results = risk_results['var_results']

print("\n" + "=" * 65)
print("FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"  Dataset        : bendgame Options Market Trades, 2017–2019")
print(f"  n (cleaned)    : {len(returns):,}")
print(f"  Return var     : Signed Moneyness (%)")
print()
print(f"  ── Full Sample ──────────────────────────────────────────")
print(f"  MLE  μ̂          : {mu_mle:.4f}%   σ̂ = {sigma_mle:.4f}%")
print(f"  SE(μ̂) = CRLB    : {crlb_se:.4f}%")
print(f"  Bootstrap 95%CI : [{boot_ci[0]:.4f}%, {boot_ci[1]:.4f}%]")
print(f"  Bayesian 95%CI  : [{cred_int[0]:.4f}%, {cred_int[1]:.4f}%]")
print(f"  t-test vs 0     : t={t_stat:.3f},  p={p_value:.2e}")
print()
print(f"  ── Subgroups ────────────────────────────────────────────")
for r in [res_call, res_put, res_itm, res_otm]:
    print(f"  {r['label']:<12} μ={r['mean']:.4f}%  "
            f"95%CI [{r['boot_ci'][0]:.3f}, {r['boot_ci'][1]:.3f}]  "
            f"p={r['p_val']:.2e}")
print(f"  Call vs Put    Welch t={t_cp:.3f},  p={p_cp:.4f}")
print(f"  ITM  vs OTM    Welch t={t_io:.3f},  p={p_io:.2e}")
print()
print(f"  ── Risk Metrics ─────────────────────────────────────────")
for a, vr in var_results.items():
    print(f"  VaR {int(a*100)}%       : {vr['var']:.3f}%   CVaR: {vr['cvar']:.3f}%")
print("=" * 65)