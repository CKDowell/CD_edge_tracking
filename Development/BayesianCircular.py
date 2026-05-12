# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:37:03 2026

@author: dowel
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
datadir = r'Y:\Data\FCI\Hedwig\68A10_60D05_FC2_GC8s_RC3\260401\f1\Trial3'
from analysis_funs.CX_analysis_col import CX_a
regions2 = ['eb_ch1','fsb1_ch1','fsb2_ch2']
cxa = CX_a(datadir,regions=regions2,yoking=True,denovo=False)
y = cxa.pdat['phase_fsb1_ch1']
x = cxa.pdat['phase_fsb2_ch2']
x2 = cxa.pdat['phase_eb_ch1']
np.save('_y_tmp.npy', y)
np.save('_x_tmp.npy', x)
np.save('_x2_tmp.npy', x2)
if __name__ == "__main__":
    y = np.load('_y_tmp.npy')
    x = np.load('_x_tmp.npy')
    x = np.load('_x2_tmp.npy')
    
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(x2)
    y, x, x2 = y[mask], x[mask], x2[mask]

    with pm.Model() as model:

        # ── mu: one amplitude+phase pair per predictor ────────────────────────
        A_mu1  = pm.HalfNormal("A_mu1",  sigma=1.0)
        phi_mu1 = pm.VonMises("phi_mu1", mu=0, kappa=0.5)

        A_mu2  = pm.HalfNormal("A_mu2",  sigma=1.0)
        phi_mu2 = pm.VonMises("phi_mu2", mu=0, kappa=0.5)

        C = A_mu1 * pt.cos(x  - phi_mu1) + A_mu2 * pt.cos(x2 - phi_mu2)
        S = A_mu1 * pt.sin(x  - phi_mu1) + A_mu2 * pt.sin(x2 - phi_mu2)

        mu = pm.Deterministic("mu", pt.arctan2(S, C))

        # ── kappa: log-linear, one harmonic per predictor ─────────────────────
        kappa_0  = pm.Normal("kappa_0",  mu=1.0, sigma=1.0)

        kappa_c1 = pm.Normal("kappa_c1", mu=0, sigma=0.5)
        kappa_s1 = pm.Normal("kappa_s1", mu=0, sigma=0.5)

        kappa_c2 = pm.Normal("kappa_c2", mu=0, sigma=0.5)
        kappa_s2 = pm.Normal("kappa_s2", mu=0, sigma=0.5)

        log_kappa = (kappa_0
                     + kappa_c1 * pt.cos(x)  + kappa_s1 * pt.sin(x)
                     + kappa_c2 * pt.cos(x2) + kappa_s2 * pt.sin(x2))

        kappa = pm.Deterministic("kappa", pt.exp(pt.clip(log_kappa, -2, 5)))

        # ── likelihood ────────────────────────────────────────────────────────
        y_obs = pm.VonMises("y_obs", mu=mu, kappa=kappa, observed=y)

    with model:
        trace = pm.sample(
            draws=2000, tune=2000,
            chains=4, target_accept=0.95,
            init="adapt_diag",
        )
    
    # with pm.Model() as model:

    #     # mu: explicit amplitude + phase — better geometry than beta_c/beta_s
    #     A_mu  = pm.HalfNormal("A_mu",  sigma=1.0)   # amplitude > 0
    #     phi_mu = pm.VonMises("phi_mu", mu=0, kappa=0.5)  # phase on circle

    #     mu = pm.Deterministic("mu", pt.arctan2(
    #             A_mu * pt.sin(x - phi_mu),
    #             A_mu * pt.cos(x - phi_mu)))

    #     # kappa: log-linear in circular basis
    #     kappa_0 = pm.Normal("kappa_0", mu=1.0, sigma=1.0)  # prior centred on exp(1)~2.7
    #     kappa_c = pm.Normal("kappa_c", mu=0,   sigma=0.5)
    #     kappa_s = pm.Normal("kappa_s", mu=0,   sigma=0.5)

    #     log_kappa = kappa_0 + kappa_c * pt.cos(x) + kappa_s * pt.sin(x)
    #     kappa     = pm.Deterministic("kappa", pt.exp(
    #                     pt.clip(log_kappa, -2, 5)))  # bounds kappa to ~[0.14, 150]

    #     y_obs = pm.VonMises("y_obs", mu=mu, kappa=kappa, observed=y)

    # with model:
    #     trace = pm.sample(
    #         draws=2000, tune=2000,
    #         chains=4, target_accept=0.95,
    #         init="adapt_diag",
    #     )
        
        
#%%
# import arviz as az
# import matplotlib.pyplot as plt
# import numpy as np

# # ── summary table ─────────────────────────────────────────────────────────────
# print(az.summary(trace, var_names=["A_mu", "phi_mu", "kappa_0", "kappa_c", "kappa_s"]))
# # r_hat should be <1.01 for all parameters
# # ess_bulk should be >400 for all parameters

# # ── trace plots — visually confirm chain mixing ───────────────────────────────
# az.plot_trace(trace, var_names=["A_mu", "phi_mu", "kappa_0", "kappa_c", "kappa_s"])
# plt.tight_layout()
# plt.show()

# # ── posterior predictive check ────────────────────────────────────────────────
# with model:
#     ppc = pm.sample_posterior_predictive(trace)

# az.plot_ppc(ppc, observed_rug=True)
# plt.show()

# # ── visualise mu(x) and kappa(x) across the circle ───────────────────────────
# x_grid = np.linspace(-np.pi, np.pi, 200)

# # posterior mean parameters
# A_mu_mean   = float(trace.posterior["A_mu"].mean())
# phi_mu_mean = float(trace.posterior["phi_mu"].mean())
# k0_mean     = float(trace.posterior["kappa_0"].mean())
# kc_mean     = float(trace.posterior["kappa_c"].mean())
# ks_mean     = float(trace.posterior["kappa_s"].mean())

# mu_grid    = np.arctan2(A_mu_mean * np.sin(x_grid - phi_mu_mean),
#                         A_mu_mean * np.cos(x_grid - phi_mu_mean))
# kappa_grid = np.exp(k0_mean + kc_mean*np.cos(x_grid) + ks_mean*np.sin(x_grid))

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# ax1.plot(x_grid, mu_grid)
# ax1.set_ylabel("μ(x)  [predicted direction]")
# ax1.axhline(0, color='k', lw=0.5, ls='--')

# ax2.plot(x_grid, kappa_grid, color='C1')
# ax2.set_ylabel("κ(x)  [concentration]")
# ax2.set_xlabel("x  [predictor phase]")
# plt.tight_layout()
# plt.show()

# #%%
# x_new = np.linspace(-np.pi, np.pi, 200)

# # pull posterior samples as numpy — shape (chains*draws,)
# A_mu_samp   = trace.posterior["A_mu"].values.reshape(-1)
# phi_mu_samp = trace.posterior["phi_mu"].values.reshape(-1)
# k0_samp     = trace.posterior["kappa_0"].values.reshape(-1)
# kc_samp     = trace.posterior["kappa_c"].values.reshape(-1)
# ks_samp     = trace.posterior["kappa_s"].values.reshape(-1)

# # broadcast: (n_samples, len(x_new))
# x_new = x_new[None, :]                          
# A   = A_mu_samp[:, None]
# phi = phi_mu_samp[:, None]

# mu_samp    = np.arctan2(A * np.sin(x_new - phi),
#                         A * np.cos(x_new - phi))   # (n_samples, len(x_new))
# kappa_samp = np.exp(np.clip(
#                 k0_samp[:,None] + kc_samp[:,None]*np.cos(x_new)
#                              + ks_samp[:,None]*np.sin(x_new),
#                 -2, 5))

# # posterior mean and 94% credible interval at each x
# mu_mean = np.arctan2(np.mean(np.sin(mu_samp), axis=0),   # circular mean
#                      np.mean(np.cos(mu_samp), axis=0))
# mu_lo   = np.percentile(mu_samp,  3, axis=0)
# mu_hi   = np.percentile(mu_samp, 97, axis=0)

# kappa_mean = np.mean(kappa_samp, axis=0)
# kappa_lo   = np.percentile(kappa_samp,  3, axis=0)
# kappa_hi   = np.percentile(kappa_samp, 97, axis=0)

# # ── plot ──────────────────────────────────────────────────────────────────────
# x_plot = x_new.squeeze()
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

# ax1.plot(x_plot, mu_mean, label="posterior mean μ")
# ax1.fill_between(x_plot, mu_lo, mu_hi, alpha=0.25, label="94% CI")
# ax1.set_ylabel("predicted μ(x)")
# ax1.legend(fontsize=9)

# ax2.plot(x_plot, kappa_mean, color="C1", label="posterior mean κ")
# ax2.fill_between(x_plot, kappa_lo, kappa_hi, alpha=0.25, color="C1", label="94% CI")
# ax2.set_ylabel("predicted κ(x)")
# ax2.set_xlabel("x")
# ax2.legend(fontsize=9)

# plt.tight_layout()
# plt.show()