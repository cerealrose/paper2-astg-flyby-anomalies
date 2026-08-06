#!/usr/bin/env python3
"""
AIC reconciliation script for the vis-viva vs. Binet-route flyby-formula
comparison, referenced in Nesvinga, Nyambuya & Jones (2026b), Sec. II.C:
"We do not report an Akaike Information Criterion comparison between the
two: several standard conventions gave inconsistent results, including
disagreement on which route they favour."

PURPOSE: an earlier draft quoted AIC = -55.3 (vis-viva) vs. -54.2 (Binet
route), favouring the vis-viva route. Attempting to reproduce this figure
turned up no convention (of the four implemented below) that recovers it,
and one convention reverses which route it favours. Rather than silently
drop the discrepancy, this script is provided so a referee -- or anyone
with access to whatever originally produced -55.3/-54.2 -- can check each
convention against that original computation directly.

Run with:  python3 04_aic_reconciliation.py
Requires:  numpy, scipy
"""
import numpy as np
from scipy import stats

GM = 3.986004418e14
R_E = 6.371e6
c0 = 2.99792458e8
s_g = 2 * GM / c0**2

flybys_corrected = {
    "Galileo I": [960.0, 8.949, 2.560, -12.52, -34.15],
    "NEAR":      [539.0, 6.851, 7.210, -20.76, -71.96],
    "Cassini":   [1171.0, 16.010, -1.683, -12.92, -4.99],
    "Rosetta I": [1955.0, 3.863, 0.660, -2.81, -34.29],
}
sigmas_corrected = {"Galileo I": 0.195, "NEAR": 0.006, "Cassini": 0.842, "Rosetta I": 0.011}


def derived(hp_km, Vinf_kms):
    rp = R_E + hp_km * 1e3
    v = Vinf_kms * 1e3
    eps = 1 + rp * v**2 / GM
    vp = np.sqrt(v**2 + 2 * GM / rp)
    return rp, eps, vp


def visviva_fit(flybys):
    """Model 1: vis-viva route, 1 parameter (lambda_1), no intercept."""
    X, Y = [], []
    for name, (hp, vinf, dVp, din, dout) in flybys.items():
        rp, eps, vp = derived(hp, vinf)
        dcos = np.cos(np.radians(din)) - np.cos(np.radians(dout))
        X.append(s_g * dcos / (rp * (1 + eps)))
        Y.append((dVp * 1e-3) / vp)
    X, Y = np.array(X), np.array(Y)
    lam = np.sum(X * Y) / np.sum(X**2)
    resid = Y - lam * X
    resid_mm_s = resid * np.array([derived(hp, vinf)[2] for hp, vinf, *_ in flybys.values()]) * 1e3
    return dict(k=1, n=len(X), resid_dimless=resid, resid_mm_s=resid_mm_s, param=lam)


def binet_fit(flybys):
    """Model 2: Binet route, 2 parameters (slope + intercept of k vs. Dcosd)."""
    ks, dcoss = [], []
    for name, (hp, vinf, dVp, din, dout) in flybys.items():
        rp, eps, vp = derived(hp, vinf)
        eta = np.arccos(-1 / eps)
        dcos = np.cos(np.radians(din)) - np.cos(np.radians(dout))
        ratio = (dVp * 1e-3) / vp
        denom = 2 * ((eps**2 + 2) * eta + 3 * np.sqrt(eps**2 - 1)) / (1 + eps)**2
        ks.append(ratio / denom)
        dcoss.append(dcos)
    ks, dcoss = np.array(ks), np.array(dcoss)
    Xmat = np.column_stack([np.ones_like(dcoss), dcoss])
    beta, *_ = np.linalg.lstsq(Xmat, ks, rcond=None)
    resid_k = ks - Xmat @ beta
    return dict(k=2, n=len(ks), resid_k=resid_k, param=beta)


def aic_ols_native(resid, nparam):
    """Standard OLS AIC: n*ln(RSS/n) + 2k, in the fit's own (native) units."""
    n = len(resid)
    rss = np.sum(resid**2)
    return n * np.log(rss / n) + 2 * nparam


def aic_gaussian_full(resid, nparam):
    """Full Gaussian log-likelihood AIC: n*ln(2*pi) + n*ln(RSS/n) + n + 2k."""
    n = len(resid)
    rss = np.sum(resid**2)
    return n * np.log(2 * np.pi) + n * np.log(rss / n) + n + 2 * nparam


def aic_chi2(resid, sigmas, nparam):
    """Chi-square-based AIC using quoted per-flyby uncertainties: chi2 + 2k."""
    chi2 = np.sum((resid / sigmas)**2)
    return chi2 + 2 * nparam


def aicc_correction(aic_value, n, nparam):
    """Small-sample corrected AIC (AICc)."""
    if n - nparam - 1 <= 0:
        return np.nan
    return aic_value + 2 * nparam * (nparam + 1) / (n - nparam - 1)


def main():
    print("=" * 78)
    print("Testing four AIC conventions against the corrected data")
    print("(none of these reproduce the earlier draft's quoted -55.3/-54.2;")
    print(" see the module docstring)")
    print("=" * 78)

    vv = visviva_fit(flybys_corrected)
    bn = binet_fit(flybys_corrected)

    print("\n-- Vis-viva route (k=1 param) --")
    for space, resid in [("dimensionless (native fit space)", vv["resid_dimless"]),
                          ("physical mm/s", vv["resid_mm_s"])]:
        a1 = aic_ols_native(resid, vv["k"])
        a2 = aic_gaussian_full(resid, vv["k"])
        print(f"  [{space}] AIC_ols={a1:8.2f}  AIC_gauss={a2:8.2f}  "
              f"AICc_ols={aicc_correction(a1, vv['n'], vv['k']):8.2f}")
    sig_arr = np.array([sigmas_corrected[n] for n in flybys_corrected])
    a3 = aic_chi2(vv["resid_mm_s"], sig_arr, vv["k"])
    print(f"  [chi2, quoted sigma] AIC_chi2={a3:10.2f}")

    print("\n-- Binet route (k=2 params) --")
    a1 = aic_ols_native(bn["resid_k"], bn["k"])
    a2 = aic_gaussian_full(bn["resid_k"], bn["k"])
    print(f"  [k-space, native fit space] AIC_ols={a1:8.2f}  AIC_gauss={a2:8.2f}  "
          f"AICc_ols={aicc_correction(a1, bn['n'], bn['k']):8.2f}")

    print("\n" + "=" * 78)
    print("If you have the ORIGINAL script/notebook that produced -55.3 and")
    print("-54.2, compare its AIC call line-by-line against the four aic_*")
    print("functions above -- that is a faster path to reconciliation than")
    print("searching for further conventions.")
    print("=" * 78)


if __name__ == "__main__":
    main()
    print("\nDone.")
