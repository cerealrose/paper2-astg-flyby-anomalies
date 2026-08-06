#!/usr/bin/env python3
"""
Reproduces the two-parameter Binet-route regression quoted in Nesvinga,
Nyambuya & Jones (2026b), Sec. II.C ("Relation to the Binet-equation
route"): R^2=0.938, t=5.53, p=0.031, n=4, 2 d.o.f.

Method: invert the boxed Binet-route formula from the companion paper
(Nesvinga et al. 2026a, Eq. B5) for a per-flyby coupling k_i, then regress
k_i against Delta cos(delta) with two free parameters (slope, intercept),
unlike the single-parameter, no-intercept vis-viva fit of script 01.

Before trusting this on the Anderson(2008)-corrected data, this script
first validates the method by reproducing the PRE-correction result
(R^2=0.936, p=0.033, using the uncorrected Cassini dVp=-0.420 mm/s that
appeared in an earlier draft) -- if the method didn't reproduce that
already-known number on the old data, it would not be trustworthy on the
new data either.

Run with:  python3 02_binet_route_two_parameter_fit.py
Requires:  numpy, scipy
"""
import numpy as np
from scipy import stats

GM = 3.986004418e14
R_E = 6.371e6
c0 = 2.99792458e8
s_g = 2 * GM / c0**2


def binet_k_extract(hp_km, Vinf_kms, dVp_mms, delta_in, delta_out):
    """Invert the boxed Binet-route formula, Eq. (B5) of the companion
    paper, for the per-flyby coupling k."""
    rp = R_E + hp_km * 1e3
    V = Vinf_kms * 1e3
    eps = 1 + rp * V**2 / GM
    Vp = np.sqrt(V**2 + 2 * GM / rp)
    eta = np.arccos(-1 / eps)
    dcos = np.cos(np.radians(delta_in)) - np.cos(np.radians(delta_out))
    ratio = (dVp_mms * 1e-3) / Vp
    denom = 2 * ((eps**2 + 2) * eta + 3 * np.sqrt(eps**2 - 1)) / (1 + eps)**2
    k = ratio / denom
    return k, dcos


def two_param_fit(flybys, label):
    """flybys: dict of name -> [hp_km, Vinf_kms, dVp_mms, delta_in, delta_out]"""
    ks, dcoss = [], []
    print(f"\n[{label}]")
    for name, (hp, vinf, dVp, din, dout) in flybys.items():
        k, dcos = binet_k_extract(hp, vinf, dVp, din, dout)
        ks.append(k)
        dcoss.append(dcos)
        print(f"  {name:12s} k={k:+.4e}  Dcosd={dcos:+.4f}")
    ks, dcoss = np.array(ks), np.array(dcoss)

    Xmat = np.column_stack([np.ones_like(dcoss), dcoss])
    beta, *_ = np.linalg.lstsq(Xmat, ks, rcond=None)
    pred = Xmat @ beta
    resid = ks - pred
    n, p_params = len(ks), 2
    dof = n - p_params
    RSS = np.sum(resid**2)
    R2 = 1 - RSS / np.sum((ks - np.mean(ks))**2)
    sigma2 = RSS / dof
    cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(cov))
    t_slope = beta[1] / se[1]
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), dof))

    print(f"  intercept = {beta[0]:.4e} +/- {se[0]:.2e}")
    print(f"  slope     = {beta[1]:.4e} +/- {se[1]:.2e}")
    print(f"  R^2 = {R2:.4f}, t(slope) = {t_slope:.3f}, "
          f"p(slope) = {p_slope:.4f}, n = {n}, dof = {dof}")
    return R2, t_slope, p_slope


def main():
    print("=" * 72)
    print("Validation step: reproduce the ORIGINAL (pre-correction) result")
    print("first, using Cassini's uncorrected dVp=-0.420 mm/s. If the")
    print("method doesn't reproduce R^2=0.936, p=0.033 here, it cannot be")
    print("trusted on the corrected data below.")
    print("=" * 72)
    flybys_original = {
        "Galileo I": [960.0, 8.949, 2.560, -12.52, -34.15],
        "NEAR":      [539.0, 6.851, 7.210, -20.76, -71.96],
        "Cassini":   [1171.0, 16.010, -0.420, -12.92, -4.99],
        "Rosetta I": [1955.0, 3.863, 0.660, -2.81, -34.29],
    }
    R2_orig, t_orig, p_orig = two_param_fit(flybys_original, "uncorrected (validation)")
    assert abs(R2_orig - 0.936) < 0.001, "validation failed to reproduce R^2=0.936"
    assert abs(p_orig - 0.033) < 0.001, "validation failed to reproduce p=0.033"
    print("\n  Validation PASSED: method reproduces the known original result.")

    print("\n" + "=" * 72)
    print("Corrected data (Cassini's dVp per Anderson et al. 2008 Table I)")
    print("=" * 72)
    flybys_corrected = {
        "Galileo I": [960.0, 8.949, 2.560, -12.52, -34.15],
        "NEAR":      [539.0, 6.851, 7.210, -20.76, -71.96],
        "Cassini":   [1171.0, 16.010, -1.683, -12.92, -4.99],
        "Rosetta I": [1955.0, 3.863, 0.660, -2.81, -34.29],
    }
    R2_corr, t_corr, p_corr = two_param_fit(flybys_corrected, "Anderson(2008)-corrected")
    print(f"\n  Matches manuscript Sec. II.C: R^2=0.938, t=5.53, p=0.031")
    assert abs(R2_corr - 0.938) < 0.001
    assert abs(t_corr - 5.53) < 0.01
    assert abs(p_corr - 0.031) < 0.001

    print("\n  Note the corrected fit barely moved from the original")
    print(f"  (R^2: {R2_orig:.3f}->{R2_corr:.3f}) -- the two-parameter fit's")
    print("  extra intercept absorbs Cassini's correction almost entirely,")
    print("  unlike script 01's zero-intercept vis-viva fit, whose R^2")
    print("  dropped from 0.920 to 0.904 under the same correction. Both")
    print("  papers report this asymmetry explicitly.")


if __name__ == "__main__":
    main()
    print("\nAll checks in this script passed.")
