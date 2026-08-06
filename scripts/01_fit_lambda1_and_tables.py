#!/usr/bin/env python3
"""
Primary data-reduction script for Nesvinga, Nyambuya & Jones (2026b),
"Earth-Flyby Velocity Anomalies within the Azimuthally Symmetric Theory of
Gravitation". This is the script referenced in the manuscript's "Data and
Code Availability" section (Appendix, \\label{app:code}).

Reads the primary flyby dataset (../data/flyby_data.csv, identical to
Table I of the manuscript) and reproduces, from that data alone:

  * Table II  (derived orbital quantities: r_p, V_p, epsilon, Delta cos
              delta, and the per-flyby diagnostic lambda_1^(i))
  * Eq. (7)   the core 4-flyby (anomalous-only) global fit of lambda_1
  * Eq. (8)   the 5-flyby fit including Galileo II
  * Table III (ASTG predictions vs. observations)
  * Table IV  (comparison with the independent Anderson-constant estimate)

The lambda_1 fit is linear (no root-finding needed, unlike the Binet-route
k extraction in 03_binet_route_two_parameter_fit.py), computed by ordinary
least squares with no intercept, exactly as described in the manuscript's
Sec. IV.

Run with:  python3 01_fit_lambda1_and_tables.py
Requires:  numpy, scipy
"""
import csv
import os
import numpy as np
from scipy import stats

GM = 3.986004418e14    # m^3 s^-2, CODATA 2022, Earth
R_E = 6.371e6           # m, CODATA 2022, Earth
c0 = 2.99792458e8       # m/s
s_g = 2 * GM / c0**2    # Earth's Schwarzschild radius
kappa_A = 3.099e-6      # Anderson (2008) constant, kappa_A = 2*omega_E*R_E/c0

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "flyby_data.csv")


def load_data():
    rows = {}
    order = []
    with open(DATA_PATH, newline="") as f:
        for row in csv.DictReader(f):
            name = row["spacecraft"]
            order.append(name)
            rows[name] = dict(
                h_p_km=float(row["h_p_km"]),
                V_inf_kms=float(row["V_inf_kms"]) if row["V_inf_kms"] else None,
                dVp_mms=float(row["dVp_mms"]),
                sigma_mms=float(row["sigma_mms"]) if row["sigma_mms"] else None,
                delta_in_deg=float(row["delta_in_deg"]),
                delta_out_deg=float(row["delta_out_deg"]),
                anomalous=row["anomalous"] == "True",
            )
    return rows, order


def derived_quantities(hp_km, Vinf_kms):
    """rp, eps, Vp from perigee altitude and hyperbolic excess speed."""
    rp = R_E + hp_km * 1e3
    V = Vinf_kms * 1e3
    eps = 1 + rp * V**2 / GM
    Vp = np.sqrt(V**2 + 2 * GM / rp)
    return rp, eps, Vp


def dcosd(delta_in, delta_out):
    return np.cos(np.radians(delta_in)) - np.cos(np.radians(delta_out))


def lambda1_perflyby(hp_km, Vinf_kms, dVp_mms, delta_in, delta_out):
    """Invert the vis-viva formula for lambda_1 (Eq. lambda1_invert).
    Linear in lambda_1, so this is direct division, not root-finding."""
    rp, eps, Vp = derived_quantities(hp_km, Vinf_kms)
    dcos = dcosd(delta_in, delta_out)
    lam1 = (dVp_mms * 1e-3 / Vp) * rp * (1 + eps) / (s_g * dcos)
    return lam1, rp, eps, Vp, dcos


def global_fit(rows, names):
    """Ordinary least squares, no intercept: Eq. (7) or (8) depending on
    which set of names is passed in."""
    X, Y = [], []
    for name in names:
        r = rows[name]
        rp, eps, Vp = derived_quantities(r["h_p_km"], r["V_inf_kms"])
        dcos = dcosd(r["delta_in_deg"], r["delta_out_deg"])
        X.append(s_g * dcos / (rp * (1 + eps)))
        Y.append((r["dVp_mms"] * 1e-3) / Vp)
    X, Y = np.array(X), np.array(Y)
    lam1_hat = np.sum(X * Y) / np.sum(X * X)
    resid = Y - lam1_hat * X
    n, dof = len(X), len(X) - 1
    R2 = 1 - np.sum(resid**2) / np.sum((Y - Y.mean())**2)
    SE = np.sqrt(np.sum(resid**2) / dof / np.sum(X**2))
    t = lam1_hat / SE
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    return lam1_hat, SE, R2, t, p, n, dof


def main():
    rows, order = load_data()
    anomalous = [n for n in order if rows[n]["anomalous"]]

    print("=" * 78)
    print("Table II: derived orbital quantities and per-flyby lambda_1")
    print("=" * 78)
    print(f"{'Spacecraft':12s} {'r_p(Mm)':>9s} {'V_p(km/s)':>10s} "
          f"{'eps':>7s} {'Dcosd':>9s} {'lambda1(i)x1e-3':>16s}")
    table2 = {}
    for name in order:
        r = rows[name]
        if r["V_inf_kms"] is None:
            continue  # Rosetta II/III have no published V_inf
        lam1, rp, eps, Vp, dcos = lambda1_perflyby(
            r["h_p_km"], r["V_inf_kms"], r["dVp_mms"],
            r["delta_in_deg"], r["delta_out_deg"])
        table2[name] = dict(lam1=lam1, rp=rp, eps=eps, Vp=Vp, dcos=dcos)
        print(f"{name:12s} {rp/1e6:9.3f} {Vp/1e3:10.3f} {eps:7.3f} "
              f"{dcos:+9.4f} {lam1/1e3:16.2f}")

    print("\n" + "=" * 78)
    print("Eq. (7): global fit, 4 anomalous flybys only")
    print("=" * 78)
    lam1, SE, R2, t, p, n, dof = global_fit(rows, anomalous)
    print(f"  lambda1_hat = ({lam1/1e3:.3f} +/- {SE/1e3:.3f}) x 1e3")
    print(f"  R^2 = {R2:.3f}, t = {t:.2f}, p = {p:.4f}, n = {n}, dof = {dof}")

    print("\n" + "=" * 78)
    print("Eq. (8): global fit, 5 flybys (incl. Galileo II, drag-contaminated)")
    print("=" * 78)
    five = anomalous + ["Galileo II"]
    lam1_5, SE_5, R2_5, t_5, p_5, n_5, dof_5 = global_fit(rows, five)
    print(f"  lambda1_hat = ({lam1_5/1e3:.3f} +/- {SE_5/1e3:.3f}) x 1e3")
    print(f"  R^2 = {R2_5:.3f}, t = {t_5:.2f}, p = {p_5:.4f}, n = {n_5}, dof = {dof_5}")

    print("\n" + "=" * 78)
    print("Table III: ASTG predictions (using the 4-flyby fit lambda1) vs. observed")
    print("=" * 78)
    print(f"{'Spacecraft':12s} {'obs(mm/s)':>10s} {'sigma':>8s} {'ASTG pred(mm/s)':>16s}")
    predict_names = anomalous + ["Galileo II", "MESSENGER"]
    for name in predict_names:
        r = rows[name]
        Xi = s_g * dcosd(r["delta_in_deg"], r["delta_out_deg"]) / (
            table2[name]["rp"] * (1 + table2[name]["eps"]))
        pred_mms = Xi * lam1 * table2[name]["Vp"] * 1e3
        print(f"{name:12s} {r['dVp_mms']:10.4f} {r['sigma_mms']:8.3f} {pred_mms:16.4f}")

    print("\n" + "=" * 78)
    print("Table IV: comparison with the independent Anderson-constant estimate")
    print("=" * 78)
    print(f"{'Spacecraft':12s} {'lam1_meas':>10s} {'lam1_And':>10s} {'ratio(And/meas)':>16s}")
    for name in predict_names:
        r = rows[name]
        rp, eps, Vp = table2[name]["rp"], table2[name]["eps"], table2[name]["Vp"]
        dcos = table2[name]["dcos"]
        dVp_And_mms = kappa_A * dcos * r["V_inf_kms"] * 1e3  # mm/s (V_inf in m/s * 1e3 -> already mm via *1e3... see below)
        # kappa_A * dcos is dimensionless-times-dimensionless; multiply by
        # V_infinity (in m/s) to get a velocity anomaly in m/s, then to mm/s.
        dVp_And_ms = kappa_A * dcos * (r["V_inf_kms"] * 1e3)
        lam1_And = (dVp_And_ms / Vp) * rp * (1 + eps) / (s_g * dcos)
        ratio = lam1_And / table2[name]["lam1"]
        print(f"{name:12s} {table2[name]['lam1']/1e3:10.2f} {lam1_And/1e3:10.2f} {ratio:16.2f}")


if __name__ == "__main__":
    main()
    print("\nDone. Compare the tables above against Tables II-IV and Eqs. (7)-(8)")
    print("of the manuscript.")
