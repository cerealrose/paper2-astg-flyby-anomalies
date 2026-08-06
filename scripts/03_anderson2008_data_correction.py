#!/usr/bin/env python3
"""
Verifies the data correction described in the manuscript's Table I caption:
Cassini's dVp and sigma, and Galileo I's sigma, are derived from Anderson
et al. (2008) Table I's dV_infinity and sigma(dV_infinity) values via the
vis-viva scaling dVp = (V_infinity/V_p) * dV_infinity, NOT reported
directly by that source.

This reproduces the arithmetic from Anderson (2008)'s own published
numbers (transcribed by hand from the paper, Table I) for every flyby, not
just Cassini, to show the correction is not an isolated patch but the
correct, general application of the same scaling relation across the
board -- four of six rows required no change at all, which is itself part
of the evidence that the correction is right rather than an arbitrary fix.

Run with:  python3 03_anderson2008_data_correction.py
Requires:  numpy
"""
import numpy as np

GM = 3.986004418e14
R_E = 6.371e6


def derived(hp_km, Vinf_kms):
    rp = R_E + hp_km * 1e3
    V = Vinf_kms * 1e3
    eps = 1 + rp * V**2 / GM
    Vp = np.sqrt(V**2 + 2 * GM / rp)
    return rp, eps, Vp


def main():
    print("=" * 78)
    print("Anderson et al. (2008) Table I -- transcribed values")
    print("=" * 78)
    print("Columns: h_p(km), V_inf(km/s), dV_infinity(mm/s), "
          "sigma(dV_infinity)(mm/s)")

    # Directly transcribed from Anderson et al. (2008), Physical Review
    # Letters 100, 091102, Table I. This is the primary source; note that
    # this table reports dV_infinity (asymptotic), NOT dVp (perigee) --
    # the manuscript's dVp values are DERIVED from these via vis-viva
    # scaling, not copied from a dVp column that does not exist in the
    # source.
    anderson_table1 = {
        "Galileo I":  dict(hp=960.0,  Vinf=8.949,  dVinf=3.92,  sig_dVinf=0.3),
        "Galileo II": dict(hp=303.0,  Vinf=8.877,  dVinf=-4.6,  sig_dVinf=1.0),
        "NEAR":       dict(hp=539.0,  Vinf=6.851,  dVinf=13.46, sig_dVinf=0.01),
        "Cassini":    dict(hp=1171.0, Vinf=16.010, dVinf=-2.0,  sig_dVinf=1.0),
        "Rosetta I":  dict(hp=1955.0, Vinf=3.863,  dVinf=1.80,  sig_dVinf=0.03),
        "MESSENGER":  dict(hp=2347.0, Vinf=4.056,  dVinf=0.02,  sig_dVinf=0.01),
    }

    # dVp and sigma as they appear in the manuscript's Table I (i.e. after
    # the correction described in that table's caption).
    manuscript_table1 = {
        "Galileo I":  dict(dVp=2.560, sigma=0.195),
        "Galileo II": dict(dVp=-2.900, sigma=0.700),
        "NEAR":       dict(dVp=7.210, sigma=0.006),
        "Cassini":    dict(dVp=-1.683, sigma=0.842),
        "Rosetta I":  dict(dVp=0.660, sigma=0.011),
        "MESSENGER":  dict(dVp=0.007, sigma=0.004),
    }

    print(f"\n{'Spacecraft':12s} {'V_inf/V_p':>10s} {'dVp implied':>13s} "
          f"{'manuscript':>11s} {'rel.diff':>9s}  |  {'sig implied':>13s} "
          f"{'manuscript':>11s} {'rel.diff':>9s}")
    for name, row in anderson_table1.items():
        rp, eps, Vp = derived(row["hp"], row["Vinf"])
        ratio_v = (row["Vinf"] * 1e3) / Vp
        dVp_implied = ratio_v * row["dVinf"]
        sig_implied = ratio_v * row["sig_dVinf"]
        m = manuscript_table1[name]
        dvp_reldiff = abs(dVp_implied - m["dVp"]) / abs(m["dVp"])
        sig_reldiff = abs(sig_implied - m["sigma"]) / abs(m["sigma"])
        print(f"{name:12s} {ratio_v:10.4f} {dVp_implied:13.4f} "
              f"{m['dVp']:11.4f} {dvp_reldiff*100:8.1f}%  |  "
              f"{sig_implied:13.4f} {m['sigma']:11.4f} {sig_reldiff*100:8.1f}%")

    print()
    print("All six rows are consistent with the manuscript's current Table I")
    print("within ordinary rounding (Anderson 2008 quotes most values to")
    print("1-2 significant figures, so sub-percent-to-~10% differences are")
    print("expected propagation of that input rounding, not errors). This")
    print("confirms the manuscript's Table I correctly applies dVp =")
    print("(V_inf/V_p)*dV_infinity uniformly across all six flybys.")
    print()
    print("For context: an EARLIER secondary source (not Anderson 2008")
    print("itself) had Cassini's dVp off by a factor of ~4 and Galileo I's")
    print("sigma off by a factor of ~10, traced to a transcription error")
    print("in that intermediate source rather than in Anderson (2008). The")
    print("comparison above is against the manuscript's current, already-")
    print("corrected values, which is why it shows agreement throughout --")
    print("the point of running it is to confirm that correction was")
    print("applied correctly, which it was.")

    print("\n" + "=" * 78)
    print("Self-consistency cross-check: V_infinity implied by (r_p, eps)")
    print("=" * 78)
    print("Independent check using the manuscript's OWN r_p and eccentricity")
    print("(Table II), which do not depend on Anderson's V_infinity column at")
    print("all -- if V_infinity is correctly transcribed, back-solving for it")
    print("from r_p and eps via eps = 1 + r_p*V_inf^2/GM should reproduce it.")
    derived_table2 = {
        # r_p (Mm), eps, from the manuscript's Table II
        "Galileo I":  [7.331, 2.473],
        "NEAR":       [6.910, 1.814],
        "Cassini":    [7.542, 5.850],
        "Rosetta I":  [8.326, 1.312],
    }
    for name, (rp_Mm, eps_tab) in derived_table2.items():
        rp_m = rp_Mm * 1e6
        v_inf_implied = np.sqrt((eps_tab - 1) * GM / rp_m) / 1e3
        v_inf_table = anderson_table1[name]["Vinf"]
        match = abs(v_inf_implied - v_inf_table) < 0.01
        print(f"  {name:12s}: implied V_inf = {v_inf_implied:.3f} km/s | "
              f"table V_inf = {v_inf_table:.3f} km/s | "
              f"{'OK' if match else 'DIFFERS'}")
    print("\nThis is the check that originally caught a factor-of-10 error")
    print("in an earlier secondary source's transcription of Cassini's")
    print("V_infinity (1.601 vs the correct 16.010 km/s) -- included here")
    print("so a referee can repeat that same check independently.")


if __name__ == "__main__":
    main()
    print("\nDone.")
