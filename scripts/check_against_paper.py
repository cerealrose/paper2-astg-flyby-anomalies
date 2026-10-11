"""Recompute the numbers of the paper and compare each with the printed value at the printed precision.
Prints one line per check and a summary; exits with status 1 if any check fails.  Runs in about half a minute
(the covariance analysis of Table 5 dominates)."""
import sys
from math import comb
import numpy as np
import astg
from galileo_projected_area import totals, largest_area, hga_disc_area
from galileo_drag import geometry, density_profile, unit_impulse, drag_cases, residual, pointing_sensitivity, appendix_c
from cassini_refit import table3
from juno_prediction import juno, z_at, appendix_a, SIGMA_ADOPTED
from juno_covariance import table5, perigee_state, check_partials
from two_term_fit import two_term_fit
from juno_candidates import multipole_l2, thermal_recoil, spin_ratios, table6, IORIO_2009
from juno_drag import juno_drag
from selection_rule import table7, null_deviations, sigma_threshold
from refit import refit, table9

results = []


def check(label, value, printed, decimals):
    ok = round(value, decimals) == round(printed, decimals) or abs(value - printed) <= 0.5 * 10**(-decimals) + 1e-12
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} computed {value:+.{decimals + 3}f}  printed {printed:+.{decimals}f}")


def check_sci(label, value, printed, sig):
    """Compare in scientific notation with sig significant figures."""
    ok = float(f"{value:.{sig - 1}e}") == printed
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label:62s} computed {value:.{sig + 1}e}  printed {printed:.{sig - 1}e}")


def check_true(label, condition, detail=''):
    results.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {label:62s} {detail}")


fl = astg.load_flybys()
four = astg.anomalous_fit(fl)

# Section 1
check('kappa_A (1e-6)', astg.KAPPA_A * 1e6, 3.099, 3)
check('R_g (1e-3 m)', astg.RG * 1e3, 8.870, 3)
check('lambda1, four anomalous encounters (1e3)', four['lam'] / 1e3, 3.902, 3)
check('standard error (1e3)', four['se'] / 1e3, 0.576, 3)

# Table 1 and Section 2.2
t = totals()
for lab, val, pr in (('axial min', t['ax'][0], 9.5), ('axial max', t['ax'][1], 14.8), ('broadside min', t['tr'][0], 10.9),
                     ('broadside max', t['tr'][1], 16.1)):
    check(f'Table 1 total {lab} (m^2)', val, pr, 1)
g = geometry(); hs, rho_h = density_profile()
cases = drag_cases(g, hs, rho_h)
check('Sec. 2.2 alpha at perigee (deg)', g['alpha_perigee'], 56.0, 1)
check('Sec. 2.2 alpha over 90% of the drag, low (deg)', cases['central']['alpha90'][0], 52.7, 1)
check('Sec. 2.2 alpha over 90% of the drag, high (deg)', cases['central']['alpha90'][1], 59.2, 1)
check('Sec. 2.2 projected area at perigee, low (m^2)', cases['lower']['area'], 14.3, 1)
check('Sec. 2.2 projected area at perigee, high (m^2)', cases['upper']['area'], 21.6, 1)
check_true('Sec. 2.2 areas close to the largest value at any angle',
           cases['lower']['area'] / largest_area(t['ax'][0], t['tr'][0]) > 0.98 and cases['upper']['area'] / largest_area(t['ax'][1], t['tr'][1]) > 0.98,
           f"{cases['lower']['area'] / largest_area(t['ax'][0], t['tr'][0]):.3f}, {cases['upper']['area'] / largest_area(t['ax'][1], t['tr'][1]):.3f} of the largest")
check('Sec. 2.2 pointing error of 5 deg, at most (%)', pointing_sensitivity(g, hs, rho_h) * 100, 1.6, 1)
check('Sec. 2.2 4.8 m dish end-on (m^2)', hga_disc_area(), 18.1, 1)
check('Fig. 1 caption, deployed HGA at alpha (m^2)', hga_disc_area() * np.cos(np.radians(g['alpha_perigee'])), 10.1, 1)

# Sections 2.1 and 2.3
r = residual(cases)
check('Sec. 2.1 measured change at perigee (mm/s)', r['measured'], -5.04, 2)
check('Sec. 2.1 its uncertainty (mm/s)', r['sigma_measured'], 0.63, 2)
check_sci('Sec. 2.3 density at closest approach (kg/m^3)', rho_h[0], 3.12e-11, 3)
check('Sec. 2.3 relative speed at perigee (km/s)', g['vr_perigee'], 14.45, 2)
check('Sec. 2.3 inertial perigee speed (km/s)', g['vp'], 14.08, 2)
check('Sec. 2.3 rp (Mm)', g['rp'] / 1e3, 6.674, 3)
check('Sec. 2.3 eps', g['ecc'], 2.319, 3)
u = unit_impulse(g, hs, rho_h)
check('Sec. 2.3 Eq. (4) for C_D A/m = 1 m^2/kg (mm/s)', -u['k_rot'], -429.7, 1)
check('Sec. 2.3 local scale height (km)', u['H'], 52.4, 1)
check('Sec. 2.3 single exponential, % less (inertial)', (1 - u['l_exp'] / u['l_eff']) * 100, 4.5, 1)
check('Sec. 2.3 single exponential, % less (relative speed)', (1 - u['k_exp_rot'] / u['k_rot']) * 100, 4.5, 1)

# Table 2
for name, area, cda, dv in (('lower', 14.3, 0.0115, -4.94), ('central', 18.0, 0.0158, -6.81), ('upper', 21.6, 0.0208, -8.93)):
    check(f'Table 2 {name} area (m^2)', cases[name]['area'], area, 1)
    check(f'Table 2 {name} C_D A/m (m^2/kg)', cases[name]['cda_m'], cda, 4)
    check(f'Table 2 {name} drag impulse (mm/s)', cases[name]['dv'], dv, 2)

# Section 2.4
check('Sec. 2.4 residual (mm/s)', r['residual'], 1.76, 2)
check('Sec. 2.4 residual range, low (mm/s)', r['low'], -0.11, 2)
check('Sec. 2.4 residual range, high (mm/s)', r['high'], 3.88, 2)
check('Sec. 2.4 ASTG prediction (mm/s)', r['astg'], -1.87, 2)
check('Sec. 2.4 dcos', r['dcos'], -0.1699, 4)
check('Sec. 2.4 uncertainty of lambda1 (%)', four['se'] / four['lam'] * 100, 15, 0)
check('Sec. 2.4 deviation from the prediction (sigma)', r['z_prediction'], 1.7, 1)
check('Sec. 2.4 deviation from zero (sigma)', r['z_zero'], 0.8, 1)
check('Sec. 2.4 Anderson drag at perigee (mm/s)', r['anderson_drag_perigee'], 2.14, 2)
check('Sec. 2.4 forward estimate / Anderson estimate ("about three")', -cases['central']['dv'] / r['anderson_drag_perigee'], 3, 0)
check('Sec. 2.4 drag range spans (mm/s)', r['drag_span'], 4.0, 1)
check_true('Sec. 2.4 span more than twice the predicted signal', r['drag_span'] > 2 * abs(r['astg']),
           f"{r['drag_span']:.2f} > {2 * abs(r['astg']):.2f}")

# Section 3 and Table 3
rows3, pred_c, s_pred_c, dcos_c = table3(fl)
check('Sec. 3 ASTG prediction for Cassini (mm/s)', pred_c, -0.137, 3)
check('Sec. 3 Cassini dcos', dcos_c, -0.0215, 4)
for row, dvp, sd, z, lam, se in ((rows3[0], -1.683, 0.842, 1.8, 3.902, 0.576), (rows3[1], -0.42, 0.42, 0.7, 3.897, 0.488)):
    src = row['source'].split()[0]
    check(f'Table 3 {src} dVp (mm/s)', row['dvp'], dvp, 3 if src == 'Anderson' else 2)
    check(f'Table 3 {src} sigma (mm/s)', row['sigma_dvp'], sd, 3 if src == 'Anderson' else 2)
    check(f'Table 3 {src} deviation (sigma)', row['deviation'], z, 1)
    check(f'Table 3 {src} lambda1 refit (1e3)', row['lam'] / 1e3, lam, 3)
    check(f'Table 3 {src} standard error (1e3)', row['se'] / 1e3, se, 3)
check('Sec. 3 Jouannic dVinf scaled to perigee (mm/s)', rows3[1]['dvp_scaled'], -0.42, 2)
check('Sec. 3 coupling moves by (%)', abs(rows3[1]['lam'] / rows3[0]['lam'] - 1) * 100, 0.14, 2)
check_sci('Sec. 3 Cassini leverage', rows3[0]['leverage'], 2e-4, 1)

# Section 4 and Table 4
j = juno(fl); d = j['d']
check('Table 4 rp (Mm)', d['rp'] / 1e6, 6.933, 3)
check('Table 4 Vp (km/s)', d['vp'] / 1e3, 14.931, 3)
check('Table 4 eps', d['eps'], 2.877, 3)
check('Table 4 dcos', d['dcos'], 0.1977, 4)
check('Table 4 dsin', d['dsin'], 0.3910, 4)
check('Eq. (5) ASTG prediction (mm/s)', j['prediction'], 1.90, 2)
check('Sec. 4 factor over the adopted uncertainty', j['factor'], 9.5, 1)
check('Sec. 4 deviation at the adopted uncertainty (sigma)', j['z_adopted'], 5.5, 1)
check('Sec. 4 Anderson relation at infinity (mm/s)', j['anderson_inf'], 6.37, 2)
check('Sec. 4 Anderson relation at perigee (mm/s)', j['anderson_p'], 4.43, 2)

# Section 4.1 and Table 5
jd, xp = perigee_state()
check_true('Sec. 4.1 partials agree with finite differences', check_partials(xp, jd) < 1e-6)
T5 = [(8613, 0.0057, 0.017), (8522, 0.0063, 0.019), (8282, 0.0064, 0.019), (2853, 0.0089, 0.027), (14373, 0.0045, 0.014),
      (8613, 0.0028, 0.0085), (8613, 0.0114, 0.034), (8613, 0.0063, 0.019), (8282, 0.0156, 0.047)]
rows5 = table5(xp, jd)
for row, (n, s1, s3) in zip(rows5, T5):
    check(f"Table 5 {row['case'][:34]} observations", row['n_obs'], n, 0)
    check(f"Table 5 {row['case'][:34]} 1 sigma (mm/s)", row['sigma'], s1, 4)
    check(f"Table 5 {row['case'][:34]} 3 sigma (mm/s)", row['three_sigma'], s3, 4 if s3 < 0.01 else 3)
worst = rows5[-1]['sigma']
check('Sec. 4.1 most pessimistic case (mm/s)', worst, 0.016, 3)
check_true('Sec. 4.1 0.05 mm/s detected at 3 sigma in every case', max(r5['three_sigma'] for r5 in rows5) < 0.05,
           f"largest 3 sigma {max(r5['three_sigma'] for r5 in rows5):.4f}")
check_true('Sec. 4.1 0.200 mm/s exceeds the pessimistic case tenfold', SIGMA_ADOPTED / worst > 10, f"ratio {SIGMA_ADOPTED / worst:.1f}")
check('Abstract, detectable change at 3 sigma (mm/s)', rows5[0]['three_sigma'], 0.02, 2)
check('Sec. 4.1 lambda1 contribution to the prediction (mm/s)', j['sigma_from_lambda'], 0.28, 2)
check('Sec. 4.1 deviation at the detection limit (sigma)', z_at(rows5[0]['sigma'], j), 6.8, 1)

# Section 5
tt = two_term_fit(fl)
check('Sec. 5.1 lambda_a (1e3)', tt['lam_a'] / 1e3, 4.48, 2)
check('Sec. 5.1 sigma of lambda_a (1e3)', tt['se_a'] / 1e3, 1.53, 2)
check('Sec. 5.1 lambda_b (1e3)', tt['lam_b'] / 1e3, 0.51, 2)
check('Sec. 5.1 sigma of lambda_b (1e3)', tt['se_b'] / 1e3, 1.21, 2)
check('Sec. 5.1 R^2, two terms', tt['r2'], 0.912, 3)
check('Sec. 5.1 R^2, single term', tt['r2_single'], 0.904, 3)
check('Sec. 5.1 Juno dsin', tt['juno_dsin'], 0.391, 3)
check('Sec. 5.1 two-term Juno prediction (mm/s)', tt['juno'], 2.68, 2)
l2 = multipole_l2(j['prediction'])
check('Sec. 5.2 lambda_12 / lambda_11', l2['ratio'], -0.08, 2)
check_sci('Sec. 5.2 GM / (rp c0^2) at a typical perigee', l2['intrinsic'], 6e-10, 1)
check_sci('Table 6 l = 2 multipole (mm/s)', l2['effect'], 1e-10, 1)
th = thermal_recoil()
check('Sec. 5.3 absorbed power (kW)', th['power'] / 1e3, 73.5, 1)
check('Sec. 5.3 flyby mass (kg)', 3625 - 752, 2873, 0)
check('Sec. 5.3 Eq. (7) thermal recoil (mm/s)', th['dv'], 0.097, 3)
check('Sec. 5.3 factor below Eq. (5)', j['prediction'] / th['dv'], 20, 0)
check('Sec. 5.3 asymmetry, dT = 5 K (mm/s)', th['asymmetry'][0], 0.004, 3)
check('Sec. 5.3 asymmetry, dT = 20 K (mm/s)', th['asymmetry'][1], 0.016, 3)
S_e, ratios = spin_ratios()
check_sci('Sec. 5.4 Earth specific spin angular momentum (m^2/s)', S_e, 9.79e8, 3)
check_sci('Sec. 5.4 Juno spin ratio', ratios['Juno'], 9e-9, 1)
check_sci('Sec. 5.4 Galileo spin ratio', ratios['Galileo'], 7e-10, 1)
jd_ = juno_drag()
check('Sec. 5.5 perigee latitude (deg S)', -jd_['perigee']['lat'], 34.4, 1)
check('Sec. 5.5 perigee longitude (deg E)', jd_['perigee']['lon'], 33.7, 1)
check_sci('Sec. 5.5 density at 561.5 km (kg/m^3)', jd_['rho_p'], 2.00e-13, 3)
check('Sec. 5.5 drag impulse (mm/s)', jd_['dv'], 0.132, 3)
check('Sec. 5.5 co-rotation lowers the impulse by (%)', (1 - jd_['dv'] / jd_['dv_inertial']) * 100, 4.5, 1)
check('Sec. 5.5 local scale height (km)', jd_['H'], 62.8, 1)
check('Sec. 5.5 single exponential, % less (inertial)', (1 - jd_['exp_inertial'] / jd_['dv_inertial']) * 100, 5, 0)
check('Sec. 5.5 single exponential, % less (relative speed)', (1 - jd_['exp_relative'] / jd_['dv']) * 100, 5, 0)
check('Sec. 5.5 / Table 6 drag, factor short of Eq. (5)', j['prediction'] / jd_['dv'], 14, 0)
check('Sec. 5.6 Iorio (2009) signed total (mm/s)', sum(IORIO_2009.values()), 0.0016, 4)
check('Sec. 5.7 required suppression (mm/s)', j['required_suppression'], 1.70, 2)
rows6, pred6, need6 = table6()
v6 = {name: val for name, kind, val in rows6}
check('Table 6 additional l = 1 term, Juno prediction (mm/s)', v6['Additional l = 1 angular term'], 2.68, 2)
check('Table 6 eclipse thermal recoil (mm/s)', v6['Eclipse thermal recoil'], 0.097, 3)
check('Table 6 atmospheric drag (mm/s)', v6['Atmospheric drag'], 0.132, 3)
check('Table 6 relativistic and classical (mm/s)', v6['Relativistic and classical'], 0.0016, 4)
check_true('Table 6 spacecraft spin of order 1e-8 mm/s', round(np.log10(v6['Spacecraft spin'])) == -8, f"{v6['Spacecraft spin']:.2e}")
check_true('Table 6 l = 2 multipole of order 1e-10 mm/s', round(np.log10(v6['l = 2 multipole'])) == -10, f"{v6['l = 2 multipole']:.2e}")
orders = {n: np.log10(need6 / v) for n, k, v in rows6 if k == 'suppression'}
check('Table 6 l = 2 multipole, orders of magnitude short', orders['l = 2 multipole'], 10, 0)
check('Table 6 relativistic and classical, orders of magnitude short', orders['Relativistic and classical'], 3, 0)
check('Sec. 5.4 / Table 6 spacecraft spin, orders of magnitude short', orders['Spacecraft spin'], 8, 0)
external = [orders[n] for n in ('Eclipse thermal recoil', 'Atmospheric drag', 'Relativistic and classical', 'Spacecraft spin')]
check('Sec. 5.7 external mechanisms, fewest orders short', min(external), 1, 0)
check('Sec. 5.7 external mechanisms, most orders short', max(external), 8, 0)

# Section 6 and Table 7
for row in table7(fl):
    expect = row['spacecraft'] in ('Galileo I', 'Galileo II', 'NEAR', 'Cassini', 'Rosetta I', 'EPOXI III')
    check_true(f"Table 7 {row['spacecraft']} both negative: {'Yes' if expect else 'No'}", row['both_negative'] == expect,
               f"{row['delta_in']:+.2f}, {row['delta_out']:+.2f}")
nd = {r6['spacecraft']: r6 for r6 in null_deviations(fl)}
check('Sec. 6 MESSENGER before the rule (sigma)', nd['MESSENGER']['z_law'], 4.5, 1)
check('Sec. 6 MESSENGER under the rule (sigma)', nd['MESSENGER']['z_rule'], 1.8, 1)
check('Sec. 6 Rosetta III before the rule (sigma)', nd['Rosetta III']['z_law'], 2.3, 1)
check('Sec. 6 EPOXI II before the rule (sigma)', nd['EPOXI II']['z_law'], 2.5, 1)
check('Sec. 6 Juno before the rule (sigma)', nd['Juno']['z_law'], 5.5, 1)
check('Sec. 6 chi^2 over the nulls, without the rule', sum(v['z_law']**2 for v in nd.values()), 81, 0)
check('Sec. 6 chi^2 over the nulls, with the rule', sum(v['z_rule']**2 for v in nd.values()), 19, 0)
check_true('Sec. 6 five of the six other nulls predicted null', sum(not v['inside'] for k, v in nd.items() if k != 'Juno') == 5)
check('Sec. 6 EPOXI III perigee altitude (km)', fl['EPOXI III']['hp_km'], 30488, 0)
check('Sec. 6 EPOXI III prediction (mm/s)', nd['EPOXI III']['astg'], -0.49, 2)
check('Sec. 6 EPOXI III deviation (sigma)', nd['EPOXI III']['z_law'], 4.0, 1)
check('Sec. 6 EPOXI III below 3 sigma above (mm/s)', sigma_threshold(nd['EPOXI III']['astg'], nd['EPOXI III']['sigma_astg'], 3), 0.15, 2)
check('Sec. 6 EPOXI III below 2 sigma above (mm/s)', sigma_threshold(nd['EPOXI III']['astg'], nd['EPOXI III']['sigma_astg'], 2), 0.23, 2)
check('Sec. 6 chance of the selection, 6/792', comb(6, 5) / comb(12, 5), 0.008, 3)

# Section 7, Tables 8 and 9
f4, f5, gg = refit(fl, r)
for lab, f, lam, se, tv, pv, r2 in (('four', f4, 3.902, 0.576, 6.77, 0.0066, 0.904), ('five', f5, 3.528, 0.913, 3.86, 0.018, 0.616)):
    check(f'Table 8 {lab} encounters lambda1 (1e3)', f['lam'] / 1e3, lam, 3)
    check(f'Table 8 {lab} encounters standard error (1e3)', f['se'] / 1e3, se, 3)
    check(f'Table 8 {lab} encounters t', f['t'], tv, 2)
    check(f'Table 8 {lab} encounters p', f['p'], pv, 4 if lab == 'four' else 3)
    check(f'Table 8 {lab} encounters R^2', f['r2_mean'], r2, 3)
check('Sec. 7 coupling moves by (%)', abs(f5['lam'] / f4['lam'] - 1) * 100, 9.6, 1)
check('Sec. 7 shift in units of the original uncertainty (sigma)', abs(f5['lam'] - f4['lam']) / f4['se'], 0.6, 1)
T9 = {'Galileo II': ((-5.04, 0.63), (1.76, 2.09), 4.6, 1.6), 'Cassini': ((-1.683, 0.842), (-0.42, 0.42), 1.8, 0.7),
      'Juno': ((0.0, 0.200), (0.0, 0.200), 5.5, 0.0)}
for row in table9(fl, f4, f5, gg):
    k = row['encounter']; (bv, bs), (av, as_), zb, za = T9[k]
    dec = 3 if k == 'Cassini' else 2
    check(f'Table 9 {k} before', row['before'][0], bv, dec); check(f'Table 9 {k} before sigma', row['before'][1], bs, dec)
    check(f'Table 9 {k} after', row['after'][0], av, 2 if k != 'Juno' else 3)
    check(f'Table 9 {k} after sigma', row['after'][1], as_, 2 if k != 'Juno' else 3)
    check(f'Table 9 {k} deviation before (sigma)', row['z_before'], zb, 1)
    check(f'Table 9 {k} deviation after (sigma)', row['z_after'], za, 1)

# Appendix A
a = appendix_a()
check('App. A rp (1e6 m)', a['rp'] / 1e6, 6.9325, 4)
check('App. A Vinf^2 (1e8 m^2/s^2)', a['vinf2'] / 1e8, 1.07931, 5)
check('App. A 2GM/rp (1e8 m^2/s^2)', a['grav'] / 1e8, 1.14995, 5)
check('App. A Vp^2 (1e8 m^2/s^2)', a['vp2'] / 1e8, 2.22926, 5)
check('App. A Vp (km/s)', a['vp'] / 1e3, 14.931, 3)
check('App. A eps - 1', a['eps_minus_1'], 1.877, 3)
check('App. A cos 14.17', a['cos_in'], 0.96957, 5)
check('App. A cos 39.48', a['cos_out'], 0.77185, 5)
check('App. A dcos', a['dcos'], 0.1977, 4)
check('App. A Eq. (A3) (mm/s)', a['dvp'], 1.90, 2)

# Appendix C
c = appendix_c(g, hs, rho_h)
check_sci('App. C density over latitude and local time, min (kg/m^3)', c['rho_min'], 1.72e-11, 3)
check_sci('App. C density over latitude and local time, max (kg/m^3)', c['rho_max'], 3.43e-11, 3)
check('App. C density along the arc changes the impulse by (%)', (c['arc_ratio'] - 1) * 100, -0.4, 1)
check('App. C grid points', c['n_points'], 40001, 0)
check_true('App. C arc above 1200 km below one part in 1e4', c['above_top'] < 1e-4, f"{c['above_top']:.1e}")
check_true('App. C halving the grid below one part in 1e6', c['grid_halving'] < 1e-6, f"{c['grid_halving']:.1e}")

n_fail = results.count(False)
print(f"\n{len(results) - n_fail} of {len(results)} checks pass")
sys.exit(1 if n_fail else 0)
