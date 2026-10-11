"""The hemispheric selection rule (Section 6, Table 7).

Rule: the law of Eq. (2) applies when delta_in < 0 and delta_out < 0, and the anomaly is zero otherwise.
For the seven null encounters the deviation before the rule is that of the companion paper (ASTG law with the
four-encounter coupling, combined uncertainty sqrt(sigma_obs^2 + (prediction * se / lambda1)^2)); under the rule an
encounter outside it is compared with zero, z = |observed| / sigma_obs."""
from math import comb
import numpy as np
import astg

ORDER = ['Galileo I', 'Galileo II', 'NEAR', 'Cassini', 'Rosetta I', 'EPOXI III', 'MESSENGER', 'Rosetta II',
         'EPOXI I', 'Rosetta III', 'EPOXI II', 'Juno']
NONZERO = ['Galileo I', 'Galileo II', 'NEAR', 'Cassini', 'Rosetta I']


def both_negative(rec):
    return rec['delta_in_deg'] < 0 and rec['delta_out_deg'] < 0


def table7(flybys=None):
    fl = flybys or astg.load_flybys()
    return [dict(spacecraft=k, delta_in=fl[k]['delta_in_deg'], delta_out=fl[k]['delta_out_deg'],
                 both_negative=both_negative(fl[k]), observed='Anomalous' if k in NONZERO else 'Null') for k in ORDER]


def null_deviations(flybys=None):
    fl = flybys or astg.load_flybys()
    fit = astg.anomalous_fit(fl); lam, se = fit['lam'], fit['se']
    rows = []
    for k in astg.NULLS:
        f = fl[k]; o, s = f['dVp_mm_s'], f['sigma_dVp_mm_s']
        a = astg.astg_dvp(f, lam); z_law = abs(a - o) / np.hypot(s, abs(a) * se / lam)
        z_rule = z_law if both_negative(f) else abs(o) / s
        rows.append(dict(spacecraft=k, astg=a, sigma_astg=abs(a) * se / lam, z_law=z_law, z_rule=z_rule, inside=both_negative(f)))
    return rows


def sigma_threshold(prediction, sigma_pred, z):
    """Observational uncertainty above which |prediction| lies less than z combined standard deviations from zero."""
    return np.sqrt((prediction / z)**2 - sigma_pred**2)


if __name__ == '__main__':
    print('Table 7. Asymptotic velocity declinations of the twelve Earth flybys')
    for r in table7():
        print(f"  {r['spacecraft']:12s} {r['delta_in']:+7.2f} {r['delta_out']:+7.2f}  {'Yes' if r['both_negative'] else 'No ':3s}  {r['observed']}")
    rows = null_deviations()
    print('\nNull encounters, deviation without and with the rule (sigma)')
    for r in rows:
        print(f"  {r['spacecraft']:12s} ASTG {r['astg']:+.3f}   law {r['z_law']:4.1f}   rule {r['z_rule']:4.1f}{'   (inside the rule)' if r['inside'] else ''}")
    print(f"  summed chi^2: {sum(r['z_law']**2 for r in rows):.0f} -> {sum(r['z_rule']**2 for r in rows):.0f}")
    e3 = [r for r in rows if r['spacecraft'] == 'EPOXI III'][0]
    print(f"\nEPOXI III: hp {astg.load_flybys()['EPOXI III']['hp_km']:.0f} km, prediction {e3['astg']:+.2f} mm/s, {e3['z_law']:.1f} sigma; "
          f"below 3 sigma for sigma_obs > {sigma_threshold(e3['astg'], e3['sigma_astg'], 3):.2f} mm/s, "
          f"below 2 sigma for sigma_obs > {sigma_threshold(e3['astg'], e3['sigma_astg'], 2):.2f} mm/s")
    n_neg = sum(r['both_negative'] for r in table7())
    print(f"chance that all five non-zero anomalies fall among the {n_neg} both-negative encounters: "
          f"C({n_neg},5)/C(12,5) = {comb(n_neg, 5)}/{comb(12, 5)} = {comb(n_neg, 5) / comb(12, 5):.3f}")
