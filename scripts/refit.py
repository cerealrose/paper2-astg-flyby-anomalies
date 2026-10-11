"""Refit with the corrected encounters (Section 7, Eq. 8, Tables 8 and 9, Appendix B).

Ordinary least squares through the origin in the variables of Eq. (2), as in the companion paper.  The five-encounter
refit adds Galileo II at its drag-corrected residual (galileo_drag.py) and replaces Cassini by the value of
Jouannic et al. (2015), -0.42 +- 0.42 mm/s.  Deviations in Table 9 are in units of the combined uncertainty of the
measurement and the coupling: before correction with the four-encounter coupling, after with the refit coupling,
and for Juno from the selection rule of Section 6."""
import numpy as np
import astg
from galileo_drag import residual
from cassini_refit import JOUANNIC
from juno_prediction import juno, SIGMA_ADOPTED

FIVE = astg.ANOMALOUS + ['Galileo II']


def refit(flybys=None, galileo=None):
    fl = flybys or astg.load_flybys()
    g = galileo or residual()
    four = astg.anomalous_fit(fl)
    five = astg.anomalous_fit(fl, dvp_override={'Cassini': JOUANNIC['dvp'], 'Galileo II': g['residual']}, names=FIVE)
    return four, five, g


def deviation(obs, s_obs, pred, lam, se):
    return abs(obs - pred) / np.hypot(s_obs, abs(pred) * se / lam)


def table9(flybys=None, four=None, five=None, g=None):
    fl = flybys or astg.load_flybys()
    if four is None:
        four, five, g = refit(fl)
    gal_b = astg.astg_dvp(fl['Galileo II'], four['lam']); gal_a = astg.astg_dvp(fl['Galileo II'], five['lam'])
    cas_b = astg.astg_dvp(fl['Cassini'], four['lam']); cas_a = astg.astg_dvp(fl['Cassini'], five['lam'])
    c = fl['Cassini']; j = juno(fl)
    return [dict(encounter='Galileo II', before=(g['measured'], g['sigma_measured']), after=(g['residual'], g['sigma_residual']),
                 z_before=deviation(g['measured'], g['sigma_measured'], gal_b, four['lam'], four['se']),
                 z_after=deviation(g['residual'], g['sigma_residual'], gal_a, five['lam'], five['se'])),
            dict(encounter='Cassini', before=(c['dVp_mm_s'], c['sigma_dVp_mm_s']), after=(JOUANNIC['dvp'], JOUANNIC['sigma_dvp']),
                 z_before=deviation(c['dVp_mm_s'], c['sigma_dVp_mm_s'], cas_b, four['lam'], four['se']),
                 z_after=deviation(JOUANNIC['dvp'], JOUANNIC['sigma_dvp'], cas_a, five['lam'], five['se'])),
            dict(encounter='Juno', before=(0.0, SIGMA_ADOPTED), after=(0.0, SIGMA_ADOPTED), z_before=j['z_adopted'], z_after=0.0)]


if __name__ == '__main__':
    four, five, g = refit()
    print('Table 8. Determinations of the coupling (OLS through the origin)')
    for name, f in (('Four anomalous encounters', four), ('With corrected Galileo II and Cassini', five)):
        print(f"  {name:38s} n = {f['dof'] + 1}  lambda1 = ({f['lam'] / 1e3:.3f} +- {f['se'] / 1e3:.3f}) e3  "
              f"t = {f['t']:.2f}  p = {f['p']:.4f}  R^2 = {f['r2_mean']:.3f}")
    print(f"\nEq. (8): lambda1 = {five['lam']:.1f} +- {five['se']:.1f} (Galileo II entered at {g['residual']:+.3f} mm/s)")
    print(f"  shift {abs(five['lam'] / four['lam'] - 1) * 100:.1f}%, or {abs(five['lam'] - four['lam']) / four['se']:.1f} sigma of the original uncertainty")
    print('\nTable 9. The three treated encounters before and after correction (mm/s; deviations in sigma)')
    for r in table9(four=four, five=five, g=g):
        print(f"  {r['encounter']:11s} before {r['before'][0]:+7.3f} +- {r['before'][1]:.3f}   after {r['after'][0]:+7.3f} +- {r['after'][1]:.3f}"
              f"   {r['z_before']:.1f} -> {r['z_after']:.1f}")
