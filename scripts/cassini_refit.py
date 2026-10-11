"""The Cassini encounter (Section 3, Table 3): the anomaly in the two published sources, its deviation from the
ASTG prediction, and the coupling refitted on the four anomalous encounters with each.

  Anderson et al. (2008): dVinf = -2.0 +- 1.0 mm/s, scaled to perigee by Vinf/Vp (data/earth_flybys.csv)
  Jouannic et al. (2015): dVinf = -0.5 +- 0.5 mm/s, listed at perigee as dVp = -0.42 +- 0.42 mm/s
Deviations are in units of the combined uncertainty sqrt(sigma_obs^2 + (prediction * se / lambda1)^2)."""
import numpy as np
import astg

JOUANNIC = dict(dvinf=-0.5, sigma_dvinf=0.5, dvp=-0.42, sigma_dvp=0.42)


def table3(flybys=None):
    fl = flybys or astg.load_flybys()
    base = astg.anomalous_fit(fl)
    pred = astg.astg_dvp(fl['Cassini'], base['lam']); s_pred = abs(pred) * base['se'] / base['lam']
    d = astg.derived(fl['Cassini'])
    rows = []
    for source, dvinf, sdvinf, dvp, sdvp in (
            ('Anderson et al. (2008)', fl['Cassini']['dVinf_mm_s'], fl['Cassini']['sigma_dVinf_mm_s'],
             fl['Cassini']['dVp_mm_s'], fl['Cassini']['sigma_dVp_mm_s']),
            ('Jouannic et al. (2015)', JOUANNIC['dvinf'], JOUANNIC['sigma_dvinf'], JOUANNIC['dvp'], JOUANNIC['sigma_dvp'])):
        fit = astg.anomalous_fit(fl, dvp_override={'Cassini': dvp})
        rows.append(dict(source=source, dvinf=dvinf, sigma_dvinf=sdvinf, dvp=dvp, sigma_dvp=sdvp,
                         dvp_scaled=dvinf * d['vinf'] / d['vp'], deviation=abs(dvp - pred) / np.hypot(sdvp, s_pred),
                         lam=fit['lam'], se=fit['se'], leverage=fit['leverage'][astg.ANOMALOUS.index('Cassini')]))
    return rows, pred, s_pred, d['dcos']


if __name__ == '__main__':
    rows, pred, s_pred, dcos = table3()
    print(f"ASTG prediction for Cassini: {pred:+.3f} +- {s_pred:.3f} mm/s (dcos = {dcos:+.4f})")
    print('\nTable 3. The Cassini anomaly in the two published sources')
    print(f"{'Source':24s} {'dVinf (mm/s)':>13s} {'dVp (mm/s)':>16s} {'Deviation':>10s} {'lambda1 refit (1e3)':>20s}")
    for r in rows:
        print(f"{r['source']:24s} {r['dvinf']:+6.1f} +- {r['sigma_dvinf']:.1f} {r['dvp']:+8.3f} +- {r['sigma_dvp']:.3f} "
              f"{r['deviation']:8.1f}   {r['lam'] / 1e3:.3f} +- {r['se'] / 1e3:.3f}")
    a, j = rows
    print(f"\nJouannic dVinf scaled to perigee: {j['dvp_scaled']:+.3f} mm/s (listed {j['dvp']:+.2f})")
    print(f"coupling moves by {abs(j['lam'] / a['lam'] - 1) * 100:.2f}%; standard error {a['se'] / 1e3:.3f} -> {j['se'] / 1e3:.3f} (1e3); "
          f"Cassini leverage {a['leverage']:.1e}")
