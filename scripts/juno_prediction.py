"""The Juno prediction (Section 4, Table 4, Eq. 5 and Appendix A).

Table 4 and Eq. (5) follow from data/earth_flybys.csv (JPL Horizons osculating state at perigee) and the coupling
fitted to the four anomalous encounters.  The deviation from the null is in units of the combined uncertainty
sqrt(sigma_obs^2 + (prediction * se / lambda1)^2), with sigma_obs the adopted 0.200 mm/s or the 1-sigma detection
limit of juno_covariance.py.  Appendix A repeats the arithmetic with the rounded values printed there."""
import numpy as np
import astg

SIGMA_ADOPTED = 0.200        # mm/s


def juno(flybys=None):
    fl = flybys or astg.load_flybys()
    fit = astg.anomalous_fit(fl)
    rec = fl['Juno']; d = astg.derived(rec)
    pred = astg.astg_dvp(rec, fit['lam']); s_lam = pred * fit['se'] / fit['lam']
    return dict(rec=rec, d=d, lam=fit['lam'], se=fit['se'], prediction=pred, sigma_from_lambda=s_lam,
                factor=pred / SIGMA_ADOPTED, z_adopted=pred / np.hypot(SIGMA_ADOPTED, s_lam),
                anderson_inf=astg.anderson_dvinf(rec), anderson_p=astg.anderson_dvp(rec),
                required_suppression=pred - SIGMA_ADOPTED)


def z_at(sigma_obs, j=None):
    j = j or juno()
    return j['prediction'] / np.hypot(sigma_obs, j['sigma_from_lambda'])


def appendix_a():
    """The arithmetic of Appendix A with the rounded inputs printed there."""
    rp = astg.R_EARTH + 561.5e3; vinf = 10.389e3
    vinf2 = vinf**2; grav = 2 * astg.GM / rp
    eps_minus_1 = rp * vinf2 / astg.GM
    c_in, c_out = np.cos(np.radians(14.17)), np.cos(np.radians(39.48))
    dvp = 14931 * 3902 * 8.870e-3 * 0.1977 / (2 * 6.9325e6 * 3.877) * 1e3
    return dict(rp=rp, vinf2=vinf2, grav=grav, vp2=vinf2 + grav, vp=np.sqrt(vinf2 + grav), eps_minus_1=eps_minus_1,
                cos_in=c_in, cos_out=c_out, dcos=c_in - c_out, rg=astg.RG, dvp=dvp)


if __name__ == '__main__':
    j = juno(); d = j['d']; rec = j['rec']
    print('Table 4. Observed and derived quantities for the Juno Earth flyby')
    print(f"  date {rec['date']}   hp {rec['hp_km']} km   Vinf {rec['Vinf_km_s']} km/s   delta_in {rec['delta_in_deg']:+.2f}   delta_out {rec['delta_out_deg']:+.2f}")
    print(f"  rp {d['rp'] / 1e6:.3f} Mm   Vp {d['vp'] / 1e3:.3f} km/s   eps {d['eps']:.3f}   dcos {d['dcos']:+.4f}   dsin {d['dsin']:+.4f}")
    print(f"  observed 0 +- {SIGMA_ADOPTED:.3f} mm/s   ASTG {j['prediction']:+.2f} mm/s")
    print(f"\nEq. (5): dVp = {j['prediction']:+.3f} mm/s with lambda1 = {j['lam']:.1f}; {j['factor']:.1f} times the adopted uncertainty")
    print(f"  lambda1 uncertainty contributes {j['sigma_from_lambda']:.2f} mm/s; deviation {j['z_adopted']:.1f} sigma at the adopted uncertainty")
    try:
        from juno_covariance import perigee_state, analyse
        jd, xp = perigee_state(); s_det = analyse(xp, jd)[0]
        print(f"  at the 1-sigma detection limit of Table 5 ({s_det:.4f} mm/s): {z_at(s_det, j):.1f} sigma")
    except ImportError:
        pass
    print(f"  Anderson relation, Eq. (1): {j['anderson_inf']:+.2f} mm/s at infinity, {j['anderson_p']:+.2f} mm/s at perigee")
    print(f"  suppression required to reach the observed limit: {j['required_suppression']:.2f} mm/s")
    a = appendix_a()
    print('\nAppendix A')
    print(f"  rp = {a['rp']:.4e} m;  Vp^2 = {a['vinf2']:.5e} + {a['grav']:.5e} = {a['vp2']:.5e} m^2/s^2;  Vp = {a['vp'] / 1e3:.3f} km/s")
    print(f"  eps = 1 + {a['eps_minus_1']:.3f} = {1 + a['eps_minus_1']:.3f};  dcos = {a['cos_in']:.5f} - {a['cos_out']:.5f} = {a['dcos']:+.4f}")
    print(f"  Rg = {a['rg']:.3e} m;  dVp = 14931 (3902)(8.870e-3)(0.1977) / [2 (6.9325e6)(3.877)] = {a['dvp']:+.2f} mm/s")
