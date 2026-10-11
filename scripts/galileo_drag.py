"""Galileo II drag impulse and residual (Sections 2.1, 2.3 and 2.4, Table 2, Appendix C).

Inputs
  * Hyperbola: rp, eps and Vp from data/earth_flybys.csv (Anderson et al. 2008: hp = 303 km, Vinf = 8.877 km/s).
  * Orientation of the orbit and the Sun direction: JPL Horizons geocentric ICRF state of Galileo (-77) near
    perigee, data/horizons_states.csv; spin axis on the Sun.
  * Closest-approach point for the density: 33.76 S, 354.42 E, 303.108 km at 1992-12-08 15:09:24.9 UTC
    (Antreasian and Guinn 1998).
  * NRLMSISE-00 with the indices in data/space_weather.csv (F10.7 of 7 December 1992 = 120.1 sfu, 81-day centred
    mean 139.9 sfu, daily Ap = 26).
  * Projected area from galileo_projected_area.py; C_D = 2.0, 2.2, 2.4; mass 2497 kg (Anderson et al. 2008).
  * Measured change: Delta Vinf = -8 +- 1 mm/s before the drag subtraction of Anderson et al. (2008), converted
    to perigee with dE = Vinf dVinf = Vp dVp.
Nothing here uses the observed velocity change except the residual itself."""
import numpy as np
import astg
import drag_model as dm
from galileo_projected_area import totals, area_cases

PERIGEE = dict(date='1992-12-08T15:09:24.9', jd_utc=2448964.5 + (15 + 9 / 60 + 24.9 / 3600) / 24,
               lat=-33.76, lon=354.42, h=303.108)
TDB_MINUS_UTC = 59.184          # s in December 1992 (TT - TAI = 32.184 s, 27 leap seconds)
MASS = 2497.0                   # kg
CD = dict(lower=2.0, central=2.2, upper=2.4)
DVINF_MEASURED, SIGMA_DVINF = -8.0, 1.0     # mm/s, Anderson et al. (2008), before their drag subtraction
ANDERSON_DRAG_INF = 3.4                     # mm/s, the drag estimate subtracted by Anderson et al. (2008)
H_TOP = 1200.0                              # km, top of the integration


def hyperbola():
    d = astg.derived(astg.load_flybys()['Galileo II'])
    return dict(rp=d['rp'] / 1e3, ecc=d['eps'], vp=d['vp'] / 1e3, vinf=d['vinf'] / 1e3)


def geometry(n=40001, h_top=H_TOP):
    """Arc of the hyperbola oriented by Horizons, the Sun-pointed spin axis and the angle alpha between it and the
    velocity relative to the co-rotating atmosphere."""
    jd_tdb, r, v = dm.state('galileo2_near_perigee')
    el = dm.elements(r, v)
    hyp = hyperbola()
    a = dm.arc(hyp['rp'], hyp['ecc'], el['p_hat'], el['q_hat'], h_top - PERIGEE['h'], n=n)
    spin = dm.sun_dir(jd_tdb - TDB_MINUS_UTC / 86400)
    vr = dm.relative_velocity(a['R'], a['V'])
    alpha = np.degrees(np.arccos(np.abs(vr @ spin) / np.linalg.norm(vr, axis=1)))
    i0 = n // 2
    lat, lon, hgt = dm.geodetic(el['p_hat'] * el['rp'], jd_tdb - TDB_MINUS_UTC / 86400)
    return dict(arc=a, spin=spin, alpha=alpha, alpha_perigee=alpha[i0], vr_perigee=np.linalg.norm(vr[i0]),
                horizons=el, horizons_perigee_point=(lat, lon, hgt), **hyp)


def density_profile(h_top=H_TOP):
    f107, f107a, ap = dm.space_weather('Galileo II')
    return dm.profile(PERIGEE['date'], PERIGEE['lat'], PERIGEE['lon'], PERIGEE['h'], f107, f107a, ap, hmax=h_top)


def unit_impulse(g, hs, rho_h):
    """Eq. (4) for C_D A / m = 1 m^2/kg, co-rotating and inertial, and the exponential check (Section 2.3)."""
    k_rot = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'])['dv_mm_s']
    k_in = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'], corot=False)['dv_mm_s']
    H = dm.scale_height(hs, rho_h)
    l_exp = dm.exponential_path_length(H, g['rp'], g['ecc'])                        # km
    l_eff = 2 * k_in * 1e-3 / (rho_h[0] * g['vp'] * 1e3) / 1e3                      # km, from the inertial integral
    k_exp_in = 0.5 * rho_h[0] * g['vp'] * 1e3 * l_exp * 1e3 * 1e3                  # mm/s, exponential, inertial
    k_exp_rot = k_exp_in * (g['vr_perigee'] / g['vp'])**2                           # same at the relative speed
    return dict(k_rot=k_rot, k_in=k_in, H=H, l_exp=l_exp, l_eff=l_eff, k_exp_in=k_exp_in, k_exp_rot=k_exp_rot)


def drag_cases(g=None, hs=None, rho_h=None):
    """Table 2: area at perigee, C_D A / m and the drag impulse (mm/s) for the three cases, with the angle range
    over the part of the arc that carries 90% of the drag."""
    g = g or geometry()
    if hs is None:
        hs, rho_h = density_profile()
    t = totals(); areas = dict(lower=(t['ax'][0], t['tr'][0]), central=(t['ax_mid'], t['tr_mid']), upper=(t['ax'][1], t['tr'][1]))
    ap = area_cases(g['alpha_perigee'])
    out = {}
    for name in ('lower', 'central', 'upper'):
        res = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'], spin=g['spin'], areas=areas[name])
        c = np.cumsum(res['integrand']); c /= c[-1]
        lo, hi = np.searchsorted(c, 0.05), np.searchsorted(c, 0.95)
        out[name] = dict(area=ap[name], cd=CD[name], cda_m=CD[name] * ap[name] / MASS,
                         dv=-res['dv_mm_s'] * CD[name] / MASS,
                         alpha90=(res['alpha'][lo:hi].min(), res['alpha'][lo:hi].max()))
    return out


def measured_at_perigee():
    hyp = hyperbola()
    f = hyp['vinf'] / hyp['vp']
    return DVINF_MEASURED * f, SIGMA_DVINF * f, ANDERSON_DRAG_INF * f


def residual(cases=None):
    """Section 2.4: residual of the measured change after the drag impulse, and its deviations."""
    cases = cases or drag_cases()
    meas, s_meas, anderson_drag_p = measured_at_perigee()
    fl = astg.load_flybys(); fit = astg.anomalous_fit(fl)
    pred = astg.astg_dvp(fl['Galileo II'], fit['lam']); s_pred = abs(pred) * fit['se'] / fit['lam']
    res = meas - cases['central']['dv']
    half = (cases['lower']['dv'] - cases['upper']['dv']) / 2
    s_res = np.hypot(s_meas, half)
    return dict(measured=meas, sigma_measured=s_meas, anderson_drag_perigee=anderson_drag_p,
                residual=res, low=meas - cases['lower']['dv'], high=meas - cases['upper']['dv'], half_width=half,
                sigma_residual=s_res, astg=pred, sigma_astg=s_pred, dcos=astg.derived(fl['Galileo II'])['dcos'],
                z_prediction=abs(res - pred) / np.sqrt(s_res**2 + s_pred**2), z_zero=abs(res) / s_res,
                z_before=abs(meas - pred) / np.hypot(s_meas, s_pred),
                drag_span=cases['lower']['dv'] - cases['upper']['dv'])


def pointing_sensitivity(g, hs, rho_h, tilt_deg=5.0, n_az=72):
    """Largest relative change of the central impulse for a spin axis tilted tilt_deg from the Sun."""
    t = totals(); areas = (t['ax_mid'], t['tr_mid']); s = g['spin']
    k0 = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'], spin=s, areas=areas)['dv_mm_s']
    u = np.cross(s, [0, 0, 1.0]); u /= np.linalg.norm(u); w = np.cross(s, u)
    worst = 0.0
    for th in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
        s2 = np.cos(np.radians(tilt_deg)) * s + np.sin(np.radians(tilt_deg)) * (np.cos(th) * u + np.sin(th) * w)
        k = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'], spin=s2, areas=areas)['dv_mm_s']
        worst = max(worst, abs(k / k0 - 1))
    return worst


def appendix_c(g, hs, rho_h):
    """Appendix C: density over all latitudes and local times at the perigee altitude, density along the arc,
    truncation of the integral and grid resolution."""
    f107, f107a, ap = dm.space_weather('Galileo II')
    lats, lons = np.arange(-90, 90.001, 1.0), np.arange(0, 360, 5.0)
    LA, LO = np.meshgrid(lats, lons, indexing='ij')
    rho_grid = dm.msis(np.datetime64(PERIGEE['date']), LO.ravel(), LA.ravel(), PERIGEE['h'], f107, f107a, ap)
    t = totals(); areas = (t['ax_mid'], t['tr_mid'])
    base = dm.drag_integral(g['arc'], hs, rho_h, PERIGEE['h'], spin=g['spin'], areas=areas)['dv_mm_s']
    # density evaluated at each point of the arc (geodetic position at its own time) instead of on the vertical
    a = g['arc']
    tsec = dm.time_from_perigee(a['phi'], a['rp'], a['ecc'])
    lat, lon, _ = dm.geodetic(a['R'].T, PERIGEE['jd_utc'] + tsec / 86400)
    dates = np.datetime64(PERIGEE['date']) + (tsec * 1000).astype('timedelta64[ms]')
    rho_arc = dm.msis(dates, lon, lat, PERIGEE['h'] + (a['r'] - a['rp']), f107, f107a, ap)
    along = dm.drag_integral(a, hs, rho_h, PERIGEE['h'], spin=g['spin'], areas=areas, rho_override=rho_arc)['dv_mm_s']
    # truncation: extend the arc to 2000 km, and cut it at 1000 km
    out = {}
    for top in (1000.0, 2000.0):
        g2 = geometry(h_top=top); hs2, rho2 = density_profile(h_top=top)
        out[top] = dm.drag_integral(g2['arc'], hs2, rho2, PERIGEE['h'], spin=g2['spin'], areas=areas)['dv_mm_s']
    g3 = geometry(n=20001)
    half = dm.drag_integral(g3['arc'], hs, rho_h, PERIGEE['h'], spin=g3['spin'], areas=areas)['dv_mm_s']
    return dict(rho_min=rho_grid.min(), rho_max=rho_grid.max(), arc_ratio=along / base,
                above_top=out[2000.0] / base - 1, cut_1000=1 - out[1000.0] / base, grid_halving=abs(half / base - 1),
                n_points=len(a['phi']))


if __name__ == '__main__':
    g = geometry(); hs, rho_h = density_profile()
    el = g['horizons']
    print(f"Horizons osculating conic near perigee: rp {el['rp']:.2f} km, eps {el['ecc']:.4f}, inclination {el['incl']:.2f} deg (retrograde)")
    print(f"Horizons perigee point: {g['horizons_perigee_point'][0]:.2f} lat, {g['horizons_perigee_point'][1]:.2f} E"
          f"  (Antreasian and Guinn 1998: {PERIGEE['lat']} lat, {PERIGEE['lon']} E)")
    meas, s_meas, adp = measured_at_perigee()
    print(f"\nSection 2.1: measured change at perigee (Vinf/Vp)(-8) = {meas:.2f} +- {s_meas:.2f} mm/s")
    print(f"Section 2.2: alpha at perigee {g['alpha_perigee']:.2f} deg")
    u = unit_impulse(g, hs, rho_h)
    print(f"\nSection 2.3: density at the closest-approach point {rho_h[0]:.3e} kg/m^3, local scale height {u['H']:.1f} km")
    print(f"  speed relative to the co-rotating atmosphere at perigee {g['vr_perigee']:.2f} km/s against inertial Vp {g['vp']:.2f} km/s")
    print(f"  Eq. (4) for C_D A/m = 1 m^2/kg: {-u['k_rot']:.1f} mm/s co-rotating ({-u['k_in']:.1f} inertial)")
    print(f"  single exponential: path length {u['l_exp']:.0f} km against {u['l_eff']:.0f} km from the integral, "
          f"{(1 - u['l_exp'] / u['l_eff']) * 100:.1f}% less (inertial); at the relative speed "
          f"{-u['k_exp_rot']:.1f} against {-u['k_rot']:.1f}, {(1 - u['k_exp_rot'] / u['k_rot']) * 100:.1f}% less")
    cases = drag_cases(g, hs, rho_h)
    print(f"\nTable 2. Drag impulse at the Galileo II perigee (alpha = {g['alpha_perigee']:.1f} deg, m = {MASS:.0f} kg)")
    for name, c in cases.items():
        print(f"  {name:8s} area {c['area']:5.2f} m^2  C_D {c['cd']:.1f}  C_D A/m {c['cda_m']:.4f} m^2/kg  dV {c['dv']:+.3f} mm/s"
              f"   alpha over 90% of the drag {c['alpha90'][0]:.1f} to {c['alpha90'][1]:.1f} deg")
    print(f"  5 deg pointing error changes the central impulse by at most {pointing_sensitivity(g, hs, rho_h) * 100:.2f}%")
    r = residual(cases)
    print(f"\nSection 2.4: residual {r['residual']:+.2f} mm/s (range {r['low']:+.2f} to {r['high']:+.2f}); "
          f"ASTG prediction {r['astg']:+.3f} +- {r['sigma_astg']:.3f} mm/s (dcos {r['dcos']:+.4f})")
    print(f"  residual uncertainty sqrt(sigma_meas^2 + half-width^2) = {r['sigma_residual']:.2f} mm/s (half-width {r['half_width']:.2f})")
    print(f"  deviation from the prediction {r['z_prediction']:.2f} sigma, from zero {r['z_zero']:.2f} sigma; "
          f"before the drag correction {r['z_before']:.1f} sigma")
    print(f"  Anderson et al. (2008) drag of {ANDERSON_DRAG_INF} mm/s at infinity is {r['anderson_drag_perigee']:.2f} mm/s at perigee; "
          f"forward estimate / theirs = {-cases['central']['dv'] / r['anderson_drag_perigee']:.1f}")
    print(f"  drag range spans {r['drag_span']:.1f} mm/s against twice the predicted signal, {2 * abs(r['astg']):.2f} mm/s")
    c = appendix_c(g, hs, rho_h)
    print(f"\nAppendix C: density over all latitudes and local times at {PERIGEE['h']} km: {c['rho_min']:.2e} to {c['rho_max']:.2e} kg/m^3")
    print(f"  density along the arc instead of the vertical: impulse ratio {c['arc_ratio']:.4f} ({(c['arc_ratio'] - 1) * 100:+.1f}%)")
    print(f"  grid of {c['n_points']} points; contribution above {H_TOP:.0f} km {c['above_top']:.1e} of the total; "
          f"1000-1200 km layer {c['cut_1000']:.1e}; grid halving changes the result by {c['grid_halving']:.1e}")
