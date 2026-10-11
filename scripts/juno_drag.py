"""Juno drag impulse (Section 5.5): Eq. (4) applied to the Juno geometry.

Inputs
  * Hyperbola: rp, eps and Vp from data/earth_flybys.csv (hp = 561.5 km, Vinf = 10.389 km/s).
  * Orientation of the orbit, time of perigee and perigee point: JPL Horizons geocentric ICRF state of Juno (-61)
    near perigee, data/horizons_states.csv.
  * NRLMSISE-00 with the indices in data/space_weather.csv (F10.7 of 8 October 2013 = 111.9 sfu, 81-day centred
    mean 126.5 sfu, daily Ap = 29).
  * A 60 m^2 array broadside to the flow, C_D = 2.2, and the flyby mass of 2873 kg (launch mass 3625 kg less the
    752 kg of propellant used in the two deep-space manoeuvres of 2012).
The density profile and the arc run from the perigee altitude to 2000 km; extending both to 4000 km adds less
than 0.1% to the impulse."""
import numpy as np
import astg
import drag_model as dm

TDB_MINUS_UTC = 67.184          # s in 2013 (TT - TAI = 32.184 s, 35 leap seconds)
AREA, CD = 60.0, 2.2            # m^2, free-molecular drag coefficient
MASS = 3625.0 - 752.0           # kg
H_TOP = 2000.0                  # km, top of the integration


def hyperbola():
    d = astg.derived(astg.load_flybys()['Juno'])
    return dict(rp=d['rp'] / 1e3, ecc=d['eps'], vp=d['vp'] / 1e3, hp=d['rp'] / 1e3 - astg.R_EARTH / 1e3)


def perigee_point():
    """Time of perigee (JD UTC and ISO string) and its geodetic position from the Horizons state."""
    jd_tdb, r, v = dm.state('juno_near_perigee')
    el = dm.elements(r, v)
    a = el['rp'] / (el['ecc'] - 1)
    F = np.arccosh((1 + np.linalg.norm(r) / a) / el['ecc']) * np.sign(r @ v)
    t_from_p = np.sqrt(a**3 / dm.GM) * (el['ecc'] * np.sinh(F) - F)
    jd_utc = jd_tdb - t_from_p / 86400 - TDB_MINUS_UTC / 86400
    lat, lon, _ = dm.geodetic(el['p_hat'] * el['rp'], jd_utc)
    secs = (jd_utc - 2456574.5) * 86400
    iso = '2013-10-09T%02d:%02d:%06.3f' % (secs // 3600, secs % 3600 // 60, secs % 60)
    return dict(el=el, jd_utc=jd_utc, iso=iso, lat=lat, lon=lon)


def juno_drag():
    pp = perigee_point(); hyp = hyperbola(); el = pp['el']
    f107, f107a, ap = dm.space_weather('Juno')
    hs, rho_h = dm.profile(pp['iso'], pp['lat'], pp['lon'], hyp['hp'], f107, f107a, ap, hmax=H_TOP)
    a = dm.arc(hyp['rp'], hyp['ecc'], el['p_hat'], el['q_hat'], H_TOP - hyp['hp'])
    rot = dm.drag_integral(a, hs, rho_h, hyp['hp'], areas=(AREA, 0.0))
    k_in = dm.drag_integral(a, hs, rho_h, hyp['hp'], areas=(AREA, 0.0), corot=False)['dv_mm_s']
    H = dm.scale_height(hs, rho_h)
    l_exp = dm.exponential_path_length(H, hyp['rp'], hyp['ecc'])
    exp_in = 0.5 * CD * AREA / MASS * rho_h[0] * hyp['vp'] * 1e3 * l_exp * 1e3 * 1e3      # mm/s
    vr_p = rot['vr'][len(a['phi']) // 2]
    return dict(perigee=pp, rho_p=rho_h[0], H=H, vr_perigee=vr_p, vp=hyp['vp'],
                dv=rot['dv_mm_s'] * CD / MASS, dv_inertial=k_in * CD / MASS,
                exp_inertial=exp_in, exp_relative=exp_in * (vr_p / hyp['vp'])**2)


if __name__ == '__main__':
    j = juno_drag(); pp = j['perigee']; el = pp['el']
    print(f"Horizons osculating conic: rp {el['rp']:.2f} km, eps {el['ecc']:.4f}, inclination {el['incl']:.2f} deg (prograde)")
    print(f"perigee {pp['iso']} UTC at {abs(pp['lat']):.2f} S, {pp['lon']:.2f} E")
    print(f"density at {hyperbola()['hp']:.1f} km {j['rho_p']:.3e} kg/m^3, local scale height {j['H']:.1f} km")
    print(f"drag impulse ({AREA:.0f} m^2, C_D {CD}, {MASS:.0f} kg): {j['dv']:.4f} mm/s co-rotating, {j['dv_inertial']:.4f} inertial "
          f"(co-rotation lowers it by {(1 - j['dv'] / j['dv_inertial']) * 100:.1f}%)")
    print(f"single exponential: {j['exp_inertial']:.4f} mm/s against the inertial integral, {(1 - j['exp_inertial'] / j['dv_inertial']) * 100:.1f}% less;"
          f" at the relative speed {j['exp_relative']:.4f} against {j['dv']:.4f}, {(1 - j['exp_relative'] / j['dv']) * 100:.1f}% less")
    pred = astg.astg_dvp(astg.load_flybys()['Juno'], astg.anomalous_fit(astg.load_flybys())['lam'])
    need = pred - 0.200
    print(f"the impulse is a factor of {pred / j['dv']:.1f} short of the prediction of Eq. (5) ({pred:.2f} mm/s) "
          f"and a factor of {need / j['dv']:.1f} short of the required suppression ({need:.2f} mm/s)")
