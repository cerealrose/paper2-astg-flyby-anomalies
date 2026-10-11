"""Drag integral of Eq. (4) along the unperturbed hyperbola, with NRLMSISE-00 densities and a co-rotating
atmosphere, shared by galileo_drag.py and juno_drag.py.

    dV_drag = -(1/2) (C_D A / m) (1/J) * integral over phi of rho(r) v_r^2 r^2 dphi,

with phi the true anomaly, r = rp (1 + eps) / (1 + eps cos phi), dt = r^2 dphi / J, J = sqrt(GM p), and v_r the
speed relative to an atmosphere rotating rigidly with the Earth, v_r = |v - omega z x r|.  The density is taken
on the vertical through the perigee point at altitude h = h_p + (r - rp).  Lengths in km and speeds in km/s
inside this module; impulses are returned in mm/s.

NRLMSISE-00 is evaluated with pymsis (version=0 selects NRLMSISE-00), with the daily Ap index for all seven
Ap inputs of the model."""
import csv
import os
import warnings
import numpy as np
import pymsis
import astg

warnings.filterwarnings('ignore', module='pymsis')
GM = astg.GM / 1e9                 # km^3 s^-2
OMEGA = astg.OMEGA                 # rad s^-1
D2R = np.pi / 180
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data')
trapezoid = getattr(np, 'trapezoid', None) or np.trapz      # numpy >= 2 renamed trapz


def load_csv(name, key):
    with open(os.path.join(DATA, name), newline='') as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


def state(label):
    """Geocentric ICRF state (km, km/s) and JD(TDB) from data/horizons_states.csv."""
    row = load_csv('horizons_states.csv', 'label')[label]
    r = np.array([float(row[k]) for k in ('x_km', 'y_km', 'z_km')])
    v = np.array([float(row[k]) for k in ('vx_km_s', 'vy_km_s', 'vz_km_s')])
    return float(row['jd_tdb']), r, v


def space_weather(encounter):
    """F10.7 of the previous day, its 81-day centred mean and the daily Ap, from data/space_weather.csv."""
    row = load_csv('space_weather.csv', 'encounter')[encounter]
    return float(row['f107_previous_day_sfu']), float(row['f107_81day_centred_mean_sfu']), float(row['ap_daily'])


def gmst_deg(jd_ut1):
    """Greenwich mean sidereal time (IAU 1982), degrees."""
    jd0 = np.floor(jd_ut1 - 0.5) + 0.5
    h = (jd_ut1 - jd0) * 24
    tu = (jd0 - 2451545.0) / 36525
    g0 = (24110.54841 + 8640184.812866 * tu + 0.093104 * tu**2 - 6.2e-6 * tu**3) % 86400 / 240
    return (g0 + 1.00273790935 * h * 15) % 360


def sun_dir(jd):
    """Low-precision geocentric unit vector to the Sun, equatorial (Astronomical Almanac formula)."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    lam = L + 1.915 * np.sin(g * D2R) + 0.020 * np.sin(2 * g * D2R)
    eps = 23.439 - 4e-7 * n
    return np.array([np.cos(lam * D2R), np.cos(eps * D2R) * np.sin(lam * D2R), np.sin(eps * D2R) * np.sin(lam * D2R)])


def elements(r, v):
    """Eccentricity, perigee radius, inclination and the in-plane unit vectors of the osculating conic."""
    h = np.cross(r, v)
    e = np.cross(v, h) / GM - r / np.linalg.norm(r)
    ecc = np.linalg.norm(e)
    p_hat = e / ecc
    q_hat = np.cross(h / np.linalg.norm(h), p_hat)
    p = h @ h / GM
    return dict(ecc=ecc, rp=p / (1 + ecc), p_hat=p_hat, q_hat=q_hat, h=h,
                incl=np.degrees(np.arccos(h[2] / np.linalg.norm(h))))


def geodetic(rvec, jd_ut1):
    """Geodetic latitude, east longitude (deg) and height (km) on WGS84 of an equatorial position
    (Earth rotation by GMST; precession and nutation neglected)."""
    a, f = 6378.137, 1 / 298.257223563
    e2 = f * (2 - f)
    th = gmst_deg(jd_ut1) * D2R
    x = np.cos(th) * rvec[0] + np.sin(th) * rvec[1]
    y = -np.sin(th) * rvec[0] + np.cos(th) * rvec[1]
    z = rvec[2]
    lon = np.degrees(np.arctan2(y, x)) % 360
    pxy = np.hypot(x, y)
    lat = np.arctan2(z, pxy * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        hgt = pxy / np.cos(lat) - N
        lat = np.arctan2(z, pxy * (1 - e2 * N / (N + hgt)))
    return np.degrees(lat), lon, hgt


def msis(dates, lons, lats, alts, f107, f107a, ap):
    """NRLMSISE-00 total mass density (kg m^-3) at points; arrays of equal length are a track, not a grid."""
    dates, lons, lats, alts = (np.atleast_1d(np.asarray(x)) for x in (dates, lons, lats, alts))
    n = max(len(dates), len(lons), len(lats), len(alts))
    dates, lons, lats, alts = (np.broadcast_to(x, (n,)) for x in (dates, lons, lats, alts))
    out = pymsis.calculate(np.asarray(dates, dtype='datetime64[ms]'), np.asarray(lons, float), np.asarray(lats, float),
                           np.asarray(alts, float), f107s=np.full(n, f107), f107as=np.full(n, f107a),
                           aps=np.full((n, 7), ap), version=0)
    return np.asarray(out).reshape(n, -1)[:, 0]


def profile(date, lat, lon, h0, f107, f107a, ap, hmax=1200.0, step=0.5):
    """Density on the vertical through (lat, lon) from h0 to hmax km."""
    hs = np.arange(h0, hmax + step / 2, step)
    return hs, msis(np.datetime64(date), lon, lat, hs, f107, f107a, ap)


def scale_height(hs, rho):
    """Local density scale height (km) at the bottom of a profile."""
    return -(hs[2] - hs[0]) / np.log(rho[2] / rho[0])


def arc(rp, ecc, p_hat, q_hat, h_top_above_perigee, n=40001):
    """Points of the hyperbola, evenly spaced in true anomaly, up to rp + h_top_above_perigee:
    true anomaly, radius, position and inertial velocity (km, km/s), semi-latus rectum."""
    p = rp * (1 + ecc)
    rmax = rp + h_top_above_perigee
    phimax = np.arccos((p / rmax - 1) / ecc)
    phi = np.linspace(-phimax, phimax, n)
    r = p / (1 + ecc * np.cos(phi))
    R = r[:, None] * (np.cos(phi)[:, None] * p_hat + np.sin(phi)[:, None] * q_hat)
    V = np.sqrt(GM / p) * (-np.sin(phi)[:, None] * p_hat + (ecc + np.cos(phi))[:, None] * q_hat)
    return dict(phi=phi, r=r, R=R, V=V, p=p, rp=rp, ecc=ecc)


def relative_velocity(R, V, corot=True):
    return V - (OMEGA * np.cross(np.array([0.0, 0.0, 1.0]), R) if corot else 0.0)


def drag_integral(a, hs, rho_h, h_p, spin=None, areas=(1.0, 0.0), corot=True, rho_override=None):
    """Eq. (4) for C_D / m = 1 kg^-1: the impulse in mm/s, its integrand, the angle alpha (deg) between the spin
    axis and the relative velocity, and the density along the arc.  The projected area is
    A = A_ax |cos alpha| + A_tr |sin alpha| (Eq. 3) when a spin axis is given, and areas[0] otherwise;
    with areas = (1, 0) and no spin axis the result is the impulse for C_D A / m = 1 m^2 kg^-1."""
    phi, r = a['phi'], a['r']
    J = np.sqrt(GM * a['p'])                                    # km^2 s^-1
    vr = relative_velocity(a['R'], a['V'], corot)
    vr2 = np.einsum('ij,ij->i', vr, vr)                         # km^2 s^-2
    if rho_override is None:
        rho = np.exp(np.interp(h_p + (r - a['rp']), hs, np.log(rho_h)))
    else:
        rho = rho_override
    if spin is None:
        A = np.full_like(r, areas[0]); alpha = None
    else:
        ca = np.abs(vr @ spin) / np.sqrt(vr2)
        alpha = np.degrees(np.arccos(ca))
        A = areas[0] * ca + areas[1] * np.sqrt(1 - ca**2)
    integrand = rho * A * (vr2 * 1e6) * (r * 1e3)**2 / (J * 1e6)   # SI
    dv = 0.5 * trapezoid(integrand, phi)                      # m/s for C_D/m = 1 kg^-1
    return dict(dv_mm_s=dv * 1e3, integrand=integrand, alpha=alpha, rho=rho, vr=np.sqrt(vr2))


def exponential_path_length(H, rp, ecc):
    """Effective path length of a hyperbolic passage through an exponential atmosphere,
    [2 pi H rp (1 + eps) / eps]^(1/2), km."""
    return np.sqrt(2 * np.pi * H * rp * (1 + ecc) / ecc)


def time_from_perigee(phi, rp, ecc):
    """Time from perigee (s) on the hyperbola, from the hyperbolic anomaly."""
    F = 2 * np.arctanh(np.sqrt((ecc - 1) / (ecc + 1)) * np.tan(phi / 2))
    a = rp / (ecc - 1)
    return np.sqrt(a**3 / GM) * (ecc * np.sinh(F) - F)
