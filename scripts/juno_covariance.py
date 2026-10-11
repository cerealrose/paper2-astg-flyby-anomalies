"""Detectability of a velocity change at Juno's perigee (Section 4.1, Table 5).

Covariance analysis in the manner of Anderson et al. (2008): tracking before perigee fixes the trajectory and
tracking after it measures an impulsive change dV applied at perigee.
  * Estimated parameters: the perigee state (6), the three components of dV, and in two cases a solar-pressure
    scale factor (nominal acceleration 1e-10 km s^-2, anti-Sun).
  * Observable: two-way range rate every count interval from whichever station sees Juno highest above the
    elevation mask, among DSN Goldstone, Canberra and Madrid and ESA New Norcia and Malargue; white noise.
  * Partials from the two-body variational equations (state transition matrix), checked against finite
    differences by check_partials(); Earth rotation by GMST; light time neglected in the partials.
  * Covariance P = (H^T W H)^-1; the printed number is the 1-sigma uncertainty of dV along the perigee velocity.
The perigee state (geocentric ICRF, TDB) is read from data/horizons_states.csv.  With --horizons it is fetched
again from the JPL Horizons API (needs network access and the requests package).

Run time is a few minutes for the full table."""
import argparse
from datetime import datetime, timedelta
import numpy as np
from scipy.integrate import solve_ivp
import drag_model as dm

GM, OMEGA = 398600.4418, 7.2921150e-5            # km^3 s^-2, rad s^-1
A_WGS, F_WGS = 6378.137, 1 / 298.257223563
STATIONS = {                                      # geodetic latitude (deg), east longitude (deg), height (km)
    'Goldstone DSS-25': (35.3376, 243.1246, 0.96), 'Canberra DSS-34': (-35.3985, 148.9819, 0.69),
    'Madrid DSS-54': (40.4256, 355.7459, 0.84), 'New Norcia (ESA)': (-31.0482, 116.1915, 0.25),
    'Malargue (ESA)': (-35.7760, 290.6713, 1.55)}
JUNO_SRP = 1.0e-10                                # km s^-2, nominal solar-pressure acceleration
TDB_MINUS_UTC = 67.184                            # s in 2013
CASES = [('Baseline', {}),
         ('2 h tracking gap at perigee', {'gap_h': 1.0}),
         ('6 h tracking gap at perigee', {'gap_h': 3.0}),
         ('Arc of +-1 day', {'days': 1.0}),
         ('Arc of +-5 days', {'days': 5.0}),
         ('Noise 0.05 mm/s', {'noise_mm': 0.05}),
         ('Noise 0.2 mm/s', {'noise_mm': 0.2}),
         ('Solar-pressure scale factor estimated', {'est_srp': True}),
         ('6 h gap, solar pressure estimated, noise 0.2 mm/s', {'gap_h': 3.0, 'est_srp': True, 'noise_mm': 0.2})]


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon); e2 = F_WGS * (2 - F_WGS)
    N = A_WGS / np.sqrt(1 - e2 * np.sin(la)**2)
    return (np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)]),
            np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]))


def gmst(jd_ut1):
    return np.radians((280.46061837 + 360.98564736629 * (jd_ut1 - 2451545.0)) % 360)


def perigee_state():
    jd, r, v = dm.state('juno_perigee')
    return jd, np.concatenate([r, v])


def horizons_perigee(sid='-61', day='2013-10-09'):
    """Geocentric ICRF perigee state (km, km/s) and its JD(TDB) from the JPL Horizons API."""
    import requests

    def query(start, stop, step):
        p = {'format': 'json', 'COMMAND': f"'{sid}'", 'OBJ_DATA': "'NO'", 'MAKE_EPHEM': "'YES'", 'EPHEM_TYPE': "'VECTORS'",
             'CENTER': "'500@399'", 'REF_PLANE': "'FRAME'", 'REF_SYSTEM': "'ICRF'", 'VEC_TABLE': "'2'", 'OUT_UNITS': "'KM-S'",
             'CSV_FORMAT': "'YES'", 'VEC_LABELS': "'NO'", 'START_TIME': f"'{start}'", 'STOP_TIME': f"'{stop}'", 'STEP_SIZE': f"'{step}'"}
        res = requests.get('https://ssd.jpl.nasa.gov/api/horizons.api', params=p, timeout=120).json()['result']
        rows = [l.split(',') for l in res.split('$$SOE')[1].split('$$EOE')[0].strip().splitlines()]
        return np.array([float(r[0]) for r in rows]), np.array([[float(x) for x in r[2:8]] for r in rows])
    d0 = datetime.strptime(day, '%Y-%m-%d'); fmt = lambda t: t.strftime('%Y-%m-%d %H:%M:%S')
    jd, s = query(fmt(d0 - timedelta(hours=12)), fmt(d0 + timedelta(hours=36)), '1m')
    i = int(np.argmin(np.linalg.norm(s[:, :3], axis=1)))
    tc = d0 - timedelta(hours=12) + timedelta(minutes=i)
    jd, s = query(fmt(tc - timedelta(minutes=2)), fmt(tc + timedelta(minutes=2)), '240')
    j = int(np.argmin(np.linalg.norm(s[:, :3], axis=1)))
    return jd[j], s[j]


def deriv(t, y, srp_hat):
    r, v = y[:3], y[3:6]; rn = np.linalg.norm(r)
    a = -GM * r / rn**3
    G = GM * (3 * np.outer(r, r) / rn**5 - np.eye(3) / rn**3)
    A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3); A[3:, :3] = G
    Phi = y[6:42].reshape(6, 6); S = y[42:48]
    dS = A @ S; dS[3:] += srp_hat * JUNO_SRP
    return np.concatenate([v, a, (A @ Phi).ravel(), dS])


def propagate(x0, span, srp_hat):
    y0 = np.concatenate([x0, np.eye(6).ravel(), np.zeros(6)])
    return solve_ivp(deriv, (0, span), y0, args=(srp_hat,), rtol=1e-11, atol=1e-12, dense_output=True, method='DOP853')


def design(xp, jd_tdb, days=3.0, count=60.0, gap_h=0.0, mask_deg=10.0, est_srp=False):
    """Rows of the design matrix (unit noise), with the time and station of each observation."""
    jd_ut = jd_tdb - TDB_MINUS_UTC / 86400
    shat = -dm.sun_dir(jd_ut)
    fw, bw = propagate(xp, days * 86400, shat), propagate(xp, -days * 86400, shat)
    sta = {k: ecef(*v) for k, v in STATIONS.items()}
    npar = 10 if est_srp else 9
    rows, info = [], []
    for t in np.arange(-days * 86400, days * 86400 + 1, count):
        if abs(t) < gap_h * 3600 or t == 0:
            continue
        y = (fw if t > 0 else bw).sol(t); r, v = y[:3], y[3:6]
        Phi = y[6:42].reshape(6, 6); S = y[42:48]
        th = gmst(jd_ut + t / 86400); c, s = np.cos(th), np.sin(th); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        best = None
        for k, (pe, up) in sta.items():
            rs = R @ pe; vs = np.cross([0, 0, OMEGA], rs); d = r - rs; rho = np.linalg.norm(d)
            el = np.degrees(np.arcsin(d @ (R @ up) / rho))
            if el >= mask_deg and (best is None or el > best[0]):
                best = (el, k, d, v - vs, rho)
        if best is None:
            continue
        el, k, d, dv, rho = best
        rr = d @ dv / rho
        dh_dx = np.concatenate([dv / rho - rr * d / rho**2, d / rho])       # d(range rate)/d(state at t)
        h = np.zeros(npar); h[:6] = dh_dx @ Phi
        if t > 0:
            h[6:9] = dh_dx @ Phi[:, 3:6]                                       # impulse at perigee
        if est_srp:
            h[9] = dh_dx @ S
        rows.append(h); info.append((t, k))
    return np.array(rows), info


def analyse(xp, jd_tdb, days=3.0, count=60.0, noise_mm=0.1, gap_h=0.0, mask_deg=10.0, est_srp=False):
    """1-sigma uncertainty (mm/s) of dV along the perigee velocity and of its magnitude, and the number of
    observations."""
    H, info = design(xp, jd_tdb, days, count, gap_h, mask_deg, est_srp)
    N = H.T @ H
    sig = noise_mm * 1e-6                                                      # mm/s -> km/s
    scale = np.sqrt(np.diag(N)); P = np.linalg.inv(N / np.outer(scale, scale)) / np.outer(scale, scale) * sig**2
    u = xp[3:] / np.linalg.norm(xp[3:])
    return np.sqrt(u @ P[6:9, 6:9] @ u) * 1e6, np.sqrt(np.trace(P[6:9, 6:9])) * 1e6, len(H)


def range_rate(x0, t, station, jd_tdb, dv=np.zeros(3)):
    """Range rate by direct two-body integration from the perigee state, with an impulse dV at perigee."""
    f = lambda tt, y: np.concatenate([y[3:], -GM * y[:3] / np.linalg.norm(y[:3])**3])
    y0 = x0.copy()
    if t > 0:
        y0[3:] += dv
    y = solve_ivp(f, (0, t), y0, rtol=1e-13, atol=1e-13, method='DOP853').y[:, -1]
    th = gmst(jd_tdb - TDB_MINUS_UTC / 86400 + t / 86400); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pe, up = ecef(*STATIONS[station]); rs = R @ pe; vs = np.cross([0, 0, OMEGA], rs)
    d = y[:3] - rs; dvv = y[3:] - vs
    return d @ dvv / np.linalg.norm(d)


def check_partials(xp, jd_tdb, samples=10):
    """Largest relative difference between the variational-equation partials and central finite differences,
    over a sample of observations of a +-1 day arc with hourly points."""
    H, info = design(xp, jd_tdb, days=1.0, count=3600.0)
    worst = 0.0
    for i in range(0, len(info), max(1, len(info) // samples)):
        t, k = info[i]
        for j, step in enumerate([1e-3] * 3 + [1e-6] * 3):
            e = np.zeros(6); e[j] = step
            fd = (range_rate(xp + e, t, k, jd_tdb) - range_rate(xp - e, t, k, jd_tdb)) / (2 * step)
            worst = max(worst, abs(fd - H[i, j]) / max(abs(H[i, j]), 1e-12))
        if t > 0:
            for j in range(3):
                e = np.zeros(3); e[j] = 1e-6
                fd = (range_rate(xp, t, k, jd_tdb, e) - range_rate(xp, t, k, jd_tdb, -e)) / 2e-6
                worst = max(worst, abs(fd - H[i, 6 + j]) / max(abs(H[i, 6 + j]), 1e-12))
    return worst


def table5(xp=None, jd=None, cases=CASES):
    if xp is None:
        jd, xp = perigee_state()
    rows = []
    for name, kw in cases:
        sa, st, n = analyse(xp, jd, **kw)
        rows.append(dict(case=name, n_obs=n, sigma=sa, three_sigma=3 * sa, sigma_total=st))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--horizons', action='store_true', help='fetch the perigee state from the JPL Horizons API')
    args = ap.parse_args()
    jd, xp = horizons_perigee() if args.horizons else perigee_state()
    print(f"Juno perigee: JD(TDB) {jd:.9f}, r = {np.linalg.norm(xp[:3]):.3f} km, v = {np.linalg.norm(xp[3:]):.4f} km/s")
    print(f"partials against finite differences: largest relative difference {check_partials(xp, jd):.1e}")
    print('\nTable 5. Noise-only uncertainty of an along-track velocity change at perigee')
    print(f"{'Case':52s} {'Observations':>12s} {'1 sigma (mm/s)':>15s} {'3 sigma (mm/s)':>15s}")
    for r in table5(xp, jd):
        print(f"{r['case']:52s} {r['n_obs']:12d} {r['sigma']:15.4f} {r['three_sigma']:15.4f}")
