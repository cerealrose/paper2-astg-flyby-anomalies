"""Projected area of the Galileo orbiter at the Galileo II encounter (Section 2.2, Table 1, Eq. 3).

The component ranges of Table 1 are read from data/galileo_area_components.csv.  With alpha the angle between
the spin axis and the velocity relative to the atmosphere, the projected area is
    A(alpha) = A_ax |cos alpha| + A_tr |sin alpha|                                   (Eq. 3)
and its largest value at any angle is (A_ax^2 + A_tr^2)^(1/2).  The 4.8 m high-gain antenna is excluded as a
disc because its deployment failed in April 1991; the area it would have presented is printed for comparison.
The perigee angle alpha = 56.0 deg comes from galileo_drag.geometry()."""
import csv
import os
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'data', 'galileo_area_components.csv')
HGA_DIAMETER = 4.8          # m
ACEDO_RANGE = (16.0, 28.1)  # m^2, cross-section bracket of Acedo (2017), from the overall dimensions


def components(path=DATA_FILE):
    with open(path, newline='') as fh:
        return [dict(component=r['component'], ax=(float(r['axial_min_m2']), float(r['axial_max_m2'])),
                     tr=(float(r['broadside_min_m2']), float(r['broadside_max_m2']))) for r in csv.DictReader(fh)]


def totals(rows=None):
    """Summed axial and broadside ranges (m^2) and the mid-range values used for the central case."""
    rows = rows or components()
    ax = tuple(sum(r['ax'][i] for r in rows) for i in (0, 1))
    tr = tuple(sum(r['tr'][i] for r in rows) for i in (0, 1))
    return dict(ax=ax, tr=tr, ax_mid=0.5 * sum(ax), tr_mid=0.5 * sum(tr))


def projected_area(alpha_deg, ax, tr):
    a = np.radians(alpha_deg)
    return ax * abs(np.cos(a)) + tr * abs(np.sin(a))


def largest_area(ax, tr):
    return np.hypot(ax, tr)


def hga_disc_area():
    return np.pi * (HGA_DIAMETER / 2)**2


def area_cases(alpha_deg):
    """Projected area at alpha for the lower, central and upper cases of Table 2."""
    t = totals()
    return dict(lower=projected_area(alpha_deg, t['ax'][0], t['tr'][0]),
                central=projected_area(alpha_deg, t['ax_mid'], t['tr_mid']),
                upper=projected_area(alpha_deg, t['ax'][1], t['tr'][1]))


if __name__ == '__main__':
    from galileo_drag import geometry
    alpha = geometry()['alpha_perigee']
    rows = components(); t = totals(rows)
    print('Table 1. Projected area of the Galileo orbiter at GEGA2 (m^2)')
    print(f"{'Component':42s} {'Axial':>11s} {'Broadside':>11s}")
    for r in rows:
        print(f"{r['component']:42s} {r['ax'][0]:4.1f}--{r['ax'][1]:<5.1f} {r['tr'][0]:4.1f}--{r['tr'][1]:<5.1f}")
    print(f"{'Total':42s} {t['ax'][0]:4.1f}--{t['ax'][1]:<5.1f} {t['tr'][0]:4.1f}--{t['tr'][1]:<5.1f}")
    c = area_cases(alpha)
    print(f"\nperigee angle between spin axis and relative velocity alpha = {alpha:.2f} deg")
    print(f"projected area at perigee, Eq. (3): lower {c['lower']:.2f}, central {c['central']:.2f}, upper {c['upper']:.2f} m^2")
    print(f"largest area at any angle, (A_ax^2 + A_tr^2)^(1/2): {largest_area(t['ax'][0], t['tr'][0]):.2f} to "
          f"{largest_area(t['ax'][1], t['tr'][1]):.2f} m^2")
    print(f"4.8 m high-gain antenna as a disc: {hga_disc_area():.1f} m^2 end-on, "
          f"{hga_disc_area() * np.cos(np.radians(alpha)):.1f} m^2 at alpha (not in the silhouette at this encounter)")
    print(f"Acedo (2017) bracket for comparison: {ACEDO_RANGE[0]} to {ACEDO_RANGE[1]} m^2")
