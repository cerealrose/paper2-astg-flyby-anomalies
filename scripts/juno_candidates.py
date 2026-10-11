"""Candidate suppressions of the Juno prediction (Sections 5.2 to 5.7, Table 6, Figure 2).

  5.1  additional l = 1 angular term            two_term_fit.py
  5.2  higher multipoles: lambda_1l = (-1)^(l+1) (S_star / S*_l)^l (Nyambuya 2015), S*_l = (G_1l M R)^(1/2) with
       G_11 = 678 G and G_12 = 6.06 G, applied to the Earth's specific spin angular momentum 0.3307 R^2 omega,
       times the intrinsic GM / (rp c0^2) suppression of the l = 2 term at a 7000 km perigee
  5.3  eclipse thermal recoil: P = alpha S A with A = 60 m^2, alpha = 0.90, S = 1361 W/m^2, released as directed
       photon thrust for the 1140 s eclipse, m = 2873 kg; Lambertian factor 2/3; asymmetry 4 dT / T at T = 330 K
  5.4  spacecraft spin: S = 0.4 r^2 omega with r = 10 m at 2 rpm (Juno, array half-span) and r = 2.3 m at 3 rpm
       (Galileo rotor), against the Earth's 0.3307 R^2 omega
  5.5  atmospheric drag                          juno_drag.py
  5.6  relativistic and classical perturbations at Juno's geometry, as evaluated by Iorio (2009)"""
import numpy as np
import astg
from juno_prediction import juno
from two_term_fit import two_term_fit
from juno_drag import juno_drag

G11_OVER_G, G12_OVER_G = 678.0, 6.06            # Nyambuya (2015)
MOI_FACTOR = 0.3307                              # Earth's moment-of-inertia factor I / (M R^2)
IORIO_2009 = {'J2 oblateness': 0.0010, 'Schwarzschild geodetic': 0.0008, 'Lense-Thirring': 0.0,
              'solar radiation pressure': -0.0002}     # mm/s at Juno's geometry; Lense-Thirring below 1e-5
SOLAR_CONSTANT, ABSORPTIVITY, ARRAY_AREA, ECLIPSE_S, MASS = 1361.0, 0.90, 60.0, 1140.0, 3625.0 - 752.0
PANEL_T, DELTA_T = 330.0, (5.0, 20.0)
SPIN = {'Juno': (10.0, 2.0), 'Galileo': (2.3, 3.0)}     # radius (m), rpm


def earth_specific_spin():
    return MOI_FACTOR * astg.R_EARTH**2 * astg.OMEGA


def multipole_l2(prediction):
    S = earth_specific_spin()
    s1 = np.sqrt(G11_OVER_G * astg.GM * astg.R_EARTH); s2 = np.sqrt(G12_OVER_G * astg.GM * astg.R_EARTH)
    lam11, lam12 = S / s1, -(S / s2)**2
    supp = astg.GM / (7.0e6 * astg.C0**2)
    return dict(lam11=lam11, lam12=lam12, ratio=lam12 / lam11, intrinsic=supp,
                effect=abs(prediction * lam12 / lam11 * supp))


def thermal_recoil():
    P = ABSORPTIVITY * SOLAR_CONSTANT * ARRAY_AREA
    dv = P * ECLIPSE_S / (MASS * astg.C0) * 1e3
    lam = dv * 2 / 3
    return dict(power=P, dv=dv, lambertian=lam, asymmetry=tuple(lam * 4 * dT / PANEL_T for dT in DELTA_T))


def spin_ratios():
    S_e = earth_specific_spin()
    return S_e, {k: 0.4 * r**2 * rpm * 2 * np.pi / 60 / S_e for k, (r, rpm) in SPIN.items()}


def table6():
    j = juno(); pred = j['prediction']
    l2 = multipole_l2(pred); th = thermal_recoil(); S_e, ratios = spin_ratios()
    dr = juno_drag()['dv']; rel = sum(IORIO_2009.values())
    rows = [('Additional l = 1 angular term', 'prediction', two_term_fit()['juno']),
            ('l = 2 multipole', 'suppression', l2['effect']),
            ('Eclipse thermal recoil', 'suppression', th['dv']),
            ('Atmospheric drag', 'suppression', dr),
            ('Relativistic and classical', 'suppression', rel),
            ('Spacecraft spin', 'suppression', pred * ratios['Juno']),
            ('Gravity-model fidelity', 'excluded by reprocessing (Antreasian and Guinn 1998)', None)]
    return rows, pred, j['required_suppression']


if __name__ == '__main__':
    j = juno(); pred = j['prediction']
    l2 = multipole_l2(pred)
    print(f"5.2  lambda_11 = {l2['lam11']:.3e}, lambda_12 = {l2['lam12']:.3e}, lambda_12/lambda_11 = {l2['ratio']:.3f}; "
          f"GM/(rp c0^2) at 7000 km = {l2['intrinsic']:.1e}; l = 2 term ~ {l2['effect']:.0e} mm/s")
    th = thermal_recoil()
    print(f"5.3  absorbed power {th['power'] / 1e3:.1f} kW; directed recoil {th['dv']:.4f} mm/s, a factor of {pred / th['dv']:.1f} below "
          f"Eq. (5); Lambertian {th['lambertian']:.4f}; asymmetry {th['asymmetry'][0]:.4f} to {th['asymmetry'][1]:.4f} mm/s "
          f"for dT = {DELTA_T[0]:.0f} to {DELTA_T[1]:.0f} K")
    S_e, ratios = spin_ratios()
    print(f"5.4  Earth's specific spin angular momentum {S_e:.3e} m^2/s; S_sc/S_earth: Juno {ratios['Juno']:.1e}, Galileo {ratios['Galileo']:.1e}")
    print("5.6  Iorio (2009) terms: " + ', '.join(f"{k} {v:+.4f}" for k, v in IORIO_2009.items()) +
          f"; signed total {sum(IORIO_2009.values()):+.4f} mm/s")
    rows, pred, need = table6()
    print(f"\nTable 6. Candidate resolutions (prediction {pred:+.2f} mm/s, required suppression {need:.2f} mm/s)")
    for name, kind, val in rows:
        if val is None:
            print(f"  {name:32s} {kind}")
        elif kind == 'prediction':
            print(f"  {name:32s} Juno prediction {val:+.2f} mm/s (raised)")
        else:
            print(f"  {name:32s} {val:9.2e} mm/s   factor {need / val:9.3g} below the required suppression "
                  f"(log10 {np.log10(need / val):.1f}), factor {pred / val:9.3g} below Eq. (5)")
