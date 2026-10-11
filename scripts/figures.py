"""Figures 1 to 3 of the paper, written to ../figures/.
  fig1.png        (a) projected area of the Galileo orbiter by component at the perigee angle; (b) drag impulse,
                  measured change at perigee and residual against the ASTG prediction (Section 2)
  fig2_juno.png   available suppression from each candidate at the Juno geometry (Section 5)
  fig3_refit.png  observed against predicted perigee anomaly and residuals, before and after correction (Section 7)
No quantity in Figure 1 is fitted to the observed anomaly."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astg
from galileo_projected_area import components, hga_disc_area
from galileo_drag import geometry, density_profile, drag_cases, residual
from juno_candidates import table6
from refit import refit, table9
from selection_rule import null_deviations

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'dejavusans'})
BLUE, RED, DARK = '#4f7cac', '#b22222', '#1f4e79'
SHORT = {'Furled high-gain antenna and sunshade': 'furled HGA\n+ sunshade', 'Rotor bus': 'rotor bus',
         'Despun section and attached probe': 'despun sec.\n+ probe', 'Science boom, 10.9 m, spin-averaged': 'science boom',
         'RTG booms and generators, spin-averaged': 'RTG booms'}


def figure1():
    g = geometry(); hs, rho_h = density_profile(); cases = drag_cases(g, hs, rho_h); r = residual(cases)
    alpha = np.radians(g['alpha_perigee']); ca, sa = np.cos(alpha), np.sin(alpha)
    rows = components()
    lo = np.array([c['ax'][0] * ca + c['tr'][0] * sa for c in rows]); hi = np.array([c['ax'][1] * ca + c['tr'][1] * sa for c in rows])
    fig, ax = plt.subplots(1, 2, figsize=(3840 / 400, 1480 / 400), dpi=400, gridspec_kw=dict(width_ratios=[1, 1.25]))
    a = ax[0]; y = np.arange(len(rows))[::-1]
    a.barh(y, hi - lo, left=lo, height=0.55, color=BLUE, edgecolor='0.2', lw=0.5)
    hga = hga_disc_area() * ca
    a.axvline(hga, color=RED, ls='--', lw=1.1); a.text(hga - 0.15, y[1] + 0.35, 'deployed HGA\n(never unfurled)', color=RED, ha='right', fontsize=8)
    a.set_yticks(y); a.set_yticklabels([SHORT[c['component']] for c in rows], fontsize=9); a.set_xlim(0, 12)
    a.set_xlabel(r'projected area at perigee, $\alpha=%.0f^{\circ}$ (m$^2$)' % g['alpha_perigee'])
    a.set_title('(a) Galileo at GEGA2, component areas', fontsize=11)
    a = ax[1]
    bars = [('drag, this work\n(%.1f to %.1f m$^2$)' % (cases['lower']['area'], cases['upper']['area']),
             cases['central']['dv'], cases['upper']['dv'], cases['lower']['dv'], BLUE),
            ('measured, Anderson et al.\n(2008), at perigee', r['measured'], r['measured'] - r['sigma_measured'],
             r['measured'] + r['sigma_measured'], '0.35'),
            ('residual,\nmeasured minus drag', r['residual'], r['low'], r['high'], '#2e7d32')]
    yy = np.arange(len(bars))[::-1]
    for (lab, c, l, h, col), yv in zip(bars, yy):
        a.plot([l, h], [yv, yv], color=col, lw=7, alpha=0.35, solid_capstyle='butt')
        a.plot(c, yv, 'o', color=col, mec='k', mew=0.6, ms=8)
    a.axvline(r['astg'], color='k', ls=':', lw=1.5); a.axvline(0, color='0.6', lw=0.8)
    a.text(r['astg'] + 0.07, yy[0] + 0.45, 'ASTG law, $%.2f$' % r['astg'], fontsize=8, ha='left')
    a.set_yticks(yy); a.set_yticklabels([b[0] for b in bars], fontsize=9); a.set_ylim(-0.6, 2.8)
    a.set_xlim(-9.8, 4.6); a.set_xticks([-8, -6, -4, -2, 0, 2, 4]); a.set_xlabel(r'$\Delta V_p$  (mm s$^{-1}$)')
    a.set_title('(b) drag and residual at perigee', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1.png'), dpi=400); plt.close(fig)


def figure2():
    rows, pred, need = table6()
    # the l = 2 and spin entries are order-of-magnitude estimates, plotted at the nearest power of ten as in Table 6
    oom = ('l = 2 multipole', 'Spacecraft spin')
    shown = [(name, 10.0**np.round(np.log10(val)) if name in oom else val) for name, kind, val in rows if kind == 'suppression']
    label = {'l = 2 multipole': r'$\ell=2$ multipole', 'Eclipse thermal recoil': 'eclipse thermal recoil',
             'Atmospheric drag': 'atmospheric drag', 'Relativistic and classical': 'relativistic and classical',
             'Spacecraft spin': 'spacecraft spin'}
    fig, a = plt.subplots(figsize=(2800 / 400, 1400 / 400), dpi=400)
    yv = np.arange(len(shown))[::-1]
    a.barh(yv, [v for _, v in shown], color=BLUE, edgecolor='0.2', lw=0.5, height=0.55)
    a.axvline(need, color=RED, ls='--', lw=1.1); a.text(need * 1.09, yv[0] + 0.1, 'required\n%.2f' % need, color=RED, fontsize=8, va='center')
    a.set_xscale('log'); a.set_xlim(1e-11, 20); a.set_yticks(yv); a.set_yticklabels([label[n] for n, _ in shown], fontsize=9)
    a.set_xlabel(r'available suppression at Juno (mm s$^{-1}$)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_juno.png'), dpi=400); plt.close(fig)


def figure3():
    fl = astg.load_flybys()
    four, five, g = refit(fl)
    L0, S0, L1, S1 = four['lam'], four['se'], five['lam'], five['se']
    pred = lambda k, lam: astg.astg_dvp(fl[k], lam)
    consistent = [k for k in ['Galileo I', 'NEAR', 'Rosetta I']] + \
                 [r['spacecraft'] for r in null_deviations(fl) if r['z_law'] < 3]
    t9 = {r['encounter']: r for r in table9(fl, four, five, g)}
    treated = [('Galileo II', t9['Galileo II']['before'], t9['Galileo II']['after']), ('Cassini', t9['Cassini']['before'], t9['Cassini']['after'])]
    fig, axs = plt.subplots(1, 2, figsize=(3840 / 400, 1750 / 400), dpi=400, gridspec_kw=dict(width_ratios=[1, 1.05]))
    a = axs[0]; lo, hi = -6, 8.5
    a.plot([lo, hi], [lo, hi], 'k--', lw=0.9, zorder=1); a.axhline(0, color='0.85', lw=0.7); a.axvline(0, color='0.85', lw=0.7)
    for i, k in enumerate(consistent):
        a.errorbar(pred(k, L1), fl[k]['dVp_mm_s'], yerr=fl[k]['sigma_dVp_mm_s'], fmt='o', color=DARK, ms=5, capsize=2, zorder=3,
                   label='consistent in companion paper' if i == 0 else None)
    a.annotate('NEAR', (pred('NEAR', L1), fl['NEAR']['dVp_mm_s']), xytext=(-36, 4), textcoords='offset points', fontsize=8)
    a.annotate('Galileo I', (pred('Galileo I', L1), fl['Galileo I']['dVp_mm_s']), xytext=(8, 4), textcoords='offset points', fontsize=8)
    a.annotate('Rosetta I', (pred('Rosetta I', L1), fl['Rosetta I']['dVp_mm_s']), xytext=(6, 4), textcoords='offset points', fontsize=8)
    a.annotate('nulls', (0.15, 0.0), xytext=(-28, 12), textcoords='offset points', fontsize=8)
    for k, (ob, sb), (oa, sa) in treated:
        pb, pa = pred(k, L0), pred(k, L1)
        a.errorbar(pb, ob, yerr=sb, fmt='s', mfc='white', mec=RED, ecolor=RED, ms=6, capsize=2, zorder=4,
                   label='before correction' if k == 'Galileo II' else None)
        a.errorbar(pa, oa, yerr=sa, fmt='s', color=RED, ms=6, capsize=2, zorder=5, label='after correction' if k == 'Galileo II' else None)
        a.annotate('', xy=(pa, oa), xytext=(pb, ob), arrowprops=dict(arrowstyle='->', color=RED, lw=1.1), zorder=4)
    a.annotate('Galileo II', (pred('Galileo II', L0), treated[0][1][0]), xytext=(8, -4), textcoords='offset points', fontsize=8, color=RED)
    a.annotate('Cassini', (pred('Cassini', L0), treated[1][1][0]), xytext=(8, -10), textcoords='offset points', fontsize=8, color=RED)
    pj = pred('Juno', L0)
    a.errorbar(pj, 0.0, yerr=[[0], [0.2]], fmt='^', mfc='white', mec='k', ecolor='k', ms=7, capsize=2, zorder=5, label='Juno, before / after selection rule')
    a.errorbar(0.0, 0.0, yerr=[[0], [0.2]], fmt='^', color='k', ms=7, capsize=2, zorder=6)
    a.annotate('', xy=(0.05, 0.0), xytext=(pj - 0.05, 0.0), arrowprops=dict(arrowstyle='->', color='k', lw=1.0), zorder=5)
    a.annotate('Juno', (pj, 0), xytext=(6, -12), textcoords='offset points', fontsize=8)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xlabel(r'ASTG prediction $\Delta V_p$ (mm s$^{-1}$)'); a.set_ylabel(r'observed $\Delta V_p$ (mm s$^{-1}$)')
    a.set_title('(a) observed against predicted', fontsize=11); a.legend(frameon=False, fontsize=8, loc='upper left')
    b = axs[1]
    names = consistent + ['Galileo II', 'Cassini', 'Juno']
    y = np.arange(len(names))[::-1]
    for k, yy in zip(consistent, y):
        p = pred(k, L1); e = np.hypot(fl[k]['sigma_dVp_mm_s'], abs(p) * S1 / L1)
        b.errorbar(fl[k]['dVp_mm_s'] - p, yy, xerr=e, fmt='o', color=DARK, ms=5, capsize=2)
    for (k, (ob, sb), (oa, sa)), yy in zip(treated, y[len(consistent):]):
        pb, pa = pred(k, L0), pred(k, L1)
        b.errorbar(ob - pb, yy + 0.18, xerr=np.hypot(sb, abs(pb) * S0 / L0), fmt='s', mfc='white', mec=RED, ecolor=RED, ms=6, capsize=2)
        b.errorbar(oa - pa, yy - 0.18, xerr=np.hypot(sa, abs(pa) * S1 / L1), fmt='s', color=RED, ms=6, capsize=2)
        b.annotate('', xy=(oa - pa, yy - 0.18), xytext=(ob - pb, yy + 0.18), arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
    b.errorbar(0 - pj, y[-1] + 0.18, xerr=np.hypot(0.2, pj * S0 / L0), fmt='^', mfc='white', mec='k', ecolor='k', ms=7, capsize=2)
    b.errorbar(0.0, y[-1] - 0.18, xerr=0.2, fmt='^', color='k', ms=7, capsize=2)
    b.annotate('', xy=(0.0, y[-1] - 0.18), xytext=(0 - pj, y[-1] + 0.18), arrowprops=dict(arrowstyle='->', color='k', lw=1.0))
    b.axvline(0, color='k', lw=0.9); b.set_yticks(y); b.set_yticklabels(names, fontsize=9)
    b.axhline(y[len(consistent)] + 0.5, color='0.8', lw=0.8, ls=':')
    b.set_xlabel(r'observed minus predicted (mm s$^{-1}$)'); b.set_title('(b) residuals by encounter', fontsize=11); b.set_xlim(-4.2, 6.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_refit.png'), dpi=400); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    figure1(); figure2(); figure3()
    print('wrote', os.path.normpath(os.path.join(OUT, 'fig1.png')), ', fig2_juno.png and fig3_refit.png')
