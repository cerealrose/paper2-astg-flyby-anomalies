"""An additional l = 1 angular term (Section 5.1, Eq. 6, Appendix B).

Ordinary least squares with no intercept on the four anomalous encounters,
    Y_i = lambda_a X_i + lambda_b W_i,
    X_i = Rg dcos_i / [2 rp_i (1 + eps_i)],  W_i = Rg dsin_i / [2 rp_i (1 + eps_i)],  Y_i = (dVp / Vp)_i,
with dsin = sin(delta_out) - sin(delta_in), standard errors from the residual variance on two degrees of freedom,
and R^2 about the mean as in the companion paper."""
import numpy as np
import astg


def two_term_fit(flybys=None):
    fl = flybys or astg.load_flybys()
    X, W, Y = [], [], []
    for k in astg.ANOMALOUS:
        d = astg.derived(fl[k]); f = astg.RG / (2 * d['semilatus'])
        X.append(f * d['dcos']); W.append(f * d['dsin']); Y.append(fl[k]['dVp_mm_s'] * 1e-3 / d['vp'])
    A = np.column_stack([X, W]); Y = np.array(Y)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ coef; dof = len(Y) - 2
    cov = res @ res / dof * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - res @ res / np.sum((Y - Y.mean())**2)
    single = astg.anomalous_fit(fl)
    dj = astg.derived(fl['Juno']); fj = astg.RG / (2 * dj['semilatus'])
    juno = (coef[0] * dj['dcos'] + coef[1] * dj['dsin']) * fj * dj['vp'] * 1e3
    return dict(lam_a=coef[0], lam_b=coef[1], se_a=se[0], se_b=se[1], corr=cov[0, 1] / (se[0] * se[1]),
                r2=r2, r2_single=single['r2_mean'], juno=juno, juno_dsin=dj['dsin'], juno_dcos=dj['dcos'])


if __name__ == '__main__':
    t = two_term_fit()
    print('Eq. (6), four anomalous encounters, no intercept')
    print(f"  lambda_a = ({t['lam_a'] / 1e3:.2f} +- {t['se_a'] / 1e3:.2f}) e3   [{t['lam_a']:.1f} +- {t['se_a']:.1f}]")
    print(f"  lambda_b = ({t['lam_b'] / 1e3:.2f} +- {t['se_b'] / 1e3:.2f}) e3   [{t['lam_b']:.1f} +- {t['se_b']:.1f}]")
    print(f"  correlation of the two coefficients {t['corr']:+.3f}")
    print(f"  R^2 = {t['r2']:.3f} against {t['r2_single']:.3f} for the single term")
    print(f"  Juno: dsin = {t['juno_dsin']:+.3f} (dcos = {t['juno_dcos']:+.4f}); two-term prediction {t['juno']:+.2f} mm/s")
