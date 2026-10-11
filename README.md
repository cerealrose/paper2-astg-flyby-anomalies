# A Drag Model for the Low-Perigee Galileo II Encounter and the Manoeuvre-Affected and Null Earth Gravity Assists

P. A. Nesvinga<sup>1,2</sup>, B. C. Jones<sup>3</sup> and G. G. Nyambuya<sup>1,2</sup>

<sup>1</sup> Fundamental Theoretical Physics Group, National University of Science and Technology, Bulawayo, Zimbabwe<br>
<sup>2</sup> Department of Applied Physics, National University of Science and Technology, Bulawayo, Zimbabwe<br>
<sup>3</sup> Department of Applied Mathematics, National University of Science and Technology, Bulawayo, Zimbabwe

Data and code for the second paper of the series, submitted to *Advances in Space Research* (2026). The paper treats the
three Earth flybys left outside the determination of the ASTG coupling: Galileo II, whose drag impulse is built forward
from the spacecraft and the NRLMSISE-00 atmosphere; Cassini, whose closest approach was accompanied by thrusting; and
Juno, the null encounter with the largest residual. It then refits the coupling on five encounters,
λ<sub>1</sub> = (3.528 ± 0.913) × 10<sup>3</sup>.

## What the paper says is here

Appendix D of the paper reads:

> The projected-area model of Table 1, the drag integration of Eq. (4) for Galileo II and Juno, the Cassini refit, the covariance analysis of Section 4.1, the regressions of Sections 5.1 and 7 and the code producing Figures 1–3 are available at https://github.com/cerealrose.

| Item in the paper | Where it is |
|---|---|
| Projected-area model, Table 1 and Eq. (3) | `data/galileo_area_components.csv`, `scripts/galileo_projected_area.py` |
| Drag integral of Eq. (4), shared by Galileo II and Juno | `scripts/drag_model.py` |
| Galileo II drag impulse and residual, Sections 2.1 to 2.4, Table 2, Appendix C | `scripts/galileo_drag.py` |
| Juno drag impulse, Section 5.5 | `scripts/juno_drag.py` |
| Cassini refit, Section 3 and Table 3 | `scripts/cassini_refit.py` |
| Covariance analysis, Section 4.1 and Table 5 | `scripts/juno_covariance.py` |
| Regression of Section 5.1, Eq. (6) | `scripts/two_term_fit.py` |
| Regression of Section 7, Eq. (8), Tables 8 and 9 | `scripts/refit.py` |
| Figures 1 to 3 | `scripts/figures.py`, written to `figures/` |

The remaining numbers have scripts of their own: the Juno prediction of Section 4, Table 4 and Appendix A in
`juno_prediction.py`; the candidate suppressions of Sections 5.2 to 5.7 and Table 6 in `juno_candidates.py`; the
selection rule of Section 6 and Table 7 in `selection_rule.py`. `check_against_paper.py` compares every number with the
printed value.

## Running it

Python 3.9 or later with the packages in `requirements.txt`:

```
pip install -r requirements.txt
cd scripts
python check_against_paper.py
```

`check_against_paper.py` recomputes the numbers in the text, Tables 1 to 9 and Appendices A and C, compares each with the
value printed in the paper at the printed precision, and prints one PASS or FAIL line per number (217 in all, in about
15 seconds). Each of the other scripts prints its own section. `python figures.py` redraws the three figures.

Everything runs offline. `juno_covariance.py --horizons` fetches the Juno perigee state again from the JPL Horizons API
instead of reading it from `data/horizons_states.csv`.

## Data

| File | Contents |
|---|---|
| `data/earth_flybys.csv` | The twelve documented Earth flybys of the first paper: anomaly, uncertainty and geometry (columns described in the [first repository](https://github.com/cerealrose/paper1-astg-flyby-foundation)) |
| `data/galileo_area_components.csv` | Axial and broadside area ranges of the Galileo orbiter by component, Table 1 |
| `data/space_weather.csv` | F10.7 of the previous day, its 81-day centred mean and the daily A<sub>p</sub> for Galileo II and Juno, from the GFZ Potsdam record |
| `data/horizons_states.csv` | JPL Horizons geocentric ICRF states (TDB) of Galileo near its second perigee and of Juno near and at perigee |

Inputs and settings of the scripts:

* NRLMSISE-00 through `pymsis` (`version=0`), with the daily A<sub>p</sub> in all seven A<sub>p</sub> inputs.
* Galileo II: closest-approach point 33.76° S, 354.42° E, 303.108 km at 1992-12-08 15:09:24.9 UTC (Antreasian and Guinn
  1998); spin axis on the Sun; C<sub>D</sub> from 2.0 to 2.4; mass 2497 kg; integration on 40 001 points in true anomaly up to
  1200 km.
* Juno: perigee point from the Horizons state; a 60 m<sup>2</sup> array broadside to the flow, C<sub>D</sub> = 2.2, and 2873 kg, the launch
  mass of 3625 kg less the 752 kg used in the deep-space manoeuvres of 2012; integration up to 2000 km.
* The atmosphere co-rotates rigidly with the Earth.
* Covariance: two-way range rate every 60 s, 0.1 mm/s noise, 10° elevation mask, DSN Goldstone, Canberra and Madrid and
  ESA New Norcia and Malargüe, ±3 days around perigee, partials from the two-body variational equations.

## Companion papers

| Paper | Repository |
|---|---|
| Application of the ASTG-Model to Earth Flyby Encounters: Plausible Origins and Velocity Anomalies | [paper1-astg-flyby-foundation](https://github.com/cerealrose/paper1-astg-flyby-foundation) |
| Application of the ASTG-Model to Earth Flyby Encounters: On a Pre-Flyby Predictive Test of the ASTG | [paper3-astg-flyby-prediction](https://github.com/cerealrose/paper3-astg-flyby-prediction) |

## Earlier version

The scripts that accompanied an earlier version of the manuscript are kept in the history of this repository under the
tag `v1`.

## Licence and citation

Code is under the MIT licence and the data files under CC BY 4.0; see `LICENSE`. Citation details are in `CITATION.cff`.

Contact: P. A. Nesvinga, praise.nesvinga@nust.ac.zw
