================================================================================
 Earth-Flyby Velocity Anomalies
 Within the Azimuthally Symmetric Theory of Gravitation
 P. A. Nesvinga, G. G. Nyambuya, B. C. Jones
 NUST Fundamental Theoretical Physics Group
================================================================================

This repository accompanies the manuscript "Earth-Flyby Velocity Anomalies
within the Azimuthally Symmetric Theory of Gravitation" (submitted to
Physical Review D). It is the repository referenced in the manuscript's
"Data and Code Availability" appendix. It contains the primary flyby
dataset, the manuscript source, and a complete, independently re-run set
of scripts reproducing every table, fit, and statistical claim in the
paper.

This is the data-confrontation paper of a companion series. The
theoretical derivation of the formula tested here is established in the
companion paper "Plausible Origins of the Earth Flyby Anomalies at the
Perigee Point" (separate repository -- see CROSS-REFERENCES below).
Dedicated treatments of the 2013 Juno flyby and the December 2026 Europa
Clipper prediction are likewise separate, forthcoming companion papers,
not included here.

--------------------------------------------------------------------------------
REPOSITORY CONTENTS
--------------------------------------------------------------------------------

data/
    flyby_data.csv          Primary dataset: all nine documented Earth
                             flybys (Table I of the manuscript), machine-
                             readable.
    DATA_DICTIONARY.txt      Column definitions, units, and full sourcing
                             for every value in flyby_data.csv, including
                             the two data-transcription errors this
                             project caught and corrected.

scripts/
    01_fit_lambda1_and_tables.py
        The primary script referenced in the manuscript's Data and Code
        Availability appendix. Reads flyby_data.csv and reproduces Table
        II (derived orbital quantities and the per-flyby diagnostic
        lambda_1), the core global fits Eq. (7) (4 anomalous flybys) and
        Eq. (8) (5 flybys including Galileo II), Table III (ASTG
        predictions vs. observations), and Table IV (comparison with the
        independent Anderson-constant estimate).

    02_binet_route_two_parameter_fit.py
        Reproduces the two-parameter Binet-route regression quoted in
        Sec. II.C (R^2=0.938, t=5.53, p=0.031) by inverting the companion
        paper's boxed Binet-route formula for a per-flyby coupling and
        regressing it against Delta cos(delta) with slope and intercept.
        Validates the method by first reproducing the known PRE-
        correction result on the uncorrected data, then applies it to the
        corrected data -- the same two-step discipline used throughout
        this project before trusting any numerical result.

    03_anderson2008_data_correction.py
        Verifies, row by row, that the manuscript's Table I values for
        dVp and sigma are correctly derived from Anderson et al. (2008)'s
        own published dV_infinity and sigma(dV_infinity) via the vis-viva
        scaling described in the Table I caption -- and includes the
        independent self-consistency check (V_infinity implied by r_p and
        eccentricity alone) that originally caught a decimal-point
        transcription error in an intermediate secondary source.

    04_aic_reconciliation.py
        The AIC-convention comparison referenced in Sec. II.C. An earlier
        draft quoted an AIC comparison (-55.3 vs. -54.2) that could not be
        reconstructed on revisiting it; this script implements four
        standard AIC conventions, none of which reproduce that figure
        (one reverses which route it favours), and is provided so a
        referee -- or anyone with the original computation -- can
        reconcile the discrepancy directly rather than take either number
        on faith. The manuscript does not report an AIC comparison as a
        result of this finding.

requirements.txt   Python dependencies (numpy, scipy)
LICENSE             MIT License (applies to all code in scripts/) plus a
                    CC-BY-4.0 grant for the contents of data/
CITATION.cff        Machine-readable citation metadata

--------------------------------------------------------------------------------
CROSS-REFERENCES
--------------------------------------------------------------------------------
The symbolic derivation this paper's central formula rests on (the exact
vis-viva relation, the Binet-route cross-check, the phi-dependent-coupling
appendix, and the lambda_2/Juno order-of-magnitude estimate) is verified
in the companion repository for "Plausible Origins of the Earth Flyby
Anomalies at the Perigee Point". That repository's script
01_verify_visviva_derivation.py also reproduces this paper's R^2=0.90 vs.
0.53 comparison between Delta cos(delta) and Delta sin(delta), referenced
in this manuscript's Sec. VI.F.

--------------------------------------------------------------------------------
REPRODUCING THE RESULTS
--------------------------------------------------------------------------------

1. Install dependencies:
       pip install -r requirements.txt

2. Run the scripts in order (each is self-contained and prints its own
   step-by-step output and pass/fail status):
       python3 scripts/01_fit_lambda1_and_tables.py
       python3 scripts/02_binet_route_two_parameter_fit.py
       python3 scripts/03_anderson2008_data_correction.py
       python3 scripts/04_aic_reconciliation.py

   All four run in well under a minute on a standard laptop.

3. To compile the manuscript itself (requires a LaTeX distribution with
   REVTeX 4-2, e.g. TeX Live's texlive-publishers package):
       cd paper
       pdflatex paper_2_corrected.tex
       bibtex paper_2_corrected
       pdflatex paper_2_corrected.tex
       pdflatex paper_2_corrected.tex

--------------------------------------------------------------------------------
WHAT THIS REPOSITORY DOES NOT CLAIM
--------------------------------------------------------------------------------
In the interest of transparency to referees:

  * The AIC comparison in an earlier draft could not be reconciled (script
    04) and is not used as a result anywhere in the manuscript.
  * The per-flyby scatter in lambda_1 (a factor of ~30 across the four
    anomalous flybys, or ~4.6 excluding Cassini; see Table II) is reported
    plainly in the manuscript as a real limitation, not smoothed over.
  * The Juno flyby (Sec. V.B) is reported as an open falsification of the
    model in its current form. No resolution is offered in this
    repository or the manuscript it accompanies.
  * With n=4 anomalous flybys, every fit in this repository is, in the
    manuscript's own words, "a correlation, not a confirmation."

--------------------------------------------------------------------------------
CONTACT
--------------------------------------------------------------------------------
P. A. Nesvinga <praise.nesvinga@nust.ac.zw>
Department of Applied Physics, National University of Science & Technology,
Bulawayo, Zimbabwe
