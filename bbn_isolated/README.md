# Isolated standard-BBN recovery

This tree is not an evidence line. It does not import the empirical suite.
The empirical BBN row stays `FAIL_INTEGRATOR`.

`tsb_on` is not a switch in this package. It is not added until G1–G5 are green.

## Gates

| ID | Gate | Pass |
|---|---|---|
| G1 | Conservation | absolute relative change in baryon number and charge, each below 1e-8. B is the sum of A_i Y_i. |
| G2 | Positivity | every abundance stays non-negative. No clip-and-continue. |
| G3 | Nonzero deuterium | Y_D at 0.01 MeV is positive, and D/H is between 1e-6 and 1e-3. |
| G4 | Published SBBN table | at the same eta, helium mass fraction within 0.005 and D/H within 20 percent of a named network. Not a PDG observation. |
| G5 | Convergence | G1–G4 still hold at half and double step, and at tighter tolerance. Relative change in Y_p and D/H below 0.05. |

## Status

The runner writes `gates.json` and `report.json` in this directory.
`status` is `FAIL_INTEGRATOR` until G3 is actually true.
No observational sigma is written.
The full reaction network is not finished in this pass, so deuterium is not claimed.

## Refuse

Do not import the empirical suite, neutron, glueball, sparc, desi, cms, casimir, or bounce.
Do not read the frozen mass floor into the Friedmann equation.
Do not score a zero deuterium yield against observations.
