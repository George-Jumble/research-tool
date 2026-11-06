# Residual analysis (live question)

## Live question

After a standard 3+1D ΛCDM (or Newtonian SPARC) reduction, do the residuals match Superblock / Mirron predictions, and not a compact extra dimension?

## Rejected questions

These are not scored on the default path:

- Does the data show a 7-fold / KK / 5D signature?
- Is there a compact S¹ of radius 7 h⁻¹ Mpc?
- Is 142857 a search target?
- Is Resonance Graph Core a cosmological backend?
- Is the bounce a 5D roll, Coleman instanton, or Φ₇ pin-drop?

The bounce stays in original TSB/CCC form: conformal aeon crossover plus Mirron / Planck-Kerr dynamic refresh. The word for a Superblock object is **Mirron**, not soliton.

## Symbol key

| Symbol | Meaning |
|---|---|
| D | Observed data vector |
| B | Conventional 3+1D baseline (ΛCDM, SM, GR, or Newtonian SPARC) |
| r | Residual, r = D − B |
| S_k | Superblock / Mirron template, k ∈ {A, B, C, D, E} |
| σ | Reported uncertainty on D |
| χ²_SB | Fit of r to S_k in 3+1D only. Numeric kernel is TODO where not yet geometry-coded |
| χ²_Λ | Fit of D to B |
| N | Number of residual bins |
| Δ | Dipole residual after CMB-frame subtraction |
| V_orb | Domain orbital-velocity eigenvalue from an E₈ projection |
| G_avg | Time-averaged gravitational coupling seen in our domain |
| M_mir | Mirron / Planck-Kerr geometric mass-gap scale (not a KK mass) |
| a | FLRW scale factor (baseline only) |
| z | Redshift (observable; not a 5D coordinate) |

## Templates

- **A** Multi-domain 3+1D kinematics. Six co-located domains. Staggered V_orb from E₈ projections. Causal isolation; only time-averaged gravity is visible here.
- **B** Conformal / twistor coupling. Apparent extra mass is geometric (G_avg), not a 5D force and not a WIMP.
- **C** Mirron / Planck-Kerr. Geometric mass gap M_mir. 4D resonance, not a KK tower.
- **D** Emergent QCD / Yang-Mills gap as a mirror result, not a 5D Wilson line.
- **E** Observational residuals: dipole Δ, SPARC outer curve, DESI BAO vs ΛCDM, CMS/LHCb 4D Mirron structure, LIGO/LISA stochastic imprint, geometric sterile state.

## Code path

`superblock_residual.run_superblock_residual` is the only default scoring path:

1. Load with the existing loader when one is available (DESI Cobaya table, SPARC CSV).
2. Subtract the conventional baseline when the table already carries it (SPARC: quadrature sum of V_gas, V_disk, V_bul).
3. Score r only against S_k. If the numeric kernel is not implemented, the verdict is **inconclusive** and the kernel field is `TODO`. No extra-dimension stand-in is called.
4. The report states residual amplitude and scale, the template letter, the verdict, and the verdict.

Boot config: `config/superblock_only.json`.
