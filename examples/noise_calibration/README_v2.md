# Calibrated coarse model, version 2

**Model.** As V1 (`README_v1.md`): `Euler_mixSD` on 64², DQ1/CG1, implicit midpoint, r = 0.01,
forcing_amp = 0.1, model step = data window, Δt = 0.08. V2 keeps the V1 SALT field and adds a non-transport
increment of q per step (`q_source=True`):

    dU_2^n        = k c ψ^n + mean_rest + Σ_j a_j^n EOF_j        (V1, unchanged)
    q_source_fn^n = source_det + Σ_j b_j^n EOFr_j

- **source_det**: a fixed pattern, the time mean of the non-transport one-step error of V1 (plus the small mean of
  what is left after it).
- **The EOFr sum**: the rest of the non-transport error, K = 100 modes with AR(1) amplitudes,
  b_j^{n+1} = φ_j b_j^n + √(λ_j(1 − φ_j²)) N(0, 1), b_j^0 ~ N(0, λ_j). The rest is slow: τ ≈ 4–5 for the leading
  modes, median 0.5.

Without the two EOF sums it is the deterministic V2 model.

**Where it comes from.** The one-step error of the V1 deterministic model is split into its transport fit and the
rest R_nt (about 35% of |R|²). R_nt is an enstrophy sink in every window. Candidate closures were fitted to R_nt
(`nt_fit.py`) and tested in the model (`nt_onestep.py`, `run_ensemble_v2.py`). Damping or diffusion remove the
enstrophy excess but make the ψ forecast worse than no correction; the fixed pattern alone (variant C) is kept.

**Pipeline**, after the V1 pipeline (all in `~/DA_Results/noise_calibration/v1/`, then `v2/`):

| step | command | output |
|---|---|---|
| non-transport split | `python nontransport.py` | `R_v1.npy`, `R_nt_v1.npy`, `windows_v1.npy` |
| candidate fit | `python nt_fit.py` | printed only |
| one-step check | `python nt_onestep.py C` | `R_C.npy` |
| rest model | `python nt_rest.py C` | `noise_model_v2_rest_C.npz` |
| ensemble test | `python run_ensemble_v2.py 10 10 100 1 1 2027 C` | `ensemble_v2C_10x10_amp1.0_1.0_seed2027.npz` |
| freeze | `python freeze_v2.py` | `v2/noise_model_v2.npz` |

**Use.** Load `noise_model_v2.npz`, build the model with `noise_type='salt'`, `noise_source='external'`,
`noise_scale = 1`, `q_source=True`, `r` from the file, and set before each step:

    noise_fn.dat.data[:]          = kc*psi + mean_rest_grid[ix, iy] + (a @ eofs_grid)[ix, iy]
    model.q_source_fn.dat.data[:] = source_det_grid[ix, iy] + (b @ eofs_rest_grid)[ix, iy]

with ψ the model's own streamfunction at the start of the step (`model.psi0_solver`), and a, b advanced by their
AR(1) after the step.

**Performance**, out of sample: 10 starts at windows 700–871, 10 members, errors relative to rms(ψ̄).

| time | no correction | V1 det | V2 det | V1 full: spread/error | V2 full: mean | V2 full: spread/error |
|---|---|---|---|---|---|---|
| 0.08 | 0.00240 | 0.00222 | 0.00211 | 0.65 | 0.00216 | 0.94 |
| 0.8 | 0.0202 | 0.0182 | 0.0175 | 0.74 | 0.0182 | 1.02 |
| 2.0 | 0.0465 | 0.0416 | 0.0418 | 0.73 | 0.0441 | 0.95 |
| 4.0 | 0.0814 | 0.0734 | 0.0783 | 0.73 | 0.0839 | 0.91 |
| 8.0 | 0.135 | 0.124 | 0.145 | 0.73 | 0.158 | 0.86 |

Calibrated spread/error for 10 members is about 0.95. Rank histogram of V2 full at t = 0.8 is flat (end bins
0.096 / 0.091, flat 0.091). Enstrophy ratio to the truth at t = 8: 1.16 (V1: 1.56).

**Known limits.**
- **Short leads only.** Up to t ≈ 2 (25 windows) V2 is the best model and the ensemble is calibrated. From t ≈ 4
  the fixed pattern makes the forecast drift: worse than V1, and worse than no correction at t = 8. Use V1 for
  long free forecasts.
- **Ensemble mean.** The V2 full ensemble mean is no better than V1 deterministic at t = 0.8.
- **Enstrophy excess** reduced by about 2/3, not removed.
- **One run.** 10 starts, one noise seed (the seed hardly changed the results of the damping variant).
- **Not tested in data assimilation.** All forecasts start from the filtered truth.
- **Setting-specific.** Fitted for 64², dt = 0.08 (one step per window), the same mesh order as V1.
