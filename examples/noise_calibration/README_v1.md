# Calibrated coarse model, version 1

**Model.** `Euler_mixSD` on 64², DQ1/CG1, implicit midpoint, r = 0.01, forcing_amp = 0.1. The model step equals the
data window, Δt = 0.08.

**Correction.** The model is run with the SALT hook: `noise_type='salt'`, `noise_source='external'`,
`noise_scale = 1`. The field passed as `dU_2` at step n is

    dU_2^n = k c ψ^n + mean_rest + Σ_j a_j^n EOF_j

The terms are:
- **k c ψ^n**: k c = 2 × 0.0084. ψ^n is the model's own streamfunction at the start of the step. This term amounts
  to about 5% extra transport velocity.
- **mean_rest**: a fixed pattern.
- **The EOF sum**: AR(1) amplitudes, a_j^{n+1} = φ_j a_j^n + √(λ_j(1 − φ_j²)) N(0, 1) and a_j^0 ~ N(0, λ_j),
  with K = 100 modes and per-mode τ_j (φ_j = exp(−Δt/τ_j)).

Without the EOF sum it is the deterministic V1 model.

**Where it comes from.** It is the one-step closure error of the coarse model, R = q̄^{m+1} − Φ(q̄^m), fitted as
transport, R ≈ Δt J(z, q̄). The data is the 128² fine run, Helmholtz-filtered with ℓ = 2h_c = 1/32 and sampled at the
64² nodes. Fitted on windows 0–699; windows 700–998 are kept for testing.

**Pipeline**, from scratch. Output goes to `~/DA_Results/noise_calibration/`, and the data folder is
`data_run128_long/displacements_ell_2hc64_w4/`.

| step | command | output |
|---|---|---|
| spin-up | `mpiexec -n 4 python spinup.py 128 ~/DA_Results/noise_calibration/spinup128_final 200 0.02` | `state_final.h5`, t = 200 |
| fine data run | `mpiexec -n 4 python data_run.py 1000 ~/DA_Results/noise_calibration/data_run128_long 128` | ψ every fine step, q every window |
| displacements | `python displacements.py 0.03125 ell_2hc64 4 data_run128_long 4000 64 0.02` | ψ̄ at window starts, nodes, ΔX |
| Lagrangian ζ | `python zeta.py ell_2hc64 4 data_run128_long` | `zeta.npy`, only for comparison |
| fine q̄ | `python qbar_fine.py 0.03125 ell_2hc64 4 data_run128_long 4000 64` | `qbar_fine_start.npy` |
| closure error | `python one_step_error.py ell_2hc64 data_run128_long 64 0.08 1` | `zeta_salt_fit.npy` |
| calibration | `python salt_generator_psi.py` | `noise_model_salt_psi.npz` |
| freeze | `python freeze_v1.py` | `v1/noise_model_v1.npz` |

**Use.**

    python freeze_v1.py              # writes ~/DA_Results/noise_calibration/v1/noise_model_v1.npz
    python run_ensemble_v1.py        # D, V1 deterministic, V1 full; 10 starts × 10 members × 100 steps (~55 min)

`noise_model_v1.npz` holds the following, with grids of size 65 × 65 and zero on the walls:
- `c`, `k_scale`
- `mean_rest_grid`, `eofs_grid`
- `lam`, `tau`, `phi`, `K`
- `ix`, `iy` (grid index of each CG1 node)
- metadata: resolution pair, ℓ, Δt, `n_train`, data folder

**Performance**, out of sample: starts at windows 700–871, errors relative to rms(ψ̄).

| time | no correction | V1 deterministic | V1 full, ensemble mean | spread/error |
|---|---|---|---|---|
| 0.8 | 0.0202 | 0.0182 | 0.0185 | 0.74 |
| 4.0 | 0.0814 | 0.0734 | 0.0771 | 0.73 |
| 8.0 | 0.135 | 0.124 | 0.132 | 0.73 |

V1 deterministic is better than no correction in 10 of 10 starts at every lead.

**Known limits.**
- **Under-dispersed.** Spread/error is about 0.73, against 0.95 for 10 members. Inflating the amplitude to 1.5
  calibrates the spread but makes the ensemble mean worse than no correction after t ≈ 2.
- **Enstrophy excess** of about 1.5× the truth at t = 8; neither the correction nor the noise fixes it.
- **Transport only.** About 1/3 of |R|² is not transport and is not represented (V2).
- **Narrow validation.** One resolution pair, one data record of 80 time units, SALT only.
- **Mesh-specific.** `ix` and `iy` assume the node order of the serial 64² `UnitSquareMesh`.
