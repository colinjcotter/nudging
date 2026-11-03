"""V2d: stochastic model of the non-transport rest of the one-step error after variant A or C (nt_onestep.py).

For every window m: R_A^m (R_A.npy) is split as in nontransport.py,
    R_rest^m = R_A^m - dt J(z^m, q_bar^m)    (CG1 projection; z = transport fit, eps_rel 1e-2)
(the transport part is what the V1 SALT noise represents). On the windows 0 to n_train - 1: time mean of R_rest,
EOFs of the fluctuation, K modes with AR(1) amplitudes, per-mode tau from a fit of log acf over the lags with
acf > acf_min (as salt_generator_psi.py). In the model the rest is an additive increment of q per window:
q_source_fn = bP P + mean_rest_nt + sum_j b_j EOF_j.
Printed: share of |R_A|^2 not transport, enstrophy effect of R_rest, variance shares, tau per mode, rho(lag).
Units check: on 5 test windows, the variant run with and without one sample of the rest added to q_source_fn;
rms(difference of the model q, CG1) / rms(sample) should be about 1.

    python nt_rest.py [variant [K]]
    (defaults: A, 100; variant A = pattern + damping, C = pattern only, as nt_onestep.py)

Reads noise_model_v1.npz, R_<variant>.npy, windows_v1.npy (~/DA_Results/noise_calibration/v1/) and qbar_fine_start.npy of
its data folder. Saves R_rest_<variant>.npy and noise_model_v2_rest_<variant>.npz (mean_grid, eofs_grid on the (n+1) x (n+1) grid,
lam, tau, phi, K, bP, gamma, r, n_train) in the v1 folder. One MPI rank.
"""
import sys
import numpy as np
import firedrake as fd
from nudging.models.stochastic_mix_euler import Euler_mixSD

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
variant = "A"
K = 100
if len(sys.argv) > 1:
    variant = sys.argv[1]
if len(sys.argv) > 2:
    K = int(sys.argv[2])
eps_rel = 1e-2
maxlag = 300
nsample = 20000
acf_min = 0.2
if variant == "A":
    bP = 0.89279988
    gamma = 0.013483674564380568
if variant == "C":
    bP = 1.0
    gamma = 0.0
r = 0.01 + gamma

# ---- V1 ----
v1 = np.load(v1_folder + "/noise_model_v1.npz")
kc = float(v1["k_scale"])*float(v1["c"])
ix = v1["ix"]
iy = v1["iy"]
n_coarse = int(v1["n_coarse"])
window_length = float(v1["window_length"])
n_train = int(v1["n_train"])
mean_rest = v1["mean_rest_grid"][ix, iy]
data_folder = str(v1["data_folder"])

qbar_fine = np.load(data_folder + "/qbar_fine_start.npy")
R_A = np.load(v1_folder + "/R_" + variant + ".npy")
windows = np.load(v1_folder + "/windows_v1.npy")
Rnt_v1 = np.load(v1_folder + "/R_nt_v1.npy")
P = Rnt_v1[windows < n_train].mean(axis=0)
nw = R_A.shape[0]
print("variant", variant, ": r =", r, " bP =", bP, " windows:", nw, " consecutive:", bool(np.all(np.diff(windows) == 1)), " K:", K, flush=True)

# ---- split R_A: transport fit and rest (CG1) ----
mesh = fd.UnitSquareMesh(n_coarse, n_coarse, quadrilateral=True, name="mesh2d_per")
Vcg = fd.FunctionSpace(mesh, "CG", 1)
q0c = fd.Function(Vcg)
R = fd.Function(Vcg)
z = fd.Function(Vcg)
Rrest = fd.Function(Vcg)
trial = fd.TrialFunction(Vcg)
test = fd.TestFunction(Vcg)
eps = fd.Constant(1.0)
Ju = -trial.dx(1)*q0c.dx(0) + trial.dx(0)*q0c.dx(1)
Jv = -test.dx(1)*q0c.dx(0) + test.dx(0)*q0c.dx(1)
a_fit = window_length**2*Ju*Jv*fd.dx + eps*fd.inner(fd.grad(trial), fd.grad(test))*fd.dx
L_fit = window_length*R*Jv*fd.dx
f_problem = fd.LinearVariationalProblem(a_fit, L_fit, z, bcs=fd.DirichletBC(Vcg, 0, "on_boundary"))
f_solver = fd.LinearVariationalSolver(f_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type": "mumps"})
Jz = -z.dx(1)*q0c.dx(0) + z.dx(0)*q0c.dx(1)
n_problem = fd.LinearVariationalProblem(trial*test*fd.dx, (R - window_length*Jz)*test*fd.dx, Rrest)
n_solver = fd.LinearVariationalSolver(n_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

Rrest_all = np.zeros_like(R_A)
nt_share = np.zeros(nw)
dZ = np.zeros(nw)
for iw in range(nw):
    m = windows[iw]
    q0c.dat.data[:] = qbar_fine[m]
    R.dat.data[:] = R_A[iw]
    grad_q2 = fd.assemble(fd.inner(fd.grad(q0c), fd.grad(q0c))*fd.dx)
    eps.assign(eps_rel*window_length**2*grad_q2)
    f_solver.solve()
    n_solver.solve()
    Rrest_all[iw] = Rrest.dat.data_ro
    nt_share[iw] = fd.assemble((R - window_length*Jz)**2*fd.dx)/fd.assemble(R*R*fd.dx)
    dZ[iw] = fd.assemble(q0c*(R - window_length*Jz)*fd.dx)/(0.5*fd.assemble(q0c*q0c*fd.dx))
np.save(v1_folder + "/R_rest_" + variant + ".npy", Rrest_all)
train = windows < n_train
test_w = windows >= n_train
print("share of |R_A|^2 not transport: fit", nt_share[train].mean(), " test", nt_share[test_w].mean(),
      " (V1: 0.39 / 0.36)")
print("dZ/Z of R_rest per window: fit", dZ[train].mean(), " test", dZ[test_w].mean(), " (R_nt of V1: -0.0035)")

# ---- mean pattern, EOFs, amplitudes (fit windows) ----
flat = Rrest_all[train]
nwindows = flat.shape[0]
mean_pattern = flat.mean(axis=0)
f = flat - mean_pattern
u, s, vt = np.linalg.svd(f, full_matrices=False)
amp = u[:, :K]*s[:K]
lam = np.sum(amp**2, axis=0)/nwindows
share = s**2/np.sum(s**2)
cumulative = np.cumsum(share)
print("rms of R_rest:", np.sqrt(np.mean(flat**2)), " of its mean pattern:", np.sqrt(np.mean(mean_pattern**2)),
      " of the fluctuation:", np.sqrt(np.mean(f**2)))
print("share of the energy in the mean pattern:", np.sum(mean_pattern**2)*nwindows/np.sum(flat**2))
print("modes 1 to 3:", share[0:3], " modes for 50%, 70%, 90%, 99%:", np.searchsorted(cumulative, 0.5) + 1,
      np.searchsorted(cumulative, 0.7) + 1, np.searchsorted(cumulative, 0.9) + 1, np.searchsorted(cumulative, 0.99) + 1)
print("variance share of the first K modes:", cumulative[K - 1])

# ---- tau overall and per mode ----
rho_all = np.zeros(maxlag + 1)
for lag in range(maxlag + 1):
    rho_all[lag] = np.sum(f[:nwindows - lag]*f[lag:])/np.sum(f[:nwindows - lag]**2)
nfit = np.argmax(rho_all[1:] < acf_min)
if nfit >= 1:
    lk = np.arange(1, nfit + 1)
    tau_all = -window_length/(np.sum(lk*np.log(rho_all[lk]))/np.sum(lk**2))
else:
    tau_all = -window_length/np.log(max(rho_all[1], 1e-3))
print("overall tau:", tau_all, " rho at lag 1:", rho_all[1])

tau = np.zeros(K)
n_fallback = 0
for k in range(K):
    a = amp[:, k]
    acf = np.zeros(maxlag + 1)
    for lag in range(maxlag + 1):
        acf[lag] = np.sum(a[:nwindows - lag]*a[lag:])/np.sum(a[:nwindows - lag]**2)
    if np.any(acf[1:] < acf_min):
        nfit_k = np.argmax(acf[1:] < acf_min)
    else:
        nfit_k = maxlag
    if nfit_k >= 2:
        lk = np.arange(1, nfit_k + 1)
        tau[k] = -window_length/(np.sum(lk*np.log(acf[lk]))/np.sum(lk**2))
    else:
        tau[k] = tau_all
        n_fallback += 1
phi = np.exp(-window_length/tau)
print("tau per mode: modes 1 to 6:", tau[0:6], " median:", np.median(tau), " min:", tau.min(), " max:", tau.max())
print("modes that used the overall tau (fit failed):", n_fallback, "of", K)

# ---- sample ----
rng = np.random.default_rng(123)
a_s = np.zeros((nsample, K))
a_s[0] = np.sqrt(lam)*rng.standard_normal(K)
for n in range(nsample - 1):
    a_s[n + 1] = phi*a_s[n] + np.sqrt(lam*(1.0 - phi**2))*rng.standard_normal(K)
print()
print("rho(lag) of the fluctuation: data all modes | data K modes | sample")
for lag in [1, 2, 5, 10, 20, 50]:
    rho_K = np.sum(amp[:nwindows - lag]*amp[lag:])/np.sum(amp[:nwindows - lag]**2)
    rho_s = np.sum(a_s[:nsample - lag]*a_s[lag:])/np.sum(a_s[:nsample - lag]**2)
    print("   lag", lag, " time", lag*window_length, ":", rho_all[lag], rho_K, rho_s)

# ---- save on the grid ----
mean_grid = np.zeros((n_coarse + 1, n_coarse + 1))
mean_grid[ix, iy] = mean_pattern
eofs_grid = np.zeros((K, n_coarse + 1, n_coarse + 1))
eofs_grid[:, ix, iy] = vt[:K]
np.savez(v1_folder + "/noise_model_v2_rest_" + variant + ".npz", mean_grid=mean_grid, eofs_grid=eofs_grid, lam=lam, tau=tau,
         phi=phi, K=K, bP=bP, gamma=gamma, r=r, n_train=n_train, window_length=window_length)
print()
print("saved", v1_folder + "/noise_model_v2_rest_" + variant + ".npz")

# ---- units check: one sample of the rest through q_source_fn ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                    solver_parameters=newton, r=r, q_source=True)
model.setup()
noise_fn = fd.Function(model.Vcg)
model.noise_fields = [noise_fn]
X0 = model.allocate()
X1 = model.allocate()
qmod = fd.Function(model.Vcg)
trial_m = fd.TrialFunction(model.Vcg)
test_m = fd.TestFunction(model.Vcg)
p_problem = fd.LinearVariationalProblem(trial_m*test_m*fd.dx, X1[0].subfunctions[0]*test_m*fd.dx, qmod)
p_solver = fd.LinearVariationalSolver(p_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
q_start = fd.Function(model.Vcg)
print()
print("units check: rms(q with sample - q without) / rms(sample)")
for m in [700, 760, 820, 880, 940]:
    sample = np.tensordot(np.sqrt(lam)*rng.standard_normal(K), vt[:K], axes=(0, 0))
    q_out = np.zeros((2, qbar_fine.shape[1]))
    for iv in range(2):
        q_start.dat.data[:] = qbar_fine[m]
        X0[0].subfunctions[0].interpolate(q_start)
        model.qpsi0.assign(X0[0])
        model.psi0_solver.solve()
        noise_fn.dat.data[:] = kc*model.psi0_tmp.dat.data_ro + mean_rest
        model.q_source_fn.dat.data[:] = bP*P + iv*sample
        model.run(X0, X1)
        p_solver.solve()
        q_out[iv] = qmod.dat.data_ro
    print("   window", m, ":", np.sqrt(np.mean((q_out[1] - q_out[0])**2))/np.sqrt(np.mean(sample**2)))
