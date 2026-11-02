"""Ensemble with calibration version 1 (noise_model_v1.npz from freeze_v1.py), out of sample.

Coarse model n_coarse^2, dt = window, SALT hook with external noise (noise_scale = 1). Start q0 = fine q_bar of the
start window, truth = psi_bar_start (from data_folder in the npz). Starts n_train, n_train + spacing, ...
Variants:
  D  no noise
  M  V1 deterministic part only: k c psi_model + mean_rest (one run)
  A  V1 full: deterministic part + amp * AR(1) fluctuations, nmembers members (seed 2026)
Printed by lead, averaged over the starts, errors relative to rms(psi_bar): error D, error M, error of the mean of
A, member error of A, spread of A, spread / error of the mean (calibrated: about sqrt(N/(N+1))), and the number of
starts where M and A are better than D.
With the defaults this gives the table in README_v1.md.

    python run_ensemble_v1.py [nstarts nmembers nsteps [amp]]
    (defaults: 10 10 100, amp = 1)

Saves ensemble_v1_<nstarts>x<nmembers>_amp<amp>.npz in ~/DA_Results/noise_calibration/v1/. One MPI rank.
"""
import sys
import time
import numpy as np
import firedrake as fd
from nudging.models.stochastic_mix_euler import Euler_mixSD

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
nstarts = 10
nmembers = 10
nsteps = 100
amp = 1.0
if len(sys.argv) > 3:
    nstarts = int(sys.argv[1])
    nmembers = int(sys.argv[2])
    nsteps = int(sys.argv[3])
if len(sys.argv) > 4:
    amp = float(sys.argv[4])

# ---- V1 ----
v1 = np.load(v1_folder + "/noise_model_v1.npz")
kc = float(v1["k_scale"])*float(v1["c"])
eofs_grid = v1["eofs_grid"]
lam = v1["lam"]
phi = v1["phi"]
K = int(v1["K"])
ix = v1["ix"]
iy = v1["iy"]
n_coarse = int(v1["n_coarse"])
window_length = float(v1["window_length"])
n_train = int(v1["n_train"])
mean_rest = v1["mean_rest_grid"][ix, iy]
data_folder = str(v1["data_folder"])

# ---- data ----
psi_bar = np.load(data_folder + "/psi_bar_start.npy")
qbar_fine = np.load(data_folder + "/qbar_fine_start.npy")
nwindows = psi_bar.shape[0]
nnodes = psi_bar.shape[1]
print("V1: k c =", kc, " K =", K, " n_train =", n_train, " amp =", amp, " starts:", nstarts, " members:", nmembers,
      " steps:", nsteps, flush=True)

# ---- model ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                    solver_parameters=newton)
model.setup()
noise_fn = fd.Function(model.Vcg)
model.noise_fields = [noise_fn]
X0 = model.allocate()
X1 = model.allocate()
q_start = fd.Function(model.Vcg)

rng = np.random.default_rng(2026)
spacing = (nwindows - 1 - nsteps - n_train)//nstarts
err_D = np.zeros((nstarts, nsteps + 1))
err_M = np.zeros((nstarts, nsteps + 1))
err_mean = np.zeros((nstarts, nsteps + 1))
err_member = np.zeros((nstarts, nsteps + 1))
spread = np.zeros((nstarts, nsteps + 1))
t_begin = time.time()

for s in range(nstarts):
    m0 = n_train + spacing*s
    truth = psi_bar[m0:m0 + nsteps + 1]
    norm_T = np.sqrt(np.mean(truth**2, axis=1))
    q_start.dat.data[:] = qbar_fine[m0]
    X0[0].subfunctions[0].interpolate(q_start)
    state0 = X0[0].copy(deepcopy=True)

    # ---- D and M (deterministic) ----
    for iv in range(2):
        X0[0].assign(state0)
        noise_fn.dat.data[:] = 0.0
        psi_run = np.zeros((nsteps + 1, nnodes))
        for j in range(nsteps):
            model.qpsi0.assign(X0[0])
            model.psi0_solver.solve()
            psi_run[j] = model.psi0_tmp.dat.data_ro
            if iv == 1:
                noise_fn.dat.data[:] = kc*psi_run[j] + mean_rest
            model.run(X0, X1)
            X0[0].assign(X1[0])
        model.qpsi0.assign(X0[0])
        model.psi0_solver.solve()
        psi_run[nsteps] = model.psi0_tmp.dat.data_ro
        if iv == 0:
            err_D[s] = np.sqrt(np.mean((psi_run - truth)**2, axis=1))/norm_T
        else:
            err_M[s] = np.sqrt(np.mean((psi_run - truth)**2, axis=1))/norm_T
    print("start", s, "window", m0, ": error at the end D", err_D[s, nsteps], " M", err_M[s, nsteps],
          " elapsed", time.time() - t_begin, flush=True)

    # ---- A: deterministic part + AR(1) fluctuations ----
    psi_ens = np.zeros((nmembers, nsteps + 1, nnodes))
    for k in range(nmembers):
        X0[0].assign(state0)
        a = np.sqrt(lam)*rng.standard_normal(K)
        for j in range(nsteps):
            model.qpsi0.assign(X0[0])
            model.psi0_solver.solve()
            psi_ens[k, j] = model.psi0_tmp.dat.data_ro
            field = np.tensordot(a, eofs_grid, axes=(0, 0))
            noise_fn.dat.data[:] = kc*psi_ens[k, j] + mean_rest + amp*field[ix, iy]
            model.run(X0, X1)
            X0[0].assign(X1[0])
            a = phi*a + np.sqrt(lam*(1.0 - phi**2))*rng.standard_normal(K)
        model.qpsi0.assign(X0[0])
        model.psi0_solver.solve()
        psi_ens[k, nsteps] = model.psi0_tmp.dat.data_ro
    ens_mean = psi_ens.mean(axis=0)
    err_mean[s] = np.sqrt(np.mean((ens_mean - truth)**2, axis=1))/norm_T
    err_member[s] = np.mean(np.sqrt(np.mean((psi_ens - truth)**2, axis=2)), axis=0)/norm_T
    spread[s] = np.sqrt(np.mean(psi_ens.var(axis=0, ddof=1), axis=1))/norm_T
    print("    A: error of the mean at the end", err_mean[s, nsteps], " spread at the end", spread[s, nsteps],
          " elapsed", time.time() - t_begin, flush=True)

np.savez(v1_folder + "/ensemble_v1_" + str(nstarts) + "x" + str(nmembers) + "_amp" + str(amp) + ".npz", err_D=err_D,
         err_M=err_M, err_mean=err_mean, err_member=err_member, spread=spread, window_length=window_length)

print()
print("averages over the", nstarts, "starts; errors relative to rms(psi_bar); N =", nmembers,
      " (calibrated: spread/error of the mean about", np.sqrt(nmembers/(nmembers + 1.0)), ")")
print("   lead | time | error D | error M | A: error of the mean | A: member error | A: spread | spread / error"
      " | starts M better than D | starts A better than D")
for lead in [0, 1, 2, 5, 10, 25, 50, 75, 100]:
    if lead <= nsteps:
        print("   ", lead, "|", lead*window_length, "|", err_D[:, lead].mean(), "|", err_M[:, lead].mean(), "|",
              err_mean[:, lead].mean(), "|", err_member[:, lead].mean(), "|", spread[:, lead].mean(), "|",
              spread[:, lead].mean()/err_mean[:, lead].mean(), "|", int((err_M[:, lead] < err_D[:, lead]).sum()),
              "|", int((err_mean[:, lead] < err_D[:, lead]).sum()))
