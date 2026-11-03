"""V2e: ensemble with V1 and V2, out of sample.

Coarse model n_coarse^2, dt = window, SALT hook with external noise (noise_scale = 1). Start q0 = fine q_bar of the
start window, truth psi_bar_start and qbar_fine_start (data folder of noise_model_v1.npz). Starts n_train,
n_train + spacing, ... (as run_ensemble_v1.py). Variants:
  D   no correction (r = 0.01)
  M   V1 deterministic: dU_2 = k c psi_model + mean_rest
  V   V2 deterministic: M + r = 0.01 + gamma + q_source_fn = bP P + mean of R_rest (noise_model_v2_rest_<variant>.npz)
  A   V1 full: M + amp * AR(1) transport fluctuations (V1 EOFs), nmembers members
  F   V2 full: V + amp * the same transport fluctuations as A + amp_rest * AR(1) rest (V2 EOFs) in q_source_fn
The transport fluctuations of A are drawn with seed 2026 in the order of run_ensemble_v1.py (A reproduces it); F
uses the same transport sequences, the rest noise has its own generator (seed).
Printed by lead, averaged over the starts, errors relative to rms(psi_bar): errors of D, M, V, error of the mean,
spread and spread / error of A and F (calibrated: about sqrt(N/(N+1))), starts better than D, enstrophy ratio
Z_model / Z_truth (Z = mean of q^2 over the nodes, model q projected to CG1; ensembles: member mean), and rank
histograms of psi_bar among the members at all interior nodes and starts (leads 10, 25, 50, 100).

    python run_ensemble_v2.py [nstarts nmembers nsteps [amp amp_rest [seed [variant]]]]
    (defaults: 10 10 100, amp = amp_rest = 1, seed = 2027, variant A; variant C = pattern only, see nt_rest.py)

Reads noise_model_v1.npz, noise_model_v2_rest_<variant>.npz, R_nt_v1.npy, windows_v1.npy (~/DA_Results/noise_calibration/v1/).
Saves ensemble_v2<variant>_<nstarts>x<nmembers>_amp<amp>_<amp_rest>_seed<seed>.npz there. One MPI rank.
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
amp_rest = 1.0
seed = 2027
variant = "A"
if len(sys.argv) > 3:
    nstarts = int(sys.argv[1])
    nmembers = int(sys.argv[2])
    nsteps = int(sys.argv[3])
if len(sys.argv) > 5:
    amp = float(sys.argv[4])
    amp_rest = float(sys.argv[5])
if len(sys.argv) > 6:
    seed = int(sys.argv[6])
if len(sys.argv) > 7:
    variant = sys.argv[7]

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

# ---- V2 (non-transport part) ----
v2 = np.load(v1_folder + "/noise_model_v2_rest_" + variant + ".npz")
bP = float(v2["bP"])
r2 = float(v2["r"])
eofs2 = v2["eofs_grid"][:, ix, iy]
lam2 = v2["lam"]
phi2 = v2["phi"]
K2 = int(v2["K"])
windows_v1 = np.load(v1_folder + "/windows_v1.npy")
P = np.load(v1_folder + "/R_nt_v1.npy")[windows_v1 < n_train].mean(axis=0)
source_det = bP*P + v2["mean_grid"][ix, iy]

# ---- data ----
psi_bar = np.load(data_folder + "/psi_bar_start.npy")
qbar_fine = np.load(data_folder + "/qbar_fine_start.npy")
nwindows = psi_bar.shape[0]
nnodes = psi_bar.shape[1]
inner = (ix > 0) & (ix < n_coarse) & (iy > 0) & (iy < n_coarse)     # rank histograms: psi = 0 on the walls
print("variant", variant, " V1: k c =", kc, " K =", K, "  V2: r =", r2, " bP =", bP, " K2 =", K2, "  amp =", amp, " amp_rest =", amp_rest,
      " seed =", seed, " starts:", nstarts, " members:", nmembers, " steps:", nsteps, flush=True)

# ---- models: 0 with r = 0.01 (D, M, A), 1 with r = 0.01 + gamma and q_source (V, F) ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model0 = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                     solver_parameters=newton)
model0.setup()
model1 = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                     solver_parameters=newton, r=r2, q_source=True, mesh=model0.mesh)
model1.setup()
models = [model0, model1]
noise_fns = [fd.Function(model0.Vcg), fd.Function(model1.Vcg)]
model0.noise_fields = [noise_fns[0]]
model1.noise_fields = [noise_fns[1]]
X0s = [model0.allocate(), model1.allocate()]
X1s = [model0.allocate(), model1.allocate()]
qprojs = [fd.Function(model0.Vcg), fd.Function(model1.Vcg)]
p_solvers = []
for im in range(2):
    trial = fd.TrialFunction(models[im].Vcg)
    test = fd.TestFunction(models[im].Vcg)
    p_problem = fd.LinearVariationalProblem(trial*test*fd.dx, X0s[im][0].subfunctions[0]*test*fd.dx, qprojs[im])
    p_solvers.append(fd.LinearVariationalSolver(p_problem,
                                                solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}))
q_start = fd.Function(model0.Vcg)

rng = np.random.default_rng(2026)
rng_rest = np.random.default_rng(seed)
spacing = (nwindows - 1 - nsteps - n_train)//nstarts
leads_rank = [10, 25, 50, 100]
names = ["D", "M", "V", "A", "F"]
err_det = np.zeros((3, nstarts, nsteps + 1))         # D, M, V
Zr_det = np.zeros((3, nstarts, nsteps + 1))
err_mean = np.zeros((2, nstarts, nsteps + 1))        # A, F
err_member = np.zeros((2, nstarts, nsteps + 1))
spread = np.zeros((2, nstarts, nsteps + 1))
Zr_ens = np.zeros((2, nstarts, nsteps + 1))
ranks = np.zeros((2, len(leads_rank), nmembers + 1))
t_begin = time.time()

for s in range(nstarts):
    m0 = n_train + spacing*s
    truth = psi_bar[m0:m0 + nsteps + 1]
    norm_T = np.sqrt(np.mean(truth**2, axis=1))
    Z_T = np.mean(qbar_fine[m0:m0 + nsteps + 1]**2, axis=1)
    q_start.dat.data[:] = qbar_fine[m0]

    # transport sequences of A (and F), drawn in the order of run_ensemble_v1.py
    a_seq = np.zeros((nmembers, nsteps, K))
    for k in range(nmembers):
        a = np.sqrt(lam)*rng.standard_normal(K)
        for j in range(nsteps):
            a_seq[k, j] = a
            a = phi*a + np.sqrt(lam*(1.0 - phi**2))*rng.standard_normal(K)

    psi_ens = np.zeros((2, nmembers, nsteps + 1, nnodes))
    Z_ens = np.zeros((2, nmembers, nsteps + 1))
    for iv in range(5):
        im = 0
        if iv == 2 or iv == 4:
            im = 1
        model = models[im]
        X0 = X0s[im]
        X1 = X1s[im]
        nrun = 1
        if iv >= 3:
            nrun = nmembers
        for k in range(nrun):
            X0[0].subfunctions[0].interpolate(q_start)
            b = np.sqrt(lam2)*rng_rest.standard_normal(K2)
            psi_run = np.zeros((nsteps + 1, nnodes))
            Z_run = np.zeros(nsteps + 1)
            for j in range(nsteps + 1):
                model.qpsi0.assign(X0[0])
                model.psi0_solver.solve()
                psi_run[j] = model.psi0_tmp.dat.data_ro
                p_solvers[im].solve()
                Z_run[j] = np.mean(qprojs[im].dat.data_ro**2)
                if j == nsteps:
                    break
                noise = np.zeros(nnodes)
                if iv >= 1:
                    noise = kc*psi_run[j] + mean_rest
                if iv >= 3:
                    field = np.tensordot(a_seq[k, j], eofs_grid, axes=(0, 0))
                    noise = noise + amp*field[ix, iy]
                noise_fns[im].dat.data[:] = noise
                if iv == 2:
                    model.q_source_fn.dat.data[:] = source_det
                if iv == 4:
                    model.q_source_fn.dat.data[:] = source_det + amp_rest*(b @ eofs2)
                    b = phi2*b + np.sqrt(lam2*(1.0 - phi2**2))*rng_rest.standard_normal(K2)
                model.run(X0, X1)
                X0[0].assign(X1[0])
            if iv < 3:
                err_det[iv, s] = np.sqrt(np.mean((psi_run - truth)**2, axis=1))/norm_T
                Zr_det[iv, s] = Z_run/Z_T
            else:
                psi_ens[iv - 3, k] = psi_run
                Z_ens[iv - 3, k] = Z_run
    for ie in range(2):
        ens_mean = psi_ens[ie].mean(axis=0)
        err_mean[ie, s] = np.sqrt(np.mean((ens_mean - truth)**2, axis=1))/norm_T
        err_member[ie, s] = np.mean(np.sqrt(np.mean((psi_ens[ie] - truth)**2, axis=2)), axis=0)/norm_T
        spread[ie, s] = np.sqrt(np.mean(psi_ens[ie].var(axis=0, ddof=1), axis=1))/norm_T
        Zr_ens[ie, s] = Z_ens[ie].mean(axis=0)/Z_T
        for il in range(len(leads_rank)):
            lead = leads_rank[il]
            if lead <= nsteps:
                rk = np.sum(psi_ens[ie, :, lead][:, inner] < truth[lead][inner], axis=0)
                ranks[ie, il] += np.bincount(rk, minlength=nmembers + 1)
    print("start", s, "window", m0, ": error at the end D", err_det[0, s, nsteps], " M", err_det[1, s, nsteps],
          " V", err_det[2, s, nsteps], " A mean", err_mean[0, s, nsteps], " F mean", err_mean[1, s, nsteps],
          " spread A", spread[0, s, nsteps], " F", spread[1, s, nsteps], " elapsed", time.time() - t_begin, flush=True)

np.savez(v1_folder + "/ensemble_v2" + variant + "_" + str(nstarts) + "x" + str(nmembers) + "_amp" + str(amp) + "_" + str(amp_rest)
         + "_seed" + str(seed) + ".npz", err_det=err_det, Zr_det=Zr_det, err_mean=err_mean, err_member=err_member,
         spread=spread, Zr_ens=Zr_ens, ranks=ranks, leads_rank=leads_rank, window_length=window_length)

print()
print("averages over the", nstarts, "starts; errors relative to rms(psi_bar); N =", nmembers,
      " (calibrated: spread/error of the mean about", np.sqrt(nmembers/(nmembers + 1.0)), ")")
print("   lead | time | error D | M | V | A: error of the mean | A: spread/error | F: error of the mean"
      " | F: spread/error | starts better than D: M V A F")
for lead in [0, 1, 2, 5, 10, 25, 50, 75, 100]:
    if lead <= nsteps:
        print("   ", lead, "|", lead*window_length, "|", err_det[0, :, lead].mean(), "|", err_det[1, :, lead].mean(),
              "|", err_det[2, :, lead].mean(), "|", err_mean[0, :, lead].mean(), "|",
              spread[0, :, lead].mean()/err_mean[0, :, lead].mean(), "|", err_mean[1, :, lead].mean(), "|",
              spread[1, :, lead].mean()/err_mean[1, :, lead].mean(), "|",
              int((err_det[1, :, lead] < err_det[0, :, lead]).sum()),
              int((err_det[2, :, lead] < err_det[0, :, lead]).sum()),
              int((err_mean[0, :, lead] < err_det[0, :, lead]).sum()),
              int((err_mean[1, :, lead] < err_det[0, :, lead]).sum()))
print()
print("enstrophy ratio Z_model / Z_truth (mean over the starts): D | M | V | A | F")
for lead in [0, 10, 25, 50, 75, 100]:
    if lead <= nsteps:
        print("   lead", lead, " time", lead*window_length, ":", Zr_det[0, :, lead].mean(), Zr_det[1, :, lead].mean(),
              Zr_det[2, :, lead].mean(), Zr_ens[0, :, lead].mean(), Zr_ens[1, :, lead].mean())
print()
print("rank histograms of psi_bar (share per rank 0 ..", nmembers, "; flat:", 1.0/(nmembers + 1), ")")
for ie in range(2):
    for il in range(len(leads_rank)):
        if leads_rank[il] <= nsteps:
            print("   ", names[3 + ie], "lead", leads_rank[il], ":", np.round(ranks[ie, il]/ranks[ie, il].sum(), 3))
