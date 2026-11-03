"""V2a: the one-step error of the V1 deterministic model and its non-transport part.

For every window m: start the coarse model (dt = window) from the fine-mesh q_bar^m with the V1 deterministic
correction on (SALT hook, dU_2 = k c psi_model + mean_rest, no noise; noise_model_v1.npz), run one window, project
the model q to CG1 and compare with the truth q_bar^{m+1}:
    R^m    = q_bar^{m+1} - Phi_V1(q_bar^m)
    R_nt^m = R^m - dt J(z^m, q_bar^m)    (CG1 projection; z = transport fit as in one_step_error.py, eps_rel 1e-2)
Per window: enstrophy effect int q_bar R dx and energy effect -int psi_bar R dx (for R, the transport part and R_nt).
Printed for the windows 0 to n_train - 1 and n_train to the end:
  |R| / |change of q_bar| (V1; the uncorrected model had 0.34), share of |R|^2 that is not transport,
  share of the energy of R_nt in its time mean,
  enstrophy and energy effect per window relative to Z = 1/2 int q_bar^2 and E = 1/2 int |grad psi_bar|^2
  (negative: a sink), and the share of the windows with a negative effect,
  spectrum of R, the transport part and R_nt by wavenumber bin (sine transform at the interior nodes,
  k = sqrt(p^2 + q^2)); the excess enstrophy of the uncorrected model sits in the bins 4.5 to 16.5.

    python nontransport.py [stride]
    (default 1: all windows)

Reads noise_model_v1.npz (~/DA_Results/noise_calibration/v1/) and qbar_fine_start.npy, psi_bar_start.npy of its
data folder. Saves R_v1.npy, R_nt_v1.npy, z_v1.npy (windows x nodes) and windows_v1.npy in the v1 folder.
One MPI rank.
"""
import sys
import time
import numpy as np
import firedrake as fd
from nudging.models.stochastic_mix_euler import Euler_mixSD

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
stride = 1
if len(sys.argv) > 1:
    stride = int(sys.argv[1])
eps_rel = 1e-2

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
h = 1.0/n_coarse

qbar_fine = np.load(data_folder + "/qbar_fine_start.npy")
psi_bar = np.load(data_folder + "/psi_bar_start.npy")
nwindows = qbar_fine.shape[0]
nnodes = qbar_fine.shape[1]
windows_list = list(range(0, nwindows - 1, stride))
nw = len(windows_list)
print("V1: k c =", kc, " n_train =", n_train, " windows used:", nw, " eps_rel:", eps_rel, flush=True)

# ---- model with the V1 deterministic correction ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                    solver_parameters=newton)
model.setup()
noise_fn = fd.Function(model.Vcg)
model.noise_fields = [noise_fn]
X0 = model.allocate()
X1 = model.allocate()
Vcg = model.Vcg

q0c = fd.Function(Vcg)          # q_bar^m
q1c = fd.Function(Vcg)          # q_bar^{m+1}
psic = fd.Function(Vcg)         # psi_bar^m
qmod = fd.Function(Vcg)         # model q after one window, projected to CG1
R = fd.Function(Vcg)
z = fd.Function(Vcg)
Rnt = fd.Function(Vcg)

trial = fd.TrialFunction(Vcg)
test = fd.TestFunction(Vcg)

# projection of the model q (DQ1) to CG1
p_problem = fd.LinearVariationalProblem(trial*test*fd.dx, X1[0].subfunctions[0]*test*fd.dx, qmod)
p_solver = fd.LinearVariationalSolver(p_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

# transport fit: (dt J(u,q), dt J(v,q)) + eps (grad u, grad v) = (R, dt J(v,q))
eps = fd.Constant(1.0)
Ju = -trial.dx(1)*q0c.dx(0) + trial.dx(0)*q0c.dx(1)
Jv = -test.dx(1)*q0c.dx(0) + test.dx(0)*q0c.dx(1)
a_fit = window_length**2*Ju*Jv*fd.dx + eps*fd.inner(fd.grad(trial), fd.grad(test))*fd.dx
L_fit = window_length*R*Jv*fd.dx
f_problem = fd.LinearVariationalProblem(a_fit, L_fit, z, bcs=fd.DirichletBC(Vcg, 0, "on_boundary"))
f_solver = fd.LinearVariationalSolver(f_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type": "mumps"})
Jz = -z.dx(1)*q0c.dx(0) + z.dx(0)*q0c.dx(1)

# non-transport part, projected to CG1
n_problem = fd.LinearVariationalProblem(trial*test*fd.dx, (R - window_length*Jz)*test*fd.dx, Rnt)
n_solver = fd.LinearVariationalSolver(n_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

R_all = np.zeros((nw, nnodes))
Rnt_all = np.zeros((nw, nnodes))
z_all = np.zeros((nw, nnodes))
rel_err = np.zeros(nw)
nt_share = np.zeros(nw)
dZ = np.zeros((3, nw))          # enstrophy effect / Z of R, transport part, R_nt
dE = np.zeros((3, nw))          # energy effect / E of R, transport part, R_nt
t_begin = time.time()

for iw in range(nw):
    m = windows_list[iw]
    q0c.dat.data[:] = qbar_fine[m]
    q1c.dat.data[:] = qbar_fine[m + 1]
    psic.dat.data[:] = psi_bar[m]
    X0[0].subfunctions[0].interpolate(q0c)
    model.qpsi0.assign(X0[0])
    model.psi0_solver.solve()
    noise_fn.dat.data[:] = kc*model.psi0_tmp.dat.data_ro + mean_rest
    model.run(X0, X1)
    p_solver.solve()
    R.assign(q1c - qmod)
    norm_R2 = fd.assemble(R*R*fd.dx)
    rel_err[iw] = np.sqrt(norm_R2/fd.assemble((q1c - q0c)**2*fd.dx))
    grad_q2 = fd.assemble(fd.inner(fd.grad(q0c), fd.grad(q0c))*fd.dx)
    eps.assign(eps_rel*window_length**2*grad_q2)
    f_solver.solve()
    n_solver.solve()
    nt_share[iw] = fd.assemble((R - window_length*Jz)**2*fd.dx)/norm_R2
    Z = 0.5*fd.assemble(q0c*q0c*fd.dx)
    E = 0.5*fd.assemble(fd.inner(fd.grad(psic), fd.grad(psic))*fd.dx)
    dZ[0, iw] = fd.assemble(q0c*R*fd.dx)/Z
    dZ[1, iw] = fd.assemble(q0c*window_length*Jz*fd.dx)/Z
    dZ[2, iw] = fd.assemble(q0c*(R - window_length*Jz)*fd.dx)/Z
    dE[0, iw] = -fd.assemble(psic*R*fd.dx)/E
    dE[1, iw] = -fd.assemble(psic*window_length*Jz*fd.dx)/E
    dE[2, iw] = -fd.assemble(psic*(R - window_length*Jz)*fd.dx)/E
    R_all[iw] = R.dat.data_ro
    Rnt_all[iw] = Rnt.dat.data_ro
    z_all[iw] = z.dat.data_ro
    if iw % 50 == 0:
        print("window", m, ": |R|/change =", rel_err[iw], " non-transport share =", nt_share[iw],
              " dZ/Z (R, transport, R_nt) =", dZ[:, iw], " elapsed", time.time() - t_begin, flush=True)

np.save(v1_folder + "/R_v1.npy", R_all)
np.save(v1_folder + "/R_nt_v1.npy", Rnt_all)
np.save(v1_folder + "/z_v1.npy", z_all)
np.save(v1_folder + "/windows_v1.npy", np.array(windows_list))

# ---- spectra: sine transform at the interior nodes, radial bins ----
x = np.arange(1, n_coarse)*h
S = np.sin(np.pi*np.outer(np.arange(1, n_coarse), x))*np.sqrt(2.0*h)
pp, qq = np.meshgrid(np.arange(1, n_coarse), np.arange(1, n_coarse), indexing='ij')
kr = np.sqrt(pp**2 + qq**2)
edges = [0.5, 4.5, 8.5, 16.5, 32.5, 100.0]
spec = np.zeros((3, nw, len(edges) - 1))
fields = [R_all, R_all - Rnt_all, Rnt_all]
for i in range(3):
    grid = np.zeros((nw, n_coarse + 1, n_coarse + 1))
    grid[:, ix, iy] = fields[i]
    coef2 = np.einsum('pi,mij,qj->mpq', S, grid[:, 1:-1, 1:-1], S)**2
    for b in range(len(edges) - 1):
        mask = (kr >= edges[b]) & (kr < edges[b + 1])
        spec[i, :, b] = np.sum(coef2[:, mask], axis=1)

# ---- summary ----
windows_arr = np.array(windows_list)
names = ["R", "transport", "R_nt"]
for block in range(2):
    if block == 0:
        sel = windows_arr < n_train
        print()
        print("windows 0 to", n_train - 1, "(", int(sel.sum()), "windows )")
    else:
        sel = windows_arr >= n_train
        print()
        print("windows", n_train, "to", nwindows - 2, "(", int(sel.sum()), "windows )")
    print("   |R| / |change of q_bar| (V1): mean", rel_err[sel].mean(), " min", rel_err[sel].min(), " max",
          rel_err[sel].max())
    print("   share of |R|^2 not transport: mean", nt_share[sel].mean(), " min", nt_share[sel].min(), " max",
          nt_share[sel].max())
    mean_nt = Rnt_all[sel].mean(axis=0)
    print("   share of the energy of R_nt in its time mean:", np.mean(mean_nt**2)/np.mean(Rnt_all[sel]**2))
    for i in range(3):
        print("   ", names[i], ": dZ/Z per window mean", dZ[i, sel].mean(), " (negative in",
              int((dZ[i, sel] < 0).sum()), "of", int(sel.sum()), ")   dE/E per window mean", dE[i, sel].mean(),
              " (negative in", int((dE[i, sel] < 0).sum()), "of", int(sel.sum()), ")")
    print("   spectrum: share of |.|^2 by bin", edges)
    for i in range(3):
        total = spec[i, sel].sum()
        print("   ", names[i], ":", spec[i, sel].sum(axis=0)/total)
print()
print("saved R_v1.npy, R_nt_v1.npy, z_v1.npy, windows_v1.npy in", v1_folder)
