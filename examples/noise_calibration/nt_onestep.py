"""V2c: one-step check of the deterministic non-transport term in the model.

Variants (coefficients from nt_fit.py, fitted on the windows 0 to n_train - 1):
  A  pattern + damping:    r = 0.01 + gamma, q_source_fn = bP * P
  B  pattern + diffusion:  nu > 0 (interior penalty), q_source_fn = bP * P
  C  pattern only:         r = 0.01, q_source_fn = P
P = time mean of R_nt_v1 over the fit windows (all nodes). Both run on top of the V1 deterministic correction
(dU_2 = k c psi_model + mean_rest). For every window: R = q_bar^{m+1} - Phi(q_bar^m), model q projected to CG1.
Printed for the fit and the test windows: |R| / |change of q_bar| for this variant and for V1 (R_v1.npy), the ratio
of the node sums of |R|^2 (new / V1), the same ratio predicted offline (R_v1 - fit, fit from the grid stencils
of nt_fit.py), and the enstrophy effect int q_bar R / Z per window.

    python nt_onestep.py A|B|C [stride]

Reads noise_model_v1.npz, R_v1.npy, R_nt_v1.npy, windows_v1.npy (~/DA_Results/noise_calibration/v1/) and
qbar_fine_start.npy of its data folder. Saves R_<variant>.npy in the v1 folder. One MPI rank.
"""
import sys
import time
import numpy as np
import firedrake as fd
from nudging.models.stochastic_mix_euler import Euler_mixSD

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
variant = sys.argv[1]
stride = 1
if len(sys.argv) > 2:
    stride = int(sys.argv[2])

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
R_v1 = np.load(v1_folder + "/R_v1.npy")
Rnt_v1 = np.load(v1_folder + "/R_nt_v1.npy")
windows_v1 = np.load(v1_folder + "/windows_v1.npy")
P = Rnt_v1[windows_v1 < n_train].mean(axis=0)

# ---- coefficients (nt_fit.py) ----
r = 0.01
nu = 0.0
if variant == "A":
    bP = 0.89279988
    gamma = 0.013483674564380568
    r = 0.01 + gamma
if variant == "B":
    bP = 0.819793075
    nu = 2.9253776576074123e-05
if variant == "C":
    bP = 1.0
print("variant", variant, ": r =", r, " nu =", nu, " bP =", bP, flush=True)

# ---- offline prediction: R_v1 - fit on all nodes ----
q_grid = np.zeros((len(windows_v1), n_coarse + 1, n_coarse + 1))
q_grid[:, ix, iy] = qbar_fine[windows_v1]
if variant == "A":
    X = -gamma*window_length*qbar_fine[windows_v1]
if variant == "B":
    lap_grid = np.zeros_like(q_grid)
    lap_grid[:, 1:-1, 1:-1] = (q_grid[:, 2:, 1:-1] + q_grid[:, :-2, 1:-1] + q_grid[:, 1:-1, 2:]
                               + q_grid[:, 1:-1, :-2] - 4.0*q_grid[:, 1:-1, 1:-1])/h**2
    X = nu*window_length*lap_grid[:, ix, iy]
if variant == "C":
    X = 0.0
R_pred = R_v1 - bP*P - X

# ---- model ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='salt', noise_source='external',
                    solver_parameters=newton, r=r, q_source=True, nu=nu)
model.setup()
noise_fn = fd.Function(model.Vcg)
model.noise_fields = [noise_fn]
model.q_source_fn.dat.data[:] = bP*P
X0 = model.allocate()
X1 = model.allocate()
Vcg = model.Vcg

q0c = fd.Function(Vcg)
q1c = fd.Function(Vcg)
qmod = fd.Function(Vcg)
trial = fd.TrialFunction(Vcg)
test = fd.TestFunction(Vcg)
p_problem = fd.LinearVariationalProblem(trial*test*fd.dx, X1[0].subfunctions[0]*test*fd.dx, qmod)
p_solver = fd.LinearVariationalSolver(p_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

windows_list = list(range(0, qbar_fine.shape[0] - 1, stride))
nw = len(windows_list)
R_all = np.zeros((nw, qbar_fine.shape[1]))
rel_err = np.zeros(nw)
rel_err_v1 = np.zeros(nw)
dZ = np.zeros(nw)
t_begin = time.time()

for iw in range(nw):
    m = windows_list[iw]
    q0c.dat.data[:] = qbar_fine[m]
    q1c.dat.data[:] = qbar_fine[m + 1]
    X0[0].subfunctions[0].interpolate(q0c)
    model.qpsi0.assign(X0[0])
    model.psi0_solver.solve()
    noise_fn.dat.data[:] = kc*model.psi0_tmp.dat.data_ro + mean_rest
    model.run(X0, X1)
    p_solver.solve()
    R_all[iw] = qbar_fine[m + 1] - qmod.dat.data_ro
    change = qbar_fine[m + 1] - qbar_fine[m]
    rel_err[iw] = np.sqrt(np.sum(R_all[iw]**2)/np.sum(change**2))
    rel_err_v1[iw] = np.sqrt(np.sum(R_v1[m]**2)/np.sum(change**2))
    q0c.dat.data[:] = qbar_fine[m]
    Rf = fd.Function(Vcg)
    Rf.dat.data[:] = R_all[iw]
    dZ[iw] = fd.assemble(q0c*Rf*fd.dx)/(0.5*fd.assemble(q0c*q0c*fd.dx))
    if iw % 100 == 0:
        print("window", m, ": |R|/change =", rel_err[iw], " V1:", rel_err_v1[iw], " dZ/Z =", dZ[iw],
              " elapsed", time.time() - t_begin, flush=True)

np.save(v1_folder + "/R_" + variant + ".npy", R_all)

windows_arr = np.array(windows_list)
for block in range(2):
    if block == 0:
        sel = windows_arr < n_train
        print()
        print("fit windows (", int(sel.sum()), ")")
    else:
        sel = windows_arr >= n_train
        print()
        print("test windows (", int(sel.sum()), ")")
    print("   |R| / |change|: variant", rel_err[sel].mean(), "  V1", rel_err_v1[sel].mean())
    print("   sum |R|^2 new / V1:", np.sum(R_all[sel]**2)/np.sum(R_v1[windows_arr[sel]]**2),
          "  predicted offline:", np.sum(R_pred[windows_arr[sel]]**2)/np.sum(R_v1[windows_arr[sel]]**2))
    print("   dZ/Z per window: mean", dZ[sel].mean(), " (V1 R: about -0.0035)")
