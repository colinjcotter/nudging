"""One-step closure error of the coarse model, and how much of it transport (SALT) can represent.

For windows m (every `stride`-th): start the coarse model (no noise, dt = window) from the fine-mesh q_bar^m
(qbar_fine_start.npy, CG1 at the coarse nodes), run one window, project the model q to CG1 and compare with the
truth q_bar^{m+1}:
    R^m = q_bar^{m+1} - Phi(q_bar^m)            (one-step closure error, in q)
Printed (means over the windows):
  |R| / |q_bar^{m+1} - q_bar^m|    size of the error relative to the one-step change of the truth
  |mean of R| / rms |R|            how much of R is a fixed bias (time-mean pattern)
  transport fit: the divergence-free velocity v = grad-perp(z) that best explains R by transport over one window,
      R ~ dt J(z, q_bar^m),  J(a, b) = -a_y b_x + a_x b_y (CG1, z = 0 on the walls),
      min |R - dt J(z, q)|^2 + eps |grad z|^2,  eps = eps_rel * dt^2 * mean |grad q|^2   (z in the kernel J(z,q)=0
      is not determined; eps picks the smallest one)
  -> share of |R|^2 explained by transport, and the SALT field it needs, zeta_salt = sqrt(dt) z, compared with the
     calibrated residual zeta_res (rms over nodes) and with the full zeta of the Lagrangian data.
A share near 1 with a moderate zeta_salt: SALT can represent the closure error. A small share: it cannot.

In-model check (every 10th window used): the same window run with the SALT hook (noise_source='external',
noise_scale = 1) and dU_2 = +sqrt(dt) z and -sqrt(dt) z (z of the first eps_rel): |q_bar^{m+1} - Phi_salt| / |R|.
The sign that gives a value well below 1 is the model-ready field. Saved (first eps_rel, that sign):
zeta_salt_fit.npy (windows x nodes, dU_2 for one window) and windows_salt.npy (the window numbers).

    python one_step_error.py [outname data_name n_coarse dt_window stride]
    (defaults: ell_2hc64 data_run128_long 64 0.08 5)

Reads qbar_fine_start.npy, psi_bar_start.npy, zeta.npy, nodes.npy of <data_name>/displacements_<outname>_w4/.
Saves zeta_salt_fit.npy and windows_salt.npy there (stride 1 for salt_generator_psi.py).
One MPI rank.
"""
import sys
import time
import numpy as np
import firedrake as fd
from nudging.models.stochastic_mix_euler import Euler_mixSD

outname = "ell_2hc64"
data_name = "data_run128_long"
n_coarse = 64
window_length = 0.08
stride = 5
eps_list = [1e-2, 1e-4]
if len(sys.argv) > 5:
    outname = sys.argv[1]
    data_name = sys.argv[2]
    n_coarse = int(sys.argv[3])
    window_length = float(sys.argv[4])
    stride = int(sys.argv[5])
folder = "/home/mkumar/DA_Results/noise_calibration/" + data_name + "/displacements_" + outname + "_w4"
h = 1.0/n_coarse

qbar_fine = np.load(folder + "/qbar_fine_start.npy")
zeta = np.load(folder + "/zeta.npy")
nodes = np.load(folder + "/nodes.npy")
nwindows = qbar_fine.shape[0]
nnodes = qbar_fine.shape[1]
windows_list = list(range(0, nwindows - 1, stride))
print("folder:", folder, " windows used:", len(windows_list), " eps_rel:", eps_list, flush=True)

# zeta_res of the Lagrangian data (5-point q_bar at the interior nodes), for comparison
ix = np.rint(nodes[:, 0]*n_coarse).astype(int)
iy = np.rint(nodes[:, 1]*n_coarse).astype(int)
psi_bar = np.load(folder + "/psi_bar_start.npy")
zeta_grid = np.zeros((nwindows, n_coarse + 1, n_coarse + 1))
psi_grid = np.zeros((nwindows, n_coarse + 1, n_coarse + 1))
zeta_grid[:, ix, iy] = zeta
psi_grid[:, ix, iy] = psi_bar
q5 = (psi_grid[:, 2:, 1:-1] + psi_grid[:, :-2, 1:-1] + psi_grid[:, 1:-1, 2:] + psi_grid[:, 1:-1, :-2]
      - 4.0*psi_grid[:, 1:-1, 1:-1])/h**2
zeta_in = zeta_grid[:, 1:-1, 1:-1]
c = np.sum(zeta_in*q5, axis=(1, 2))/np.sum(q5**2, axis=(1, 2))
zeta_res = zeta_in - c[:, None, None]*q5
print("Lagrangian data: rms zeta =", np.sqrt(np.mean(zeta**2)), " rms zeta_res =", np.sqrt(np.mean(zeta_res**2)),
      flush=True)

# ---- model ----
newton = {"snes_type": "newtonls", "snes_rtol": 1e-12, "snes_atol": 1e-12, "snes_max_it": 20}
model = Euler_mixSD(n_coarse, 1, window_length, 1.0, noise_type='none', solver_parameters=newton)
model.setup()
X0 = model.allocate()
X1 = model.allocate()
Vcg = model.Vcg

q0c = fd.Function(Vcg)          # q_bar^m
q1c = fd.Function(Vcg)          # q_bar^{m+1}
qmod = fd.Function(Vcg)         # model q after one window, projected to CG1
R = fd.Function(Vcg)
z = fd.Function(Vcg)

# projection of the model q (DQ1) to CG1
trial = fd.TrialFunction(Vcg)
test = fd.TestFunction(Vcg)
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

# second model with the SALT hook, for the in-model check
model_s = Euler_mixSD(n_coarse, 1, window_length, 1.0, mesh=model.mesh, noise_type='salt',
                      noise_source='external', solver_parameters=newton)
model_s.setup()
noise_fn = fd.Function(model_s.Vcg)
model_s.noise_fields = [noise_fn]
Y0 = model_s.allocate()
Y1 = model_s.allocate()
qsalt = fd.Function(model_s.Vcg)
s_problem = fd.LinearVariationalProblem(fd.TrialFunction(model_s.Vcg)*fd.TestFunction(model_s.Vcg)*fd.dx,
                                        Y1[0].subfunctions[0]*fd.TestFunction(model_s.Vcg)*fd.dx, qsalt)
s_solver = fd.LinearVariationalSolver(s_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
signs = [1.0, -1.0]

nw = len(windows_list)
z_first = np.zeros((nw, nnodes))
check = []
rel_err = np.zeros(nw)
R_all = np.zeros((nw, nnodes))
explained = np.zeros((len(eps_list), nw))
zeta_salt_rms = np.zeros((len(eps_list), nw))
t_begin = time.time()

for iw in range(nw):
    m = windows_list[iw]
    q0c.dat.data[:] = qbar_fine[m]
    q1c.dat.data[:] = qbar_fine[m + 1]
    X0[0].subfunctions[0].interpolate(q0c)
    model.run(X0, X1)
    p_solver.solve()
    R.assign(q1c - qmod)
    norm_R = fd.assemble(R*R*fd.dx)**0.5
    change = fd.assemble((q1c - q0c)**2*fd.dx)**0.5
    rel_err[iw] = norm_R/change
    R_all[iw] = R.dat.data_ro
    grad_q2 = fd.assemble(fd.inner(fd.grad(q0c), fd.grad(q0c))*fd.dx)
    for ie in range(len(eps_list)):
        eps.assign(eps_list[ie]*window_length**2*grad_q2)
        f_solver.solve()
        rest = fd.assemble((R - window_length*Jz)**2*fd.dx)
        explained[ie, iw] = 1.0 - rest/norm_R**2
        zeta_salt_rms[ie, iw] = np.sqrt(window_length)*np.sqrt(np.mean(z.dat.data_ro**2))
        if ie == 0:
            z_first[iw] = z.dat.data_ro
    if iw % 10 == 0:
        row = []
        for sg in signs:
            noise_fn.dat.data[:] = sg*np.sqrt(window_length)*z_first[iw]
            Y0[0].subfunctions[0].interpolate(q0c)
            model_s.run(Y0, Y1)
            s_solver.solve()
            row.append(fd.assemble((q1c - qsalt)**2*fd.dx)**0.5/norm_R)
        check.append(row)
    if iw % 40 == 0:
        print("window", m, ": |R|/change =", rel_err[iw], " explained by transport (eps_rel", eps_list, "):",
              explained[:, iw], " elapsed", time.time() - t_begin, flush=True)

mean_R = R_all.mean(axis=0)
print()
print("one-step closure error |R| / |change of q_bar|: mean =", rel_err.mean(), " min =", rel_err.min(),
      " max =", rel_err.max())
print("bias: |mean of R| / rms |R| (nodal) =", np.sqrt(np.mean(mean_R**2))/np.sqrt(np.mean(R_all**2)),
      "  share of the energy of R in its time mean =", np.mean(mean_R**2)/np.mean(R_all**2))
for ie in range(len(eps_list)):
    print("transport fit, eps_rel =", eps_list[ie], ": share of |R|^2 explained: mean =", explained[ie].mean(),
          " min =", explained[ie].min(), " max =", explained[ie].max(),
          "  rms zeta_salt =", zeta_salt_rms[ie].mean())
check = np.array(check)
print("in-model check, |q_bar^{m+1} - Phi_salt(q_bar^m)| / |R| over", check.shape[0], "windows:  dU_2 = +sqrt(dt) z:",
      check[:, 0].mean(), "  dU_2 = -sqrt(dt) z:", check[:, 1].mean())
best = 0
if check[:, 1].mean() < check[:, 0].mean():
    best = 1
np.save(folder + "/zeta_salt_fit.npy", signs[best]*np.sqrt(window_length)*z_first)
np.save(folder + "/windows_salt.npy", np.array(windows_list))
print("saved zeta_salt_fit.npy with the sign", signs[best], "(eps_rel", eps_list[0], ") and windows_salt.npy")
print("(for comparison: the residual noise zeta_res changes q_bar by about 0.55% of the one-step change; rms of")
print(" zeta_res and zeta of the Lagrangian data are printed at the top)")
