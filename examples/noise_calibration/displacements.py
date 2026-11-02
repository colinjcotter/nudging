"""2d, part 3: Lagrangian displacements DeltaX for all windows.

Stage 1: for every saved psi snapshot (201 of them) compute the fine velocity
(vector CG1 projection on the fine mesh), filter psi with the Helmholtz filter
(length ell) on the fine mesh, sample it on the coarse mesh and compute the
coarse velocity (vector CG1 projection on the coarse mesh).
Stage 2: for every window, move a particle from every coarse node with the fine
velocities and with the coarse velocities (RK2 midpoint loop). DeltaX = fine end - coarse end.

    python displacements.py

Settings are below: ell, outname (folder name of this run), steps_per_window.
Saved in <outdir>/ : delta_x.npy (windows x nodes x 2), end_fine.npy,
end_coarse.npy (same shape), nodes.npy (nodes x 2), psi_bar_start.npy
(windows x nodes: coarse psi at each window start), info.txt.

Run on one MPI rank.
"""
import os
import sys
import numpy as np
import firedrake as fd

# ---- settings ----
steps_per_window = 4           # window length in fine steps (4: Delta t = 0.16)
nsteps_data = 200              # number of fine steps saved by the data run
dt = 0.04                      # fine time step
n_coarse = 16
ell = 1.0/16                   # filter length: 1/16 = h_c
outname = "ell_hc"             # name of the folder for this ell
data_name = "data_run64"       # folder of the data run in DA_Results/noise_calibration
# optional command line:
#   python displacements.py <ell> <outname> <steps_per_window> [data_name nsteps_data n_coarse dt]
if len(sys.argv) > 3:
    ell = float(sys.argv[1])
    outname = sys.argv[2]
    steps_per_window = int(sys.argv[3])
if len(sys.argv) > 5:
    data_name = sys.argv[4]
    nsteps_data = int(sys.argv[5])
if len(sys.argv) > 7:
    n_coarse = int(sys.argv[6])
    dt = float(sys.argv[7])
data_dir = "/home/mkumar/DA_Results/noise_calibration/" + data_name
psi_file = data_dir + "/psi_snapshots.h5"
outdir = data_dir + "/displacements_" + outname + "_w" + str(steps_per_window)

nwindows = nsteps_data//steps_per_window
nsnap = nsteps_data + 1
os.makedirs(outdir, exist_ok=True)

# ---- coarse mesh, spaces and start points (the coarse nodes) ----
mesh_c = fd.UnitSquareMesh(n_coarse, n_coarse, quadrilateral=True)
Vc = fd.FunctionSpace(mesh_c, "CG", 1)
Wc = fd.VectorFunctionSpace(mesh_c, "CG", 1)
xc = fd.SpatialCoordinate(mesh_c)
x_coord = fd.Function(Vc)
x_coord.interpolate(xc[0])
y_coord = fd.Function(Vc)
y_coord.interpolate(xc[1])
npoints = len(x_coord.dat.data_ro)
start = np.zeros((npoints, 2))
start[:, 0] = x_coord.dat.data_ro
start[:, 1] = y_coord.dat.data_ro
print("number of start points:", npoints, " windows:", nwindows,
      " steps per window:", steps_per_window, " ell:", ell)

# ---- stage 1: velocities for all snapshots ----
u_fine = []
u_coarse = []
psi_bar_all = np.zeros((nsnap, npoints))
with fd.CheckpointFile(psi_file, "r") as psi_h5:
    mesh_f = psi_h5.load_mesh("mesh2d_per")
    W_f = fd.VectorFunctionSpace(mesh_f, "CG", 1)

    # the Helmholtz filter problem on the fine mesh
    psi_0 = psi_h5.load_function(mesh_f, "psi", idx=0)
    V_f = psi_0.function_space()
    psi_f = fd.Function(V_f)          # input of the filter
    psi_tilde = fd.Function(V_f)      # output of the filter
    trial = fd.TrialFunction(V_f)
    phi = fd.TestFunction(V_f)
    a = trial*phi*fd.dx + ell**2*fd.inner(fd.grad(trial), fd.grad(phi))*fd.dx
    L = psi_f*phi*fd.dx
    bc = fd.DirichletBC(V_f, 0, "on_boundary")
    sp = {"ksp_type": "preonly", "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}

    for k in range(nsnap):
        psi_k = psi_h5.load_function(mesh_f, "psi", idx=k)

        # fine velocity
        u_k = fd.project(fd.as_vector((-psi_k.dx(1), psi_k.dx(0))), W_f)
        u_fine.append(u_k)

        # filter, sample on the coarse mesh, coarse velocity
        psi_f.interpolate(psi_k)
        fd.solve(a == L, psi_tilde, bcs=bc, solver_parameters=sp)
        psi_bar = fd.assemble(fd.interpolate(psi_tilde, Vc))
        psi_bar_all[k, :] = psi_bar.dat.data_ro
        u_c = fd.project(fd.as_vector((-psi_bar.dx(1), psi_bar.dx(0))), Wc)
        u_coarse.append(u_c)
print("stage 1 done: velocities of", nsnap, "snapshots")

# ---- stage 2: the paths of every window ----
end_fine_all = np.zeros((nwindows, npoints, 2))
end_coarse_all = np.zeros((nwindows, npoints, 2))
clamp_fine = 0
clamp_fine_max = 0.0
clamp_coarse = 0
clamp_coarse_max = 0.0

for m in range(nwindows):
    first = m*steps_per_window

    # fine path
    points = start.copy()
    for step in range(steps_per_window):
        k = first + step
        evaluator = fd.PointEvaluator(mesh_f, points)
        u1 = evaluator.evaluate(u_fine[k])

        mid = points + 0.5*dt*u1
        mid_clamped = np.clip(mid, 0.0, 1.0)
        clamp_fine = clamp_fine + np.sum(mid != mid_clamped)
        clamp_fine_max = max(clamp_fine_max, abs(mid - mid_clamped).max())
        mid = mid_clamped

        evaluator = fd.PointEvaluator(mesh_f, mid)
        u2 = 0.5*(evaluator.evaluate(u_fine[k]) + evaluator.evaluate(u_fine[k + 1]))

        new = points + dt*u2
        new_clamped = np.clip(new, 0.0, 1.0)
        clamp_fine = clamp_fine + np.sum(new != new_clamped)
        clamp_fine_max = max(clamp_fine_max, abs(new - new_clamped).max())
        points = new_clamped
    end_fine_all[m] = points

    # coarse path
    points = start.copy()
    for step in range(steps_per_window):
        k = first + step
        evaluator = fd.PointEvaluator(mesh_c, points)
        u1 = evaluator.evaluate(u_coarse[k])

        mid = points + 0.5*dt*u1
        mid_clamped = np.clip(mid, 0.0, 1.0)
        clamp_coarse = clamp_coarse + np.sum(mid != mid_clamped)
        clamp_coarse_max = max(clamp_coarse_max, abs(mid - mid_clamped).max())
        mid = mid_clamped

        evaluator = fd.PointEvaluator(mesh_c, mid)
        u2 = 0.5*(evaluator.evaluate(u_coarse[k]) + evaluator.evaluate(u_coarse[k + 1]))

        new = points + dt*u2
        new_clamped = np.clip(new, 0.0, 1.0)
        clamp_coarse = clamp_coarse + np.sum(new != new_clamped)
        clamp_coarse_max = max(clamp_coarse_max, abs(new - new_clamped).max())
        points = new_clamped
    end_coarse_all[m] = points

    if (m + 1) % 10 == 0:
        print("window", m + 1, "of", nwindows, "done")

delta_x_all = end_fine_all - end_coarse_all

# ---- save ----
psi_bar_start = psi_bar_all[0:nwindows*steps_per_window:steps_per_window, :]
np.save(outdir + "/delta_x.npy", delta_x_all)
np.save(outdir + "/end_fine.npy", end_fine_all)
np.save(outdir + "/end_coarse.npy", end_coarse_all)
np.save(outdir + "/nodes.npy", start)
np.save(outdir + "/psi_bar_start.npy", psi_bar_start)
info = open(outdir + "/info.txt", "w")
info.write("ell = " + str(ell) + "\n")
info.write("dt (fine) = " + str(dt) + "\n")
info.write("steps_per_window = " + str(steps_per_window) + "\n")
info.write("window length Delta t = " + str(steps_per_window*dt) + "\n")
info.write("nwindows = " + str(nwindows) + "\n")
info.write("n_coarse = " + str(n_coarse) + "\n")
info.close()
print("saved to", outdir)

# ---- a few numbers over all windows ----
h_c = 1.0/n_coarse
size_dx = np.sqrt(delta_x_all[:, :, 0]**2 + delta_x_all[:, :, 1]**2)
disp_fine = end_fine_all - start
size_fine = np.sqrt(disp_fine[:, :, 0]**2 + disp_fine[:, :, 1]**2)
mean_vector = delta_x_all.mean(axis=(0, 1))
print("clamps fine path:", clamp_fine, " largest:", clamp_fine_max)
print("clamps coarse path:", clamp_coarse, " largest:", clamp_coarse_max)
print("mean fine displacement size:", size_fine.mean())
print("DeltaX size: mean =", size_dx.mean(), " max =", size_dx.max(),
      " mean / h_c =", size_dx.mean()/h_c)
print("DeltaX size / fine displacement size:", size_dx.mean()/size_fine.mean())
print("mean DeltaX vector over all windows and nodes:", mean_vector)
print("rms of DeltaX minus that mean:",
      np.sqrt(np.mean((delta_x_all - mean_vector)**2)))
