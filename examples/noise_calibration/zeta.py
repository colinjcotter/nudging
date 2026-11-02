"""Streamfunction zeta of every displacement field DeltaX.

For each window m: d = DeltaX[m] / sqrt(Delta t) as a vector CG1 field on the
coarse mesh (nodal values), then zeta in CG1 with zeta = 0 on the boundary
minimising int |grad_perp(zeta) - d|^2, by one Poisson solve:
    int grad(zeta).grad(phi) dx = int d . grad_perp(phi) dx.
Captured fraction of each window = int |grad zeta|^2 / int |d|^2.

    python zeta.py <outname> <steps_per_window> [data_name]     e.g.  python zeta.py ell_hc 4
(defaults: ell_hc, 4, data_run64)

Reads  <data_name>/displacements_<outname>_w<steps_per_window>/delta_x.npy, nodes.npy
Saves  zeta.npy (windows x nodes, nodal values of zeta) and
       captured_fraction.npy (windows) in the same folder.
Run on one MPI rank.
"""
import sys
import numpy as np
import firedrake as fd

# ---- settings ----
outname = "ell_hc"
steps_per_window = 4
data_name = "data_run64"       # folder of the data run in DA_Results/noise_calibration
if len(sys.argv) > 2:
    outname = sys.argv[1]
    steps_per_window = int(sys.argv[2])
if len(sys.argv) > 3:
    data_name = sys.argv[3]
dt = 0.04                      # fine time step
n_coarse = 16
if len(sys.argv) > 5:          # python zeta.py <outname> <steps> <data_name> <n_coarse> <dt>
    n_coarse = int(sys.argv[4])
    dt = float(sys.argv[5])
folder = "/home/mkumar/DA_Results/noise_calibration/" + data_name \
    + "/displacements_" + outname + "_w" + str(steps_per_window)
window_length = steps_per_window*dt
sqrt_window = np.sqrt(window_length)

delta_x = np.load(folder + "/delta_x.npy")      # windows x nodes x 2
nodes = np.load(folder + "/nodes.npy")          # nodes x 2
nwindows = delta_x.shape[0]
npoints = delta_x.shape[1]
print("folder:", folder)
print("windows:", nwindows, " nodes:", npoints, " Delta t =", window_length)

# ---- coarse mesh, spaces and the problem for zeta ----
mesh_c = fd.UnitSquareMesh(n_coarse, n_coarse, quadrilateral=True)
V = fd.FunctionSpace(mesh_c, "CG", 1)
W = fd.VectorFunctionSpace(mesh_c, "CG", 1)

# check: the node order of the vector space must be the order of the saved nodes
coords = fd.Function(W)
coords.interpolate(fd.SpatialCoordinate(mesh_c))
print("node order check: max |node coordinates of W - saved nodes| =",
      abs(coords.dat.data_ro - nodes).max())

d = fd.Function(W)
zeta = fd.Function(V)
trial = fd.TrialFunction(V)
phi = fd.TestFunction(V)
a = fd.inner(fd.grad(trial), fd.grad(phi))*fd.dx
L = fd.inner(d, fd.as_vector((-phi.dx(1), phi.dx(0))))*fd.dx
bc = fd.DirichletBC(V, 0, "on_boundary")
sp = {"ksp_type": "preonly", "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps"}

# ---- one solve per window ----
zeta_all = np.zeros((nwindows, npoints))
fraction = np.zeros(nwindows)
d_rms = np.zeros(nwindows)
for m in range(nwindows):
    d.dat.data[:, 0] = delta_x[m, :, 0]/sqrt_window
    d.dat.data[:, 1] = delta_x[m, :, 1]/sqrt_window
    fd.solve(a == L, zeta, bcs=bc, solver_parameters=sp)

    zeta_all[m, :] = zeta.dat.data_ro
    energy_zeta = fd.assemble(fd.inner(fd.grad(zeta), fd.grad(zeta))*fd.dx)
    energy_d = fd.assemble(fd.inner(d, d)*fd.dx)
    fraction[m] = energy_zeta/energy_d
    d_rms[m] = np.sqrt(energy_d)      # the domain has area 1

np.save(folder + "/zeta.npy", zeta_all)
np.save(folder + "/captured_fraction.npy", fraction)
print("saved zeta.npy and captured_fraction.npy")

# ---- a few numbers ----
print("captured fraction: mean =", fraction.mean(), " min =", fraction.min(),
      " max =", fraction.max())
print("rms of d = DeltaX/sqrt(Delta t) over the domain: mean over windows =", d_rms.mean())
print("zeta: max |zeta| =", abs(zeta_all).max(),
      " rms over nodes and windows =", np.sqrt(np.mean(zeta_all**2)))
