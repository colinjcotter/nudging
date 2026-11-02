"""q_bar from the FINE mesh at every window start (for the wall-artefact test).

The Helmholtz filter psi_tilde - ell^2 laplace(psi_tilde) = psi gives
laplace(psi_tilde) = (psi_tilde - psi)/ell^2 exactly. So the vorticity of the filtered flow is computed
on the fine mesh (no coarse 5-point stencil) and sampled at the coarse nodes with
assemble(interpolate(., V_coarse)) (same node order as nodes.npy).

    python qbar_fine.py <ell> <outname> <steps_per_window> <data_name> <nsteps_data> <n_coarse>

Reads <data_name>/psi_snapshots.h5, saves qbar_fine_start.npy (windows x nodes) in
<data_name>/displacements_<outname>_w<steps_per_window>/. Run on one MPI rank.
"""
import sys
import numpy as np
import firedrake as fd

ell = float(sys.argv[1])
outname = sys.argv[2]
steps_per_window = int(sys.argv[3])
data_name = sys.argv[4]
nsteps_data = int(sys.argv[5])
n_coarse = int(sys.argv[6])
data_dir = "/home/mkumar/DA_Results/noise_calibration/" + data_name
outdir = data_dir + "/displacements_" + outname + "_w" + str(steps_per_window)
nwindows = nsteps_data//steps_per_window

mesh_c = fd.UnitSquareMesh(n_coarse, n_coarse, quadrilateral=True)
Vc = fd.FunctionSpace(mesh_c, "CG", 1)
npoints = len(fd.Function(Vc).dat.data_ro)
qbar_all = np.zeros((nwindows, npoints))

with fd.CheckpointFile(data_dir + "/psi_snapshots.h5", "r") as psi_h5:
    mesh_f = psi_h5.load_mesh("mesh2d_per")
    psi_0 = psi_h5.load_function(mesh_f, "psi", idx=0)
    V_f = psi_0.function_space()
    psi_f = fd.Function(V_f)
    psi_tilde = fd.Function(V_f)
    qbar_f = fd.Function(V_f)
    trial = fd.TrialFunction(V_f)
    phi = fd.TestFunction(V_f)
    a = trial*phi*fd.dx + ell**2*fd.inner(fd.grad(trial), fd.grad(phi))*fd.dx
    L = psi_f*phi*fd.dx
    bc = fd.DirichletBC(V_f, 0, "on_boundary")
    sp = {"ksp_type": "preonly", "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}

    for m in range(nwindows):
        psi_k = psi_h5.load_function(mesh_f, "psi", idx=m*steps_per_window)
        psi_f.interpolate(psi_k)
        fd.solve(a == L, psi_tilde, bcs=bc, solver_parameters=sp)
        qbar_f.assign((psi_tilde - psi_f)/ell**2)
        qbar_c = fd.assemble(fd.interpolate(qbar_f, Vc))
        qbar_all[m, :] = qbar_c.dat.data_ro
        if m % 100 == 0:
            print("window", m, "of", nwindows, flush=True)

np.save(outdir + "/qbar_fine_start.npy", qbar_all)
print("saved", outdir + "/qbar_fine_start.npy")
