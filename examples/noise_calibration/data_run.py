"""Data run: deterministic fine run that saves snapshots.

Starts from the equilibrated state of the spin-up (state_final.h5, t = 200)
and runs nwindows windows of steps_per_window fine steps, same model and
solver as the spin-up (noise_type='none', full Newton).

Saved with CheckpointFile (mesh saved once in each file):
  <outdir>/psi_snapshots.h5   psi (CG1) at every fine step, idx = fine step,
                              idx 0 is the initial state
  <outdir>/q_snapshots.h5     q (DQ1) at every window start (and the last
                              end), idx = window number, idx 0 is the initial state
  <outdir>/data_run_n<n>.csv  one row per step: step, t, KE, enstrophy, umax,
                              courant, newton_its, step_s
No VTK output.

    mpiexec -n 4 python data_run.py [nwindows outdir]

Settings (n, dt, windows, folders) are set below; the number of windows and the
output folder can also be given on the command line.
"""
import os
import sys
import time
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.models.stochastic_mix_euler import Euler_mixSD

# ---- settings ----
n = 64
dt = 0.04                  # fine time step (64^2: 0.04, 128^2: 0.02)
steps_per_window = 4       # window length = 4 fine steps (coarse Delta t)
nwindows = 50
t_start = 200.0            # the spin-up state is the flow at t = 200
state_file = "/home/mkumar/DA_Results/noise_calibration/spinup64_final/state_final.h5"
outdir = "/home/mkumar/DA_Results/noise_calibration/data_run64"
# optional command line: python data_run.py <nwindows> <outdir> [n]
if len(sys.argv) > 2:
    nwindows = int(sys.argv[1])
    outdir = sys.argv[2]
if len(sys.argv) > 3:
    n = int(sys.argv[3])
if n == 128:
    dt = 0.02
    state_file = "/home/mkumar/DA_Results/noise_calibration/spinup128_final/state_final.h5"
nsteps_total = steps_per_window*nwindows

comm = fd.COMM_WORLD
if comm.rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

# ---- solver: as in the spin-up ----
newton = {"snes_type": "newtonls",
          "snes_rtol": 1e-12,
          "snes_atol": 1e-12,
          "snes_max_it": 20}
if n <= 64:
    solver_parameters = newton
else:
    solver_parameters = dict(newton, ksp_type="preonly", pc_type="lu",
                             pc_factor_mat_solver_type="mumps",
                             mat_type="aij")

# ---- load the spun-up state ----
with fd.CheckpointFile(state_file, "r") as afile:
    mesh = afile.load_mesh("mesh2d_per")
    q_loaded = afile.load_function(mesh, "q")
    psi_loaded = afile.load_function(mesh, "psi")

# ---- model on the loaded mesh ----
model = Euler_mixSD(n, 1, dt, 0.0, mesh=mesh, noise_type='none',
                    solver_parameters=solver_parameters)
model.setup()

X = model.allocate()
q, psi = X[0].subfunctions
q.interpolate(q_loaded)        # psi is solved from q inside run()

# ---- diagnostics (as in the spin-up) ----
Vdq = fd.FunctionSpace(mesh, "DQ", 1)
speed = fd.Function(Vdq)
u = model.gradperp(psi)
speed_expr = fd.sqrt(fd.inner(u, u))
KE_form = 0.5*fd.inner(fd.grad(psi), fd.grad(psi))*fd.dx
enstrophy_form = 0.5*q*q*fd.dx
h = 1.0/n

# ---- plain functions that are written to the files ----
Vcg = fd.FunctionSpace(mesh, "CG", 1)
psi_out = fd.Function(Vcg, name="psi")
q_out = fd.Function(Vdq, name="q")

psi_path = os.path.join(outdir, "psi_snapshots.h5")
q_path = os.path.join(outdir, "q_snapshots.h5")

if comm.rank == 0:
    csv_file = open(os.path.join(outdir, "data_run_n" + str(n) + ".csv"), "w")
    csv_file.write("step,t,KE,enstrophy,umax,courant,newton_its,step_s\n")

PETSc.Sys.Print("data run: n =", n, " dt =", dt, " windows =", nwindows,
                " steps per window =", steps_per_window,
                " total steps =", nsteps_total, " ranks =", comm.size)

with fd.CheckpointFile(psi_path, "w") as psi_h5, fd.CheckpointFile(q_path, "w") as q_h5:
    psi_h5.save_mesh(mesh)
    q_h5.save_mesh(mesh)

    # snapshot at step 0: the loaded state
    psi_out.interpolate(psi_loaded)
    psi_h5.save_function(psi_out, idx=0)
    q_out.interpolate(q)
    q_h5.save_function(q_out, idx=0)

    t_wall = time.perf_counter()
    for step in range(1, nsteps_total + 1):
        t0 = time.perf_counter()
        model.run(X, X)
        step_s = time.perf_counter() - t0
        newton_its = model.qphi_solver.snes.getIterationNumber()

        # diagnostics
        KE = fd.assemble(KE_form)
        enstrophy = fd.assemble(enstrophy_form)
        speed.interpolate(speed_expr)
        umax = comm.allreduce(speed.dat.data_ro.max(), op=MPI.MAX)
        courant = umax*dt/h
        if comm.rank == 0:
            csv_file.write(",".join(str(v) for v in [
                step, t_start + step*dt, KE, enstrophy, umax, courant,
                newton_its, step_s]) + "\n")

        # psi at every fine step
        psi_out.interpolate(psi)
        psi_h5.save_function(psi_out, idx=step)

        # q at every window start (step 4, 8, ... is the start of the next window)
        if step % steps_per_window == 0:
            q_out.interpolate(q)
            q_h5.save_function(q_out, idx=step//steps_per_window)

        if step % 20 == 0:
            PETSc.Sys.Print("step", step, " t =", t_start + step*dt,
                            " KE =", KE, " enstrophy =", enstrophy,
                            " umax =", umax, " Courant =", courant,
                            " elapsed (s) =", time.perf_counter() - t_wall)

if comm.rank == 0:
    csv_file.close()
    print("psi_snapshots.h5 size (MB):", os.path.getsize(psi_path)/1e6)
    print("q_snapshots.h5 size (MB):", os.path.getsize(q_path)/1e6)
PETSc.Sys.Print("done: snapshots saved to", outdir)
