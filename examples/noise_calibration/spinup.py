"""Deterministic spin-up from omega_spin.

    mpiexec -n 4 python spinup.py <n> <outdir> [tend] [dt] [restart]

    n      : 64 or 128
    outdir : all output goes here
    tend   : end time, default 150
    dt     : time step, default 0.01
    restart: the word restart continues a run from <outdir>/state_latest.h5
             (use the same n, outdir, tend, dt and number of ranks);
             the csv is cut back to the step of the checkpoint and continued

Model: Euler_mixSD, noise_type='none', full Newton. Linear solver chosen
by timing tests: fieldsplit for n <= 64, monolithic MUMPS LU for larger n.
Initial vorticity omega_spin, eq. (3.5) of Cotter et al. (SIAM MMS 2019).
psi0 is solved from q0 inside run() in every step.

Written to <outdir>:
  spinup_n<n>.csv        one row per step: step, t, KE, enstrophy, umax,
                         courant, newton_its, step_s
  state_latest.h5        q and psi, overwritten every nsave steps, with the
                         step number in state_latest_step.txt
  state_final.h5         q and psi at the end
  vtk/                   q and psi every vtk_interval = 0.2 time units, for
                         visual checks only (about 0.8 MB per frame at 64^2)
KE = 1/2 int |grad psi|^2, enstrophy = 1/2 int q^2, umax = max |grad_perp psi|
(largest value at the cell vertices), courant = umax*dt/h with h = 1/n.
"""
import os
import sys
import time
import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.models.stochastic_mix_euler import Euler_mixSD

n = int(sys.argv[1])
outdir = sys.argv[2]
if len(sys.argv) > 3:
    tend = float(sys.argv[3])
else:
    tend = 150.0
if len(sys.argv) > 4:
    dt = float(sys.argv[4])
else:
    dt = 0.01
restart = len(sys.argv) > 5 and sys.argv[5] == "restart"
nsteps_total = int(round(tend/dt))
nsave = 500      # steps between restart checkpoints
vtk_interval = 0.2          # time units between vtk frames
nvtk = int(round(vtk_interval/dt))   # steps between vtk frames
nprint = 100     # steps between progress lines and csv flushes

comm = fd.COMM_WORLD
if comm.rank == 0:
    os.makedirs(os.path.join(outdir, "vtk"), exist_ok=True)
comm.Barrier()

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

# restart: read the step number, the mesh and q from the last checkpoint
start_step = 0
mesh = False        # False: the model builds its own mesh
if restart:
    with open(os.path.join(outdir, "state_latest_step.txt")) as step_file:
        start_step = int(step_file.read())
    with fd.CheckpointFile(os.path.join(outdir, "state_latest.h5"),
                           "r") as afile:
        mesh = afile.load_mesh("mesh2d_per")
        q_restart = afile.load_function(mesh, "q")

model = Euler_mixSD(n, 1, dt, 0.0, mesh=mesh, noise_type='none',
                    solver_parameters=solver_parameters)
model.setup()
mesh = model.mesh

X = model.allocate()
q, psi = X[0].subfunctions
x = fd.SpatialCoordinate(mesh)
if restart:
    # psi is not needed, run() solves psi0 from q0 at the start of every step
    q.interpolate(q_restart)
else:
    q.interpolate(fd.sin(8*fd.pi*x[0])*fd.sin(8*fd.pi*x[1])
                  + 0.4*fd.cos(6*fd.pi*x[0])*fd.cos(6*fd.pi*x[1])
                  + 0.3*fd.cos(10*fd.pi*x[0])*fd.cos(4*fd.pi*x[1])
                  + 0.02*fd.sin(2*fd.pi*x[1]) + 0.02*fd.sin(2*fd.pi*x[0]))

# diagnostics
Vdq = fd.FunctionSpace(mesh, "DQ", 1)
speed = fd.Function(Vdq)
u = model.gradperp(psi)
speed_expr = fd.sqrt(fd.inner(u, u))
KE_form = 0.5*fd.inner(fd.grad(psi), fd.grad(psi))*fd.dx
enstrophy_form = 0.5*q*q*fd.dx
h = 1.0/n

# fields for checkpoint and vtk (plain functions, not parts of the mixed one)
Vcg = fd.FunctionSpace(mesh, "CG", 1)
q_out = fd.Function(Vdq, name="q")
psi_out = fd.Function(Vcg, name="psi")
if restart:
    pvd_name = "spinup_from" + str(start_step) + ".pvd"   # keeps the old pvd
else:
    pvd_name = "spinup.pvd"
vtk = fd.VTKFile(os.path.join(outdir, "vtk", pvd_name))


def save_state(filename, step):
    # write to a temporary file and rename, so that a crash during the
    # write cannot destroy the previous checkpoint
    q_out.interpolate(q)
    psi_out.interpolate(psi)
    tmp_path = os.path.join(outdir, "state_tmp.h5")
    with fd.CheckpointFile(tmp_path, "w") as afile:
        afile.save_mesh(mesh)
        afile.save_function(q_out)
        afile.save_function(psi_out)
    if comm.rank == 0:
        os.replace(tmp_path, os.path.join(outdir, filename))
        step_path = os.path.join(outdir, filename[:-3] + "_step.txt")
        with open(step_path, "w") as step_file:
            step_file.write(str(step))
    comm.Barrier()


csv_path = os.path.join(outdir, "spinup_n" + str(n) + ".csv")
if comm.rank == 0:
    if restart:
        # keep the header and the rows up to the checkpoint, drop the rest
        with open(csv_path) as old_csv:
            old_lines = old_csv.readlines()
        kept = [old_lines[0]]
        for line in old_lines[1:]:
            if line.strip() and int(line.split(",")[0]) <= start_step:
                kept.append(line)
        csv_file = open(csv_path, "w")
        csv_file.writelines(kept)
    else:
        csv_file = open(csv_path, "w")
        csv_file.write("step,t,KE,enstrophy,umax,courant,newton_its,step_s\n")

PETSc.Sys.Print("spin-up: n =", n, " dt =", dt, " steps =", nsteps_total,
                " ranks =", comm.size, " starting after step", start_step)
t_start = time.perf_counter()
for step in range(start_step + 1, nsteps_total + 1):
    t0 = time.perf_counter()
    model.run(X, X)
    step_s = time.perf_counter() - t0
    newton_its = model.qphi_solver.snes.getIterationNumber()

    KE = fd.assemble(KE_form)
    enstrophy = fd.assemble(enstrophy_form)
    speed.interpolate(speed_expr)
    umax = comm.allreduce(speed.dat.data_ro.max(), op=MPI.MAX)
    courant = umax*dt/h

    if comm.rank == 0:
        csv_file.write(",".join(str(v) for v in [
            step, step*dt, KE, enstrophy, umax, courant, newton_its,
            step_s]) + "\n")
    if step % nprint == 0:
        PETSc.Sys.Print("step", step, " t =", step*dt, " KE =", KE,
                        " enstrophy =", enstrophy, " umax =", umax,
                        " Courant =", courant,
                        " elapsed (s) =", time.perf_counter() - t_start)
        if comm.rank == 0:
            csv_file.flush()
    if step % nvtk == 0:
        q_out.interpolate(q)
        psi_out.interpolate(psi)
        vtk.write(q_out, psi_out, time=step*dt)
    if step % nsave == 0:
        save_state("state_latest.h5", step)

save_state("state_final.h5", nsteps_total)
if comm.rank == 0:
    csv_file.close()
PETSc.Sys.Print("done: final state saved to", outdir)
