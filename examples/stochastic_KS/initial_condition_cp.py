from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
from nudging.models.stochastic_KS import KS
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
n = 300
nsteps = 5
xpoints = 40
model = KS(n, nsteps, xpoints)

comm = MPI.COMM_WORLD
mesh = PeriodicIntervalMesh(n, 40.0, comm = comm)

model.setup(mesh)
X_truth = model.allocate()
u0 = X_truth[0]
x, = SpatialCoordinate(model.mesh)

#Initial conditions
u0.project(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

Q = FunctionSpace(model.mesh, 'CG', 1)
u_out = Function(Q, name="u_out")
u_out.interpolate(X_truth[0])
N_obs = 100

for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step

    u_out.interpolate(X_truth[0])

    #store numerical solution as a checkpoint file
    with CheckpointFile("initial_sol_cp.h5", 'w') as afile:
        afile.save_mesh(model.mesh)
        afile.save_function(u_out, idx=i+1)
