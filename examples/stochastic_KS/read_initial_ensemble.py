import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc

import pickle

with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]

verbose = True
model = ndg.KS(nsteps, xpoints, seed=12353, lambdas=False,
               dt=dt, nu=nu, dc=dc, L=L)
with fd.CheckpointFile("ks_ensemble.h5", "r") as afile:
    mesh = afile.load_mesh("ksmesh")

model.setup(mesh)
file0 = fd.VTKFile("initial_ensemble.pvd")
X = model.allocate()
u = X[0]

with fd.CheckpointFile("ks_ensemble.h5", "r") as afile:
    for i in fd.ProgressBar("").iter(range(200)):
        u0 = afile.load_function(mesh, "u", idx=i)
        u.assign(u0)
        file0.write(u)
