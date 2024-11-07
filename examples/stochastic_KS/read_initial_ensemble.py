import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc
from nudging.models.stochastic_KS_CIP import KS_CIP
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
model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=False,
               dt=dt, nu=nu, dc=dc, L=L)
model.setup()

with fd.CheckpointFile("ks_ensemble.h5", "r") as afile:
    mesh = afile.load_mesh("ksmesh")


#file0 = fd.VTKFile("initial_ensemble.pvd")
X = model.allocate()
u = X[0]

with fd.CheckpointFile("ks_ensemble.h5", "r") as afile:
    for i in fd.ProgressBar("").iter(range(20)):
        u0 = afile.load_function(mesh, "u", idx=i)
        u.interpolate(u0)
        #file0.write(u)
