import firedrake as fd
import nudging as ndg
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP
from firedrake.adjoint import continue_annotation, pause_annotation
import firedrake.adjoint as fadj

import pickle

with open("params.pickle", "rb") as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]
# dc = 2.5

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True, dt=dt, nu=nu, dc=dc, L=L)
model.setup()
X_start = model.allocate()
X_out = model.allocate()
u_in = X_start[0]

with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r") as afile:
    mesh = afile.load_mesh("ksmesh")
    u0 = afile.load_function(mesh, "u", idx=1)
    u_in.interpolate(u0)


def log_likelihood(y, Y):
    ll = (y - Y) ** 2 / 5.5**2 / 2 * fd.dx
    return ll


# Load data
y_exact = np.load("../../DA_KS/y_true.npy")
y = np.load("../../DA_KS/y_obs.npy")
yVOM = fd.Function(model.VVOM)
yVOM.dat.data[:] = y[0, :]

continue_annotation()

scale = []
scale_controls = []
for step in range(nsteps):
    s = fd.Function(model.R)
    s.assign(1.0)
    scale.append(s)
    scale_controls.append(fadj.Control(s))

model.run(X_start, X_out, s=scale)
Y = model.obs()
nudge_J = fd.assemble(log_likelihood(yVOM, Y))
nudge_J += model.lambda_functional(scale)
m = model.controls() + [fadj.Control(yVOM)] + scale_controls
fnl = fadj.ReducedFunctional(nudge_J, m)
pause_annotation()

X_use = model.allocate()
X_use[0].assign(X_start[0])

for step in range(nsteps):
    X_use[nsteps + 1 + step].assign(0.5)
    scale[step].assign(0.0)

before = fnl(X_use + [yVOM] + scale)

for step in range(nsteps):
    X_use[nsteps + 1 + step].assign(0 * X_use[nsteps + 1 + step])
    # scale[step].assign(1.0)

after = fnl(X_use + [yVOM] + scale)

print(before, after)
