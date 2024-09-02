import firedrake as fd
import nudging as ndg
import numpy as np

# create some synthetic data/observation data at T_1 ---- T_Nobs
# Pick initial conditon
# run model, get obseravation
# add observation noise N(0, sigma^2)

params = {}

nsteps = 5
params["nsteps"] = nsteps
xpoints = 40
params["xpoints"] = xpoints
L = 10.
params["L"] = L
dt = 0.01
params["dt"] = dt
nu = 0.02923
params["nu"] = nu
dc = 0.01
params["dc"] = dc

model = ndg.KS(nsteps, xpoints, seed=12353, lambdas=False,
               dt=dt, nu=nu, dc=dc, L=L)
model.setup()
X_start = model.allocate()
u = X_start[0]
x, = fd.SpatialCoordinate(model.mesh)

CG3 = fd.FunctionSpace(model.mesh, "CG", 3)
u0 = model.rg.normal(CG3, 0., 1.)
u0 -= fd.assemble(u0*fd.dx)/L
u0 *= L**0.5/fd.norm(u0)
u.project(u0)

print("Finding an initial state.")
for i in fd.ProgressBar("").iter(range(2000)):
    model.randomize(X_start)
    model.run(X_start, X_start)  # run method for every time step

print("generating ensemble.")

Nensemble = 200  # size of the ensemble
import math
spread_steps = math.ceil(2./dt/nsteps)

Hermite = fd.FunctionSpace(model.mesh, "Hermite", 3)
uout = fd.Function(Hermite, name="u")

with fd.CheckpointFile("ks_ensemble.h5", 'w') as afile:
    afile.save_mesh(model.mesh)

    for i in range(Nensemble+1):
        if i < Nensemble:
            print("Generating ensemble member", i)
        else:
            print("Generating 'true' value")
        X = model.allocate()
        X[0].assign(X_start[0])

        for step in fd.ProgressBar("").iter(range(spread_steps)):
            model.randomize(X)
            model.run(X, X)  # run method for every time step

        uout.assign(X[0])
        afile.save_function(uout, idx=i)

print("Generating the observational data.")
N_obs = 10
params["N_obs"] = N_obs

y_true = model.obs().dat.data[:]
y_true_full = np.zeros((y_true.size, N_obs))
y_obs_full = np.zeros((y_true.size, N_obs))

noise_var = 0.1**2
params["noise_var"] = noise_var

for i in fd.ProgressBar("").iter(range(N_obs)):
    model.randomize(X)
    model.run(X, X)  # run method for every time step
    y_true = model.obs().dat.data[:]
    y_true_full[:, i] = y_true.reshape((y_true.size,))
    y_noise = np.random.normal(0.0, noise_var**0.5, y_true.shape)
    y_obs_full[:, i] = (y_true + y_noise).reshape((y_true.size,))
np.save("y_true.npy", y_true_full)
np.save("y_obs.npy", y_obs_full)

import pickle
with open('params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
