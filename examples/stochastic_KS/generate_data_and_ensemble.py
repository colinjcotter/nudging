import firedrake as fd
from firedrake.output import VTKFile
import numpy as np
import math
import nudging as ndg
from nudging.models.stochastic_KS_CIP import KS_CIP

import os
os.makedirs('../../DA_KS/', exist_ok=True)
os.makedirs('../../DA_KS/checkpoint_files/', exist_ok=True)
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
truth_init = VTKFile("../../DA_KS/truth_init.pvd")
truth = VTKFile("../../DA_KS/truth.pvd")
particle_init = VTKFile("../../DA_KS/particle_init.pvd")

params = {}

nsteps = 10
params["nsteps"] = nsteps
xpoints = 10
params["xpoints"] = xpoints
L = 10.
params["L"] = L
dt = 0.005
params["dt"] = dt
nu = 0.02923
params["nu"] = nu
dc = 2.5
params["dc"] = dc

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True,
               dt=dt, nu=nu, dc=dc, L=L)
model.setup()
X_start = model.allocate()
u_in = X_start[0] # u_initilization
x, = fd.SpatialCoordinate(model.mesh)

# Setup initilization
u_in.project(0.2*2/(fd.exp(x-403./15.) + fd.exp(-x+403./15.)) + 0.5*2/(fd.exp(x-203./15.)+fd.exp(-x+203./15.)))


print("Finding an initial state.")
for i in fd.ProgressBar("").iter(range(200)):
    model.randomize(X_start)
    model.run(X_start, X_start)  # run method for every time step
    u = fd.Function(model.Vdg)
    u.rename('state')
    u.interpolate(model.un)
    truth_init.write(u)

print("generating ensemble.")

Nensemble = 90  # size of the ensemble

spread_steps = math.ceil(4./dt/nsteps)



X = model.allocate()
CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
uout = fd.Function(CG2, name="u")
ndump = 20  # dump data
p_dump = 0

y_true = model.obs().dat.data[:]
particle_in = np.zeros((y_true.size, Nensemble+1))


with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", 'w') as afile:
    afile.save_mesh(model.mesh)

    
    X[0].assign(X_start[0])
    for step in fd.ProgressBar("").iter(range(spread_steps)):
        model.randomize(X)
        model.run(X, X)
    for i in range(Nensemble+1):
        if i < Nensemble:
            print("Generating ensemble member", i)
        else:
            print("Generating 'true' value")
        particle_in[:,i] = model.obs().dat.data[:]
        u = fd.Function(model.Vdg)
        u.rename('state')
        u.interpolate(model.un)
        particle_init.write(u)

        uout.interpolate(X[0])
        afile.save_function(uout, idx=i)
    np.save("../../DA_KS/particle_in.npy", particle_in)

print("Generating the observational data.")
N_obs = 100
params["N_obs"] = N_obs


y_true_full = np.zeros((N_obs, y_true.size))
y_obs_full = np.zeros((N_obs, y_true.size))

noise_var = 1.5**2
params["noise_var"] = noise_var

for i in fd.ProgressBar("").iter(range(N_obs)):
    model.randomize(X)
    model.run(X, X)  # run method for every time step
    u = fd.Function(model.Vdg)
    u.rename('state')
    u.interpolate(model.un)
    truth.write(u)
    y_true = model.obs().dat.data[:]
    y_true_full[i, :] = y_true
    y_max = np.abs(y_true).max()
    y_true_data = np.save("y_true.npy", y_true_full)
    y_noise = np.random.normal(0.0, noise_var**0.5)
    y_obs = y_true + y_noise
    y_obs_full[i,:] = y_obs
np.save("../../DA_KS/y_true.npy", y_true_full)
np.save("../../DA_KS/y_obs.npy", y_obs_full)

import pickle
with open('params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
