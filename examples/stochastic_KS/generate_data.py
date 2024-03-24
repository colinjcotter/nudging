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
nsteps = 5
xpoints = 40
model = KS(100, nsteps, xpoints)
model.setup(nu=0.01)
X_truth = model.allocate()
u0 = X_truth[0]
x, = SpatialCoordinate(model.mesh)
u0.project(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

Q = FunctionSpace(model.mesh, 'CG', 1)
u_out = Function(Q)

N_obs = 50

y_true = model.obs().dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

file0 = File('true.pvd')
u_out.interpolate(X_truth[0])
file0.write(u_out)

for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    y_true = model.obs().dat.data[:]

    u_out.interpolate(X_truth[0])
    file0.write(u_out)


    y_true_full[i,:] = y_true
    y_true_data = np.save("y_true.npy", y_true_full)

    y_noise = np.random.normal(0.0, 0.05, xpoints)

    y_obs = y_true + y_noise

    y_obs_full[i,:] = y_obs

# need to save into rank 0
y_obsdata = np.save("y_obs.npy", y_obs_full)
