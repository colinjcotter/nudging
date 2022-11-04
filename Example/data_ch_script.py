from ctypes import sizeof
from fileinput import filename
from firedrake import *
from nudging import *
import numpy as np
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
#np.random.seed(138)

model = Camsholm(100)
X_truth = model.allocate()
_, u0 = X_truth.split()
x, = SpatialCoordinate(model.mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

dt = 0.01
nsteps = 5 
N_obs = 5
DT = nsteps*dt 
nobs_step = nsteps*N_obs 

y_true = model.obs(X_truth)
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

for i in range(N_obs):
    W_truth = np.random.randn(nsteps, 10)
    model.run(nsteps, W_truth, X_truth, X_truth)
    y_true = model.obs(X_truth)

    y_true_full[i,:] = y_true
    y_true_data = np.save("y_true.npy", y_true_full)

    y_noise = np.random.normal(0.0, 0.05, 40)  

    y_obs = y_true + y_noise
    
    y_obs_full[i,:] = y_obs 

    y_obsdata = np.save("y_obs.npy", y_obs_full)