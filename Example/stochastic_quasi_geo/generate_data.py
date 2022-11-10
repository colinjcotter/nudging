from ctypes import sizeof
from fileinput import filename
from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt

from nudging.Quasi_Geostrophic_solve import QuasiGeostrophic
#from nudging import Quasi_Geostrophic_solve
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
#np.random.seed(138)


model = QuasiGeostrophic(40)
X_truth = model.allocate()
q0 = X_truth
x = SpatialCoordinate(model.mesh)
q0.interpolate(0.1*sin(x[0])*sin(x[1]))

dt = 0.1
nsteps = 5 
N_obs = 10
DT = nsteps*dt 
nobs_step = nsteps*N_obs 

# Exact numerical approximation 
q_true = model.obs(X_truth)
q_obs_full = np.zeros((N_obs, np.size(q_true)))
q_true_full = np.zeros((N_obs, np.size(q_true)))

for i in range(N_obs):
    W_truth = np.random.randn(nsteps, 10)
    model.run(nsteps, W_truth, X_truth, X_truth)
    q_true = model.obs(X_truth)

    q_true_full[i,:] = q_true
    
    #print(np.shape(q_true_data))
    #plt.plot(q_true_data[5,:], label='q_true')
    q_true_txt = np.savetxt("q_true.txt", q_true_full)
    q_true_data = np.save("q_true.npy", q_true_full)
    # Fix the simailrity with q_true
    q_noise = np.random.normal(0.1, 0.1, 40)  

    q_obs = q_true + q_noise
    
    q_obs_full[i,:] = q_obs 

    q_obsdata = np.save("q_obs.npy", q_obs_full)

q_num_exct = np.load('q_true.npy')
q_obs = np.load('q_obs.npy')
# plt.plot(q_num_exct[5,:], 'r-', label='q_true')
# plt.plot(q_obs[5,:], 'b-', label='q_obs')
# plt.legend()
# plt.show()