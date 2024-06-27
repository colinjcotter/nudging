from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat
from nudging.models.stochastic_KS import KS

nsteps = 5
xpoints = 40
model = KS(300, nsteps, xpoints)

#Load initial condition from a checkpoint file after some time idx
with CheckpointFile("initial_sol_mixing.h5", 'r') as afile:
    mesh = afile.load_mesh()
model.setup(mesh)

sample_prior = model.allocate()
sample_posterior = model.allocate()
proposal = model.allocate()

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

def sample_initial_dist(i):
    with CheckpointFile("initial_sol_mixing.h5", 'r') as afile:
        u0_read = afile.load_function(model.mesh, name="u_out", idx=i*1000)
    return u0_read

rho = 0.9998
N_steps = 1000
y = np.load('y_obs.npy')
yVOM = Function(model.VVOM)
yVOM.dat.data[:] = y[0, :] #index set at 0

f_list = []
for m in range(N_steps):
    #new sample
    sample_prior[0].assign(sample_initial_dist(m))
    model.randomize(sample_prior)
    #construct the proposal
    for i, component in enumerate(sample_posterior):
        if m == 0:
            proposal[i].assign(sample_prior[i])
        else:
            proposal[i].assign(rho * component + (1-rho**2)**(0.5) * sample_prior[i])
    #compute the log likelihood
    model.run(proposal, sample_posterior) #use sample_posterior as working memory
    simulated_obs = model.obs()
    new_likelihood = assemble(log_likelihood(yVOM, simulated_obs)) #first argument
    
    #acceptance step
    if m > 0:
        prob = min(1, math.exp(new_likelihood - old_likelihood))
        if random.random() < prob:
            for i, component in enumerate(sample_posterior):
                component.assign(proposal[i])
    old_likelihood = new_likelihood

    model.run(sample_posterior, proposal) #use proposal as working memory
    simulated_obs = model.obs()
    f_list.append(simulated_obs.dat.data[:])

stat = 0
for el in f_list:
    stat += el[0]
stat /= len(f_list)
print(stat)
