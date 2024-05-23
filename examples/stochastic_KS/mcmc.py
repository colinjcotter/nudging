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

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy')
#N_obs = y.shape[0]

def sample_initial_dist():
    i = random.randint(1, 40)
    with CheckpointFile("initial_sol_mixing.h5", 'r') as afile:
        u0_read = afile.load_function(model.mesh, name="u_out", idx=i*2000)
    return u0_read

rho = 0.9998
N_steps = 1000

u=[sample_initial_dist()]
for m in range(1, N_steps):
    proposal = rho * u[m-1] + (1-rho**2)**(0.5) * sample_initial_dist()
    #acceptance step
    prob = min(1, math.exp(log_likelihood(y_exact, proposal)) / math.exp(log_likelihood(y_exact, u[m-1])))
    if random.random() > prob:
        u.append(proposal)
    else:
        u.append(u[m-1])

burnout = 100
#store the solutions in a checkpoint file
with CheckpointFile("mcmc_functions.h5", 'w') as afile:
    afile.save_mesh(model.mesh)
for i in range(burnout, len(u)):
    with CheckpointFile("mcmc_functions.h5", 'w') as afile:
        afile.save_function(u[i])
