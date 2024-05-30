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

def sample_initial_dist(i):
    #i = random.randint(1, 100)
    with CheckpointFile("initial_sol_mixing.h5", 'r') as afile:
        u0_read = afile.load_function(model.mesh, name="u_out", idx=i*2000)
    return u0_read

rho = 0.9998
N_steps = 100

V = FunctionSpace(model.mesh, 'CG', 1)
proposal = Function(V)
u = sample_initial_dist(1)

#Load data
y = np.load('y_obs.npy')
yVOM = Function(model.VVOM)
#yVOM = Function(V)
yVOM.dat.data[:] = y[0, :] #set at time 0


for m in range(1, N_steps):
    prop = rho * u + (1-rho**2)**(0.5) * sample_initial_dist(m+1)
    proposal.assign(prop)
    #acceptance step
    prob = min(1, math.exp(log_likelihood(yVOM, proposal)) / math.exp(log_likelihood(yVOM, u)))
    if random.random() > prob:
        u = proposal

burnout = 100
#store the solutions in a checkpoint file
with CheckpointFile("mcmc_functions.h5", 'w') as afile:
    afile.save_mesh(model.mesh)
    afile.save_function(u) #compare this with the initial output of the particle filter
