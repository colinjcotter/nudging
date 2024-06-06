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

X_truth = model.allocate()
#u0 = X_truth[0]

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

def sample_initial_dist(i):
    #i = random.randint(1, 100)
    with CheckpointFile("initial_sol_mixing.h5", 'r') as afile:
        u0_read = afile.load_function(model.mesh, name="u_out", idx=i*2000)
    return u0_read

rho = 0.9998
N_steps = 50

V = FunctionSpace(model.mesh, 'CG', 1)
proposal = Function(V)

X_truth[0] = sample_initial_dist(1)

for m in range(1, N_steps):
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    y_true = model.obs().dat.data[:]

    prop = rho * X_truth[0] + (1-rho**2)**(0.5) * sample_initial_dist(m+1)
    proposal.assign(prop)
    #acceptance step
    prob = min(1, math.exp(log_likelihood(y_true, proposal)) / math.exp(log_likelihood(y_true, X_truth[0])))
    if random.random() > prob:
        X_truth[0] = proposal

#burnout = 100
#store the solutions in a checkpoint file
with CheckpointFile("mcmc_functions.h5", 'w') as afile:
    afile.save_mesh(model.mesh)
    afile.save_function(X_truth[0]) #compare this with the initial output of the particle filter
