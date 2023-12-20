from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

from nudging.models.stochastic_KS import KS

""" read obs from saved file
    Do assimilation step for tempering and jittering steps
"""

nsteps = 5
xpoints = 40
model = KS(100, nsteps, xpoints)
MALA = False
verbose = True
jtfilter = jittertemp_filter(n_jitt=4, verbose=verbose, MALA=MALA)

nensemble = [5,5,5,5]
jtfilter.setup(nensemble, model)

x, = SpatialCoordinate(model.mesh)

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, 0., 0.05)
    dx1 = model.rg.normal(model.R, 0., 0.05)
    a = model.rg.uniform(model.R, 0., 1.0)
    b = model.rg.uniform(model.R, 0., 1.0)
    u0_exp = (1+a)*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
        + (1+b)*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))

    _, u = jtfilter.ensemble[i][0].split()
    u.interpolate(u0_exp)


def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

#Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy')
N_obs = y.shape[0]

yVOM = Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
for m in range(y.shape[1]):
    y_e_shared = SharedArray(partition=nensemble,
                                  comm=jtfilter.subcommunicators.ensemble_comm)
    y_sim_obs_shared = SharedArray(partition=nensemble,
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    y_e_list.append(y_e_shared)
    y_sim_obs_list.append(y_sim_obs_shared)

ys = y.shape
if COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_e_asmfwd = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim_obs_alltime_step = np.zeros((np.sum(nensemble),nsteps,  ys[1]))
    y_sim_obs_allobs_step = np.zeros((np.sum(nensemble),nsteps*N_obs,  ys[1]))

mylist = []
def mycallback(ensemble):
   xpt = np.arange(0.5,40.0) # need to change the according to generate data
   X = ensemble[0]
   mylist.append(X.at(xpt))

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[k, :]

    # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood)


    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

if COMM_WORLD.rank == 0:
    print("Time shape", y_sim_obs_alltime_step.shape)
    print("Obs shape", y_sim_obs_allobs_step.shape)
    print("Ensemble member", y_e.shape)
    np.save("assimilated_ensemble.npy", y_e)
    np.save("simualated_all_time_obs.npy", y_sim_obs_allobs_step)
