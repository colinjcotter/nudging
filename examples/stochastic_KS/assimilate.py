import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP

import pickle

with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]

nensemble = [10]*1

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True,
               dt=dt, nu=nu, dc=dc, L=L)

nudging = True
jtfilter = ndg.jittertemp_filter(n_jitt=0, delta=0.15,
                             verbose=2, MALA=False,
                             visualise_tape=False, nudging=True, sigma=0.01)



jtfilter.setup(nensemble, model)

comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm
# load the initial ensemble
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("ks_ensemble.h5", "r", comm=comm) as afile:
    mesh = afile.load_mesh("ksmesh")
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "u", idx=i)
        u.interpolate(u0)

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*fd.dx
    return ll


# Load data
y_exact = np.load('y_true.npy')
y = np.load('y_obs.npy')
N_obs = y.shape[0]

yVOM = fd.Function(model.VVOM)

# prepare shared arrays for data
y_e_list = []
y_sim_obs_list = []
for m in range(y.shape[1]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    y_e_list.append(y_e_shared)

ys = y.shape
if fd.COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    yVOM.dat.data[:] = y[:,k]


    # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood)

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.w0.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

if fd.COMM_WORLD.rank == 0:
    print("ensemble shape", y_e.shape)
    if not nudging:
        np.save("tempjitt_assimilated_Vorticity_ensemble.npy", y_e)
    if nudging:
        np.save("nudge_assimilated_Vorticity_ensemble.npy", y_e)