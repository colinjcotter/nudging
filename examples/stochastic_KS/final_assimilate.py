import firedrake as fd
import nudging as ndg
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP

import pickle

with open("params.pickle", "rb") as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]
# dc = 2.5

nensemble = [1] * 30

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True, dt=dt, nu=nu, dc=dc, L=L)

nudging = True
jtfilter = ndg.jittertemp_filter(
    n_jitt=5,
    delta=0.15,
    verbose=2,
    MALA=False,
    visualise_tape=False,
    nudging=nudging,
    sigma=0.01,
)
# jtfilter = ndg.bootstrap_filter(verbose=2)


jtfilter.setup(nensemble, model)


# load mesh
with fd.CheckpointFile(
    "../../DA_KS/ks_ensemble.h5", "r", comm=jtfilter.subcommunicators.comm
) as afile:
    mesh = afile.load_mesh("ksmesh")

comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm
# load the initial ensemble
erank = jtfilter.ensemble_rank
filename = f"../../DA_KS/checkpoint_files/ensemble_temp{erank}.h5"
with fd.CheckpointFile(filename, "r", comm=jtfilter.subcommunicators.comm) as afile:
    for i in range(nensemble[erank]):
        u = jtfilter.new_ensemble[i][0]
        u0 = afile.load_function(mesh, "u", idx=i)  # read locally
        u.interpolate(u0)


def log_likelihood(y, Y):
    ll = (y - Y) ** 2 / 5.5**2 / 2 * fd.dx
    return ll


# Load data
y_exact = np.load("../../DA_KS/y_true.npy")
y = np.load("../../DA_KS/y_obs.npy")

PETSc.Sys.Print("shape of y", y.shape)
N_obs = y.shape[0]
# N_obs = 1

yVOM = fd.Function(model.VVOM)
# print('yvomShpe', yVOM.dat.data.shape)

# prepare shared arrays for data
y_e_list = []
y_sim_list = []
for m in range(y.shape[0]):
    y_e_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    y_e_list.append(y_e_shared)
    y_sim_list.append(y_e_shared)

ys = y.shape
if fd.COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    y_sim = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# diagnostic


class samples(ndg.base_diagnostic):
    def compute_diagnostic(self, particle):
        model.un.assign(particle[0])
        return model.obs().dat.data[0]


nolambdasamples = samples(
    ndg.Stage.WITHOUT_LAMBDAS, jtfilter.subcommunicators, nensemble
)

resamplingsamples = samples(
    ndg.Stage.AFTER_ASSIMILATION_STEP, jtfilter.subcommunicators, nensemble
)
nudgingsamples = samples(ndg.Stage.AFTER_NUDGING, jtfilter.subcommunicators, nensemble)
aftertempering = samples(
    ndg.Stage.AFTER_TEMPER_RESAMPLE, jtfilter.subcommunicators, nensemble
)
afterjittering = samples(
    ndg.Stage.AFTER_ONE_JITTER_STEP, jtfilter.subcommunicators, nensemble
)

diagnostics = [aftertempering, afterjittering, resamplingsamples]

# diagnostics = [resamplingsamples]
tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-2,
    "tao_grtol": 1.0e-3,
    "tao_gttol": 1.0e-3,
}


N_obs = 1
# do assimiliation step
# PETSc.Sys.Print('assimilation step')
for k in range(N_obs):
    PETSc.Sys.Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k, :]

    # # # actually do the data assimilation step
    jtfilter.assimilation_step(
        yVOM,
        log_likelihood,
        diagnostics=diagnostics,
        ess_tol=-990.8,
        taylor_test=False,
        tao_params=tao_params,
    )

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.un.assign(jtfilter.ensemble[i][0])
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
        np.save("../../DA_KS/final_tempjitt_assimilated_ensemble.npy", y_e)
    if nudging:
        np.save("../../DA_KS/final_nudge_assimilated_ensemble.npy", y_e)
