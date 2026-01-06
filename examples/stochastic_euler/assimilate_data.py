import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc

# read obs from saved file
# Do assimilation step for tempering and jittering steps

n = 4
nsteps = 10
model = ndg.Euler_SD(n, 10)
bfilter = ndg.bootstrap_filter()

nensemble = [2, 2, 2, 2, 2]
bfilter.setup(nensemble, model)
x = fd.SpatialCoordinate(model.mesh)

sin = fd.sin
for i in range(nensemble[bfilter.ensemble_rank]):
    a = model.rg.uniform(model.R, 0.0, 1.0)
    q0_in = 0.1 * (1 + a) * sin(x[0]) * sin(x[1])
    q = bfilter.ensemble[i][0]
    q.interpolate(q0_in)


def log_likelihood(y, Y):
    ll = 0.5 * (1 / 0.05**2) * ((y[0] - Y[0]) ** 2 + (y[1] - Y[1]) ** 2) * fd.dx
    return ll


# #Load data
u1_exact = np.load("u_true_1.npy")
u2_exact = np.load("u_true_2.npy")
u_vel_1 = np.load("u_obs_true_1.npy")
u_vel_2 = np.load("u_obs_true_2.npy")

N_obs = u_vel_1.shape[0]

u1_VOM = fd.Function(model.VVOM)
u2_VOM = fd.Function(model.VVOM)
u_VOM = [u1_VOM]
u_VOM.append(u2_VOM)

# prepare shared arrays for data
u1_e_list = []
u2_e_list = []
for m in range(u_vel_1.shape[1]):
    u1_e_shared = ndg.SharedArray(
        partition=nensemble, comm=bfilter.subcommunicators.ensemble_comm
    )
    u2_e_shared = ndg.SharedArray(
        partition=nensemble, comm=bfilter.subcommunicators.ensemble_comm
    )
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)


ushape = u_vel_1.shape
if fd.COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))


# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    u_VOM[0].dat.data[:] = u_vel_1[k, :]
    u_VOM[1].dat.data[:] = u_vel_2[k, :]
    bfilter.assimilation_step(u_VOM, log_likelihood)

    for i in range(nensemble[bfilter.ensemble_rank]):
        model.q0.assign(bfilter.ensemble[i][0])
        obsdata_1 = model.obs()[0].dat.data[:]
        obsdata_2 = model.obs()[1].dat.data[:]
        for m in range(u_vel_1.shape[1]):
            u1_e_list[m].dlocal[i] = obsdata_1[m]
            u2_e_list[m].dlocal[i] = obsdata_2[m]

    for m in range(u_vel_1.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u1_e_list[m].data()

if fd.COMM_WORLD.rank == 0:
    np.save("Velocity_1_ensemble_simulated_obs.npy", u1_e)
    np.save("Velocity_2_ensemble_simulated_obs.npy", u2_e)
