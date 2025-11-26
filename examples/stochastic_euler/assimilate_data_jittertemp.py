import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc

# read obs from saved file
# Do assimilation step for tempering and jittering steps

n = 8
nsteps = 5
model = ndg.Euler_SD(n, nsteps=nsteps)

MALA = False
verbose = True
jtfilter = ndg.jittertemp_filter(
    n_temp=4, n_jitt=4, rho=0.99, verbose=verbose, MALA=MALA
)
# Load data
u_exact = np.load("u_true_data.npy")
u_vel = np.load("u_obs_data.npy")

nensemble = [5, 5, 5, 5]

jtfilter.setup(nensemble, model)

# prepare the initial ensemble
x = fd.SpatialCoordinate(model.mesh)
sin = fd.sin
pi = fd.pi
cos = fd.cos
for i in range(nensemble[jtfilter.ensemble_rank]):
    a = model.rg.uniform(model.R, 0.0, 1.0)
    b = model.rg.uniform(model.R, 0.0, 1.0)
    q0_in = (
        a * sin(8 * pi * x[0]) * sin(8 * pi * x[1])
        + 0.4 * b * cos(6 * pi * x[0]) * cos(6 * pi * x[1])
        + 0.02 * a * sin(2 * pi * x[0])
        + 0.02 * a * sin(2 * pi * x[1])
        + 0.3 * b * cos(10 * pi * x[0]) * cos(4 * pi * x[1])
    )
    q = jtfilter.ensemble[i][0]
    q.interpolate(q0_in)


def log_likelihood(y, Y):
    ll = (y - Y) ** 2 / 0.05**2 / 2 * fd.dx
    return ll


N_obs = u_vel.shape[0]

# VVOM Function
u_VOM = fd.Function(model.VVOM)

# prepare shared arrays for data
u1_e_list = []
u2_e_list = []
u1_sim_list = []
u2_sim_list = []

ecomm = jtfilter.subcommunicators.ensemble_comm
for m in range(u_vel.shape[1]):
    u1_e_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    u2_e_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    u1_sim_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    u2_sim_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)
    u1_sim_list.append(u1_sim_shared)
    u2_sim_list.append(u2_sim_shared)

ushape = u_vel.shape
if fd.COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u1_sim_obs_alltime_step = np.zeros((np.sum(nensemble), nsteps, ushape[1]))
    u2_sim_obs_alltime_step = np.zeros((np.sum(nensemble), nsteps, ushape[1]))
    u1_sim_obs_allobs_step = np.zeros(
        (np.sum(nensemble), nsteps * ushape[0], ushape[1])
    )
    u2_sim_obs_allobs_step = np.zeros(
        (np.sum(nensemble), nsteps * ushape[0], ushape[1])
    )

# do assimiliation steps
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    # this won't work in space parallel
    u_VOM.dat.data[:, 0] = u_vel[k, :, 0]
    u_VOM.dat.data[:, 1] = u_vel[k, :, 1]

    # make a copy so that we don't overwrite the initial condition
    # in the next step
    for i in range(nensemble[jtfilter.ensemble_rank]):
        for p in range(len(jtfilter.new_ensemble[i])):
            jtfilter.new_ensemble[i][p].assign(jtfilter.ensemble[i][p])
        model.randomize(jtfilter.new_ensemble[i])

    # print simulated data for velocity
    for step in range(nsteps):
        for i in range(nensemble[jtfilter.ensemble_rank]):
            model.run(jtfilter.new_ensemble[i], jtfilter.new_ensemble[i])
            #  this won't work in space parallel
            fwd_simdata1 = model.obs().dat.data[:][:, 0]
            fwd_simdata2 = model.obs().dat.data[:][:, 1]
            for m in range(u_vel.shape[1]):
                u1_sim_list[m].dlocal[i] = fwd_simdata1[m]
                u2_sim_list[m].dlocal[i] = fwd_simdata2[m]
        # store all timesteps values
        for m in range(u_vel.shape[1]):
            u1_sim_list[m].synchronise()
            u2_sim_list[m].synchronise()
            if fd.COMM_WORLD.rank == 0:
                u1_sim_obs_alltime_step[:, step, m] = u1_sim_list[m].data()
                u1_sim_obs_allobs_step[:, nsteps * k + step, m] = (
                    u1_sim_obs_alltime_step[:, step, m]
                )
                u2_sim_obs_alltime_step[:, step, m] = u2_sim_list[m].data()
                u2_sim_obs_allobs_step[:, nsteps * k + step, m] = (
                    u2_sim_obs_alltime_step[:, step, m]
                )

    jtfilter.assimilation_step(u_VOM, log_likelihood)

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.q0.assign(jtfilter.ensemble[i][0])
        # this won't work in space parallel
        obsdata1 = model.obs().dat.data[:][:, 0]
        obsdata2 = model.obs().dat.data[:][:, 1]

        for m in range(u_vel.shape[1]):
            u1_e_list[m].dlocal[i] = obsdata1[m]
            u2_e_list[m].dlocal[i] = obsdata2[m]

    for m in range(u_vel.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()

        if fd.COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u2_e_list[m].data()

if fd.COMM_WORLD.rank == 0:
    u_e = np.stack((u1_e, u2_e), axis=-1)
    u_sim_allobs_step = np.stack(
        (u1_sim_obs_allobs_step, u2_sim_obs_allobs_step), axis=-1
    )
    np.save("Velocity_ensemble.npy", u_e)
    np.save("Velocity simalated_all_time.npy", u_sim_allobs_step)
