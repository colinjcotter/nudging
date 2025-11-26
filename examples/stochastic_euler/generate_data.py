import firedrake as fd
import numpy as np
from nudging.models.stochastic_euler import Euler_SD

# create some synthetic data/observation data at T_1 ---- T_Nobs
# Pick initial conditon
# run model, get true value and obseravation and use paraview for viewing
# add observation noise N(0, sigma^2)

n = 8
nsteps = 5
model = Euler_SD(n, nsteps=nsteps)
model.setup()
X_truth = model.allocate()
q0 = X_truth[0]
x = fd.SpatialCoordinate(model.mesh)
sin = fd.sin
pi = fd.pi
cos = fd.cos
q0.interpolate(
    sin(8 * pi * x[0]) * sin(8 * pi * x[1])
    + 0.4 * cos(6 * pi * x[0]) * cos(6 * pi * x[1])
    + 0.02 * sin(2 * pi * x[0])
    + 0.02 * sin(2 * pi * x[1])
    + 0.3 * cos(10 * pi * x[0]) * cos(4 * pi * x[1])
)

N_obs = 5

model.randomize(X_truth)  # populating noise term with PV
model.run(X_truth, X_truth)  # use noise term to solve for PV
u_true_VOM = model.obs()  # use PV to get streamfunction and velocity
u_true = u_true_VOM.dat.data[:]
u1_true = u_true[:, 0]
u2_true = u_true[:, 1]

u1_true_all = np.zeros((N_obs, np.size(u1_true)))
u2_true_all = np.zeros((N_obs, np.size(u2_true)))
u1_obs_all = np.zeros((N_obs, np.size(u1_true)))
u2_obs_all = np.zeros((N_obs, np.size(u2_true)))


# simulated observations
for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    u_VOM = model.obs()

    u = u_VOM.dat.data[:]

    u1_true_all[i, :] = u[:, 0]
    u2_true_all[i, :] = u[:, 1]

    u_1_noise = np.random.normal(0.0, 0.05, (n + 1) ** 2)
    u_2_noise = np.random.normal(0.0, 0.05, (n + 1) ** 2)
    u1_obs = u[:, 0] + u_1_noise
    u2_obs = u[:, 1] + u_2_noise
    u1_obs_all[i, :] = u1_obs
    u2_obs_all[i, :] = u2_obs

u_true_all = np.stack((u1_true_all, u2_true_all), axis=-1)
u_obs_all = np.stack((u1_obs_all, u2_obs_all), axis=-1)

np.save("u_true_data.npy", u_true_all)
np.save("u_obs_data.npy", u_obs_all)
