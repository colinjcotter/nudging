import firedrake as fd
import nudging as ndg
import numpy as np

# create some synthetic data/observation data at T_1 ---- T_Nobs
# Pick initial conditon
# run model, get obseravation
# add observation noise N(0, sigma^2)

nsteps = 5
xpoints = 40
model = ndg.Camsholm(100, nsteps, xpoints)
model.setup()
X_truth = model.allocate()
_, u0 = X_truth[0].split()
(x,) = fd.SpatialCoordinate(model.mesh)
exp = fd.exp
u0.interpolate(
    0.2 * 2 / (exp(x - 403.0 / 15.0) + exp(-x + 403.0 / 15.0))
    + 0.5 * 2 / (exp(x - 203.0 / 15.0) + exp(-x + 203.0 / 15.0))
)
N_obs = 50

y_true = model.obs().dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth)  # run method for every time step
    y_true = model.obs().dat.data[:]
    y_true_full[i, :] = y_true
    y_true_data = np.save("y_true.npy", y_true_full)
    y_noise = np.random.normal(0.0, 0.05, xpoints)
    y_obs = y_true + y_noise
    y_obs_full[i, :] = y_obs

# need to save into rank 0
y_obsdata = np.save("y_obs.npy", y_obs_full)
