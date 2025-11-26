import numpy as np
import matplotlib.pyplot as plt

u_e = np.load("Velocity_ensemble.npy")
u_obs_alltime = np.load("Velocity simulated_all_time.npy")
n_ensemble = np.shape(u_e)[0]
u_alltime = np.transpose(u_obs_alltime, (1, 0, 2, 3))

xi = 21

plt.plot(u_alltime[:, :, xi, 1], "g-")
plt.show()
