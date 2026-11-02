"""Freeze calibration version 1 (plan §4, V1): one file with everything run_ensemble_v1.py needs.

V1 = SALT field added to the transport of the coarse model (noise_source='external', noise_scale = 1, dt = window):
    dU_2^n = k c psi_model^n + mean_rest + sum_j a_j^n EOF_j
  deterministic part: k c psi_model (recomputed every step from the model's own psi) + mean_rest (fixed pattern)
  fluctuations:       a_j^{n+1} = phi_j a_j^n + sqrt(lam_j (1 - phi_j^2)) N(0, 1),  a_j^0 ~ N(0, lam_j)
Fitted on windows 0 to n_train - 1 of the 128 -> 64 data (ell = 2 h_c = 1/32, window 0.08) by one_step_error.py
(zeta_salt, eps_rel 1e-2, sign -1) and salt_generator_psi.py (k = 2, K = 100, acf_min 0.2). Nothing is refitted
here: noise_model_salt_psi.npz is copied and the metadata added.

Saved: ~/DA_Results/noise_calibration/v1/noise_model_v1.npz
  c, k_scale, mean_rest_grid, eofs_grid, lam, tau, phi, K   (grids (n_coarse + 1)^2, zero on the walls)
  ix, iy        grid index of each CG1 node of the coarse model (node order of the serial UnitSquareMesh)
  n_coarse, n_fine, ell, window_length, n_train, data_folder

    python freeze_v1.py

One MPI rank, numpy only.
"""
import os
import numpy as np

n_coarse = 64
n_fine = 128
ell = 1.0/32.0
data_folder = "/home/mkumar/DA_Results/noise_calibration/data_run128_long/displacements_ell_2hc64_w4"
outdir = "/home/mkumar/DA_Results/noise_calibration/v1"
os.makedirs(outdir, exist_ok=True)

noise = np.load(data_folder + "/noise_model_salt_psi.npz")
nodes = np.load(data_folder + "/nodes.npy")
ix = np.rint(nodes[:, 0]*n_coarse).astype(int)
iy = np.rint(nodes[:, 1]*n_coarse).astype(int)

print("c =", float(noise["c"]), " k =", float(noise["k"]), " n_train =", int(noise["n_train"]), " K =", int(noise["K"]),
      " window =", float(noise["window_length"]))
print("rms of the mean pattern of the fluctuations (should be ~0):", np.sqrt(np.mean(noise["mean_grid"]**2)))
print("rms mean_rest:", np.sqrt(np.mean(noise["mean_rest_grid"][ix, iy]**2)), " tau modes 1 to 3:", noise["tau"][0:3])

np.savez(outdir + "/noise_model_v1.npz", c=float(noise["c"]), k_scale=float(noise["k"]),
         mean_rest_grid=noise["mean_rest_grid"], eofs_grid=noise["eofs_grid"], lam=noise["lam"], tau=noise["tau"],
         phi=noise["phi"], K=int(noise["K"]), ix=ix, iy=iy, n_coarse=n_coarse, n_fine=n_fine, ell=ell,
         window_length=float(noise["window_length"]), n_train=int(noise["n_train"]), data_folder=data_folder)
print("saved", outdir + "/noise_model_v1.npz")
