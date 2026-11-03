"""Freeze calibration version 2 (V2 = V1 + non-transport part, variant C of nt_onestep.py): one file with everything
the coarse model needs.

V2 = V1 SALT field in the transport + an additive increment of q per window (Euler_mixSD q_source=True, r = 0.01):
    dU_2^n         = k c psi_model^n + mean_rest + sum_j a_j^n EOF_j            (V1, unchanged)
    q_source_fn^n  = source_det + sum_j b_j^n EOFr_j
  source_det: time mean of R_nt (one-step error of V1 minus its transport fit, nontransport.py) over the fit windows
              + time mean of the rest after it (nt_rest.py C)
  b_j^{n+1} = phir_j b_j^n + sqrt(lamr_j (1 - phir_j^2)) N(0, 1),  b_j^0 ~ N(0, lamr_j)
Fitted on windows 0 to n_train - 1, for dt = window = 0.08 (one model step per window) on 64^2. Nothing is refitted
here: noise_model_v1.npz, R_nt_v1.npy, windows_v1.npy and noise_model_v2_rest_C.npz are combined.

Saved: ~/DA_Results/noise_calibration/v2/noise_model_v2.npz
  all keys of noise_model_v1.npz
  r, source_det_grid, eofs_rest_grid, lam_rest, tau_rest, phi_rest, K_rest   (grids (n_coarse + 1)^2)

    python freeze_v2.py

One MPI rank, numpy only.
"""
import os
import numpy as np

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
outdir = "/home/mkumar/DA_Results/noise_calibration/v2"
os.makedirs(outdir, exist_ok=True)

v1 = np.load(v1_folder + "/noise_model_v1.npz")
rest = np.load(v1_folder + "/noise_model_v2_rest_C.npz")
ix = v1["ix"]
iy = v1["iy"]
n_coarse = int(v1["n_coarse"])
n_train = int(v1["n_train"])
windows = np.load(v1_folder + "/windows_v1.npy")
P = np.load(v1_folder + "/R_nt_v1.npy")[windows < n_train].mean(axis=0)
bP = float(rest["bP"])
r = float(rest["r"])

source_det_grid = np.zeros((n_coarse + 1, n_coarse + 1))
source_det_grid[ix, iy] = bP*P + rest["mean_grid"][ix, iy]

print("V1: c =", float(v1["c"]), " k =", float(v1["k_scale"]), " K =", int(v1["K"]), " n_train =", n_train)
print("V2: r =", r, " bP =", bP, " K_rest =", int(rest["K"]), " n_train of the rest fit =", int(rest["n_train"]))
print("rms source_det:", np.sqrt(np.mean(source_det_grid[ix, iy]**2)), " rms P:", np.sqrt(np.mean(P**2)),
      " tau_rest modes 1 to 3:", rest["tau"][0:3])

np.savez(outdir + "/noise_model_v2.npz", c=float(v1["c"]), k_scale=float(v1["k_scale"]),
         mean_rest_grid=v1["mean_rest_grid"], eofs_grid=v1["eofs_grid"], lam=v1["lam"], tau=v1["tau"],
         phi=v1["phi"], K=int(v1["K"]), ix=ix, iy=iy, n_coarse=n_coarse, n_fine=int(v1["n_fine"]),
         ell=float(v1["ell"]), window_length=float(v1["window_length"]), n_train=n_train,
         data_folder=str(v1["data_folder"]), r=r, source_det_grid=source_det_grid, eofs_rest_grid=rest["eofs_grid"],
         lam_rest=rest["lam"], tau_rest=rest["tau"], phi_rest=rest["phi"], K_rest=int(rest["K"]))
print("saved", outdir + "/noise_model_v2.npz")
