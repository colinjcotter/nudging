"""Calibration of the SALT field: state-dependent deterministic part + AR(1) noise for the rest.

zeta_salt^m (zeta_salt_fit.npy from one_step_error.py) is the model-ready dU_2 (noise_scale = 1, dt = window) that
best removes the one-step closure error of window m by transport. Only the windows before n_train are used.
Deterministic part: d^m = k c psi_bar^m + mean of (zeta_salt - k c psi_bar), c the least-squares fit
zeta_salt ~ c psi_bar, k = 2 (best scale in out-of-sample tests). The rest zeta_salt - d is modelled as time-mean
pattern (about 0) + K EOF modes with AR(1) amplitudes, per-mode tau from a fit of log acf over the lags with
acf > acf_min, phi_k = exp(-dt/tau_k).
Printed: variance shares, modes for 50/70/90/99%, tau per mode, rho(lag) data | data K modes | sample, rms.

    python salt_generator_psi.py [outname data_name n_coarse dt_window K]
    (defaults: ell_2hc64 data_run128_long 64 0.08 100; n_train = 700, k = 2)

Reads zeta_salt_fit.npy, windows_salt.npy, nodes.npy of <data_name>/displacements_<outname>_w4/ (the windows
must be consecutive: one_step_error.py with stride 1) and psi_bar_start.npy. Saves noise_model_salt_psi.npz there
(mean_grid, eofs_grid on the (n+1) x (n+1) grid with zero walls, lam, tau, phi, window_length, K, c, k,
mean_rest_grid, n_train).
"""
import sys
import numpy as np

outname = "ell_2hc64"
data_name = "data_run128_long"
n_coarse = 64
window_length = 0.08
K = 100
if len(sys.argv) > 5:
    outname = sys.argv[1]
    data_name = sys.argv[2]
    n_coarse = int(sys.argv[3])
    window_length = float(sys.argv[4])
    K = int(sys.argv[5])
folder = "/home/mkumar/DA_Results/noise_calibration/" + data_name + "/displacements_" + outname + "_w4"
maxlag = 300
nsample = 20000
acf_min = 0.2
n_train = 700
k_scale = 2.0

zs = np.load(folder + "/zeta_salt_fit.npy")
windows = np.load(folder + "/windows_salt.npy")
nodes = np.load(folder + "/nodes.npy")
psi_bar = np.load(folder + "/psi_bar_start.npy")
zs = zs[windows < n_train]
psi_train = psi_bar[windows[windows < n_train]]
windows = windows[windows < n_train]
c = np.sum(zs*psi_train)/np.sum(psi_train**2)
mean_rest = (zs - k_scale*c*psi_train).mean(axis=0)
flat = zs - k_scale*c*psi_train - mean_rest
print("c =", c, " k =", k_scale, " rms zeta_salt:", np.sqrt(np.mean(zs**2)), " rms of the rest:", np.sqrt(np.mean(flat**2)))
nwindows = flat.shape[0]
print("folder:", folder, " windows:", nwindows, " consecutive:", bool(np.all(np.diff(windows) == 1)), " K:", K)
ix = np.rint(nodes[:, 0]*n_coarse).astype(int)
iy = np.rint(nodes[:, 1]*n_coarse).astype(int)

# ---- mean pattern, EOFs, amplitudes ----
mean_pattern = flat.mean(axis=0)
f = flat - mean_pattern
u, s, vt = np.linalg.svd(f, full_matrices=False)
amp = u[:, :K]*s[:K]
lam = np.sum(amp**2, axis=0)/nwindows
share = s**2/np.sum(s**2)
cumulative = np.cumsum(share)
print("rms of zeta_salt:", np.sqrt(np.mean(flat**2)), " of its mean pattern:", np.sqrt(np.mean(mean_pattern**2)),
      " of the fluctuation:", np.sqrt(np.mean(f**2)))
print("share of the energy in the mean pattern:", np.sum(mean_pattern**2)*nwindows/np.sum(flat**2))
print("modes 1 to 3:", share[0:3], " modes for 50%, 70%, 90%, 99%:", np.searchsorted(cumulative, 0.5) + 1,
      np.searchsorted(cumulative, 0.7) + 1, np.searchsorted(cumulative, 0.9) + 1, np.searchsorted(cumulative, 0.99) + 1)
print("variance share of the first K modes:", cumulative[K - 1])

# ---- tau overall and per mode ----
rho_all = np.zeros(maxlag + 1)
for lag in range(maxlag + 1):
    rho_all[lag] = np.sum(f[:nwindows - lag]*f[lag:])/np.sum(f[:nwindows - lag]**2)
nfit = np.argmax(rho_all[1:] < acf_min)
lk = np.arange(1, nfit + 1)
tau_all = -window_length/(np.sum(lk*np.log(rho_all[lk]))/np.sum(lk**2))
print("overall tau:", tau_all, " rho at lag 1:", rho_all[1])

tau = np.zeros(K)
n_fallback = 0
for k in range(K):
    a = amp[:, k]
    acf = np.zeros(maxlag + 1)
    for lag in range(maxlag + 1):
        acf[lag] = np.sum(a[:nwindows - lag]*a[lag:])/np.sum(a[:nwindows - lag]**2)
    if np.any(acf[1:] < acf_min):
        nfit_k = np.argmax(acf[1:] < acf_min)
    else:
        nfit_k = maxlag
    if nfit_k >= 2:
        lk = np.arange(1, nfit_k + 1)
        tau[k] = -window_length/(np.sum(lk*np.log(acf[lk]))/np.sum(lk**2))
    else:
        tau[k] = tau_all
        n_fallback += 1
phi = np.exp(-window_length/tau)
print("tau per mode: modes 1 to 6:", tau[0:6], " median:", np.median(tau), " min:", tau.min(), " max:", tau.max())
print("modes that used the overall tau (fit failed):", n_fallback, "of", K)

# ---- sample ----
rng = np.random.default_rng(123)
a_s = np.zeros((nsample, K))
a_s[0] = np.sqrt(lam)*rng.standard_normal(K)
for n in range(nsample - 1):
    a_s[n + 1] = phi*a_s[n] + np.sqrt(lam*(1.0 - phi**2))*rng.standard_normal(K)
print()
print("variance per mode, sample / data:")
for k in [0, 1, 2, 4, 9, 19, K - 1]:
    print("   mode", k + 1, " lambda =", lam[k], " sample/data =", a_s[:, k].var()/lam[k])
print()
print("rho(lag) of the fluctuation: data all modes | data K modes | sample")
for lag in [1, 2, 5, 10, 20, 30, 50, 100]:
    rho_K = np.sum(amp[:nwindows - lag]*amp[lag:])/np.sum(amp[:nwindows - lag]**2)
    rho_s = np.sum(a_s[:nsample - lag]*a_s[lag:])/np.sum(a_s[:nsample - lag]**2)
    print("   lag", lag, " time", lag*window_length, ":", rho_all[lag], rho_K, rho_s)

# ---- save on the grid ----
mean_grid = np.zeros((n_coarse + 1, n_coarse + 1))
mean_grid[ix, iy] = mean_pattern
eofs_grid = np.zeros((K, n_coarse + 1, n_coarse + 1))
eofs_grid[:, ix, iy] = vt[:K]
mean_rest_grid = np.zeros((n_coarse + 1, n_coarse + 1))
mean_rest_grid[ix, iy] = mean_rest
np.savez(folder + "/noise_model_salt_psi.npz", mean_grid=mean_grid, eofs_grid=eofs_grid, lam=lam, tau=tau, phi=phi,
         window_length=window_length, K=K, c=c, k=k_scale, mean_rest_grid=mean_rest_grid, n_train=n_train)
print()
print("saved", folder + "/noise_model_salt_psi.npz")
