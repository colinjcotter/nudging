"""V2b: deterministic part of the non-transport one-step error R_nt (R_nt_v1.npy from nontransport.py).

Candidates, each computed from the coarse state q_bar^m only (5-point stencils on the 64^2 grid):
  damping        X = q_bar               R_nt ~ b X,  gamma = -b/dt
  diffusion      X = Lap5 q_bar          R_nt ~ b X,  nu    =  b/dt
  hyperdiffusion X = Lap5 Lap5 q_bar     R_nt ~ b X,  kappa = -b/dt
  pattern        P = time mean of R_nt over the fit windows
and the combinations pattern + one of the others (coefficients fitted together). Least squares on the windows
0 to n_train - 1, tested on n_train to the end. All sums over the nodes at least 2 away from the wall (the
hyperdiffusion stencil needs them). Printed: coefficients, share of |R_nt|^2 explained (fit and test), and the
enstrophy effect int q_bar fit / Z per window against that of R_nt (test windows).
Plan rule: if no candidate explains more than about 20% out of sample, stop V2b and report.

    python nt_fit.py

Reads R_nt_v1.npy, windows_v1.npy, noise_model_v1.npz (~/DA_Results/noise_calibration/v1/) and qbar_fine_start.npy
of its data folder. Saves nothing. numpy only.
"""
import numpy as np

v1_folder = "/home/mkumar/DA_Results/noise_calibration/v1"
v1 = np.load(v1_folder + "/noise_model_v1.npz")
ix = v1["ix"]
iy = v1["iy"]
n_coarse = int(v1["n_coarse"])
window_length = float(v1["window_length"])
n_train = int(v1["n_train"])
data_folder = str(v1["data_folder"])
h = 1.0/n_coarse

Rnt = np.load(v1_folder + "/R_nt_v1.npy")
windows = np.load(v1_folder + "/windows_v1.npy")
q = np.load(data_folder + "/qbar_fine_start.npy")[windows]
nw = Rnt.shape[0]
train = windows < n_train
test = windows >= n_train
print("windows:", nw, " fit:", int(train.sum()), " test:", int(test.sum()))

# ---- fields on the grid, inner region (nodes 2 .. n-2) ----
q_grid = np.zeros((nw, n_coarse + 1, n_coarse + 1))
q_grid[:, ix, iy] = q
r_grid = np.zeros((nw, n_coarse + 1, n_coarse + 1))
r_grid[:, ix, iy] = Rnt
lap_grid = np.zeros((nw, n_coarse + 1, n_coarse + 1))
lap_grid[:, 1:-1, 1:-1] = (q_grid[:, 2:, 1:-1] + q_grid[:, :-2, 1:-1] + q_grid[:, 1:-1, 2:] + q_grid[:, 1:-1, :-2]
                           - 4.0*q_grid[:, 1:-1, 1:-1])/h**2
bih_grid = np.zeros((nw, n_coarse + 1, n_coarse + 1))
bih_grid[:, 1:-1, 1:-1] = (lap_grid[:, 2:, 1:-1] + lap_grid[:, :-2, 1:-1] + lap_grid[:, 1:-1, 2:]
                           + lap_grid[:, 1:-1, :-2] - 4.0*lap_grid[:, 1:-1, 1:-1])/h**2
r = r_grid[:, 2:-2, 2:-2].reshape(nw, -1)
qi = q_grid[:, 2:-2, 2:-2].reshape(nw, -1)
X_damp = qi
X_diff = lap_grid[:, 2:-2, 2:-2].reshape(nw, -1)
X_hyp = bih_grid[:, 2:-2, 2:-2].reshape(nw, -1)
P = np.tile(r[train].mean(axis=0), (nw, 1))
Z = 0.5*np.sum(qi**2, axis=1)
dZ_data = np.sum(qi*r, axis=1)/Z
print("share of |R_nt|^2 in the inner region:", np.sum(r**2)/np.sum(Rnt**2))
print("enstrophy effect of R_nt per window (inner region, test windows): mean", dZ_data[test].mean())

names = ["damping", "diffusion", "hyperdiffusion", "pattern", "pattern + damping", "pattern + diffusion",
         "pattern + hyperdiffusion", "pattern + all three"]
sets = [[X_damp], [X_diff], [X_hyp], [P], [P, X_damp], [P, X_diff], [P, X_hyp], [P, X_damp, X_diff, X_hyp]]
print()
print("candidate | coefficients b | share explained: fit | test | dZ/Z of the fit (test) | physical coefficient")
for ic in range(len(sets)):
    B = np.stack(sets[ic], axis=2)
    A = np.einsum('mni,mnj->ij', B[train], B[train])
    rhs = np.einsum('mni,mn->i', B[train], r[train])
    b = np.linalg.solve(A, rhs)
    fit = B @ b
    share_fit = 1.0 - np.sum((r[train] - fit[train])**2)/np.sum(r[train]**2)
    share_test = 1.0 - np.sum((r[test] - fit[test])**2)/np.sum(r[test]**2)
    dZ_fit = np.sum(qi*fit, axis=1)/Z
    phys = ""
    if ic == 0:
        phys = "gamma = " + str(-b[0]/window_length)
    if ic == 1:
        phys = "nu = " + str(b[0]/window_length)
    if ic == 2:
        phys = "kappa = " + str(-b[0]/window_length)
    if ic == 4:
        phys = "gamma = " + str(-b[1]/window_length)
    if ic == 5:
        phys = "nu = " + str(b[1]/window_length)
    if ic == 6:
        phys = "kappa = " + str(-b[1]/window_length)
    print("   ", names[ic], "|", b, "|", share_fit, "|", share_test, "|", dZ_fit[test].mean(), "|", phys)

# reference scales: grid diffusion nu h^2-like and the damping r of the model
print()
print("for comparison: model damping r = 0.01; h^2/dt =", h**2/window_length, "; h^4/dt =", h**4/window_length)
