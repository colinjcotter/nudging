import firedrake as fd
import nudging as ndg
import numpy as np

# create some synthetic data/observation data at T_1 ---- T_Nobs
# Pick initial conditon
# run model, get obseravation
# add observation noise N(0, sigma^2)

nsteps = 5
xpoints = 40
model = ndg.KS(n, nsteps, xpoints, seed=12353, lambdas=False,
               dt=0.01, nu=0.02923, dc=0.01, L=10.)
model.setup()
X_start = model.allocate()
u = X_start[0]
x, = fd.SpatialCoordinate(model.mesh)
pi = fd.pi
sin = fd.sin
cos = fd.cos
u.project(sin(pi*2*x) + 0.2*cos(pi*x))

for i in ProgressBar.iter(range(200)):
    model.randomize(X_start)
    model.run(X_start, X_start)  # run method for every time step
