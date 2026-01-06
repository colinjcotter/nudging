from firedrake import dx
from nudging import LSDEModel, jittertemp_filter, base_diagnostic, Stage
import numpy as np
from firedrake.petsc import PETSc
import yaml
import sys
import os
from rich.table import Table

Print = PETSc.Sys.Print  # Set up printing only on the first rank
# Read the configuration from a yaml file. If a yaml file is not provided, use default settings
opts = PETSc.Options()
filename = opts.getString("-f", default=None)
final_time = opts.getReal("-T", default=1.0)
nsteps = opts.getInt("-n", default=5)
parameter_A = opts.getReal("-A", default=1.0)
parameter_D = opts.getReal("-D", default=1.0)
nranks = opts.getInt("-r", default=8)
particles_per_rank = opts.getInt("-p", default=10)
verbose = opts.getBool("-v", default=False)
if filename is not None:
    with open(filename, "r") as f:
        config = yaml.safe_load(f)
        Print("Configuration loaded from", filename)
else:
    config = {}
    config["T"] = final_time
    config["nsteps"] = nsteps
    config["A"] = parameter_A
    config["D"] = parameter_D
    config["nranks"] = nranks
    config["p_per_rank"] = particles_per_rank
    config["verbose"] = verbose
    Print("Using configuration from command line arguments or default values.")

# model
# The one-dimensional linear SDE dx = -A*x*dt + D*dW
# multiply by A and add D
T = config["T"]
nsteps = config["nsteps"]
dt = T / nsteps
A = config["A"]  # positive, constant parameter
D = config["D"]  # positive, constant parameter

# Instantiate the model
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=True, seed=7123)

p_per_rank = config["p_per_rank"]  # number of particles per rank
nranks = config["nranks"]  # total number of ranks
max_n_rank = os.cpu_count()
if nranks > max_n_rank:
    Print(
        f"Requested nranks {nranks} exceeds available cpu count {max_n_rank}, setting nranks to {max_n_rank}"
    )
    nranks = max_n_rank
nensemble = [p_per_rank] * nranks  # list with ensemble size per rank


myfilter = jittertemp_filter(
    n_jitt=0,
    delta=0.15,
    verbose=2,
    MALA=False,
    visualise_tape=False,
    nudging=False,
    sigma=0.01,
)
myfilter.setup(nensemble=nensemble, model=model, residual=False)

# data
# The ensemble is initialised as samples from N(0, D^2/(2A))
c = 0.0
d = D**2 / 2 / A
y0 = np.random.normal(loc=c, scale=np.sqrt(d))
if config["verbose"]:
    Print("Initial observation value:", y0)


y = model.obs()
y0 = -0.05563397349186569  # need to update from invariant distribution
y.dat.data[:] = y0


# prepare the initial ensemble
# Ensemble is initialized as samples from N(0, D^2/(2A))

c = 0.0
d = D**2 / 2 / A

#
for i in range(nensemble[myfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, c, d)
    Print(f"dx0: {dx0}")
    u = myfilter.ensemble[i][0]
    Print(f"u : {u}")
    u.assign(dx0)

# observation noise standard deviation
S = 0.1


def log_likelihood(y, Y):
    ll = (y - Y) ** 2 / S**2 / 2 * dx
    return ll


# results in a diagnostic
class samples(base_diagnostic):
    def compute_diagnostic(self, particle):
        model.u.assign(particle[0])
        return model.obs().dat.data[0]


# wihout nudging
nolambdasamples = samples(Stage.WITHOUT_LAMBDAS, myfilter.subcommunicators, nensemble)

# with nudging
nudgingsamples = samples(Stage.AFTER_NUDGING, myfilter.subcommunicators, nensemble)
# after computing filteing step
resamplingsamples = samples(
    Stage.AFTER_ASSIMILATION_STEP, myfilter.subcommunicators, nensemble
)

diagnostics = [nudgingsamples, resamplingsamples, nolambdasamples]

tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-4,
    "tao_grtol": 1.0e-50,
    "tao_gttol": 1.0e-5,
}


myfilter.assimilation_step(
    y,
    log_likelihood,
    diagnostics=diagnostics,
    ess_tol=-9000.8,
    taylor_test=False,
    tao_params=tao_params,
)

if myfilter.subcommunicators.global_comm.rank == 0:
    before, descriptors = nolambdasamples.get_archive()
    after, descriptors = nudgingsamples.get_archive()
    resampled, descriptors = resamplingsamples.get_archive()

    np.save("before", before)
    np.save("after", after)
    np.save("resampled", resampled)
    bs_mean = np.mean(resampled)
    bs_var = np.var(resampled)

    sigsq = D**2 / 2 / A * (1 - np.exp(-2 * A * T))
    Sigsq = sigsq + np.exp(-2 * A * T) * d
    tmean = (Sigsq * y0 + np.exp(-A * T) * S**2 * c) / (Sigsq + S**2)
    tvar = Sigsq * S**2 / (Sigsq + S**2)

    print(
        "True mean",
        tmean,
        "ensemble mean",
        bs_mean,
        "true var",
        tvar,
        "ensemble var",
        bs_var,
        "diffmean",
        abs(tmean - bs_mean),
        "diffvar",
        abs(tvar - bs_var),
    )
