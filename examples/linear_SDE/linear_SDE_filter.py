from firedrake import dx, exp
from nudging import LSDEModel, \
    jittertemp_filter, base_diagnostic, Stage
import numpy as np

# model
# multiply by A and add D
T = 1.
nsteps = 10
dt = T/nsteps
A = 1.
D = 1.0
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=True, seed=7123)

p_per_rank = 1
nranks = 32
nensemble = [p_per_rank]*nranks

myfilter = jittertemp_filter(n_jitt=0, delta=0.15,
                             verbose=2, MALA=False,
                             visualise_tape=False, nudging=True, sigma=0.01)
myfilter.setup(nensemble=nensemble, model=model,
               residual=False)

# data
y = model.obs()
y0 = -0.05563397349186569 #  need to update from invariant distribution
y.dat.data[:] = y0

# prepare the initial ensemble
c = 0.
d = D**2/2/A
for i in range(nensemble[myfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, c, d)
    u = myfilter.ensemble[i][0]
    u.assign(dx0)

# observation noise standard deviation
S = 0.01


def log_likelihood(y, Y):
    ll = (y-Y)**2/S**2/2*dx
    return ll


# results in a diagnostic
class samples(base_diagnostic):
    def compute_diagnostic(self, particle):
        model.u.assign(particle[0])
        return model.obs().dat.data[0]


resamplingsamples = samples(Stage.AFTER_ASSIMILATION_STEP,
                            myfilter.subcommunicators,
                            nensemble)
nudgingsamples = samples(Stage.AFTER_NUDGING,
                         myfilter.subcommunicators,
                         nensemble)
nolambdasamples = samples(Stage.WITHOUT_LAMBDAS,
                          myfilter.subcommunicators,
                          nensemble)

diagnostics = [nudgingsamples,
               resamplingsamples,
               nolambdasamples]

tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-4,
    "tao_grtol": 1.0e-50,
    "tao_gttol": 1.0e-5,
}


myfilter.assimilation_step(y, log_likelihood,
                           diagnostics=diagnostics,
                           ess_tol=-666,
                           taylor_test=False,
                           tao_params=tao_params)

if myfilter.subcommunicators.global_comm.rank == 0:
    before, descriptors = nolambdasamples.get_archive()
    after, descriptors = nudgingsamples.get_archive()
    resampled, descriptors = resamplingsamples.get_archive()

    np.save("before", before)
    np.save("after", after)
    np.save("resampled", resampled)
    bs_mean = np.mean(resampled)
    bs_var = np.var(resampled)

    sigsq = D**2/2/A*(1 - np.exp(-2*A*T))
    Sigsq = sigsq + np.exp(-2*A*T)*d
    tmean = (Sigsq*y0 + np.exp(-A*T)*S**2*c)/(Sigsq + S**2)
    tvar = Sigsq*S**2/(Sigsq + S**2)

    print(tmean, bs_mean, tvar, bs_var)
