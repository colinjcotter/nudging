from firedrake import dx, exp
from nudging import LSDEModel, \
    jittertemp_filter, base_diagnostic, Stage
import numpy as np

# model
# multiply by A and add D
T = 1.
nsteps = 5
dt = T/nsteps
A = 1.
D = 2.
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=True, seed=7123)

p_per_rank = 100  # 10000
nranks = 10
nensemble = [p_per_rank]*nranks

myfilter = jittertemp_filter(n_jitt=0, delta=0.15,
                             verbose=2, MALA=False,
                             nudging=True)
myfilter.setup(nensemble=nensemble, model=model, residual=False)

# data
y = model.obs()
y0 = 1.2
y.dat.data[:] = y0

# prepare the initial ensemble
c = 1.
d = 1.
for i in range(nensemble[myfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, c, d**2)
    u = myfilter.ensemble[i][0]
    u.assign(dx0)

# observation noise standard deviation
S = 0.3


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
myfilter.assimilation_step(y, log_likelihood,
                           diagnostics=diagnostics,
                           ess_tol=-666)

if myfilter.subcommunicators.global_comm.rank == 0:
    before, descriptors = nolambdasamples.get_archive()
    after, descriptors = nudgingsamples.get_archive()
    resampled, descriptors = resamplingsamples.get_archive()

    np.save("before", before)
    np.save("after", after)
    np.save("resampled", resampled)
    bs_mean = np.mean(resampled)
    bs_var = np.var(resampled)

    # analytical formula
    # x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))
    # where a = c*exp(A), b = (sig^2+d^2*exp(2A))
    # sig^2 = (D^2/2A)*(e^{2A} - 1)
    a = c*exp(A)
    sigsq = D**2/2/A*(exp(2*A) - 1)  # t = 1
    b = sigsq + d**2*exp(2*A)
    tmean = (b**2*y0 + S**2*a)/(b**2 + S**2)
    tvar = b**2*S**2/(b**2 + S**2)

    print(tmean, bs_mean, tvar, bs_var)
