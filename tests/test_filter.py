from firedrake import dx, exp
from nudging import LSDEModel, bootstrap_filter, SharedArray, \
    jittertemp_filter
import numpy as np
import pytest


def filter_linear_sde(testfilter, filterargs, mtol, vtol,
                      p_per_rank, nranks, lambdas=False):
    # model
    # multiply by A and add D
    T = 1.
    nsteps = 10
    dt = T/nsteps
    A = 1.
    D = 2.
    model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=lambdas)

    # solving
    # dx = A*x*dt + D*dW
    # integrating factor e^{-At}
    # e^{-At}dx = e^{-At}*x*dt + De^{-t}dW
    # we have d(e^{-At}x) = e^{-At}dx - e^{-At}x*dt
    # so
    # d(e^{-At}x) = De^{-At}dW
    # i.e.
    # x(t) = e^{At}x(0) + int_0^t D e^{A(t-s)}dW(s)

    # E(int_0^t D e^{A(t-s)}dW(s)) = 0
    # E[(int_0^t D e^{A(t-s)}dW(s))^2]
    #    = int_0^t D^2 e^{2A(t-s)} ds (Ito isometry)
    # = (-D^2/(2A))*[e^{2A(t-s)}]_{s=0}^{s=t}
    # = (D^2/(2A))*(e^{2At} - 1)
    # so pi(x(t)|x(0)) ~ N(e^{at}x(0), sig^2)
    # sig^2 = (D^2/2A)*(e^{2At} - 1)

    # take x(0) ~ N(c, d^2)

    # write y(t) = x(t) - e^{At}x(0),
    # then
    # z = (y(t), x(0))^T
    # and z ~ N( (0, c)^T, Sigma )
    # with
    # Sigma = (sig^2   0)
    #         (0     d^2)

    # (x(t), x(0))^T = Bz with
    # B = (1  exp(At))
    #     (0        1)

    # (x(t), x(0))^T ~ N(B(0, c)^T, B Sigma B^T)

    # Sigma B^T = (sig^2   0)(1       0) = (sig^2         0)
    #             (0     d^2)(exp(At) 1)   (d^2*exp(At) d^2)

    # B Sigma B^T = (1 exp(At))(sig^2         0)
    #               (0       1)(d^2*exp(At) d^2)

    #             = (sig^2 + d^2*exp(2At) d^2*exp(At))
    #               (d^2*exp(At)                  d^2)

    # B(0, c)^T = (1 exp(At))(0) = c(exp(At))
    #             (0       1)(c)    (      1)

    # The marginal distribution for x(t) is then
    # x(t) ~ N(c*exp(At), sig^2 + d^2*exp(2At))

    # now we have an observation y = x(1) + e, e ~ N(0, S^2)
    # Reverend Bayes
    # pi(x(1)|y) propto pi(y|x(1))*pi(x(1))
    # = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-c*exp(A))^2/(sig^2+d^2*exp(2A))])
    # = exp( -(1/2)[(y-x(1))^2/S^2 + (x(1)-a)^2/b^2])
    # where a = c*exp(A), b = (sig^2+d^2*exp(2A))
    # completing the square
    # (y-x(1))^2/S^2 + (x(1)-a)^2/b^2
    # = (x(1)^2-2x(1)y + y^2)/S^2 + (x(1)^2 - 2ax(1) + a^2)/b^2
    # = x(1)**2(b^2 + S^2)/(b^2S^2) - 2x(1)(b^2y + S^2a)/(b^2S^2) + stuff
    # = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
    # = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - 2x(1)*(b^2y + S^2a)/(b^2+S^2)] + stuff
    # = (b^2 + S^2)/(b^2S^2)[ x(1)^2 - (b^2y + S^2a)/(b^2+S^2)]^2 + stuff
    # where stuff is anything independent of x(1)

    # then
    # x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))

    print(p_per_rank, nranks)
    nensemble = [p_per_rank]*nranks
    filterargs["nensemble"] = nensemble
    filterargs["model"] = model
    testfilter.setup(**filterargs)

    # data
    y = model.obs()
    y0 = 1.2
    y.dat.data[:] = y0

    # prepare the initial ensemble
    c = 1.
    d = 1.
    for i in range(nensemble[testfilter.ensemble_rank]):
        dx0 = model.rg.normal(model.R, c, d**2)
        u = testfilter.ensemble[i][0]
        u.assign(dx0)

    # observation noise standard deviation
    S = 0.3

    def log_likelihood(y, Y):
        ll = (y-Y)**2/S**2/2*dx
        return ll

    testfilter.assimilation_step(y, log_likelihood)

    # results in a shared array
    posterior = SharedArray(partition=nensemble,
                            comm=testfilter.subcommunicators.ensemble_comm)
    for i in range(nensemble[testfilter.ensemble_rank]):
        model.u.assign(testfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        posterior.dlocal[i] = obsdata
    posterior.synchronise()

    pvals = posterior.data()
    bs_mean = np.mean(pvals)
    bs_var = np.var(pvals)

    # analytical formula
    # x(1)|y ~ N((b^2y + S^2a)/(b^2+S^2), (b^2S^2)/(b^2 + S^2))
    # where a = c*exp(A), b = (sig^2+d^2*exp(2A))
    # sig^2 = (D^2/2A)*(e^{2A} - 1)
    a = c*exp(A)
    sigsq = D**2/2/A*(exp(2*A) - 1)  # t = 1
    b = sigsq + d**2*exp(2*A)
    tmean = (b**2*y0 + S**2*a)/(b**2 + S**2)
    tvar = b**2*S**2/(b**2 + S**2)

    assert np.abs(tmean - bs_mean) < mtol
    assert np.abs(tvar - bs_var) < vtol


@pytest.mark.parallel(nprocs=5)
def test_bsfilter():
    filter_linear_sde(bootstrap_filter(), {"residual": False},
                      mtol=0.005, vtol=0.005,
                      p_per_rank=200, nranks=5)


@pytest.mark.parallel(nprocs=10)
def test_jtfilter():
    jtfilter = jittertemp_filter(n_jitt=5, delta=0.15,
                                 verbose=False, MALA=False)
    filter_linear_sde(jtfilter, {"residual": False},
                      mtol=0.015, vtol=0.08,
                      p_per_rank=10, nranks=10)


@pytest.mark.parallel(nprocs=10)
def test_nudgingfilter():
    jtfilter = jittertemp_filter(n_jitt=5, delta=0.15,
                                 verbose=False, MALA=False,
                                 nudging=True)
    filter_linear_sde(jtfilter, {"residual": False},
                      mtol=0.1, vtol=0.1,
                      p_per_rank=10, nranks=10,
                      lambdas=True)
