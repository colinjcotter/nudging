from abc import ABCMeta, abstractmethod
import firedrake as fd
import firedrake.adjoint as fadj
from pyadjoint import exp as pexp
from pyadjoint import log as plog
from pyadjoint import OverloadedType
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from .resampling import residual_resampling
from .diagnostics import compute_diagnostics, Stage, archive_diagnostics
import numpy as np
from .parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray
from firedrake.adjoint import pause_annotation, continue_annotation, \
    get_working_tape
from .global_optimisation import ensemble_tao_solver, \
        ParameterisedEnsembleReducedFunctional


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def logsumexp_adjfloat(x, factor=1.0):
    c = factor*x[0]
    for i in range(1, len(x)):
        c = max(c, factor*x[i])
    sumexp = pexp(factor*x[0] - c)
    for i in range(1, len(x)):
        sumexp += pexp(factor*x[0] - c)
    c += plog(sumexp)
    return c


class base_filter(object, metaclass=ABCMeta):
    ensemble = []
    new_ensemble = []

    def __init__(self):
        pass

    def setup(self, nensemble, model, resampler_seed=34343,
              residual=False):
        """
        Construct the ensemble

        nensemble - a list of the number of ensembles on each ensemble rank
        model - the model to use
        """
        self.model = model
        self.nensemble = nensemble
        n_ensemble_partitions = len(nensemble)
        self.nspace = int(MPI.COMM_WORLD.size/n_ensemble_partitions)
        assert self.nspace*n_ensemble_partitions == MPI.COMM_WORLD.size

        self.subcommunicators = fd.Ensemble(MPI.COMM_WORLD, self.nspace)
        # model needs to build the mesh in setup
        self.model.setup(self.subcommunicators.comm)
        if isinstance(nensemble, int):
            nensemble = tuple(nensemble for _ in
                              range(self.subcommunicators.comm.size))

        # setting up ensemble
        self.ensemble_rank = self.subcommunicators.ensemble_comm.rank
        self.ensemble_size = self.subcommunicators.ensemble_comm.size
        self.ensemble = []
        self.new_ensemble = []
        self.proposal_ensemble = []
        for i in range(self.nensemble[self.ensemble_rank]):
            self.ensemble.append(model.allocate())
            self.new_ensemble.append(model.allocate())
            self.proposal_ensemble.append(model.allocate())

        # some numbers for shared array and owned array
        self.nlocal = self.nensemble[self.ensemble_rank]
        self.nglobal = int(np.sum(self.nensemble))

        # Shared array for the potentials
        ecomm = self.subcommunicators.ensemble_comm
        self.potential_arr = SharedArray(partition=self.nensemble, dtype=float,
                                         comm=ecomm)
        # Owned array for the resampling protocol
        self.s_arr = OwnedArray(size=self.nglobal, dtype=int,
                                comm=ecomm,
                                owner=0)
        # data layout for coordinating resampling communication
        self.layout = DistributedDataLayout1D(self.nensemble,
                                              comm=ecomm)

        # offset_list
        self.offset_list = []
        for i_rank in range(len(self.nensemble)):
            self.offset_list.append(sum(self.nensemble[:i_rank]))
        # a resampling method
        self.resampler = residual_resampling(seed=resampler_seed,
                                             residual=residual)

    def index2rank(self, index):
        for rank in range(len(self.offset_list)):
            if self.offset_list[rank] - index > 0:
                rank -= 1
                break
        return rank

    def parallel_resample(self, dtheta=1, s=None):

        if s:
            s_copy = s
            self.s_copy = s
        else:
            self.potential_arr.synchronise(root=0)
            if self.ensemble_rank == 0:
                potentials = self.potential_arr.data()
                # renormalise
                weights = np.exp(-dtheta*potentials
                                 - logsumexp(-dtheta*potentials))
                assert np.abs(np.sum(weights)-1) < 1.0e-8
                self.ess = 1/np.sum(weights**2)
                if self.verbose:
                    PETSc.Sys.Print("ESS "
                                    + str(100*self.ess/np.sum(self.nensemble))
                                    + "%")

            # compute resampling protocol on rank 0
                s = self.resampler.resample(weights, self.model)
                for i in range(self.nglobal):
                    self.s_arr[i] = s[i]

            # broadcast protocol to every rank
            self.s_arr.synchronise()
            s_copy = self.s_arr.data()
            self.s_copy = s_copy

        mpi_requests = []

        for ilocal in range(self.nensemble[self.ensemble_rank]):
            iglobal = self.layout.transform_index(ilocal, itype='l',
                                                  rtype='g')
            # add to send list
            targets = []
            for j in range(self.s_arr.size):
                if s_copy[j] == iglobal:
                    targets.append(j)

            for target in targets:
                if isinstance(self.ensemble[ilocal], list):
                    for k in range(len(self.ensemble[ilocal])):
                        request_send = self.subcommunicators.isend(
                            self.ensemble[ilocal][k],
                            dest=self.index2rank(target),
                            tag=1000*target+k)
                        mpi_requests.extend(request_send)
                else:
                    request_send = self.subcommunicators.isend(
                        self.ensemble[ilocal],
                        dest=self.index2rank(target),
                        tag=target)
                    mpi_requests.extend(request_send)

            source_rank = self.index2rank(s_copy[iglobal])
            if isinstance(self.ensemble[ilocal], list):
                for k in range(len(self.ensemble[ilocal])):
                    request_recv = self.subcommunicators.irecv(
                        self.new_ensemble[ilocal][k],
                        source=source_rank,
                        tag=1000*iglobal+k)
                    mpi_requests.extend(request_recv)
            else:
                request_recv = self.subcommunicators.irecv(
                    self.new_ensemble[ilocal],
                    source=source_rank,
                    tag=iglobal)
                mpi_requests.extend(request_recv)

        MPI.Request.Waitall(mpi_requests)
        for i in range(self.nlocal):
            for j in range(len(self.ensemble[i])):
                self.ensemble[i][j].assign(self.new_ensemble[i][j])

    @abstractmethod
    def assimilation_step(self, y, log_likelihood, diagnostics):
        """
        Advance the ensemble to the next assimilation time
        and apply the filtering algorithm
        y - a k-dimensional numpy array containing the observations
        log_likelihood - a function that computes -log(Pi(y|x))
                         for computing the filter weights
        diagnostics - a list of diagnostics to compute
        """
        pass


class sim_filter(base_filter):

    def __init__(self):
        super().__init__()

    def assimilation_step(self, s):
        for i in range(self.nensemble[self.ensemble_rank]):
            # set the particle value to the global index
            self.ensemble[i][0].assign(self.offset_list[self.ensemble_rank]+i)
        self.parallel_resample(s=s)


class bootstrap_filter(base_filter):

    def __init__(self, verbose=0, residual=False):
        super().__init__()
        self.verbose = verbose
        self.residual = residual

    def assimilation_step(self, y, log_likelihood, diagnostics):
        N = self.nensemble[self.ensemble_rank]
        # forward model step
        for i in range(N):
            self.model.randomize(self.ensemble[i])
            self.model.run(self.ensemble[i], self.ensemble[i])

            Y = self.model.obs()
            self.potential_arr.dlocal[i] = fd.assemble(log_likelihood(y, Y))
        self.parallel_resample()
        compute_diagnostics(diagnostics,
                            self.ensemble,
                            stage=Stage.AFTER_ASSIMILATION_STEP,
                            descriptor=None)
        archive_diagnostics(diagnostics)


class jittertemp_filter(base_filter):
    def __init__(self, n_jitt, delta,
                 verbose=0, MALA=False, nudging=False,
                 visualise_tape=False, sigma=0.1):
        self.delta = delta
        self.verbose = verbose
        self.MALA = MALA
        self.model_taped = False
        self.nudging = nudging
        self.visualise_tape = visualise_tape
        self.n_jitt = n_jitt
        self.sigma = sigma  # nudging parameter

        if MALA:
            PETSc.Sys.Print("Warning, we are not currently "
                            + "computing the Metropolis correction for MALA."
                            + " Choose a small delta.")

    def setup(self, nensemble, model, resampler_seed=34343, residual=False):
        super(jittertemp_filter, self).setup(
            nensemble, model, resampler_seed=resampler_seed,
            residual=residual)
        # Owned array for sending dtheta
        ecomm = self.subcommunicators.ensemble_comm
        self.dtheta_arr = OwnedArray(size=self.nglobal, dtype=float,
                                     comm=ecomm, owner=0)

    def adaptive_dtheta(self, dtheta, theta, ess_tol):
        self.potential_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            potentials = self.potential_arr.data()
            ess = 0.
            while ess < ess_tol*sum(self.nensemble):
                # renormalise
                weights = np.exp(-dtheta*potentials
                                 - logsumexp(-dtheta*potentials))
                weights /= np.sum(weights)
                ess = 1/np.sum(weights**2)
                if ess < ess_tol*sum(self.nensemble):
                    dtheta = 0.5*dtheta

            # abuse owned array to broadcast dtheta
            for i in range(self.nglobal):
                self.dtheta_arr[i] = dtheta

        # broadcast dtheta to every rank
        self.dtheta_arr.synchronise()
        dtheta = self.dtheta_arr.data()[0]
        return dtheta

    def assimilation_step(self, y, log_likelihood,
                          diagnostics=[],
                          ess_tol=0.8, tao_params=None):
        if not tao_params:
            tao_params = {
                "tao_type": "lmvm",
                "tao_cg_type": "pr",
                "tao_monitor": None,
                "tao_converged_reason": None
            }

        N = self.nensemble[self.ensemble_rank]
        potentials = np.zeros(N)
        new_potentials = np.zeros(N)
        self.ess_temper = []
        self.theta_temper = []
        nsteps = self.model.nsteps

        # tape the forward model
        if not self.model_taped:
            self.model_taped = True
            continue_annotation()
            if self.MALA:
                if self.verbose > 0:
                    PETSc.Sys.Print("taping forward model for MALA")
                self.model.run(self.ensemble[0],
                               self.new_ensemble[0])
                # set the controls
                if isinstance(y, fd.Function):
                    m = self.model.controls() + [fadj.Control(y)]
                else:
                    m = self.model.controls()
                # requires log_likelihood to return symbolic
                Y = self.model.obs()
                MALA_J = fd.assemble(log_likelihood(y, Y))
                # functional for MALA
                cpts = [j for j in range(1, nsteps+1)]
                self.Jhat_dW = fadj.ReducedFunctional(
                    MALA_J, m, derivative_components=cpts)

            if self.nudging:
                if self.verbose > 0:
                    PETSc.Sys.Print("taping forward model for nudging")
                self.y = y
                Js = []  # list of lists of functionals
                Controls = [[]]*N  # things to pass to RF constructor
                self.Control_inputs = [[]]*N  # things to pass to RF.__call__
                Parameters = [[]]*(N+1+N-1)  # things to pass to RF constructor
                self.Parameter_inputs = [[]]*(N+1+N-1)  # pass to RF.update_...
                assert self.model.lambdas  # can't nudge without lambdas
                BigJ_floats = []  # inputs for functional that takes
                #                   in all the Js
                for i in range(N):  # build functionals for each particle
                    BigJ_floats.append(fadj.AdjFloat(1.0))  # needs > 0
                    for step in range(nsteps):
                        #  adding Lambda to the controls for this step
                        self.Control_inputs[step].append(
                            self.ensemble[i][nsteps+1+step])
                        Controls.append(fadj.Control(
                            self.ensemble[i][nsteps+1+step]))
                        #  adding model state to the parameters
                        self.Parameter_inputs[step].append(
                            self.ensemble[i][0])
                        Parameters[step].append(
                            fadj.Control(self.ensemble[i][0]))
                        #  adding noise values to the parameters
                        for step2 in range(nsteps):
                            self.Parameter_inputs[step].append(
                                self.ensemble[i][1+step])
                            Parameters[step].append(
                                fadj.Control(self.ensemble[i][1+step]))
                        #  adding Lambda for other steps as parameters
                        for step2 in range(nsteps):
                            if step2 == step:
                                continue
                            self.Parameter_inputs[step].append(
                                self.ensemble[i][nsteps+1+step2])
                            Parameters.append(fadj.Control(
                                self.ensemble[i][nsteps+1+step2]))

                        # tape model for local particle i
                        self.model.run(self.ensemble[i],
                                       self.new_ensemble[i])
                        Y = self.model.obs()
                        nudge_J = fd.assemble(log_likelihood(y, Y))
                        nudge_J += self.model.lambda_functional()
                        Js.append(nudge_J)
                        assert isinstance(nudge_J, OverloadedType)
                #  adding in the data as a parameter
                self.Parameter_inputs.append(self.y)
                Parameters.append(self.y)

                # build the RF that maps from the Js to the BigJ
                BigJ = -(2 + self.sigma)*logsumexp_adjfloat(BigJ_floats,
                                                            factor=-1.0)
                BigJ += logsumexp_adjfloat(BigJ_floats, factor=-2.0)
                BigJ_Controls = [fadj.Control(fl) for fl in BigJ_floats]
                BigJhat = fadj.ReducedFunctional(BigJ, BigJ_Controls)
                # reduced functionals for each step
                # they differ by the derivative components
                self.Jhat_solvers = []  # list of Tao solvers
                self.rfs = []
                for step in range(nsteps+1, nsteps*2+1):
                    # 0 component is state
                    # 1 .. step is noise
                    # step + 1 .. 2*step is lambdas
                    offset = 0
                    cpts = []
                    for i in range(N):
                        cpts.append(offset + step)
                        offset += len(self.ensemble[i])
                    # we only update lambdas[step] on timestep step
                    rf = ParameterisedEnsembleReducedFunctional(
                        Js, Controls[step], Parameters[step], self.ensemble,
                        scatter_control=False,
                        gather_functional=BigJhat)
                    self.rfs.append(rf)
                    solver = ensemble_tao_solver(
                        rf, self.ensemble, solver_parameters=tao_params)
                    self.Jhat_solvers.append(solver)

            if self.visualise_tape:
                tape = get_working_tape()
                assert isinstance(self.visualise_tape, str)
                tape.visualise_pdf(self.visualise_tape)
            pause_annotation()

        if self.nudging:
            self.y.assign(y)
            if self.verbose > 0:
                PETSc.Sys.Print("Starting nudging")
            for i in range(N):
                # zero the noise and lambdas in preparation for nudging
                for step in range(nsteps):
                    self.ensemble[i][step+1].assign(0.)  # the noise
                    self.ensemble[i][nsteps+step+1].assign(0.)  # the nudging
            # nudging one step at a time
            for step in range(nsteps):
                # update with current noise and lambda values
                self.rfs[step].update_parameters(self.Parameter_inputs[step])
                self.rfs[step](self.Control_inputs[step])
                # get the minimum over current lambda
                if self.verbose > 1:
                    PETSc.Sys.Print("Solving for Lambda step ", step)

                Xopt = self.Jhat_solvers[step].solve()
                assert isinstance(Xopt[0], fd.Function)
                # place the optimal value of lambda into ensemble
                offset = 0
                for i in range(N):
                    self.ensemble[i][nsteps+1+step].assign(Xopt[0])
                    # get the randomised noise for this step
                    self.model.randomize(
                        self.new_ensemble[i])  # not efficient!
                    # just copy in the current component
                    self.ensemble[i][1+step].assign(
                        self.new_ensemble[i][1+step])
            PETSc.garbage_cleanup(PETSc.COMM_SELF)

            compute_diagnostics(diagnostics,
                                self.ensemble,
                                descriptor=None,
                                stage=Stage.AFTER_NUDGING,
                                run=self.model.run,
                                new_ensemble=self.new_ensemble)

            self.model.lambdas = False
            compute_diagnostics(diagnostics,
                                self.ensemble,
                                descriptor=None,
                                stage=Stage.WITHOUT_LAMBDAS,
                                run=self.model.run,
                                new_ensemble=self.new_ensemble)
            self.model.lambdas = True
        else:
            for i in range(N):
                # generate the initial noise variables
                self.model.randomize(self.ensemble[i])

        theta = .0
        temper_count = 0
        while theta < 1.:  # Tempering loop
            dtheta = 1.0 - theta

            # Compute initial potentials
            for i in range(N):
                # put result of forward model into new_ensemble
                self.model.run(self.ensemble[i], self.new_ensemble[i])
                Y = self.model.obs()
                self.potential_arr.dlocal[i] = \
                    fd.assemble(log_likelihood(y, Y))
                if self.nudging:
                    self.potential_arr.dlocal[i] += \
                        self.model.lambda_functional()

            # adaptive dtheta choice
            dtheta = self.adaptive_dtheta(dtheta, theta, ess_tol)
            theta += dtheta
            self.theta_temper.append(theta)
            if self.verbose > 1:
                PETSc.Sys.Print("theta", theta, "dtheta", dtheta)

            # resampling BEFORE jittering
            self.parallel_resample(dtheta)
            compute_diagnostics(diagnostics,
                                self.ensemble,
                                descriptor=(dtheta),
                                stage=Stage.AFTER_TEMPER_RESAMPLE,
                                run=self.model.run,
                                new_ensemble=self.new_ensemble)
            temper_count += 1

            for jitt_step in range(self.n_jitt):  # Jittering loop
                if self.verbose > 1:
                    PETSc.Sys.Print("Jitter step", jitt_step)

                for i in range(N):
                    if jitt_step == 0:
                        # Compute initial potentials
                        self.model.run(self.ensemble[i],
                                       self.new_ensemble[i])
                        Y = self.model.obs()
                        potentials[i] = fd.assemble(
                            log_likelihood(y, Y))
                        if self.nudging:
                            potentials[i] += self.model.lambda_functional()
                        potentials[i] *= theta

                    if self.MALA:
                        # run the model and get the functional value with
                        # ensemble[i]
                        self.Jhat_dW(self.ensemble[i]+[y])
                        # use the taped model to get the derivative
                        g = self.Jhat_dW.derivative()
                        # proposal
                        self.model.copy(self.ensemble[i],
                                        self.proposal_ensemble[i])
                        delta = self.delta
                        self.model.randomize(self.proposal_ensemble[i],
                                             (
                                                 (2-delta)/(2+delta)),
                                             (
                                                 (8*delta)**0.5/(2+delta)),
                                             gscale=-2*delta/(2+delta), g=g)
                    else:
                        # proposal PCN
                        self.model.copy(self.ensemble[i],
                                        self.proposal_ensemble[i])
                        delta = self.delta
                        self.model.randomize(self.proposal_ensemble[i],
                                             (2-delta)/(2+delta),
                                             (8*delta)**0.5/(2+delta))
                    # put result of forward model into new_ensemble
                    self.model.run(self.proposal_ensemble[i],
                                   self.new_ensemble[i])

                    # particle potentials
                    Y = self.model.obs()
                    new_potentials[i] = fd.assemble(
                        log_likelihood(y, Y))
                    if self.nudging:
                        new_potentials[i] += self.model.lambda_functional()
                    new_potentials[i] *= theta
                    # accept reject of MALA and Jittering
                    # Metropolis MCMC
                    if self.MALA:
                        p_accept = 1
                    else:
                        p_accept = min(1,
                                       np.exp(potentials[i]
                                              - new_potentials[i]))
                        # accept or reject tool
                        u = self.model.rg.uniform(self.model.R, 0., 1.0)
                        if u.dat.data[:] < p_accept:
                            potentials[i] = new_potentials[i]
                            self.model.copy(self.proposal_ensemble[i],
                                            self.ensemble[i])
                compute_diagnostics(diagnostics,
                                    self.ensemble,
                                    descriptor=(dtheta, jitt_step),
                                    stage=Stage.AFTER_ONE_JITTER_STEP,
                                    run=self.model.run,
                                    new_ensemble=self.new_ensemble)

            compute_diagnostics(diagnostics,
                                self.ensemble,
                                descriptor=(dtheta),
                                stage=Stage.AFTER_JITTERING,
                                run=self.model.run,
                                new_ensemble=self.new_ensemble)

        if self.verbose > 0:
            PETSc.Sys.Print(str(temper_count)+" tempering steps")
            PETSc.Sys.Print("Advancing ensemble")
        for i in range(N):
            self.model.run(self.ensemble[i], self.ensemble[i])
        if self.verbose > 0:
            PETSc.Sys.Print("assimilation step complete")
        # trigger garbage cleanup
        PETSc.garbage_cleanup(PETSc.COMM_SELF)
        compute_diagnostics(diagnostics,
                            self.ensemble,
                            descriptor=None,
                            stage=Stage.AFTER_ASSIMILATION_STEP)
        archive_diagnostics(diagnostics)
