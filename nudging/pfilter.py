from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from .resampling import *
import numpy as np
import pyadjoint
from .parallel_arrays import DistributedDataLayout1D, SharedArray, OwnedArray
from firedrake_adjoint import *
pyadjoint.tape.pause_annotation()

class base_filter(object, metaclass=ABCMeta):
    ensemble = []
    new_ensemble = []

    def __init__(self):
        pass

    def setup(self, nensemble, model, resampler_seed=34343):
        """
        Construct the ensemble

        nensemble - a list of the number of ensembles on each ensemble rank
        model - the model to use
        """
        self.model = model
        self.nensemble = nensemble
        n_ensemble_partitions = len(nensemble)
        self.nspace = int(COMM_WORLD.size/n_ensemble_partitions)
        #print(self.nspace, n_ensemble_partitions, COMM_WORLD.size, 'bsdfdf')
        assert(self.nspace*n_ensemble_partitions == COMM_WORLD.size)

        self.subcommunicators = Ensemble(COMM_WORLD, self.nspace)
        # model needs to build the mesh in setup
        self.model.setup(self.subcommunicators.comm)
        if isinstance(nensemble, int):
            nensemble = tuple(nensemble for _ in range(self.subcommunicators.comm.size))
        
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
            
        # Shared array for the weights
        self.weight_arr = SharedArray(partition=self.nensemble, dtype=float,
                                      comm=self.subcommunicators.ensemble_comm)
        # Owned array for the resampling protocol
        self.s_arr = OwnedArray(size = self.nglobal, dtype=int,
                                comm=self.subcommunicators.ensemble_comm,
                                owner=0)
        # data layout for coordinating resampling communication
        self.layout = DistributedDataLayout1D(self.nensemble,
                                         comm=self.subcommunicators.ensemble_comm)

        # offset_list
        self.offset_list = []
        for i_rank in range(len(self.nensemble)):
            self.offset_list.append(sum(self.nensemble[:i_rank]))
        #print(self.offset_list)
        #a resampling method
        self.resampler = residual_resampling(seed=resampler_seed)

    def index2rank(self, index):
        for rank in range(len(self.offset_list)):
            if self.offset_list[rank] - index > 0:
                rank -= 1
                break
        return rank
        
    def parallel_resample(self, dtheta=1):
        
        self.weight_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            weights = self.weight_arr.data()
            # renormalise
            weights = np.exp(-dtheta*weights)
            weights /= np.sum(weights)
            self.ess = 1/np.sum(weights**2)

        # compute resampling protocol on rank 0
        if self.ensemble_rank == 0:
            s = self.resampler.resample(weights, self.model)
            for i in range(self.nglobal):
                self.s_arr[i]=s[i]

        # broadcast protocol to every rank
        self.s_arr.synchronise()
        s_copy = self.s_arr.data()
        self.s_copy = s_copy
        #print('=========================================Rank====================================', self.ensemble_rank)
        #print("S", s_copy)

        mpi_requests = []
        
        for ilocal in range(self.nensemble[self.ensemble_rank]):
            iglobal = self.layout.transform_index(ilocal, itype='l',
                                             rtype='g')
            # add to send list
            targets = []
            for j in range(self.s_arr.size):
                if s_copy[j] == iglobal:
                    targets.append(j)
            # print('Target', "rank", self.ensemble_rank,
            #       "ilocal", ilocal,
            #       "iglobal", iglobal,
            #       targets, flush=True)

            for target in targets:
                if type(self.ensemble[ilocal] == 'list'):
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
            if type(self.ensemble[ilocal] == 'list'):
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
            #print(i, self.subcommunicators.ensemble_comm.rank, self.subcommunicators.comm.rank)
            for j in range(len(self.ensemble[i])):
                self.ensemble[i][j].assign(self.new_ensemble[i][j])



        
    @abstractmethod
    def assimilation_step(self, y, log_likelihood):
        """
        Advance the ensemble to the next assimilation time
        and apply the filtering algorithm
        y - a k-dimensional numpy array containing the observations
        log_likelihood - a function that computes -log(Pi(y|x))
                         for computing the filter weights
        """
        pass

class sim_filter(base_filter):

    def __init__(self):
        super().__init__()

    def assimilation_step(self, y, log_likelihood):
        for i in range(self.nensemble[self.ensemble_rank]):
            # set the particle value to the global index
            self.ensemble[i].assign(self.offset_list[self.ensemble_rank]+i)

            Y = self.model.obs()
            self.weight_arr.dlocal[i] = log_likelihood(y,Y)
        self.parallel_resample()

class bootstrap_filter(base_filter):
    def assimilation_step(self, y, log_likelihood):
        N = self.nensemble[self.ensemble_rank]
        # forward model step
        for i in range(N):
            self.model.randomize(self.ensemble[i])
            self.model.run(self.ensemble[i], self.ensemble[i])   

            Y = self.model.obs()
            self.weight_arr.dlocal[i] = log_likelihood(y,Y)
        self.parallel_resample()


class jittertemp_filter(base_filter):
    def __init__(self, n_temp, n_jitt, rho,
                 verbose=False, MALA=False):
        self.n_temp = n_temp
        self.n_jitt = n_jitt
        self.rho = rho
        self.verbose=verbose
        self.MALA = MALA
        self.model_taped = False

    def setup(self, nensemble, model, resampler_seed=34343):
        super(jittertemp_filter, self).setup(
            nensemble, model, resampler_seed=34343)
        # Owned array for sending dtheta
        self.dtheta_arr = OwnedArray(size = self.nglobal, dtype=float,
                                     comm=self.subcommunicators.ensemble_comm,
                                     owner=0)

    def adaptive_dtheta(self, dtheta, theta, ess_tol):
        N = self.nensemble[self.ensemble_rank]
        dtheta_list = []
        ttheta_list = []
        ess_list = []
        esstheta_list = []
        ttheta = 0
        #M = sum(self.nensemble)
        #print(M)
        self.weight_arr.synchronise(root=0)
        if self.ensemble_rank == 0:
            logweights = self.weight_arr.data()
            ess =0.
            ess_theta = 0.
            theta_weight = []
            while ess < ess_tol*sum(self.nensemble):
                # renormalise using dtheta
                weights = np.exp(-dtheta*logweights)
                weights /= np.sum(weights)
                ess = 1/np.sum(weights**2)
                ess_list.append(ess)
            

                if ess < ess_tol*sum(self.nensemble):
                    dtheta = 0.5*dtheta
                ttheta += dtheta
                ttheta_list.append(ttheta)
                theta_weight = np.exp(-ttheta*logweights)
                theta_weight /= np.sum(theta_weight)
                ess_theta = 1/np.sum(theta_weight**2)
                esstheta_list.append(ess_theta)
                dtheta_list.append(dtheta)
            #print("ess_theta", esstheta_list)
            # abusing owned array to send dtheta
            # to all ensemble members
            for i in range(self.nglobal):
                self.dtheta_arr[i]=dtheta

        # broadcast dtheta to every rank
        self.dtheta_arr.synchronise()
        dtheta = self.dtheta_arr.data()[0]
        theta += dtheta
        #print(theta)
        return dtheta

        
    def assimilation_step(self, y, log_likelihood, ess_tol=0.8):
        N = self.nensemble[self.ensemble_rank]
        weights = np.zeros(N)
        new_weights = np.zeros(N)
        self.ess_temper = []
        self.theta_temper = []

        theta = .0
        while theta <1.: #  Tempering loop
            dtheta = 1.0 - theta
            # forward model step
            for i in range(N):
                # put result of forward model into new_ensemble
                self.model.run(self.ensemble[i], self.new_ensemble[i])
                Y = self.model.obs()
                self.weight_arr.dlocal[i] = log_likelihood(y,Y)

            # adaptive dtheta choice
            dtheta = self.adaptive_dtheta(dtheta, theta,  ess_tol)
            theta += dtheta
            self.theta_temper.append(theta)
            PETSc.Sys.Print("theta", theta, "dtheta", dtheta)

            #print("theta_list", self.theta_temper)

            # resampling BEFORE jittering
            self.parallel_resample()

            for l in range(self.n_jitt): # Jittering loop
                if self.verbose:
                    PETSc.Sys.Print("Jitter, Temper step", l, k)
                    
                # forward model step
                for i in range(N):
                    if self.MALA:
                        if not self.model_taped:
                            self.model_taped = True
                            pyadjoint.tape.continue_annotation()
                            self.model.run(self.ensemble[i],
                                           self.new_ensemble[i])
                            obs_list = self.model.obs_symbolic()
                            #set the controls
                            self.m = self.model.controls() + [y]
                            #requires log_likelihood to return symbolic
                            Y = self.model.obs()
                            self.MALA_J = log_likelihood(y,Y)
                            self.Jhat = ReducedFunctional(self.MALA_J, self.m)
                            pyadjoint.tape.pause_annotation()
                        #pyadjoint.get_working_tape().visualise(open_in_browser=True)
                        # run the model and get the functional value with
                        # ensemble[i]
                        self.Jhat(self.ensemble[i]+[y])
                        # use the taped model to get the derivative
                        g = self.Jhat.derivative()
<<<<<<< HEAD
                        print("rank "+str(self.ensemble_rank)+
                              " gmax/min "+str(g[1].dat.data[:].max())
                              +" "+str(g[1].dat.data[:].min()))
=======
                        #print(g[1].at(0.1))
>>>>>>> bfa454ba6ddbefb8556f6aae22410b809d23fb7d
                        # proposal
                        self.model.copy(self.ensemble[i],
                                        self.proposal_ensemble[i])
                        self.model.randomize(self.proposal_ensemble[i],
                                             Constant((2-self.rho)/(2+self.rho)),
                                             Constant((8*self.rho)**0.5/(2+self.rho)),
                                             gscale=Constant(-2*self.rho/(2+self.rho)),g=g)
                    else:
                        # proposal PCN
                        self.model.copy(self.ensemble[i],
                                        self.proposal_ensemble[i])
                        self.model.randomize(self.proposal_ensemble[i],
                                             self.rho,
                                             (1-self.rho**2)**0.5)
                    # put result of forward model into new_ensemble
                    self.model.run(self.proposal_ensemble[i],
                                   self.new_ensemble[i])

                    # particle weights
                    Y = self.model.obs()
                    new_weights[i] = exp(-theta*log_likelihood(y,Y))
                    #accept reject of MALA and Jittering 
                    if l == 0:
                        weights[i] = new_weights[i]
                    else:
                        # Metropolis MCMC
                        if self.MALA:
                            p_accept = 1
                        else:
                            p_accept = min(1, new_weights[i]/weights[i])
                        # accept or reject tool
                        u = self.model.rg.uniform(self.model.R, 0., 1.0)
                        if u.dat.data[:] < p_accept:
                            weights[i] = new_weights[i]
                            self.model.copy(self.proposal_ensemble[i],
                                            self.ensemble[i])

        if self.verbose:
            print("Advancing ensemble")
        self.model.run(self.ensemble[i], self.ensemble[i])
        if self.verbose:
            print("assimilation step complete")
