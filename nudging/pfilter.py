from abc import ABCMeta, abstractmethod, abstractproperty
from firedrake import *
from .resampling import *
import numpy as np
from scipy.special import logsumexp


class base_filter(object, metaclass=ABCMeta):
    ensemble = []
    #new_ensemble = []

    def __init__(self):
        pass
        
    def setup(self, nensemble, model):
        """
        Construct the ensemble

        nensemble - number of ensemble members
        model - the model to use
        """
        self.model = model
        self.nensemble = nensemble
        for i in range(nensemble):
            self.ensemble.append(model.allocate())
            #self.new_ensemble.append(model.allocate())

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


class bootstrap_filter(base_filter):
    def __init__(self, nsteps, noise_shape):
        self.nsteps = nsteps
        self.noise_shape = noise_shape

    def setup(self, nensemble, model):
        super().setup(nensemble, model)
        # allocate working memory for resampling
        self.new_ensemble = []
        for i in range(self.nensemble):
            self.new_ensemble.append(self.model.allocate())

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        weights = np.zeros(N)
        # forward model step
        for i in range(N):
            W = np.random.randn(*(self.noise_shape))
            self.model.run(self.nsteps, W,
                           self.ensemble[i], self.ensemble[i])   # solving FEM with ensemble as input and final sol ensemble
        
            # particle weights
            Y = self.model.obs(self.ensemble[i])
            weights[i] = log_likelihood(y-Y)

        # renormalise
        weights = np.exp(-weights)
        weights /= np.sum(weights)
        self.ess = 1/np.sum(weights**2)

        s = residual_resampling(weights)
        for i in range(N):
            self.new_ensemble[i].assign(self.ensemble[s[i]])
        for i in range(N):
            self.ensemble[i].assign(self.new_ensemble[i])


class jittertemp_filter(base_filter):
    def __init__(self, nsteps, noise_shape, n_temp, n_jitt, rho,
                 verbose=False):
        self.nsteps = nsteps
        self.noise_shape = noise_shape
        self.n_temp = n_temp
        self.n_jitt = n_jitt
        self.rho = rho
        self.verbose=verbose

    def setup(self, nensemble, model):
        super().setup(nensemble, model)
        # allocate working memory for resampling and forward model
        self.new_ensemble = []
        for i in range(self.nensemble):
            self.new_ensemble.append(self.model.allocate())

    def assimilation_step(self, y, log_likelihood):
        N = len(self.ensemble)
        weights = np.zeros(N)
        weights[:] = 1/N
        new_weights = np.zeros(N)
        self.ess = []
        W = np.random.randn(N, *(self.noise_shape))
        for k in range(self.n_temp): #  Tempering loop
            for l in range(self.n_jitt): # Jittering loop
                if self.verbose:
                    print("Jitter, Temper step", l, k)
                # proposal
                Wnew = self.rho*W + (1-self.rho**2)**0.5*np.random.randn(N, *(self.noise_shape))

                # forward model step
                for i in range(N):
                    # put result of forward model into new_ensemble
                    self.model.run(self.nsteps, Wnew[i, :],
                                   self.ensemble[i], self.new_ensemble[i])
        
                    # particle weights
                    Y = self.model.obs(self.new_ensemble[i])
                    new_weights[i] = exp(-((k+1)/self.n_temp)*log_likelihood(y-Y))
                    if l == 0:
                        weights[i] = new_weights[i]
                    else:
                        #  Metropolis MCMC
                        p_accept = min(1, new_weights[i]/weights[i])
                        #accept or reject tool
                        if np.random.rand() < p_accept:
                            weights[i] = new_weights[i]
                            W[i,:] = Wnew[i,:]

                weights /= np.sum(weights)
                self.e_weight = weights
                self.ess.append(1/np.sum(weights**2))

            # resampling after jittering
            s = residual_resampling(weights)
            self.e_s = s
            if self.verbose:
                print("Updating ensembles")
            for i in range(N):
                self.new_ensemble[i].assign(self.ensemble[s[i]])
                Wnew[i, :] = W[s[i], :]
            for i in range(N):
                self.ensemble[i].assign(self.new_ensemble[i])
                W[i, :] = Wnew[i, :]

        if self.verbose:
            print("Advancing ensemble")
        self.model.run(self.nsteps, W[i, :],
                                   self.ensemble[i], self.ensemble[i])
        if self.verbose:
            print("assimilation step complete")
