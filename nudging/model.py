from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property
from firedrake import Function, FunctionSpace, PCG64, RandomGenerator
import firedrake as fd
import numpy as np

class base_model(object, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, comm):
        """
        comm - the MPI communicator used to build the mesh object

        This method should build the mesh and everything else that
        needs the mesh
        """
        pass

    @abstractmethod
    def run(self, X0, X1):
        """
        X0 - a Firedrake Function containing the initial condition
        X1- a Firedrake Function to copy the result into
        """
        pass

    @abstractmethod
    def obs(self):
        """
        Observation operator
        returns

        obs - a numpy array of the observations from current model state
        """
        pass
    

    @abstractmethod
    def allocate(self):
        """
        Allocate a function to store a model state
        
        returns
        X - a Function of the required type
        """
        pass

    @abstractmethod
    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        """
        input X: a list containing the state 
        plus the brownian motions in a list
        it is assumed that X[0] is the state and the rest of
        the list is the Brownian motions

        replace dW_o <- c1*dW_o + c2*dW_n 
        where dW_o is the old BM and 
        dW_n is the new BM
        if g is present then add gscale*g
        where g is the derivative of the functional w.r.t. 
        the Brownian motion 
        noting that we have the state and the observations in
        the controls as well
        i.e. g[0] - state (ignore for randomize)
        g[1...m] - brownian motions
        g[m+1] - observations (ignore for randomize)
        """
        pass

        @abstractmethod

        rho_uv, rho_vu = self.model.rhos_MALA(potentials[i],
                                              new_potentials[i],
                                              g, new_g, theta)

    @cached_property
    def R(self):
        """
        An R space to deal with uniform random numbers
        for resampling etc
        """
        R = FunctionSpace(self.mesh, "R", 0)
        return R

    @cached_property
    def U(self):
        """
        An R space function to deal with uniform random numbers
        for resampling
        """
        U = Function(self.R)
        return U

    @abstractmethod
    def controls(self):
        """
        Return a list of the inputs to the model as Controls.
        """

    @cached_property
    def rg(self):
        pcg = PCG64(seed=self.seed)
        return RandomGenerator(pcg)

    def copy(self, Xin, Xout):
        for i in range(len(Xin)):
            Xout[i].assign(Xin[i])
