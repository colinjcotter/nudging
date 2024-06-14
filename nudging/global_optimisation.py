import firedrake as fd
from pyop2.mpi import MPI
import warnings

class optimisation_vector(object):
    def __init__(self, allocate, ensemble):
        """
        A class for doing optimisation
        on lists of pyadjoint overloaded types. We provide a minimal
        amount of inner product space operations required for optimisation
        algorithms.

        allocate - a method/function that allocates a new list of the 
        required types.
        ensemble - a Firedrake Ensemble object for ensemble MPI communication.
        """
        self.allocate = allocate
        self.vec = allocate()
        self.ensemble = ensemble

    def copy(self, scale=1):
        vec = self.allocate()
        for i, item in enumerate(self.vec):
            if isinstance(item, float):
                vec[i] = scale*self.vec[i]
            elif isinstance(item, Fd.Function):
                vec[i].assign(scale*self.vec[i])
            else:
                raise NotImplementedError("This type of vector is not supported.")
        return vec
    
    def __iadd__(self, other):
        for i, item in enumerate(self.vec):
            if isinstance(item, float):
                self.vec[i] += other[i]
            elif isinstance(item, Fd.Function):
                self.vec[i] += other[i]
            else:
                raise NotImplementedError("This type of vector is not supported.")
        return self

    def __imul__(self, other):
        for i, item in enumerate(self.vec):
            if isinstance(item, float):
                self.vec[i] *= other
            elif isinstance(item, Fd.Function):
                self.vec[i] *= other
            else:
                raise NotImplementedError("This type of vector is not supported.")
        return self

    def inner(self, other):
        val = 0.
        for i, item in enumerate(self.vec):
            if isinstance(item, float):
                val += self.vec[i]*other[i]
            elif isinstance(item, Fd.Function):
                inner = fd.inner(self.vec[i], other[i])
                val += fd.assemble(inner*fd.dx)
            else:
                raise NotImplementedError("This type of vector is not supported.")
        # allreduce over communicator
        ensemble_comm = self.ensemble.ensemble_comm
        val = ensemble_comm.allreduce(sendobj=local_functional, op=MPI.SUM)
        return val

    def norm(self):
        return (self.inner(self, self))**0.5

class gradient_descent_solver(object):
    def __init__(self, allocate, J, gain):
        """
        A class implementing a basic gradient descent optimisation solver
        for EnsembleReducedFunctional.

        allocate - a method/function that allocates a new list of the 
        required types that are provided to f.
        J - the EnsembleReducedFunctional
        gain - the gain/damping parameter/learning rate for the
        gradient descent.
        """
        self.allocate = allocate
        self.J = J
        self.gain = gain

    def minimise(self, x, tol=1.0e-3, maxits=1000, verbose=False):
        """
        Minimise via gradient descent until the norm of the gradient
        is below the tolerance, or the maximum number of iterations is reached.

        x - an optimisation_vector containing the initial guess which
        will be overwritten with the gradient descent steps.
        tol - the tolerance.
        maxits - the maximum number of iterations.
        """
        X = optimisation_vector(x)
        self.J(X.vec)
        # actually returns a gradient
        der = optimisation_vector(self.J.derivative())
        residual = der.norm()
        for i in range(maxits):
            if residual < tol:
                break
            X -= der*self.gain
            der = optimisation_vector(self.J.derivative())
        if i == maxits - 1 and verbose:
            warnings.warn("Maximum iterations reached before reaching tolerance in gradient descent.")
