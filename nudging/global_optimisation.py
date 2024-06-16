import firedrake as fd
from operator import mul
import petsc4py.PETSc as PETSc
from functools import reduce
from pyadjoint.enlisting import Enlist
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters


class ensemble_petsc_interface:
    def __init__(self, X, ensemble):
        """
        Build a PETSc vec over the global communicator
        using an example set of list of overloaded types
        (one for each ensemble rank)

        inputs:
        X - a list of Controls (must be Functions currently)
        ensemble - Firedrake.Ensemble ensemble communication object
        """

        # use X and ensemble to compute size and distribution of
        # PETSc vec

        X = Enlist(X)
        self.X = X
        self.ensemble = ensemble

        # WE ARE GOING TO CHEAT AND USE A MIXED FUNCTION SPACE TO
        # STORE ALL THE ENSEMBLE LOCAL DATA THIS MEANS THAT THIS WILL
        # ONLY WORK IF ALL X ARE FIREDRAKE FUNCTIONS

        function_spaces = []
        for x in X:
            fn = x.tape_value()
            if not isinstance(fn, fd.Function):
                raise NotImplementedError("Controls must be Firedrake Functions")
            function_spaces.append(fn.ufl_function_space())
        # This will flatten mixed spaces into one mixed space
        mixed_function_space = reduce(mul, function_spaces)
        self.mixed_function_space = mixed_function_space
        w = fd.Function(mixed_function_space)

        # sniff the sizes to create the global PETSc Vec
        with w.dat.vec_ro as wvec:
            local_size = wvec.local_size
            global_size = wvec.size
        sizes = (local_size, global_size)

        with w.dat.vec as wvec:
            self.Vec = PETSc.Vec().createWithArray(wvec.array,
                                                   size=sizes,
                                                   comm=ensemble.global_comm)

        # some useful working memory
        self.w = w

    def vec2list(self, vec):
        """
        Transfer contents of vec to list of same types as X and return it.
        (makes a new vec)
        vec - a PETSc vec

        returns
        X - a list of Firedrake.Function of same types as self.X
        """

        with self.w.dat.vec_wo as wvec:
            # PETSc Vec copies into input to copy method
            vec.copy(wvec)
        X_out = []
        index = 0.
        ws = self.w.subfunctions
        idx = 0
        for X in self.X:
            Xo = X.tape_value().copy()
            for fn in Xo.subfunctions:
                fn.assign(ws[idx])
                idx += 1
            X_out.append(Xo)
        return X_out

    def list2vec(self, X):
        """
        Transfer contents of list of same types as X to PETSc vec
        and return it.
        (Makes a new list of new functions)

        X - a list of Firedrake.Function with same types as self.X

        returns 
        vec - a PETSc vec
        """

        # copy list into self.w
        idx = 0
        for x in X:
            for fn in x.subfunctions:
                self.w.sub(idx).assign(fn)
                idx += 1

        # get copy of self.w vec and return
        with self.w.dat.vec_ro as wvec:
            vec = PETSc.Vec(wvec)
        return vec


class ensemble_tao_solver:
    def __init__(self, Jhat, ensemble,
                 solver_parameters, options_prefix=""):
        """
        Jhat - firedrake.EnsembleReducedFunctional
        ensemble - Firedrake.Ensemble ensemble communication object
        solver_parameters - a dictionary of solver parameters
        """

        X = Jhat.controls
        interface = ensemble_petsc_interface(X, ensemble)
        tao = PETSc.TAO().create(comm=ensemble.global_comm)

        def objective_gradient(tao, x, g):
            X = interface.vec2list()
            J_val = Jhat(X)
            dJ = Jhat.derivative()
            return interface.list2vec(dJ)

        tao.setObjectiveGradient(objective_gradient, None)

        # using L2 norm/inner product
        W = interface.mixed_function_space
        u = fd.TrialFunction(W)
        v = fd.TestFunction(W)
        M = fd.assemble(fd.inner(u, v)*fd.dx).petscmat
        tao.setGradientNorm(M)

        flat_solver_parameters = flatten_parameters(solver_parameters)
        options = OptionsManager(flat_solver_parameters,
                                      options_prefix)
        tao.setOptionsPrefix(options.options_prefix)
        tao.setFromOptions()

        x = interface.list2vec(interface.X)
        tao.setSolution(x)
        tao.setUp()
        self.x = x
        self.tao = tao
        self.interface = interface

    def solve(self):
        """
        optimisation solve

        Returns:
            List of OverloadedType
        """
        self.tao.solve()
        X = self.interface.vec2list(self.x)
        return X
