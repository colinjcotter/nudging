import firedrake as fd
from operator import mul
from functools import reduce
from pyadjoint.enlisting import Enlist
from firedrake.petsc import PETSc, OptionsManager, flatten_parameters
import firedrake.adjoint as fadj


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
                raise NotImplementedError(
                    "Controls must be Firedrake Functions")
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
        self.sizes = sizes
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
        ws = self.w.subfunctions
        idx = 0
        for X in self.X:
            Xo = X.tape_value().copy(deepcopy=True)
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
        # we have to do it this way so we can copy from
        # local to global correctly.
        w1 = self.w.copy(deepcopy=True)
        gcomm = self.ensemble.global_comm
        with w1.dat.vec as fvec:
            vec = PETSc.Vec().createWithArray(fvec.array,
                                              size=self.sizes,
                                              comm=gcomm)
        vec.setFromOptions()
        return vec


class ParameterisedEnsembleReducedFunctional:
    def __init__(self, Js, Controls, Parameters, ensemble,
                 gather_functional):
        self.controls = Controls
        full_Controls = Controls + Parameters
        self.Parameters = []
        for i, parameter in enumerate(Parameters):
            self.Parameters.append(parameter.tape_value())
        derivative_components = [i for i in range(len(Controls))]
        self.rf = fadj.EnsembleReducedFunctional(
            Js, full_Controls, ensemble, scatter_control=False,
            gather_functional=gather_functional,
            derivative_components=derivative_components)
        self.derivative_components = derivative_components

    def update_parameters(self, Parameters):
        for i, parameter in enumerate(Parameters):
            self.Parameters[i].assign(parameter)

    def __call__(self, inputs):
        full_inputs = inputs + self.Parameters
        return self.rf(full_inputs)

    def derivative(self):
        return self.rf.derivative()[self.derivative_components]


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
            X = interface.vec2list(x)
            J_val = Jhat(X)
            dJ = Jhat.derivative()
            interface.list2vec(dJ).copy(g)
            return J_val

        tao.setObjectiveGradient(objective_gradient, None)

        class inner_mat:
            def __init__(self, interface):
                self.interface = interface
                # using L2 norm/inner product
                W = interface.mixed_function_space
                self.v = fd.TestFunction(W)

            def mult(self, mat, X, Y):
                # abusing vec2list side effect of copying to interface.w
                self.interface.vec2list(X)
                ycofunc = fd.assemble(fd.inner(self.v,
                                               self.interface.w)*fd.dx)
                gcomm = ensemble.global_comm
                with ycofunc.dat.vec as fvec:
                    vec = PETSc.Vec().createWithArray(fvec.array,
                                                      size=interface.sizes,
                                                      comm=gcomm)
                vec.setFromOptions()
                vec.copy(Y)

        sizes = interface.sizes
        M = PETSc.Mat().createPython([sizes, sizes],
                                     comm=ensemble.global_comm)
        M.setPythonContext(inner_mat(interface))
        M.setUp()
        tao.setGradientNorm(M)

        flat_solver_parameters = flatten_parameters(solver_parameters)
        options = OptionsManager(flat_solver_parameters,
                                 options_prefix)
        tao.setOptionsPrefix(options.options_prefix)
        options.set_from_options(tao)

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
        x = self.tao.solve()
        X = self.interface.vec2list(x)
        return X
