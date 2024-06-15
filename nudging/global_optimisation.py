import firedrake as fd
from operator import mul
import petsc4py.PETSc as PETSc
from pyadjoint.enlisting import Enlist

class ensemble_petsc_interface:
    def init__(self, X, ensemble):
        """
        Build a PETSc vec over the global communicator
        using an example set of list of overloaded types
        (one for each ensemble rank)

        inputs:
        X - a list of overloaded types
        ensemble - Firedrake.Ensemble ensemble communication object
        """

        # use X and ensemble to compute size and distribution of
        # PETSc vec

        X = Enlist(X)
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
        mixed_function_space = mul(function_spaces)
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
