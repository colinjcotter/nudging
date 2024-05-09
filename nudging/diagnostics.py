from abc import ABCMeta, abstractmethod
import firedrake as fd

class base_diagnostic(object, metaclass=ABCMeta):
    """
    Base class for diagnostics.

    :arg dtype: the dtype for the diagnostic computed for each particle.
    :arg comm: the ensemble subcommunicator to communicate over.
    :arg stage: the stage of the assimilation step to compute the diagnostic.
    """
    
    def __init__(self, stage, comm, dtype=None):
        self.stage = stage
        self.potential_arr = SharedArray(partition=self.nensemble,
                                         dtype=dtype,
                                         comm=comm)


    def apply(self, ensemble):
        
