from abc import ABCMeta, abstractmethod
from enum import Flag, auto
from .parallel_arrays import SharedArray


class Stage(Flag):
    AFTER_NUDGING = auto()
    AFTER_TEMPER_RESAMPLE = auto()
    AFTER_ONE_JITTER_STEP = auto()
    AFTER_JITTERING = auto()
    AFTER_ASSIMILATION_STEP = auto()


class base_diagnostic(object, metaclass=ABCMeta):
    """
    Base class for diagnostics.

    :arg dtype: the dtype for the diagnostic computed for each particle.
    :arg ecomm: the ensemble MPI subcommunicator to communicate over.
    :arg stage: the stage of the assimilation step to compute the diagnostic.
    :arg nensemble: the time partition
    """

    def __init__(self, stage, ecomm, nensemble, dtype=None):
        self.stage = stage
        self.shared_arr = SharedArray(partition=nensemble,
                                      dtype=dtype,
                                      comm=ecomm)
        self.grank = ecomm.global_comm.rank
        self.N = nensemble[ecomm.rank]

        # list of diagnostic values is only stored on global rank 0
        if self.grank == 0:
            self.values = []
            self.archive = []

    @abstractmethod
    def compute_diagnostic(self, particle):
        """
        Take in a particle and return a diagnostic value.
        """
        pass

    def gather_diagnostics(self, ensemble, descriptor):
        """
        Loop over all ensemble members, compute their
        diagnostics and gather them to rank zero
        where they are stored as a length N array
        (N is number of particles) which is placed in
        a tuple with the descriptor (a string/int/float/etc)
        and appended to the list.
        The descriptor is used to record when the
        diagnostic was taken.
        """

        # compute local values
        for i in range(self.N):
            self.shared_arr.dlocal[i] = self.compute(ensemble[i])

        # gather to ensemble rank 0
        self.potential_arr.synchronise(root=0)

        # add to the list
        if self.grank == 0:
            val = self.potential_arr.data()
            self.values.append((val, descriptor))

    def archive(self):
        """
        Put the current values into the archive and
        empty out values. This is done at the end
        of an assimilation step.
        """
        if self.grank == 0:
            self.archive.append(self.values)
            self.values = []


def compute_diagnostics(diagnostic_list, ensemble, descriptor, stage,
                        other_data={}):
    """
    Compute all diagnostics in diagnostic_list labelled with stage

    arg: diagnostic_list - a list of diagnostics
    (these are inherited from base_diagnostic)
    arg: stage - a string indicating the stage where diagnostics are called.
    arg: ensemble - the (local part of) the ensemble of particles (list)
    arg: descriptor - a descriptive string describing when the diagnostic
    was called
    """

    for diagnostic in diagnostic_list:
        if diagnostic.stage == stage:
            diagnostic.gather_diagnostics(ensemble, descriptor)
