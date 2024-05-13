from abc import ABCMeta, abstractmethod
from enum import Flag, auto
from .parallel_arrays import SharedArray
import numpy as np


class Stage(Flag):
    AFTER_NUDGING = auto()
    WITHOUT_LAMBDAS = auto()  # get diagnostics with lambda set to zero
    AFTER_TEMPER_RESAMPLE = auto()
    AFTER_ONE_JITTER_STEP = auto()
    AFTER_JITTERING = auto()
    AFTER_ASSIMILATION_STEP = auto()


class base_diagnostic(object, metaclass=ABCMeta):
    """
    Base class for diagnostics.

    :arg dtype: the dtype for the diagnostic computed for each particle.
    :arg ecomm: the Ensemble MPI communicator object.
    :arg stage: the stage of the assimilation step to compute the diagnostic.
    :arg nensemble: the time partition
    """

    def __init__(self, stage, ecomm, nensemble, dtype=None):
        self.stage = stage
        self.shared_arr = SharedArray(partition=nensemble,
                                      dtype=dtype,
                                      comm=ecomm.ensemble_comm)
        self.grank = ecomm.global_comm.rank
        self.N = nensemble[ecomm.ensemble_comm.rank]
        self.dtype = dtype
        self.Ntot = np.sum(nensemble)

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

    def get_archive(self):
        """
        Serialise the archive data and descriptors and return.
        """
        if self.grank == 0:
            # count up all the rows in the archive
            count = 0
            for i in range(len(self.archive)):
                count += len(self.archive[i])
            # create the archive array
            descriptors = []
            archive = np.zeros((count, self.Ntot), dtype=self.dtype)

            # copy the archive across
            count = 0
            for i in range(len(self.archive)):
                for j in range(len(self.archive[i])):
                    archive[count] = self.archive[i][j][0]
                    descriptors.append(self.archive[i][j][1])
                    count += 1
            return archive, descriptors
        else:
            return None

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
            self.shared_arr.dlocal[i] = self.compute_diagnostic(ensemble[i])

        # gather to ensemble rank 0
        self.shared_arr.synchronise(root=0)

        # add to the list
        if self.grank == 0:
            val = self.shared_arr.data()
            self.values.append((val, descriptor))

    def archive_diagnostic(self):
        """
        Put the current values into the archive and
        empty out values. This is done at the end
        of an assimilation step.
        """
        if self.grank == 0:
            self.archive.append(self.values)
            self.values = []


def compute_diagnostics(diagnostic_list, ensemble,
                        descriptor, stage,
                        other_data={}, run=None,
                        new_ensemble=None):
    """
    Compute all diagnostics in diagnostic_list labelled with stage

    arg: diagnostic_list - a list of diagnostics
    (these are inherited from base_diagnostic)
    arg: stage - a string indicating the stage where diagnostics are called.
    arg: ensemble - the (local part of) the ensemble of particles (list)
    - the values at the start of the assimilation window
    arg: new_ensemble - the (local part of) the ensemble of particles (list)
    - space to write output of run to
    arg: run - the run method - model is only run if we need it. If
    None, we use ensemble.
    arg: descriptor - a descriptive string describing when the diagnostic
    was called
    """

    ndiagnostics = 0
    for diagnostic in diagnostic_list:
        if diagnostic.stage == stage:
            ndiagnostics += 1

    if ndiagnostics > 0:
        if run:
            assert(new_ensemble)
            for i in range(len(ensemble)):
                run(ensemble[i], new_ensemble[i])
            use_ensemble = new_ensemble
        else:
            use_ensemble = ensemble

        for diagnostic in diagnostic_list:
            if diagnostic.stage == stage:
                diagnostic.gather_diagnostics(use_ensemble, descriptor)


def archive_diagnostics(diagnostic_list):
    for diagnostic in diagnostic_list:
        diagnostic.archive_diagnostic()
