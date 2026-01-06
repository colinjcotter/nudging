from nudging.models.sim_model import SimModel
from nudging import sim_filter

import pytest


def parallel_resample():
    nensemble = [2, 2, 2, 2, 2]

    model = SimModel()

    simfilter = sim_filter()
    simfilter.setup(nensemble, model)
    model.ensemble_rank = simfilter.ensemble_rank

    s = [4, 3, 7, 0, 1, 5, 2, 6, 9, 8]
    simfilter.assimilation_step(s=s)
    for i in range(len(simfilter.ensemble)):
        iglobal = simfilter.layout.transform_index(i, itype="l", rtype="g")
        s_val = simfilter.s_copy[iglobal]
        e_val = simfilter.ensemble[i][0]
        assert s_val - int(e_val.dat.data[:].min()) == 0
        assert s_val - int(e_val.dat.data[:].max()) == 0


@pytest.mark.parallel(nprocs=5)
def test_parallel_resample_1():
    parallel_resample()


@pytest.mark.parallel(nprocs=10)
def test_parallel_resample_2():
    parallel_resample()
