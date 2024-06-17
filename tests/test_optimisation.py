import firedrake as fd
import firedrake.adjoint as fadj
import pytest
from nudging import ensemble_tao_solver
from pyop2.mpi import MPI


@pytest.mark.parallel(nprocs=4)
def test_ensemble_tao_solver():
    fadj.continue_annotation()

    ensemble = fd.Ensemble(MPI.COMM_WORLD, 2)
    rank = ensemble.ensemble_comm.rank
    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    R = fd.FunctionSpace(mesh, "R", 0)

    n_Js = [2, 3]
    Js_offset = [0, 2]
    Js = []
    Controls = []
    xs = []
    for i in range(n_Js[rank]):
        val = Js_offset[rank]+i+1
        x = fd.Function(R, val=val)
        J = fd.assemble(x * x * fd.dx(domain=mesh))
        Js.append(J)
        Controls.append(fadj.Control(x))
        xs.append(x)

    Jg_m = []
    as1 = []
    for i in range(5):
        a = fadj.AdjFloat(1.0)
        as1.append(a)
        Jg_m.append(fadj.Control(a))
    Ja = as1[0]**2
    for i in range(1, 5):
        Ja += as1[i]**2
    Jg = fadj.ReducedFunctional(Ja, Jg_m)
    val = 1.0**2 + 2.0**2 + 3.0**2 + 4.0**2 + 5.0**2
    assert Jg([1., 2., 3., 4., 5.]) == val
    rf = fadj.EnsembleReducedFunctional(Js, Controls, ensemble,
                                        scatter_control=False,
                                        gather_functional=Jg)
    fadj.stop_annotating()

    solver_parameters = {
        "tao_type": "lmvm",
        "tao_cg_type": "pr",
    }

    solver = ensemble_tao_solver(rf, ensemble,
                                 solver_parameters=solver_parameters)
    solver.solve()
    its = solver.tao.getIterationNumber()
    assert its < 30
