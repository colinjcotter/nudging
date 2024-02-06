import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class Camsholm(base_model):
    def __init__(self, n, nsteps, xpoints, seed=12353, lambdas=False,
                 dt=0.025, alpha=1.0, mu=0.01, salt=False):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.mu = mu
        self.dt = dt
        self.seed = seed
        self.salt = salt
        self.xpoints = xpoints
        self.lambdas = lambdas  # include lambdas in allocate

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = fd.PeriodicIntervalMesh(self.n, 40.0, comm=comm)
        x, = fd.SpatialCoordinate(self.mesh)

        self.V = fd.FunctionSpace(self.mesh, "CG", 1)
        V = fd.FunctionSpace(self.mesh, "CG", 1)
        self.W = fd.MixedFunctionSpace((V, V))
        self.w0 = fd.Function(self.W)
        m0, u0 = self.w0.split()
        One = fd.Function(V).assign(1.0)
        dx = fd.dx
        self.Area = fd.assemble(One*dx)

        # Solver for the initial condition for m.
        alphasq = self.alpha**2
        p = fd.TestFunction(V)
        m = fd.TrialFunction(V)

        am = p*m*dx
        Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx
        mprob = fd.LinearVariationalProblem(am, Lm, m0)
        sp = {'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = fd.LinearVariationalSolver(mprob,
                                                 solver_parameters=sp)

        # Build the weak form of the timestepping algorithm.
        p, q = fd.TestFunctions(self.W)
        self.w1 = fd.Function(self.W)
        self.w1.assign(self.w0)
        m1, u1 = fd.split(self.w1)   # for n+1 th time
        m0, u0 = fd.split(self.w0)   # for n th time

        # Setup noise term using Matern formula
        self.W_F = fd.FunctionSpace(self.mesh, "DG", 0)
        self.dW = fd.Function(self.W_F)
        dphi = fd.TestFunction(V)
        du = fd.TrialFunction(V)

        cell_area = fd.CellVolume(self.mesh)
        alpha_w = (1/cell_area**0.5)
        kappa_inv_sq = 2*cell_area**2

        dU_1 = fd.Function(V)
        dU_2 = fd.Function(V)
        dU_3 = fd.Function(V)
        a_w = (dphi*du + kappa_inv_sq*dphi.dx(0)*du.dx(0))*dx
        L_w0 = alpha_w*dphi*self.dW*dx
        w_prob0 = fd.LinearVariationalProblem(a_w, L_w0, dU_1)
        self.wsolver0 = fd.LinearVariationalSolver(w_prob0,
                                                   solver_parameters=sp)
        L_w1 = dphi*dU_1*dx
        w_prob1 = fd.LinearVariationalProblem(a_w, L_w1, dU_2)
        self.wsolver1 = fd.LinearVariationalSolver(w_prob1,
                                                   solver_parameters=sp)
        L_w = dphi*dU_2*dx
        w_prob = fd.LinearVariationalProblem(a_w, L_w, dU_3)
        self.wsolver = fd.LinearVariationalSolver(w_prob,
                                                  solver_parameters=sp)

        # finite element linear functional
        Dt = self.dt
        mh = 0.5*(m1 + m0)
        uh = 0.5*(u1 + u0)

        if self.salt:
            # SALT noise
            v = uh*Dt+dU_3*Dt**0.5
        else:
            # additive noise
            v = uh*Dt
        L = ((q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx
             + (p*(m1-m0) + (p*v.dx(0)*mh - p.dx(0)*v*mh)
                + self.mu*Dt*p.dx(0)*mh.dx(0))*dx)

        if self.salt:
            L += p*dU_3*Dt**0.5*dx

        # timestepping solver
        uprob = fd.NonlinearVariationalProblem(L, self.w1)
        sp = {'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.usolver = fd.NonlinearVariationalSolver(uprob,
                                                     solver_parameters=sp)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.linspace(0, 40, num=self.xpoints, endpoint=False)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = fd.VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = fd.FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        # copy input into model variables for taping
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        # copy initial condition into model variable
        self.w0.assign(self.X[0])

        # ensure momentum and velocity are syncronised
        self.msolve.solve()

        # do the timestepping
        for step in range(self.nsteps):
            # get noise variables and lambdas
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # solve  dW --> dU0 --> dU1 --> dU
            self.wsolver0.solve()
            self.wsolver1.solve()
            self.wsolver.solve()
            # advance in time
            self.usolver.solve()
            # copy output to input
            self.w0.assign(self.w1)

        # return outputs
        X1[0].assign(self.w0)  # save sol from the nstep th time

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fd.adjoint.Control(self.X[i]))
        return controls_list

    def obs(self):
        m, u = self.w0.split()
        Y = fd.Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [fd.Function(self.W)]
        for i in range(self.nsteps):
            dW = fd.Function(self.W_F)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = fd.Function(self.W_F)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.W_F, 0., 0.5))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        nsteps = self.nsteps
        dt = self.dt

        # This should have the effect of returning
        # sum_n sum_i (dt*lambda_i^2/2 -  lambda_i*dW_i)
        # where dW_i are the contributing Brownian increments
        # and lambda_i are the corresponding Girsanov variables

        # in the case of our DG0 Gaussian random fields, there
        # is one per cell, so we can formulate this for UFL in a
        # volume integral by dividing by cell volume.

        dx = fd.dx
        for step in range(nsteps):
            lambda_step = self.X[nsteps + 1 + step]
            dW_step = self.X[1 + step]
            cv = fd.CellVolume(self.mesh)
            dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx
                                 - (1/cv)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
