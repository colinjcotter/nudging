import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class KS(base_model):
    def __init__(self, nsteps, xpoints, n=100, seed=12353, lambdas=False,
                 dt=0.01, nu=0.02923, dc=0.01, L=10.):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed
        self.nu = nu #  viscosity
        self.dc = dc #  noise coefficient
        self.L = L #  domain width
        self.xpoints = xpoints
        self.lambdas = lambdas  # include lambdas in allocate

    def setup(self, mesh=None, comm=MPI.COMM_WORLD):
        if not mesh:
            mesh = fd.PeriodicIntervalMesh(self.n, self.L,
                                           comm=comm, name="ksmesh")
        self.mesh = mesh
        x, = fd.SpatialCoordinate(mesh)

        V = fd.FunctionSpace(mesh, "Hermite", 3)
        self.V = V

        un = fd.Function(V)
        self.un = un
        unp1 = fd.Function(V)
        self.unp1 = unp1
        uh = (un + unp1)/2
        
        v = fd.TestFunction(V)
        
        dt = 0.01
        dT = fd.Constant(dt)

        self.DG0 = fd.FunctionSpace(mesh, "DG", 0)
        dW = fd.Function(self.DG0)
        self.dW = dW
        alpha = fd.Constant(1.0) # viscosity
        beta = fd.Constant(0.02923) # hyperviscosity
        gamma = fd.Constant(1.) # advection
        dc = fd.Constant(0.001) # diffusion coefficient for noise
        area = fd.CellVolume(mesh)
        dx = fd.dx

        eqn = (
            v*(unp1 - un)*dx
            - dT*alpha*v.dx(0)*uh.dx(0)*dx
            + dT*beta*(
                v.dx(0).dx(0)*uh.dx(0).dx(0)*dx
            )
            - dT*gamma*0.5*v.dx(0)*uh*uh*dx
            - (dT/area)**0.5*dc*dW*v*dx
        )

        params = {
            "snes_atol": 1.0e-50,
            "snes_rtol": 1.0e-6,
            "snes_stol": 1.0e-50,
            "ksp_type":"preonly",
            "pc_type":"lu"
        }

        #make the solver
        KSProb = fd.NonlinearVariationalProblem(eqn, unp1)
        self.KSSolver = fd.NonlinearVariationalSolver(KSProb,
                                                      solver_parameters=params)

        #stuff for interpolation to VOM
        CG3 = fd.FunctionSpace(mesh, "CG", 3)
        self.uout = fd.Function(CG3)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.linspace(0, self.L, num=self.xpoints, endpoint=False)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = fd.VertexOnlyMesh(mesh, x_obs_list)
        self.VVOM = fd.FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        # copy input into model variables for taping
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        # copy initial condition into model variable
        self.un.assign(self.X[0])
        self.unp1.assign(self.un)
        # do the timestepping
        for step in range(self.nsteps):
            # get noise variables and lambdas
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # advance in time
            self.KSSolver.solve()
            # copy output to input
            self.un.assign(self.unp1)

        # return outputs
        X1[0].assign(self.un)  # save sol from the nstep th time

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fd.adjoint.Control(self.X[i]))
        return controls_list

    def obs(self):
        Y = fd.Function(self.VVOM)
        self.uout.interpolate(self.un)
        Y.interpolate(self.uout)
        return Y

    def allocate(self):
        particle = [fd.Function(self.V)]
        for i in range(self.nsteps):
            dW = fd.Function(self.DG0)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = fd.Function(self.DG0)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.DG0, 0., 1.))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        nsteps = self.nsteps
        dt = fd.Constant(self.dt)

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
            cv = fd.CellVolume(mesh)
            dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx
                                 - (1/cv)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
