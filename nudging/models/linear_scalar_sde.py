import firedrake as fd
import firedrake.adjoint as fadj
from nudging import base_model
from pyop2.mpi import MPI


class LSDEModel(base_model):
    def __init__(self, A, D, nsteps, dt, seed=36353, lambdas=False):
        """
        Discretisation of 1D linear SDE,
        dx = A*x*dt + D*dW
        x(t) + exp(At)*x(0) + int_0^t exp(A(t-s))dW
        """
        self.A = A
        self.D = D
        self.nsteps = nsteps
        self.lambdas = lambdas
        self.dt = dt
        self.seed = seed

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = fd.UnitIntervalMesh(1, comm=comm)

        self.R = fd.FunctionSpace(self.mesh, "R", 0)
        self.V = fd.FunctionSpace(self.mesh, "DG", 0)
        self.u = fd.Function(self.V)
        self.dW = fd.Function(self.V)
        self.Lambda = fd.Function(self.V)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs_list = [[0.2]]
        self.VOM = fd.VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = fd.FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1, s=None):
        if not s:
            s = [1.0]*self.nsteps
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        self.u.assign(self.X[0])

        u = self.u
        dW = self.dW
        dt = self.dt
        A = self.A
        D = self.D

        if self.lambdas:
            self.Lambda.assign(0.)
        for step in range(self.nsteps):
            if self.lambdas:
                self.Lambda.assign(self.Lambda + s[step]*self.X[self.nsteps+step+1])
                dW.assign(self.X[step+1] + dt**0.5*self.Lambda)
            else:
                dW.assign(self.X[step+1])
            u.assign(u*(1 - dt*A/2) + D*dt**0.5*dW)/(1 + dt*A/2)
        X1[0].assign(self.u)

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fadj.Control(self.X[i]))
        return controls_list

    def obs(self):
        u = self.u
        Y = fd.Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [fd.Function(self.V)]
        for i in range(self.nsteps):
            dW = fd.Function(self.V)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = fd.Function(self.V)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.V, 0., 1.))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self, s=None):
        nsteps = self.nsteps
        dt = self.dt
        dx = fd.dx
        cv = 1.0  # should be fd.CellVolume(self.mesh)

        self.Lambda.assign(0.)
        for step in range(nsteps):
            # X[0] is the model state
            # X[1], .., X[nsteps] are the dWs
            # X[nsteps+1], .., X[2*nsteps] are the lambdas
            if s:
                self.Lambda.assign(self.Lambda + s[step]*self.X[nsteps + 1 + step])
            else:
                self.Lambda.assign(self.Lambda + self.X[nsteps + 1 + step])
            lambda_step = self.Lambda
            dW_step = self.X[1 + step]
            dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx
                                - (1/cv)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
