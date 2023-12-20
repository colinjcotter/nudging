import firedrake as fd
from nudging import base_model
from pyop2.mpi import MPI


class LGModel(base_model):
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

    def setup(self, comm=MPI.COMM_WORLD):
        self.mesh = fd.UnitIntervalMesh(2, comm=comm)

        self.R = fd.FunctionSpace(self.mesh, "R", 0)
        self.u = fd.Function(self.R)
        self.dW = fd.Function(self.R)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs_list = [[0.2]]
        self.VOM = fd.VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = fd.FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        self.u.assign(self.X[0])
        for step in range(self.nsteps):
            self.dW.assign(self.X[step+1])
            self.u.assign(self.A*self.u + self.dt**2*self.dW)
        X1[0].assign(self.u)

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fd.Control(self.X[i]))
        return controls_list

    def obs(self):
        u = self.u
        Y = fd.Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [fd.Function(self.R)]
        for i in range(self.nsteps):
            dW = fd.Function(self.R)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = fd.Function(self.R)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.R, 0., 1.))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        raise NotImplementedError
