from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self, n, nsteps, xpoints, lambdas=False,
                 dt = 0.01, alpha=1.0, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.dt = dt
        self.seed = seed
        self.xpoints = xpoints
        self.lambdas = lambdas # include lambdas in allocate

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel, width =4 and cell = self.n
        self.x, = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()       
        One = Function(self.V).assign(1.0)
        self.Area = assemble(One*dx)
        
        #Interpolate the initial condition

        #Solve for the initial condition for m.
        alphasq = self.alpha**2
        self.p = TestFunction(self.V)
        self.m = TrialFunction(self.V)
        
        self.am = self.p*self.m*dx
        self.Lm = (self.p*self.u0 + alphasq*self.p.dx(0)*self.u0.dx(0))*dx
        mprob = LinearVariationalProblem(self.am, self.Lm, self.m0)
        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = LinearVariationalSolver(mprob,
                                              solver_parameters=solver_parameters)
        
        #Build the weak form of the timestepping algorithm. 

        self.p, self.q = TestFunctions(self.W)

        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        self.m1, self.u1 = split(self.w1)   # for n+1 the  time
        self.m0, self.u0 = split(self.w0)   # for n th time 
        
        #Adding extra term included random number
        self.fx1 = Function(self.V)
        self.fx2 = Function(self.V)
        self.fx3 = Function(self.V)
        self.fx4 = Function(self.V)

        self.fx1.interpolate(0.1*sin(pi*self.x/8.))
        self.fx2.interpolate(0.1*sin(2.*pi*self.x/8.))
        self.fx3.interpolate(0.1*sin(3.*pi*self.x/8.))
        self.fx4.interpolate(0.1*sin(4.*pi*self.x/8.))

        # with added term
        self.noise_space = VectorFunctionSpace(self.mesh, "R", 0, dim=4)
        self.dW = Function(self.noise_space)
        self.Ln = (
            self.fx1*self.dW[0]+
            self.fx2*self.dW[1]+
            self.fx3*self.dW[2]+
            self.fx4*self.dW[3]
            )

        # finite element linear functional 
        Dt = Constant(self.dt)
        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*Dt+self.Ln*Dt**0.5

        self.L = ((self.q*self.u1 + alphasq*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        #def Linearfunc

        # solver

        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        
        x_obs = np.arange(0.5,self.xpoints)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1, operation = None):
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
            # advance in time
            self.usolver.solve()
            # copy output to input
            self.w0.assign(self.w1)

        # enact callbacks
        if operation:
            operation(self.w0)

        # return outputs
        X1[0].assign(self.w0) # save sol at the nstep th time 

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list
        
    def obs(self):
        m, u = self.w0.split()
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            dW = self.rg.normal(self.noise_space, 0., 1.0)
            particle.append(dW)
        if self.lambdas:
            for i in range(self.nsteps):
                dW = self.rg.normal(self.noise_space, 0., 1.0)
                particle.append(dW)
        return particle

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        assert(self.noise)
        for i in range(self.nsteps):
            count += 1
            X[count].assign(c1*X[count] + c2*rg.normal(
                self.noise_space, 0., 1.0))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        nsteps = self.nsteps
        dt = self.dt

        for step in range(nsteps):
            lambda_step = self.X[nsteps + 1 + step]
            dW_step = self.X[1 + step]
            dlfunc = assemble(lambda_step**2*dt/2*dx
                              - lambda_step*dW_step*dt**0.5*dx)
            dlfunc /= Area
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc

        return lfunc
