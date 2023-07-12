from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self, n, nsteps, xpoints, dt = 0.01, alpha=1.0, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.alpha = alpha
        self.dt = dt
        self.seed = seed
        self.xpoints = xpoints

    def setup(self, comm = MPI.COMM_WORLD):
        self.mesh = PeriodicIntervalMesh(self.n, 40.0, comm = comm) # mesh need to be setup in parallel
        self.x, = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()       

        self.p = TestFunction(self.V)
        self.m = TrialFunction(self.V)
        
        self.am = self.p*self.m*dx
        self.Lm = (self.p*self.u0 + self.alpha**2*self.p.dx(0)*self.u0.dx(0))*dx
        #Solve for the initial condition for m
        mprob = LinearVariationalProblem(self.am, self.Lm, self.m0)
        solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}
        self.msolve = LinearVariationalSolver(mprob, solver_parameters=solver_parameters)
        
        #Build the weak form of the timestepping algorithm. 
        self.p, self.q = TestFunctions(self.W)

        self.w1 = Function(self.W)
        self.w1.assign(self.w0)
        self.m1, self.u1 = split(self.w1)   # for n+1 the  time
        self.m0, self.u0 = split(self.w0)   # for n th time 
        
        #Adding modulated amplitudes
        fx1 = Function(self.V)
        fx2 = Function(self.V)
        fx3 = Function(self.V)
        fx4 = Function(self.V)

        fx1.interpolate(0.1*sin(pi*self.x/8.))
        fx2.interpolate(0.1*sin(2.*pi*self.x/8.))
        fx3.interpolate(0.1*sin(3.*pi*self.x/8.))
        fx4.interpolate(0.1*sin(4.*pi*self.x/8.))

        # with added term
        self.R = FunctionSpace(self.mesh, "R", 0)

        self.sqrt_dt = self.dt**0.5
        # noise term
        self.dW1 = Function(self.R)
        self.dW2 = Function(self.R)
        self.dW3 = Function(self.R)
        self.dW4 = Function(self.R)
        Ln = fx1*self.dW1+fx2*self.dW2+fx3*self.dW3+fx4*self.dW4

        # finite element linear functional 
        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*self.dt+Ln*self.sqrt_dt

        self.L = ((self.q*self.u1 + self.alpha**2*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        #solver
        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.linspace(0.5,39.5, self.xpoints)
        x_obs_list = []
        for i in x_obs:
            x_obs_list.append([i])
        self.VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = FunctionSpace(self.VOM, "DG", 0)

    def run(self, X0, X1, Nudge = False, operation = None):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
        self.w0.assign(self.X[0])
        self.msolve.solve()
        for step in range(self.nsteps):
            self.dW1.assign(self.X[4*step+1])
            self.dW2.assign(self.X[4*step+2])
            self.dW3.assign(self.X[4*step+3])
            self.dW4.assign(self.X[4*step+4])      
            
            self.usolver.solve()
            self.w0.assign(self.w1)

            if operation:
               operation(self.w0)
        X1[0].assign(self.w0) 
        
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
        count = 0
        particle = [Function(self.W)]
        for i in range(self.nsteps):
            for j in range(4):
                count +=1 
                dW = Function(self.R)
                particle.append(dW) 
        return particle 

    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
            for j in range(4):
                count += 1
                X[count].assign(c1*X[count] + c2*rg.normal(self.R, 0., 2.0))    
                if g:
                    X[count] += gscale*g[count]

    def lambda_functional_1(self):
        for step in range(self.nsteps):
            dW1 = self.X[4*step+1]
            dW2 = self.X[4*step+2]
            dW3 = self.X[4*step+3]
            dW4 = self.X[4*step+4]
            if step == 0:
               lambda_func = 0.5*(dW1**2+dW2**2+dW3**2+dW4**2)*dx
            else:
                lambda_func += 0.5*(dW1**2+dW2**2+dW3**2+dW4**2)*dx
        return assemble(lambda_func)/40
    
    def lambda_functional_2(self, lambda_opt):
        for step in range(self.nsteps):
            dW1 = self.X[4*step+1]
            dW2 = self.X[4*step+2]
            dW3 = self.X[4*step+3]
            dW4 = self.X[4*step+4]

            dl1 = lambda_opt[4*step+1]
            dl2 = lambda_opt[4*step+2]
            dl3 = lambda_opt[4*step+3]
            dl4 = lambda_opt[4*step+4]

            if step == 0:
               lambda_func = -(dW1*dl1+dW2*dl2+dW3*dl3+dW4*dl4)*dx # sort out dt
            else:
                lambda_func -= (dW1*dl1+dW2*dl2+dW3*dl3+dW4*dl4)*dx # sort out dt
        return assemble(lambda_func)/40
