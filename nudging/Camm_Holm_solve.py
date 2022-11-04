from ast import Constant
from pyclbr import Function
from firedrake import *
from .model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self,n):
        self.n = n
        self.mesh = PeriodicIntervalMesh(n, 40.0)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()
        self.x, = SpatialCoordinate(self.mesh)

        self.n_noise = 10

        alpha = 1.0
        alphasq = Constant(alpha**2)
        dt = 0.01
        self.dt = dt
        Dt = Constant(dt)
        #Solve for the initial condition for m.
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
        self.f = []
        self.dW = []
        for i in range(self.n_noise):
            self.dW_temp = Constant(0)
            self.dW.append(self.dW_temp)
            self.f_temp = Function(self.V)
            self.f_temp.interpolate(0.1*sin((i+1)*pi*self.x/20.))
            self.f.append(self.f_temp)

        self.Ln = np.dot(self.f, self.dW)
      
        self.mh = 0.5*(self.m1 + self.m0)
        self.uh = 0.5*(self.u1 + self.u0)
        self.v = self.uh*Dt+self.Ln*Dt**0.5

        #def Linearfunc
        self.L = ((self.q*self.u1 + alphasq*self.q.dx(0)*self.u1.dx(0) - self.q*self.m1)*dx +(self.p*(self.m1-self.m0) + (self.p*self.v.dx(0)*self.mh -self.p.dx(0)*self.v*self.mh))*dx)

        # solver
        self.uprob = NonlinearVariationalProblem(self.L, self.w1)
        self.usolver = NonlinearVariationalSolver(self.uprob, solver_parameters={'mat_type': 'aij', 'ksp_type': 'preonly','pc_type': 'lu'})

        # Data save
        self.m0, self.u0 = self.w0.split()
        self.m1, self.u1 = self.w1.split()
    
    def run(self, nsteps, W, X0, X1):
        self.w0.assign(X0)
        self.msolve.solve()
        for step in range(nsteps):
            for i in range(self.n_noise):
                self.dW[i].assign(W[step, i])
            self.usolver.solve()
            self.w0.assign(self.w1)
        X1.assign(self.w0) # save sol at the nstep th time 


    def obs(self, X0):
        m, u = X0.split()
        x_obs = np.arange(0.0,40.0)
        return np.array(u.at(x_obs))


    def allocate(self):        
        return Function(self.W)
