from firedrake import *
from nudging.model import *
import numpy as np

class Camsholm(base_model):
    def __init__(self,n, dt = 0.01, alpha=1.0):
        self.n = n
        self.mesh = PeriodicIntervalMesh(n, 40.0)
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace((self.V, self.V))
        self.w0 = Function(self.W)
        self.m0, self.u0 = self.w0.split()
        self.x, = SpatialCoordinate(self.mesh)

        self.alpha = alpha
        alphasq = Constant(alpha**2)
        self.dt = dt
        Dt = Constant(dt)
        
        #Interpolate the initial condition

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
        self.fx1 = Function(self.V)
        self.fx2 = Function(self.V)
        self.fx3 = Function(self.V)
        self.fx4 = Function(self.V)

        self.fx1.interpolate(0.1*sin(pi*self.x/8.))
        self.fx2.interpolate(0.1*sin(2.*pi*self.x/8.))
        self.fx3.interpolate(0.1*sin(3.*pi*self.x/8.))
        self.fx4.interpolate(0.1*sin(4.*pi*self.x/8.))

        # with added term
        self.dW1 = Constant(0)
        self.dW2 = Constant(0)
        self.dW3 = Constant(0)
        self.dW4 = Constant(0)


        self.Ln = self.fx1*self.dW1+self.fx2*self.dW2+self.fx3*self.dW3+self.fx4*self.dW4
        
        # finite element linear functional 

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
    
    def run(self, nsteps, W, X0, X1):
        self.w0.assign(X0)
        self.msolve.solve()
        for step in range(nsteps):
            self.dW1.assign(W[4*step])
            self.dW2.assign(W[4*step+1])
            self.dW3.assign(W[4*step+2])
            self.dW4.assign(W[4*step+3])

            self.usolver.solve()
            self.w0.assign(self.w1)
        X1.assign(self.w0) # save sol at the nstep th time 


    def obs(self, X0):
        m, u = X0.split()
        x_obs = np.arange(0.0,40.0)
        return np.array(u.at(x_obs))


    def allocate(self):        
        return Function(self.W)
