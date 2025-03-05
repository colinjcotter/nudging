import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class KS_CIP(base_model):
    def __init__(self, nsteps, xpoints, n=100, seed=12353,  mesh=False,lambdas=False,
                 dt=0.01, nu=0.02923, dc=0.01, L=10.):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed
        self.nu = nu #  viscosity
        self.dc = dc #  noise coefficient
        self.L = L #  domain width
        self.mesh = mesh
        self.xpoints = xpoints
        self.lambdas = lambdas  # include lambdas in allocate

    def setup(self, comm=MPI.COMM_WORLD):
        if not self.mesh:
            self.mesh = fd.PeriodicIntervalMesh(self.n, self.L,
                                           comm=comm, name="ksmesh")
        #self.mesh = mesh
        x, = fd.SpatialCoordinate(self.mesh)

        self.V = fd.FunctionSpace(self.mesh, "CG", 2)
        self.Vdg = fd.FunctionSpace(self.mesh, "DG", 1) # for VTK output 
       

        un = fd.Function(self.V)
        self.un = un
        unp1 = fd.Function(self.V)
        self.unp1 = unp1
        uh = (un + unp1)/2 # midpoint
        
        v = fd.TestFunction(self.V)
        
        dT = fd.Constant(self.dt)

        # Setup noise term and lambdas
        self.W_F = fd.FunctionSpace(self.mesh, "DG", 0)
        self.dW = fd.Function(self.W_F)
        self.Lambda = fd.Function(self.W_F)

        # model coefficient 
        alpha = fd.Constant(1.1) # viscosity
        beta = fd.Constant(0.02923) # hyperviscosity
        gamma = fd.Constant(1.) # advection

        eta = fd.Constant(5.) # penalty term
        area = fd.CellVolume(self.mesh)
        dx = fd.dx
        dS = fd.dS
        avg = fd.avg
        jump = fd.jump

        def a(u, v):
            h = avg(fd.CellVolume(self.mesh))/fd.FacetArea(self.mesh)
            eqn = v.dx(0).dx(0)*u.dx(0).dx(0)*dx # diffusion
            eqn += avg(u.dx(0).dx(0))*jump(v.dx(0))*dS # <avg(u_xx), jump(v_x)>
            eqn += avg(v.dx(0).dx(0))*jump(u.dx(0))*dS # <avg(v_xx), jump(u_x)>
            eqn += eta/h*jump(v.dx(0))*jump(u.dx(0))*dS # eth/h*<jump(v_x), jump(u_x)>
            return eqn

        eqn = (
            v*(unp1 - un)*dx
            - dT*alpha*v.dx(0)*uh.dx(0)*dx
            + a(dT*beta*uh, v)
            - dT*gamma*0.5*v.dx(0)*uh*uh*dx
            - (dT/area)**0.5*self.dc*self.dW*v*dx
            )

        linear_snes_params = {
                'lag_preconditioner': 5,
                'lag_preconditioner_persists': None,
                            }
        params = {
            'snes': linear_snes_params,
            "snes_atol": 1.0e-50,
            "snes_rtol": 1.0e-6,
            "snes_stol": 1.0e-50,
            "ksp_type":"gmres",
            "pc_type":"lu"
        }

        #make the solver
        KSProb = fd.NonlinearVariationalProblem(eqn, unp1)
        self.KSSolver = fd.NonlinearVariationalSolver(KSProb,
                                                      solver_parameters=params)

        # state for controls
        self.X = self.allocate()

        # vertex only mesh for observations
        x_obs = np.linspace(0.0, self.L, num=self.xpoints, endpoint=False)
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
        self.un.assign(self.X[0])
        self.unp1.assign(self.un)

        if self.lambdas:
            self.Lambda.assign(0.)
        # do the timestepping
        for step in range(self.nsteps):
            # get noise variables and lambdas
            if self.lambdas:
                self.Lambda.assign(self.Lambda + self.X[self.nsteps+step+1])
                self.dW.assign(self.X[step+1] + self.dt**0.5*self.Lambda)
            else:
                self.dW.assign(self.X[step+1])
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
        Y.interpolate(self.un)
        return Y

    def allocate(self):
        particle = [fd.Function(self.V)]
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
                self.W_F, 0., 1.))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self, reg_scale=False):
        nsteps = self.nsteps
        dt = self.dt
        dx = fd.dx
        cv = fd.CellVolume(self.mesh)

        self.Lambda.assign(0.)
        for step in range(nsteps):
            # X[0] is the model state
            # X[1], .., X[nsteps] are the dWs
            # X[nsteps+1], .., X[2*nsteps] are the lambdas
            self.Lambda.assign(self.Lambda + self.X[nsteps + 1 + step])
            lambda_step = self.Lambda
            dW_step = self.X[1 + step]
            #dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx)
            if reg_scale:
                dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx)
            else:
                dlfunc = fd.assemble((1/cv)*lambda_step**2*dt/2*dx
                                     - (1/cv)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
