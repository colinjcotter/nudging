from firedrake import *
from .model import *
import numpy as np

class QuasiGeostrophic(base_model):
    def __init__(self,n):
        self.n = n
        Lx = 2.0*pi  # Zonal length
        Ly = 2.0*pi  # Meridonal length
        self.mesh = PeriodicRectangleMesh(n, n, Lx, Ly, direction="x", quadrilateral=True)
        self.V = FunctionSpace(self.mesh, "CG", 1) # for extra noise term
        self.Vdg = FunctionSpace(self.mesh, "DQ", 1)
        self.Vcg = FunctionSpace(self.mesh, "CG", 1)
        self.Vu = VectorFunctionSpace(self.mesh, "DQ", 0)  # DQ elements for velocity
        
        self.x = SpatialCoordinate(self.mesh)
        self.q0 = Function(self.Vdg).interpolate(0.1*sin(self.x[0])*sin(self.x[1]))
        
        #some physical parameters
        alpha = Constant(1.0)
        beta = Constant(0.1)
        Dt = 0.1
        dt = Constant(Dt)

        # Define function to store the fields
        self.dq1 = Function(self.Vdg)  # PV fields for different time steps
        self.qh = Function(self.Vdg)
        self.q1 = Function(self.Vdg)

        self.psi0 = Function(self.Vcg)  # Streamfunctions for different time steps
        self.psi1 = Function(self.Vcg)

        # Define the variational form 
        self.psi = TrialFunction(self.Vcg)  # Streamfunctions for different time steps
        self.phi = TestFunction(self.Vcg)

        
        # Build the weak form for the inversion
        self.Apsi = (inner(grad(self.psi), grad(self.phi)) +  self.psi * self.phi) * dx
        self.Lpsi = -self.q1 * self.phi * dx

        # We impose homogeneous dirichlet boundary conditions on the stream
        # function at the top and bottom of the domain. ::

        bc1 = DirichletBC(self.Vcg, 0.0, (1, 2))

        self.psi_problem = LinearVariationalProblem(self.Apsi, self.Lpsi, self.psi0, bcs=bc1, constant_jacobian=True)
        self.psi_solver = LinearVariationalSolver(self.psi_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})

        # setup the second equation
        gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
        # upwinding terms
        n_F = FacetNormal(self.mesh)
        un = 0.5 * (dot(gradperp(self.psi0), n_F) + abs(dot(gradperp(self.psi0), n_F)))
        
        self.q = TrialFunction(self.Vdg)
        self.p = TestFunction(self.Vdg)
        

        
        #Adding extra term included random number
        self.n_noise = 10
        self.f = []
        self.dW = []
        for i in range(self.n_noise):
            self.dW_temp = Constant(0)
            self.dW.append(self.dW_temp)
            self.f_temp = Function(self.Vdg)
            self.f_temp.interpolate(0.1*cos((i+1)*pi*self.x[0]/20.)*cos((i+1)*pi*self.x[1]/20.))
            self.f.append(self.f_temp)

        self.Fn = np.dot(self.f, self.dW)
        self.Ln = self.Fn*dt**0.5

        a_mass = self.p * self.q * dx
        a_int = (dot(grad(self.p), -gradperp(self.psi0) * self.q) + beta * self.p * self.psi0.dx(0)) * dx
        a_flux = (dot(jump(self.p), un("+") * self.q("+") - un("-") * self.q("-")))*dS
        a_noise = self.p*self.Ln *dx
        arhs = a_mass - dt*(a_int+ a_flux + a_noise) 
        #a_mass = a_mass + a_noise
      
        self.q_prob = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)
        self.q_solver = LinearVariationalSolver(self.q_prob,
                                   solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "bjacobi",
                                                      "sub_pc_type": "ilu"})

    
    def run(self, nsteps, W, X0, X1):
        
        for step in range(nsteps):

            np.random.seed(138)
            # Compute the streamfunction for the known value of q0
            self.q1.assign(X0)
            self.psi_solver.solve()
            for i in range(self.n_noise):
                self.dW[i].assign(W[step, i])
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            for i in range(self.n_noise):
                self.dW[i].assign(W[step, i])
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.q1.assign(0.75 * self.q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            for i in range(self.n_noise):
                self.dW[i].assign(W[step, i])
            self.q_solver.solve()

            # Find new solution q^(n+1)
            self.q0.assign(self.q0 / 3 + 2*self.dq1 /3)
        X1.assign(self.q0) # save sol at the nstep th time 

    # Fix observation points
    def obs(self, X0):
        q = X0
        #x_obs = np.arange(0.0, 2.0, self.n)
        #y_obs = np.arange(0.0, 2.0, self.n)
        xy_point = np.linspace((0,0), (2.0*pi, 2.0*pi), self.n)
        return np.array(q.at(xy_point))


    def allocate(self):        
        return Function(self.Vdg)