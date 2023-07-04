from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from nudging.model import *
import numpy as np

class Euler_SD(base_model):
    def __init__(self, n, nsteps, dt = 0.01, seed=12353):

        self.n = n
        self.nsteps = nsteps
        self.dt = dt
        self.seed = seed

    def setup(self, comm = MPI.COMM_WORLD):

        alpha = Constant(1.0)
        beta = Constant(0.1)
        self.Lx = 2.0*pi  # Zonal length
        self.Ly = 2.0*pi  # Meridonal length
        self.mesh = PeriodicRectangleMesh(self.n, self.n, self.Lx, self.Ly, direction="x", quadrilateral=True, comm = comm)
        self.x = SpatialCoordinate(self.mesh)

        #FE spaces
        self.V = FunctionSpace(self.mesh, "CG", 1) #  noise term
        self.Vcg = FunctionSpace(self.mesh, "CG", 1) # Streamfunctions
        self.Vdg = FunctionSpace(self.mesh, "DQ", 1) # potential vorticity (PV)
        self.Vu = FunctionSpace(self.mesh, "DQ", 0) # velocity
        
        self.q0 = Function(self.Vdg)
        self.q1 = Function(self.Vdg)
        
        self.u0 = Function(self.Vu)

        # Define function to store the fields
        self.dq1 = Function(self.Vdg)  # PV fields for different time steps
        self.q1 = Function(self.Vdg)
       
        ##################################  Bilinear form for Stream function ##########################################
        # Define the weakfunction for stream functions
        psi = TrialFunction(self.Vcg)  
        phi = TestFunction(self.Vcg)
        self.psi0 = Function(self.Vcg) 
        

        # Build the weak form for the inversion
        Apsi = (inner(grad(psi), grad(phi)) +  psi*phi) * dx
        Lpsi = -self.q1 * phi * dx

        bc1 = DirichletBC(self.Vcg, 0.0, (1, 2))

        psi_problem = LinearVariationalProblem(Apsi, Lpsi, self.psi0, bcs=bc1, constant_jacobian=True)
        self.psi_solver = LinearVariationalSolver(psi_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})

        ################################### Setup for  velocity #####################################################################

        self.gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
        # upwinding terms
        n_F = FacetNormal(self.mesh)
        un = 0.5 * (dot(self.gradperp(self.psi0), n_F) + abs(dot(self.gradperp(self.psi0), n_F)))
        
        
        #####################################   Bilinear form  for noise variable  ################################################
        dW_phi = TestFunction(self.V)
        dW = TrialFunction(self.V)
        self.dW_n = Function(self.V)
        # Fix the right hand side white noise
        self.dXi = Function(self.V)
        self.dXi.assign(self.rg.normal(self.V, 0., 1.0))

        #### Define Bilinear form with Dirichlet BC 
        bcs_dw = DirichletBC(self.V,  zero(), ("on_boundary"))
        a_dW = inner(grad(dW), grad(dW_phi))*dx + dW*dW_phi*dx
        L_dW = self.dt*self.dXi*dW_phi*dx

        
        #make a solver 
        dW_problem = LinearVariationalProblem(a_dW, L_dW, self.dW_n, bcs=bcs_dw)
        self.dW_solver = LinearVariationalSolver(dW_problem, solver_parameters={"ksp_type": "cg", "pc_type": "hypre"})


        #####################################   Bilinear form  for PV  ################################################
        
        q = TrialFunction(self.Vdg)
        p = TestFunction(self.Vdg)

        a_mass = p*q*dx
        a_int = (dot(grad(p), -self.gradperp(self.psi0) *q) + beta*p*self.psi0.dx(0)) * dx  # with stream function
        a_flux = (dot(jump(p), un("+") * q("+") - un("-") * q("-")))*dS                          # with velocity
        a_noise = p*self.dW_n *dx                                                                          # with noise term
        arhs = a_mass - self.dt*(a_int+ a_flux+a_noise) 

        #print(type(action(arhs, self.q1)), 'action')
        q_prob = LinearVariationalProblem(a_mass, action(arhs, self.q1), self.dq1)
        self.q_solver = LinearVariationalSolver(q_prob,
                                   solver_parameters={"ksp_type": "preonly",
                                                      "pc_type": "bjacobi",
                                                      "sub_pc_type": "ilu"})

        ############################################ state for controls  ###################################
        self.X = self.allocate()

        ################################ Setup VVOM for vectorfunctionspace ###################################
        x_point = np.linspace(0.0, self.Lx, self.n+1 )
        y_point = np.linspace(0.0, self.Ly, self.n+1 )
        xv, yv  = np.meshgrid(x_point, y_point)
        x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
        VOM = VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = VectorFunctionSpace(VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])
            
        self.q0.assign(self.X[0])
        for step in range(self.nsteps):
            #compute the noise term
            self.dXi.assign(self.X[step+1]) # why this term ?
            self.dW_solver.solve()

            # Compute the streamfunction for the known value of q0
            self.q1.assign(self.q0)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(1)
            self.q1.assign(self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find intermediate solution q^(2)
            self.q1.assign(0.75 * self.q0 + 0.25 * self.dq1)
            self.psi_solver.solve()
            self.q_solver.solve()

            # Find new solution q^(n+1)
            self.q0.assign(self.q0 / 3 + 2*self.dq1 /3)
        X1[0].assign(self.q0) # save sol at the nstep th time 

    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(Control(self.X[i]))
        return controls_list
        

    def obs(self):
        self.q1.assign(self.q0)
        self.psi_solver.solve()
        u  = self.gradperp(self.psi0)
        Y = Function(self.VVOM)
        Y.interpolate(u)
        return Y


    def allocate(self):
        particle = [Function(self.Vdg)]
        for i in range(self.nsteps):
            dW_star = Function(self.V)
            particle.append(dW_star) 
        return particle 



    def randomize(self, X, c1=0, c2=1, gscale=None, g=None):
        rg = self.rg
        count = 0
        for i in range(self.nsteps):
               self.dXi.assign(rg.normal(self.V, 0., 1.0))
               self.dW_solver.solve()
               count += 1
               X[count].assign(c1*X[count] + c2*self.dW_n)
               if g:
                    X[count] += gscale*g[count]

    def lambda_functional_1(self):
        
        for step in range(self.nsteps):
            self.dXi.assign(self.X[step+1])

            if step == 0:
                lambda_func = 0.5*dot(self.dXi, self.dXi)*dx
            else:
                lambda_func += 0.5*dot(self.dXi, self.dXi)*dx
        return assemble(lambda_func)/assemble(1 * dx(self.mesh))
    

    def lambda_functional_2(self, lambda_opt):
        for step in range(self.nsteps):
            self.dXi.assign(self.X[step+1])
            #dLi.assign(lambda_opt[step+1])

            if step == 0:
                lambda_func = 0.5*dot(self.dXi, self.dXi)*dx
            else:
                lambda_func -= 0.5*dot(self.dXi, self.dXi)*dx
        return assemble(lambda_func)/assemble(1 * dx(self.mesh))

               
