import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class Euler_SD(base_model):
    def __init__(self, n_xy_pts, nsteps, dt=0.5, lambdas=False, seed=12353):
        self.n = n_xy_pts
        self.nsteps = nsteps
        self.dt = dt
        self.lambdas = lambdas  # include lambdas in allocate
        self.seed = seed

    def setup(self, comm=MPI.COMM_WORLD):
        r = 0.01
        self.Lx = 2.0*fd.pi  # Zonal length
        self.Ly = 2.0*fd.pi  # Meridonal length
        self.mesh = fd.PeriodicRectangleMesh(self.n, self.n,
                                             self.Lx, self.Ly,
                                             direction="x",
                                             quadrilateral=True,
                                             comm=comm)
        dx = fd.dx
        dS = fd.dS
        # FE spaces
        self.Vcg = fd.FunctionSpace(self.mesh, "CG", 1)  # Streamfunctions
        self.Vdg = fd.FunctionSpace(self.mesh, "DQ", 1)  # PV space

        self.q0 = fd.Function(self.Vdg)
        self.q1 = fd.Function(self.Vdg)
        # Define function to store the fields
        self.dq1 = fd.Function(self.Vdg)  # PV fields for different time steps

        # Define the weakfunction for stream functions
        psi = fd.TrialFunction(self.Vcg)
        phi = fd.TestFunction(self.Vcg)
        self.psi0 = fd.Function(self.Vcg)

        # Build the weak form for the inversion
        
        Apsi = (fd.inner(fd.grad(psi), fd.grad(phi)) + psi*phi)*dx
        Lpsi = -self.q1 * phi * dx

        bc1 = fd.DirichletBC(self.Vcg, 0.0, (1, 2))

        psi_problem = fd.LinearVariationalProblem(Apsi, Lpsi,
                                                  self.psi0,
                                                  bcs=bc1,
                                                  constant_jacobian=True)
        sp = {"ksp_type": "cg", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}
        self.psi_solver = fd.LinearVariationalSolver(psi_problem,
                                                     solver_parameters=sp)
        self.W_F = fd.FunctionSpace(self.mesh, "DG", 0)
        self.dW = fd.Function(self.W_F)

        dW_phi = fd.TestFunction(self.Vcg)
        dw = fd.TrialFunction(self.Vcg)
        self.alpha_w = fd.CellVolume(self.mesh)

        # to store noise data
        du_w = fd.Function(self.Vcg)

        bcs_dw = fd.DirichletBC(self.Vcg,  fd.zero(), ("on_boundary"))
        a_dW = fd.inner(fd.grad(dw), fd.grad(dW_phi))*dx + dw*dW_phi*dx
        L_dW = self.dW*dW_phi*dx

        dW_problem = fd.LinearVariationalProblem(a_dW, L_dW, du_w,
                                                 bcs=bcs_dw)
        self.dW_solver = fd.LinearVariationalSolver(dW_problem,
                                                    solver_parameters=sp)

        # Add noise with stream function to get stochastic velocity
        
        psi_mod = self.psi0+du_w

        def gradperp(u):
            return fd.as_vector((-u.dx(1), u.dx(0)))
        self.gradperp = gradperp
        # upwinding terms
        n_F = fd.FacetNormal(self.mesh)
        un = 0.5 * (fd.dot(gradperp(psi_mod), n_F) +
                    abs(fd.dot(gradperp(psi_mod), n_F)))

        q = fd.TrialFunction(self.Vdg)
        p = fd.TestFunction(self.Vdg)
        Q = fd.Function(self.Vdg)

        # timestepping equation
        a_mass = p*q*dx
        a_int = (fd.dot(fd.grad(p), -q*gradperp(psi_mod)) - p*(Q-r*q)) * dx
        a_flux = (fd.dot(fd.jump(p), un("+") * q("+") - un("-") * q("-"))) * dS
        arhs = a_mass - self.dt * (a_int + a_flux)

        q_prob = fd.LinearVariationalProblem(a_mass,
                                             fd.action(arhs, self.q1),
                                             self.dq1)
        dgsp = {"ksp_type": "preonly",
                "pc_type": "bjacobi",
                "sub_pc_type": "ilu"}
        self.q_solver = fd.LinearVariationalSolver(q_prob,
                                                   solver_parameters=dgsp)

        # internal state for controls
        self.X = self.allocate()

        # observations
        x_point = np.linspace(0.0, self.Lx, self.n+1)
        y_point = np.linspace(0.0, self.Ly, self.n+1)
        xv, yv = np.meshgrid(x_point, y_point)
        x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
        VOM = fd.VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM = fd.VectorFunctionSpace(VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        self.q0.assign(self.X[0])
        for step in range(self.nsteps):
            # compute the noise term
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
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
            self.q0.assign(self.q0/3 + 2 * self.dq1/3)
        X1[0].assign(self.q0)  # save sol at the nstep th time

    # control PV
    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fd.adjoint.Control(self.X[i]))
        return controls_list

    def obs(self):
        self.q1.assign(self.q0)  # assigned at time t+1
        self.psi_solver.solve()  # solved at t+1 for psi
        u = self.gradperp(self.psi0)  # evaluated velocity at time t+1
        Y = fd.Function(self.VVOM)
        Y.interpolate(u)
        return Y

    def allocate(self):
        particle = [fd.Function(self.Vdg)]
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
                self.W_F, 0., 1.0))
            if g:
                X[count] += gscale*g[count]

    def lambda_functional(self):
        nsteps = self.nsteps
        dt = self.dt
        dx = fd.dx
        cv = fd.CellVolume(self.mesh)
        for step in range(nsteps):
            lambda_step = self.X[nsteps + 1 + step]
            dW_step = self.X[1 + step]
            dlfunc = fd.assemble(
                (1/cv)*lambda_step**2*dt/2*dx
                - (1/cv)*lambda_step*dW_step*dt**0.5*dx)
            if step == 0:
                lfunc = dlfunc
            else:
                lfunc += dlfunc
        return lfunc
