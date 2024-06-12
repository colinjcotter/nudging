import firedrake as fd
from pyop2.mpi import MPI
from nudging.model import base_model
import numpy as np


class Euler_mixSD(base_model):
    def __init__(self, n_xy_pts, nsteps, dt,
                 noise_scale, mesh=False,
                 salt=False, lambdas=False, seed=12353):
        self.n = n_xy_pts
        self.nsteps = nsteps
        self.dt = dt
        self.noise_scale = noise_scale
        self.mesh = mesh
        self.salt = salt
        self.lambdas = lambdas
        self.seed = seed

    def setup(self, comm=MPI.COMM_WORLD):
        r = 0.01
        self.Lx = 1.0  # Zonal length
        self.Ly = 1.0  # Meridonal length
        if not self.mesh:
            self.mesh = fd.UnitSquareMesh(self.n, self.n,
                                          quadrilateral=True,
                                          comm=comm)
        x = fd.SpatialCoordinate(self.mesh)
        dx = fd.dx
        dS = fd.dS

        def gradperp(u):
            return fd.as_vector((-u.dx(1), u.dx(0)))
        self.gradperp = gradperp

        # solver_parameters
        sp = {"ksp_type": "cg", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}

        # Setup noise term using Matern formula
        self.Vcg = fd.FunctionSpace(self.mesh, "CG", 1)  # Streamfunctions
        self.W_F = fd.FunctionSpace(self.mesh, "DG", 0)
        self.dW = fd.Function(self.W_F)
        dW_phi = fd.TestFunction(self.Vcg)
        dU = fd.TrialFunction(self.Vcg)

        # cell_area = fd.CellVolume(self.mesh)
        # alpha_w =(1/cell_area**0.5)
        kappa_inv_sq = fd.Constant(1/30.0**2)

        self.dU_1 = fd.Function(self.Vcg)
        self.dU_2 = fd.Function(self.Vcg)
        self.dU_3 = fd.Function(self.Vcg)
        # zero boundary condition for noise
        bc_w = fd.DirichletBC(self.Vcg, 0, "on_boundary")
        a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx \
            + dU*dW_phi*dx
        L_w1 = self.dW*dW_phi*dx
        w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, self.dU_1, bcs=bc_w,
                                              constant_jacobian=True)
        self.wsolver1 = fd.LinearVariationalSolver(w_prob1,
                                                   solver_parameters=sp)
        L_w2 = self.dU_1*dW_phi*dx
        w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, self.dU_2, bcs=bc_w,
                                              constant_jacobian=True)
        self.wsolver2 = fd.LinearVariationalSolver(w_prob2,
                                                   solver_parameters=sp)
        L_w3 = self.dU_2*dW_phi*dx
        w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, self.dU_3, bcs=bc_w,
                                              constant_jacobian=True)
        self.wsolver3 = fd.LinearVariationalSolver(w_prob3,
                                                   solver_parameters=sp)

        # FE spaces for forward solver
        Vcg = fd.FunctionSpace(self.mesh, "CG", 1)  # Streamfunctions
        Vdg = fd.FunctionSpace(self.mesh, "DQ", 1)  # PV space
        self.V_mix = fd.MixedFunctionSpace((Vdg, Vcg))
        # self.Vu = fd.VectorFunctionSpace(self.mesh, "DQ", 1) # velocity
        # self.u_vel = fd.Function(self.Vu)
        bc = fd.DirichletBC(self.V_mix.sub(1), fd.zero(), ("on_boundary"))

        # mix function
        self.qpsi0 = fd.Function(self.V_mix)    # n time
        self.q0, self.psi0 = fd.split(self.qpsi0)

        self.qpsi1 = fd.Function(self.V_mix)    # n+1 time
        self.q1, self.psi1 = fd.split(self.qpsi1)

        # test functions
        p, phi = fd.TestFunctions(self.V_mix)
        # # trial functions
        # q, psi =  fd.TrialFunctions(self.V_mix)

        # timestepping equation
        Dt = self.dt

        # mid points
        # mid point formulation
        qh = 0.5*(self.q1+self.q0)
        psih = 0.5*(self.psi1+self.psi0)

        # SALT noise
        if self.salt:
            psi_mod = psih + self.noise_scale*self.dU_3*Dt**2
        else:
            psi_mod = psih

        # upwinding terms
        n_F = fd.FacetNormal(self.mesh)
        un = 0.5 * (fd.dot(gradperp(psi_mod), n_F) +
                    abs(fd.dot(gradperp(psi_mod), n_F)))

        # source term
        Q = fd.Function(Vdg)
        Q.interpolate(0.1*fd.sin(8*fd.pi*x[0]))

        F = (self.q1-self.q0)*p*dx + Dt*p*(r*self.q1-Q)*dx\
            + Dt*(fd.dot(fd.grad(p), -qh*gradperp(psi_mod)))*dx\
            + Dt*(fd.dot(fd.jump(p), un("+")*qh("+") - un("-")*qh("-")))*dS\
            + (fd.inner(fd.grad(self.psi1), fd.grad(phi)))*dx\
            + self.psi1*phi*dx + self.q1*phi*dx
        if not self.salt:
            F += self.noise_scale*p*self.dU_3*Dt**2*dx

        # timestepping solver
        qphi_prob = fd.NonlinearVariationalProblem(F, self.qpsi1, bcs=bc)

        self.qphi_solver = fd.NonlinearVariationalSolver(qphi_prob,
                                                         solver_parameters=sp)

        # internal state for controls
        self.X = self.allocate()

        # observations
        x_point = np.linspace(0.0, self.Lx, self.n+1)
        y_point = np.linspace(0.0, self.Ly, self.n+1)
        xv, yv = np.meshgrid(x_point, y_point)
        x_obs_list = np.vstack([xv.ravel(), yv.ravel()]).T.tolist()
        VOM = fd.VertexOnlyMesh(self.mesh, x_obs_list)
        self.VVOM_out = fd.FunctionSpace(VOM.input_ordering, "DG", 0)
        self.VVOM = fd.FunctionSpace(VOM, "DG", 0)

    def run(self, X0, X1):
        for i in range(len(X0)):
            self.X[i].assign(X0[i])

        self.qpsi0.assign(self.X[0])

        for step in range(self.nsteps):
            # setup the noise and lambda term
            self.dW.assign(self.X[step+1])
            if self.lambdas:
                self.dW += self.X[step+1+self.nsteps]*(self.dt)**0.5
            # solve  dW --> dU0 --> dU1 --> dU3
            self.wsolver1.solve()
            self.wsolver2.solve()
            self.wsolver3.solve()
            # advance in time
            self.qphi_solver.solve()
            # copy output to input
            self.qpsi0.assign(self.qpsi1)

        # return outputs
        X1[0].assign(self.qpsi0)  # save sol at the nstep th time

    # control PV
    def controls(self):
        controls_list = []
        for i in range(len(self.X)):
            controls_list.append(fd.adjoint.Control(self.X[i]))
        return controls_list

    def obs(self):
        Y = fd.Function(self.VVOM)
        Y.interpolate(self.qpsi0[0])
        return Y

    def allocate(self):
        particle = [fd.Function(self.V_mix)]
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
