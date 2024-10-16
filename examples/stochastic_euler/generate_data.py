from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from nudging.models.stochastic_mix_euler import Euler_mixSD
import os
os.makedirs('../../DA_Euler/', exist_ok=True)
os.makedirs('../../DA_Euler/checkpoint_files/', exist_ok=True)
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
truth_init = VTKFile("../../DA_Euler/truth_init.pvd")
truth = VTKFile("../../DA_Euler/truth.pvd")
truth_init_ptb = VTKFile("../../DA_Euler/truth_init_ptb.pvd")
particle_init = VTKFile("../../DA_Euler/particle_init.pvd")

nensemble = [1]*30
N_obs = 100
N_init = 250
n = 16
nsteps = 5
dt = 1/40


comm=fd.COMM_WORLD
#mesh = fd.UnitSquareMesh(n, n, quadrilateral = True, comm=comm, name ="mesh2d_per")

model = Euler_mixSD(n, nsteps=nsteps,  dt = dt, noise_scale=1.25, salt=False,  lambdas=True)

model.setup(comm=fd.COMM_WORLD)
mesh = model.mesh
x = SpatialCoordinate(mesh)
############################# initilisation ############
X0_truth = model.allocate()
q0,psi0 = X0_truth[0].subfunctions
q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
                +0.3*cos(10*pi*x[0])*cos(4*pi*x[1]) +0.02*sin(2*pi*x[0])+ 0.02*sin(2*pi*x[1]))

def gradperp(u):
    return fd.as_vector((-u.dx(1), u.dx(0)))
# #To store vorticity values 
psi_VOM = model.obs()
psi_VOM_out = Function(model.VVOM_out)
psi_VOM_out.interpolate(psi_VOM)
psi_true = psi_VOM_out.dat.data_ro.copy()
# To store inilization of  vorticity compoenents
psi_init = np.zeros((np.size(psi_true)))

# To check the energy functional
Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
v = Function(Vu, name="gradperp(stream function)")
u_energy = []
### To forward run for creating initlization  for truth and particles
for i in range(N_init):
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)
    # psi_VOM = model.obs()
    if i % 10 == 0:
        PETSc.Sys.Print('========Step=============', i)
        PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1), 'noise', fd.norm(model.dU_3))
    q,psi = model.qpsi1.subfunctions
    v.project(gradperp(model.qpsi0[1]))
    u_energy.append(0.5*norm(v))
    dU = model.dU_3
    q.rename("Vorticity")
    psi.rename("stream function")
    dU.rename("noise_var")
    truth_init.write(q, psi, dU)
    # to store init data
    psi_init_VOM = model.obs()
    psi_init_VOM_out = Function(model.VVOM_out)
    psi_init_VOM_out.interpolate(psi_init_VOM)
    psinit = psi_init_VOM_out.dat.data_ro.copy()
    if comm.rank == 0:
        np.save("../../DA_Euler/u_energy.npy", u_energy)
    if i == N_init-1:
        if comm.rank == 0:
            ps_init = psinit
            np.save("../../DA_Euler/psi_init.npy", ps_init)

################### Initialization  for all particles
X_particle = model.allocate()
Vcg = fd.FunctionSpace(mesh, "CG", 1)  # Streamfunctions
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
psi_chp = Function(Vcg, name="psi_chp") # checkpoint streamfunc
pv_chp = Function(Vdg, name="pv_chp")   # checkpoint vorticity

ndump = 20  # dump data
p_dump = 0

psi_particle_init = np.zeros((sum(nensemble), np.size(psi_true)))
with fd.CheckpointFile("../../DA_Euler/ensemble_init.h5", 
                       'w') as afile:
    #afile.save_mesh(mesh)
    for i in range(sum(nensemble)+1):
        if i < (sum(nensemble)):
            print("Generating ensemble member", i)
        else:
            print("Generating 'true' value")
        X_particle[0].assign(X0_truth[0])
        for j in range(ndump):
            model.randomize(X_particle)
            model.run(X_particle, X_particle)
        q,psi = model.qpsi1.subfunctions
        q.rename("Vorticity")
        psi.rename("stream function")
        PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1))
        psi_chp.interpolate(psi)
        pv_chp.interpolate(q)

        particle_init.write(q, psi)
        afile.save_function(psi_chp, idx=i)
        afile.save_function(pv_chp, idx=i)
        #print('iglobal', i, norm(psi_chp))
        psi_particle_VOM = model.obs()
        psi_particle_VOM_out = Function(model.VVOM_out)
        psi_particle_VOM_out.interpolate(psi_particle_VOM)
        psi_particle = psi_particle_VOM_out.dat.data_ro.copy()# #To store streamfunc values 

        if comm.rank == 0:
            psi_particle_init[p_dump,:] = psi_particle
    np.save("../../DA_Euler/psi_particle_init.npy", psi_particle_init)

### To store initlization  for truth 
PETSc.Sys.Print('=============Generating the observational data.=================')
X_truth = model.allocate()
X_truth[0].assign(X_particle[0])


psi_true_all = np.zeros((N_obs, np.size(psi_true)))
psi_obs_all = np.zeros((N_obs, np.size(psi_true)))

for i in range(N_obs):
    PETSc.Sys.Print('=============In N_obs step=================', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1))
    q,psi = model.qpsi1.subfunctions
    q.rename("Vorticity")
    psi.rename("stream function")
    truth.write(q, psi)

    psi_VOM = model.obs()
    psi_VOM_out = Function(model.VVOM_out)
    psi_VOM_out.interpolate(psi_VOM)
    psi_true = psi_VOM_out.dat.data_ro.copy()
    print('psi_true_squared', fd.assemble(psi_VOM_out**2*fd.dx))
    if comm.rank == 0:
        psi_true_all[i,:]= psi_true
        PETSc.Sys.Print('psi_true_ABS', psi_true.max(), psi_true.min())
        psi_noise = np.random.normal(0.0, 0.001, (int(n/2+1)**2 ) )# mean = 0, sd = 0.05
        PETSc.Sys.Print('Noise', psi_noise.max(), psi_noise.min())
        psi_max = np.abs(psi_true).max()
        psi_obs = psi_true + (1/psi_max)*psi_noise*psi_true # To get similar boundary values as truth 
        psi_obs_all[i,:] = psi_obs
    np.save("../../DA_Euler/psi_true_data.npy", psi_true_all)
    np.save("../../DA_Euler/psi_obs_data.npy", psi_obs_all)

