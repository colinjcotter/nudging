from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyadjoint import AdjFloat

# import time
# start_time = time.time()



from nudging.models.stochastic_euler import Euler_SD

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""

n = 8
nsteps = 5
model = Euler_SD(n, nsteps=nsteps)

MALA = False
verbose = True
jtfilter = jittertemp_filter(n_temp=4, n_jitt = 4, rho= 0.99,
                            verbose=verbose, MALA=MALA)

# jtfilter = bootstrap_filter()

# Load data
u_exact = np.load('u_true_data.npy')
u_vel = np.load('u_obs_data.npy') 

nensemble = [5,5,5,5]


jtfilter.setup(nensemble, model)

x = SpatialCoordinate(model.mesh) 

#prepare the initial ensemble
for i in range(nensemble[jtfilter.ensemble_rank]):
    a = model.rg.uniform(model.R, 0., 1.0) 
    b = model.rg.uniform(model.R, 0., 1.0)
    q0_in = a*sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*b*cos(6*pi*x[0])*cos(6*pi*x[1])\
                +0.02*a*sin(2*pi*x[0])+0.02*a*sin(2*pi*x[1])+0.3*b*cos(10*pi*x[0])*cos(4*pi*x[1]) 

   
    q = jtfilter.ensemble[i][0]
    q.interpolate(q0_in)
   

def log_likelihood(y, Y):
    ll = (y-Y)**2/0.05**2/2*dx
    return ll

    


N_obs = u_vel.shape[0]

# VVOM Function
u_VOM = Function(model.VVOM) 


# prepare shared arrays for data
u1_e_list = []
u2_e_list = []

for m in range(u_vel.shape[1]):        
    u1_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
  
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)
  


ushape = u_vel.shape
if COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    u_VOM.dat.data[:,0] = u_vel[k,:,0]
    u_VOM.dat.data[:,1] = u_vel[k,:,1]

    jtfilter.assimilation_step(u_VOM, log_likelihood)
    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.q0.assign(jtfilter.ensemble[i][0])
        obsdata1 = model.obs().dat.data[:][:,0]
        obsdata2 = model.obs().dat.data[:][:,1]
        

        for m in range(u_vel.shape[1]):
            u1_e_list[m].dlocal[i] = obsdata1[m]
            u2_e_list[m].dlocal[i] = obsdata2[m]

    for m in range(u_vel.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()

        if COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u2_e_list[m].data()
            

#PETSc.Sys.Print("--- %s seconds ---" % (time.time() - start_time))

if COMM_WORLD.rank == 0:
    u_e = np.stack((u1_e,u2_e), axis = -1)
    np.save("Velocity_ensemble_simulated_obs.npy", u_e)


