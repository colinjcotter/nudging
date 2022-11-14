from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt

from nudging.Quasi_Geostrophic_solve import QuasiGeostrophic

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
model = QuasiGeostrophic(40)
x = SpatialCoordinate(model.mesh) 

#bfilter = bootstrap_filter(5, (5, 10))
bfilter = jittertemp_filter(5, (5, 10), n_temp=20, n_jitt=5, rho=0.99)
nensemble = 5
bfilter.setup(nensemble, model)

dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble):
    dx0.assign(np.random.randn())
    dx1.assign(np.random.randn())
    a.assign(np.random.rand())
    b.assign(np.random.rand())

    q0_int = a*b*sin(x[0] + dx0)*sin(x[1]+dx1)


    
    q = bfilter.ensemble[i]
    q.interpolate(q0_int)  

def log_likelihood(dY):
    return np.dot(dY, dY)/0.1**2
    
#Load data

#N_obs = 10
q_exact= np.load('q_true.npy')
N_obs = q_exact.shape[0]

station_view = 5
plt.plot(q_exact[:,station_view], 'r-', label='q_true')

q = np.load('q_obs.npy') 

q_e = np.zeros((N_obs, nensemble, q.shape[1]))
q_e_mean = np.zeros((N_obs, nensemble))



# do assimiliation step
for k in range(N_obs):
    bfilter.assimilation_step(q[k,:], log_likelihood)
    print(k)
    print('theta:', bfilter.theta_temper)
    print('Temp_ess:', bfilter.ess_temper)
    for e in range(nensemble):
        q_e[k,e,:] = model.obs(bfilter.ensemble[e])


q_e_mean = np.mean(q_e[:,:,station_view], axis=1)

plt.plot(q_e[:,:,station_view], 'y-')
plt.plot(q_e_mean, 'g-', label='q_ensemble_mean')
plt.legend()
plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
plt.show()