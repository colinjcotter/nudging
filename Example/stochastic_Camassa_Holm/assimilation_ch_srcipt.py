from firedrake import *
from nudging import *
import numpy as np
import matplotlib.pyplot as plt

""" read obs from saved file 
    Do assimilation step for tempering and jittering steps 
"""
model = Camsholm(100)
x, = SpatialCoordinate(model.mesh) 

bfilter = jittertemp_filter(5, (5, 10), n_temp=5, n_jitt=5, rho=0.995)
nensemble = 5
bfilter.setup(nensemble, model)

dx0 = Constant(0.)
dx1 = Constant(0.)
a = Constant(0.)
b = Constant(0.)

for i in range(nensemble):
    dx0.assign(np.random.randn()*0.1)
    dx1.assign(np.random.randn()*0.1)
    a.assign(np.random.rand())
    b.assign(np.random.rand())

    u0_exp = a*0.2*2/(exp(x-403./15. + dx0) + exp(-x+403./15. + dx0)) \
           + b*0.5*2/(exp(x-203./15. + dx1)+exp(-x+203./15. + dx1))

    _, u = bfilter.ensemble[i].split()
    u.interpolate(u0_exp)  

def log_likelihood(dY):
    return np.dot(dY, dY)/0.1**2
    
#Load data
N_obs = 5
y_exact = np.load('y_true.npy')
#plt.plot(y_exact[:,10], 'r-', label='Y_true')

y = np.load('y_obs.npy') 
#plt.plot(y[:,10], 'b-', label='Y_true')
y_e = np.zeros((N_obs, nensemble, y.shape[1]))
y_e_mean = np.zeros((N_obs, nensemble))

print(y_e.shape)

# do assimiliation step
for k in range(N_obs):
    bfilter.assimilation_step(y[k,:], log_likelihood)
    #print(k)
    #print(bfilter.ess)
    #print('del_theta:', bfilter.arr_deltheta)
    #print('theta:', bfilter.arr_theta)
    print('e_weight ', bfilter.e_weight)
    print('pre_ess ', bfilter.pre_ess)
    #print('d_weight: ', bfilter.d_weight)
    print('check_ess: ', bfilter.check_ess)
    for e in range(nensemble):
        y_e[k,e,:] = model.obs(bfilter.ensemble[e])
# print(bfilter.temp_ess)
# print(bfilter.jitt_ess)

# plt.plot(bfilter.ess, label='N_ess')
# plt.title('Tempering ess: Camassa-Holm equation with N_ensemble = ' +str(nensemble))
# plt.legend()
# plt.show()


# y_e_mean = np.mean(y_e[:,:,10], axis=1)

# plt.plot(y_e[:,:,10], 'y-')
# plt.plot(y_e_mean, 'g-', label='Y_ensemble_mean')
# plt.legend()
# plt.title('Ensemble prediction with N_ensemble = ' +str(nensemble))
# plt.show()