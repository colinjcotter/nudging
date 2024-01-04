import numpy as np
from functools import cached_property
import firedrake as fd


class residual_resampling(object):
    def __init__(self, seed=34523, residual=False):
        self.seed = seed
        self.initialised = False
        self.residual = residual

    @cached_property
    def rg(self):
        pcg = fd.PCG64(seed=self.seed)
        return fd.RandomGenerator(pcg)

    def resample(self, weights, model):
        """
        :arg weights : a numpy array of normalised weights, size N

        returns
        :arg s: an array of integers, size N. X_i will be replaced
        with X_{s_i}.
        """
        if not self.initialised:
            self.initialised = True
            self.R = fd.FunctionSpace(model.mesh, "R", 0)
        N = weights.size

        if self.residual:
            copies = np.array(np.floor(weights*N), dtype=int)
            L = N - np.sum(copies)
            residual_weights = N*weights - copies
            residual_weights /= np.sum(residual_weights)

            for i in range(L):
                u = self.rg.uniform(self.R, 0., 1.0)
                u0 = u.dat.data[:]
                cs = np.cumsum(residual_weights)
                istar = -1
                while cs[istar+1] < u0:
                    istar += 1
                copies[istar] += 1

            count = 0
            s = np.zeros(N, dtype=int)
            for i in range(N):
                for j in range(copies[i]):
                    s[count] = i
                    count += 1

        else:  # systematic resampling
            cumsum_weight = np.cumsum(weights)
            u = self.rg.uniform(self.R, 0., 1.0)
            u0 = u.dat.data[:]

            ensembl_pos = (u0 + np.arange(N))/N
            s = np.zeros(N, dtype=int)
            i, j = 0, 0
            while i < N:
                if ensembl_pos[i] < cumsum_weight[j]:
                    s[i] = j
                    i += 1
                else:
                    j += 1
        return s
