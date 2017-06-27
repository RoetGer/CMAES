from operator import itemgetter

import numpy as np
import scipy.special as spsp

class CovMatAdapt:

    def __init__(self, func, mean_vec, step_size, pop_size=None,
                     elite_size=None, weights=None,
                     c_sig=None, d_sig=None, c_c=None,
                     c_1=None, c_mu=None, c_m=None, abs_tol=-1e3,
                     rel_tol=1e-4, maxiter=50):

        self.func = func
        self.mean_vec = mean_vec
        self.step_size = step_size
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.maxiter = maxiter
        self.ndim = mean_vec.size

        # Default values for population sizes and weights
        if pop_size:
            self.pop_size = pop_size
        else:
            self.pop_size = 4 + np.floor(3*np.log(self.ndim))

        if elite_size:
            self.elite_size = elite_size
        else:
            self.elite_size = self.pop_size//2

        if weights:
            self.weights = weights
        else:
            self.weights = (np.log(0.5*(pop_size + 1)) -
                            np.log(np.arange(1, pop_size+1)))


        norm_pos_weights = (self.weights[:self.elite_size] /
                            sum(self.weights[:self.elite_size]))
        self.mu_eff = (sum(norm_pos_weights)**2 /
                       sum(norm_pos_weights**2))

        # Initializes the step size parameters
        if c_sig:
            self.c_sig = c_sig
        else:
            self.c_sig = (self.mu_eff + 2) / (self.ndim + self.mu_eff + 5)

        if d_sig:
            self.d_sig = d_sig
        else:
            self.d_sig = (1 + 2*max(0,
                         np.sqrt((self.mu_eff-1)/(self.ndim+1)) - 1)
                         + self.c_sig)

        # Initializes parameters for covariance adaption
        if c_c:
            self.c_c = c_c
        else:
            self.c_c = ((4 + self.mu_eff/self.ndim) /
                        (self.ndim + 4 + 2*self.mu_eff/self.ndim))

        if c_1:
            self.c_1 = c_1
        else:
            self.c_1 = 2/((self.ndim + 1.3)**2 + self.mu_eff)

        if c_mu:
            self.c_mu = c_mu
        else:
            self.c_mu = min(1 - self.c_1, 2*((self.mu_eff - 2 + 1/self.mu_eff) /
                                   ((self.ndim + 2)**2 + self.mu_eff)))

        if c_m:
            self.c_m = c_m
        else:
            self.c_m = 1

        # Set the negative weights, if not provided:
        if not weights:
            neg_weights = self.weights[self.elite_size:]
            mu_eff_neg = (sum(neg_weights)**2 /
                          sum(neg_weights**2))

            alpha_mu_neg = 1 + self.c_1/self.c_mu
            alpha_mu_eff_neg = 1 + 2*mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def_neg = ((1 - self.c_1 - self.c_mu) /
                                  (self.ndim*self.c_mu))
            print(alpha_mu_neg, alpha_mu_eff_neg, alpha_pos_def_neg)
            self.weights[:self.elite_size] = norm_pos_weights
            denom = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_pos_def_neg)
            self.weights[self.elite_size:] = (denom * neg_weights /
                                              -sum(neg_weights))

        self.cov_mat = np.identity(len(self.mean_vec))

        # Set the evolution paths
        self.p_sig = 0
        self.p_c = 0

    def sample_and_evaluate(self, func, ndim, mean_vec,
                            cov_mat, pop_size, step_size):
        y = np.random.multivariate_normal(np.zeros(ndim),
                                          cov_mat,
                                          size=pop_size)
        x = mean_vec + step_size*y
        func_values = list(map(func, x))

        return (y, x, func_values)

    def zip_and_sort(self, func_values, x, y):
        zipped = list(zip(func_values, x, y))
        zipped.sort(key=itemgetter(0))
        return zipped


    def minimize(self):

        expec_norm_gaussian = (np.sqrt(2) * spsp.gamma(0.5*(self.ndim + 1)) /
                               spsp.gamma(0.5*self.ndim))
        y, x, func_values = sample_and_evaluate(self.func, self.ndim,
                                                self.mean_vec, self.cov_mat,
                                                self.pop_size, self.step_size)
        sorted_values = zip_and_sort(func_values, x, y)

        y = sorted_values[2]
        x = sorted_values[1]
        func_values = sorted_values[0]

        best_fvalue = sort_values[0][0]
        best_param = sort_values[0][1]

        for i in range(self.maxiter):
            y_weighted = sum(y[:self.elite_size] *
                             self.weights[:self.elite_size])

        else:
            print('Failed to converge within maximum iteration bound.')

def test_func(vec):
    x, y = vec
    return x**2 + y**2

if __name__ == '__main__':
    mean_vec = np.array([1, 2])

    cma_obj = CovMatAdapt(test_func, mean_vec, step_size=1, pop_size=9)

    print(cma_obj.minimize())
