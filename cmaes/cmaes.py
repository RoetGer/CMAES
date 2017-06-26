import numpy as np

class CovMatAdapt:

    def __init__(self, func, mean_vec, step_size, pop_size=None,
                     elite_size=None, weights=None,
                     c_sig=None, d_sig=None, c_c=None,
                     c_1=None, c_mu=None, abs_tol=-1e3,
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
        print(self.mu_eff)
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
        print(np.sqrt((self.mu_eff-1)/(self.ndim+1)))
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

        # Set the negative weights, if not provided:
        if not weights:
            neg_weights = self.weights[self.elite_size:]
            mu_eff_neg = (sum(neg_weights)**2 /
                          sum(neg_weights**2))

            alpha_mu_neg = 1 + self.c_1/self.c_mu
            alpha_mu_eff_neg = 1 + 2*mu_eff_neg / (self.mu_eff + 2)
            alpha_pos_def_neg = ((1 - self.c_1 - self.c_mu) /
                                  (self.ndim*self.c_mu))

            self.weights[:self.elite_size] = norm_pos_weights
            denom = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_pos_def_neg)
            self.weights[self.elite_size:] = denom * neg_weights / sum(neg_weights)

        self.cov_mat = np.identity(len(self.mean_vec))

def test_func():
    return x**2 + y**2

if __name__ == '__main__':
    mean_vec = np.array([1, 2])

    cma_obj = CovMatAdapt(test_func, mean_vec, step_size=1, pop_size=9)
