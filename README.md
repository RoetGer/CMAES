# CMAES

This package is an implementation of the covariance matrix adaption (CMA) evolution strategy based on the paper 'The CMA Evolution Strategy: A Tutorial' by Nikolaus Hansen.

CMA is a stochastic optimization method for non-convex functions with continuous parameters.


Example application:

```python
import numpy as np

from cmaes import CovMatAdapt

def simple_func(vec):
    x, y = vec
    return x**2 + y**2

# Defines the starting location of the search
mean_vec = np.array([1, 2])

# Rule of thumb: step_size should be set such that the optimum
# is within mean_vec +/- 3*step_size
cma_obj = CovMatAdapt(simple_func, mean_vec, step_size=1)

cma_obj.minimize()
```

## References
Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).
