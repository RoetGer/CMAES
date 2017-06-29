import numpy as np
import pytest

from cmaes import CovMatAdapt

def simple_func(vec):
    x, y = vec
    return x**2 + y**2

def test_simple_func_optim():
    np.random.seed(5)
    mean_vec = np.array([1, 2])

    cma_obj = CovMatAdapt(simple_func, mean_vec, step_size=1)

    opt_dict = cma_obj.minimize()

    assert opt_dict['best fvalue'] == pytest.approx(0, 0.01)
