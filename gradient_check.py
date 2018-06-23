from __future__ import division
import numpy as np


def f(x):
    b = np.ones_like(x)
    return x * x + b


def eval_numerical_gradient(f, x):
    """ a naive implementation of numerical gradient of f at x

    - f should be a function that takes a single argument

    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001

    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index

        old_value = x[ix]

        x[ix] = old_value + h
        fxh_left = f(x)

        x[ix] = old_value - h
        fxh_right = f(x)

        x[ix] = old_value

        grad[ix] = (fxh_left - fxh_right) / (2 * h)
        it.iternext()

    return grad


if __name__ == '__main__':
    x = np.array([2.0])
    grad = eval_numerical_gradient(f, x)
    print(grad)
