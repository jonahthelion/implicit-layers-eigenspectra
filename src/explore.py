import jax.numpy as jnp
from jax import grad,hessian
import numpy as onp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from spectral_density.jax import density


def g(v, w, x):
    n = len(x)
    return jnp.linalg.norm( jnp.eye(n) - v.reshape((n, 1)) @ w.reshape((1, n)) )


def analytic_fwd(v, c, w, b):
    n = len(c)
    return jnp.dot(c, jnp.linalg.solve(jnp.eye(n) - v.reshape((n, 1)) @ w.reshape((1, n)), b))


def analytic_grad(v, c, w, b):
    n = len(v)
    matinv = jnp.linalg.inv(jnp.eye(n) - v.reshape((n, 1)) @ w.reshape((1, n)))
    return c @ matinv * (jnp.ones((n, 1)) @ w.reshape((1, n)) @ matinv @ b)


def gen_problem(n):
    c = onp.random.rand(n)
    v = onp.random.rand(n)*0.4
    w = onp.random.rand(n)*0.4
    b = onp.random.rand(n)
    return c, v, w, b


def toy_model(rnd_seed=42, imname='check.jpg'):
    onp.random.seed(rnd_seed)

    c, v, w, b = gen_problem(5)

    jax_hess = hessian(analytic_fwd)(v, c, w, b)



