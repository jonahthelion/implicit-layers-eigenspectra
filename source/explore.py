import os
import sys
import jax.numpy as jnp
import jax.random as jaxrnd
from jax import grad, hessian, jacfwd, jvp, jacrev
import numpy as onp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# hacky but fine for now
sys.path.append('./source/spectral-density/jax')
import density as density_lib
import lanczos
import hessian_computation


def loss(z, c):
    return jnp.dot(z, c)


def f_theta(z, v, b, w):
    return b + v * jnp.dot(w, z)


def analytic_fixed_point(v, b, w):
    n = len(v)
    return jnp.linalg.solve(jnp.eye(n) - v.reshape((n, 1)) @ w.reshape((1, n)), b)


def gen_problem(n):
    c = onp.random.rand(n)
    v = onp.random.rand(n)*0.4
    w = onp.random.rand(n)*0.4
    b = onp.random.rand(n)
    return jnp.array(c), jnp.array(v), jnp.array(w), jnp.array(b)


def dzdtheta(z, v, b, w):
    n = len(z)
    mat = jnp.eye(n) - jacfwd(f_theta, argnums=0)(z, v, b, w)
    return jnp.linalg.inv(mat) @ jacfwd(f_theta, argnums=1)(z, v, b, w)


def ana_dzdtheta(v, b, w):
    n = len(v)
    matinv = jnp.linalg.inv(jnp.eye(n) - v.reshape((n, 1)) @ w.reshape((1, n)))
    return matinv * (w.reshape((1, n)) @ matinv @ b)


def dtheta_op(f, b, w):
    """assumes f(z(theta), theta) form of f
    """
    return lambda z,v: jacfwd(f, 1)(z,v) + jacfwd(f, 0)(z,v) @ jacfwd(analytic_fixed_point)(v, b, w)


def toy_model(rnd_seed=80, imname='check.jpg', nparams=3):
    """We have a loss function

        L = loss(f_theta(z))

        where z* is defined implicitly

        z = f_theta(z).

        We want to compute a hessian-vector product where the hessian is of the loss wrt
        the parameters theta.
    """
    onp.random.seed(rnd_seed)

    c, v, w, b = gen_problem(nparams)
    queryv = onp.random.rand(nparams)
    zstar = analytic_fixed_point(v, b, w)

    # check that 0th order lines up
    assert(jnp.allclose(zstar, f_theta(zstar, v, b, w)))

    # check that 1st order lines up 
    predjac = dtheta_op(lambda z,v: f_theta(z, v, b, w), b, w)(zstar, v)
    predjac2 = jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(zstar, v, b, w)) @ jacfwd(f_theta, 1)(zstar, v, b, w)
    gtjac = jacfwd(analytic_fixed_point)(v, b, w)
    assert(jnp.allclose(predjac, gtjac))
    assert(jnp.allclose(predjac2, gtjac))

    # check that 2nd order lines up (d2z / d2theta)
    # z* = f_theta(z*)
    # d z* / d theta = df / d theta + df / dz * dz/d theta
    # d z* / d theta = dtheta_op(f)
    # d^2 z* / d theta^2 = dtheta_op(dtheta_op(f))
    predhess = dtheta_op(dtheta_op(lambda z,v: f_theta(z, v, b, w), b, w), b, w)(zstar, v)
    gthess = jacfwd(jacfwd(analytic_fixed_point))(v, b, w)
    assert(jnp.allclose(predhess, gthess))

    predhess2 = dtheta_op(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), b, w)(zstar, v) +\
                dtheta_op(lambda z,v: jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v) @ jacfwd(analytic_fixed_point)(v, b, w), b, w)(zstar, v)
    assert(jnp.allclose(predhess2, gthess))

    predhess3 = jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 1)(zstar, v) + jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w) +\
                dtheta_op(lambda z,v: jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v) @ jacfwd(analytic_fixed_point)(v, b, w), b, w)(zstar, v)
    assert(jnp.allclose(predhess3, gthess))

    predhess4 = jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 1)(zstar, v) + jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w) +\
                dtheta_op(lambda z,v: jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z,v) @ jnp.linalg.inv(jnp.eye(nparams) - jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z,v)) @ jacfwd(lambda x,y: f_theta(x, y, b, w), 1)(z,v), b, w)(zstar, v)
    assert(jnp.allclose(predhess4, gthess))

    predhess5 = jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 1)(zstar, v) + jacfwd(jacfwd(lambda z,v: f_theta(z, v, b, w), 1), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w) +\
                jacfwd(lambda z,v: jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v) @ jnp.linalg.inv(jnp.eye(nparams) - jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v)) @ jacfwd(lambda x,y: f_theta(x, y, b, w), 1)(z, v), 1)(zstar, v) +\
                jacfwd(lambda z,v: jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v) @ jnp.linalg.inv(jnp.eye(nparams) - jacfwd(lambda x,y: f_theta(x, y, b, w), 0)(z, v)) @ jacfwd(lambda x,y: f_theta(x, y, b, w), 1)(z, v), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w)
    assert(jnp.allclose(predhess5, gthess))

    predhess6 = dtheta_op(lambda z,v: jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(z,v,b,w)) @ jacfwd(f_theta, 1)(z,v,b,w), b, w)(zstar, v)
    assert(jnp.allclose(predhess6, gthess))

    predhess7 = jacfwd(lambda z,v: jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(z,v,b,w)) @ jacfwd(f_theta, 1)(z,v,b,w), 1)(zstar, v) +\
                jacfwd(lambda z,v: jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(z,v,b,w)) @ jacfwd(f_theta, 1)(z,v,b,w), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w)
    assert(jnp.allclose(predhess7, gthess))

    predhess8 = jacfwd(lambda z,v: jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(z,v,b,w)), 1)(zstar, v) @ jacfwd(f_theta, 1)(zstar,v,b,w) + jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(zstar,v,b,w)) @ jacfwd(lambda y: jacfwd(f_theta, 1)(zstar,y,b,w))(v) +\
                jacfwd(lambda z,v: jnp.linalg.inv(jnp.eye(nparams) - jacfwd(f_theta)(z,v,b,w)) @ jacfwd(f_theta, 1)(z,v,b,w), 0)(zstar, v) @ jacfwd(analytic_fixed_point)(v, b, w)
    assert(jnp.allclose(predhess8, gthess))
    print('checks passed!')


def check_inverse(nparams=3, rnd_seed=80):
    onp.random.seed(rnd_seed)
    c, v, w, b = gen_problem(nparams)

    f = lambda v: v.reshape((nparams, 1)) * w.reshape((1, nparams))
    g = lambda v: jnp.linalg.inv(jnp.eye(nparams) - f(v))
    queryv = onp.random.rand(nparams)

    gtjac = jvp(g, [v], [queryv])[1]
    predjacfwd = jnp.linalg.inv(jnp.eye(nparams) - f(v)).T @ jvp(f, [v], [jnp.linalg.inv(jnp.eye(nparams) - f(v)).T @ queryv])[1]

    print(gtjac)
    print(predjacfwd)

