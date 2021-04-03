import os
import sys
import jax.numpy as jnp
import jax.random as jaxrnd
from jax import grad,hessian
import numpy as onp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# hacky but fine for now
sys.path.append('./src/spectral-density/jax')
import density as density_lib
import lanczos
import hessian_computation


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


def toy_model(rnd_seed=42, imname='check.jpg', nparams=100):
    onp.random.seed(rnd_seed)

    grad_loss = grad(analytic_fwd)
    hessian_loss = hessian(analytic_fwd)

    # optimize v for a few steps
    for step in range(20):
        c, v, w, b = gen_problem(nparams)

        jax_hess = hessian_loss(v, c, w, b)
        true_eigvals,_ = jnp.linalg.eigh(jax_hess)

        # lanczos algorithm eigenspectrum
        hvp = lambda v : jax_hess @ v
        tridiag, vecs = lanczos.lanczos_alg(hvp, nparams, order=90, rng_key=jaxrnd.PRNGKey(rnd_seed+100))
        density, grids = density_lib.tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

        xlim = (min(grids) - (max(grids) - min(grids))*0.2, max(grids) + (max(grids) - min(grids))*0.2)

        fig = plt.figure(figsize=(10, 5))
        gs = mpl.gridspec.GridSpec(1, 2)

        ax = plt.subplot(gs[0, 0])
        plt.title('Lanczos Spectrum')
        plt.plot(grids, density)
        plt.ylabel('Density')
        plt.xlabel('Eigenvalue')
        # plt.xlim(xlim)

        ax = plt.subplot(gs[0, 1])
        plt.title('Ground Truth Spectrum')
        plt.hist(true_eigvals, bins=100)
        # plt.xlim(xlim)

        imname = f'{step:04}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close(fig)



