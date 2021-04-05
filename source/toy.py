import numpy as onp
import jax.numpy as jnp
from jax import grad, value_and_grad, hessian, jacfwd
import jax.random as jaxrnd
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# hacky but fine for now
sys.path.append('./source/spectral-density/jax')
import density as density_lib
import lanczos
import hessian_computation


def generate_problem(ntrain, ntest):
    """X is ntrain x 2
    Y is ntrain
    Xgt is ntest x 2
    Ygt is ntest
    """
    X = onp.random.uniform(-1, 1, (ntrain, 2))
    X[:, 1] = 1.0  # bias term is fixed to 1.0 (we don't learn this one)
    Y = onp.zeros(ntrain)
    Xgt = onp.random.uniform(-1, 1, (ntest, 2))
    Xgt[:, 1] = 1.0
    Ygt = Xgt @ onp.array([1.3, -0.4])
    return jnp.array(X), jnp.array(Y), jnp.array(Xgt), jnp.array(Ygt)


def analytic_fixed_point(X, Y):
    zstar,_,_,_ = jnp.linalg.lstsq(X, Y)
    return zstar


def loss(z, Xgt, Ygt):
    return jnp.power(Xgt @ z - Ygt, 4).mean()


def impl_f(theta, zz, X):
    return zz - jacfwd(lambda y,z: jnp.square(X @ z - y).mean(), 1)(theta, zz)


def analytic_trajectory(X, Y, Xgt, Ygt, T, eta):
    Ys = []
    grad_fn = value_and_grad(lambda y: loss(analytic_fixed_point(X, y), Xgt, Ygt))
    for step in range(T):
        if step == 0:
            Ys.append(Y)
        else:
            l,grady = grad_fn(Ys[-1])
            Ys.append(Ys[-1] - eta*grady)
            print('Loss:', l)
    return Ys


def implicit_trajectory(X, Y, Xgt, Ygt, T, eta):
    Ys = []

    # the function f(theta, z(theta)) that z(theta) is the fixed point of
    df_dtheta = jacfwd(impl_f, 0)
    df_dz = jacfwd(impl_f, 1)
    dl_dz = jacfwd(lambda l: loss(l, Xgt, Ygt))

    for step in range(T):
        if step == 0:
            Ys.append(Y)
        else:
            zstar = analytic_fixed_point(X, Ys[-1])
            grady = dl_dz(zstar) @ jnp.linalg.inv(jnp.eye(len(zstar)) - df_dz(Ys[-1], zstar, X)) @ df_dtheta(Ys[-1], zstar, X)
            Ys.append(Ys[-1] - eta*grady)
    return Ys


def analytic_eigenspectra(X, Ys, Xgt, Ygt):
    print('computing brute force hessian spectra')
    eigvals = []
    spectra = []
    hess_func = hessian(lambda y: loss(analytic_fixed_point(X, y), Xgt, Ygt))
    for Y in Ys:
        hess = hess_func(Y)

        # true eigenvalues
        vals,vecs = jnp.linalg.eigh(hess)
        eigvals.append(vals)

        # lanczos
        tridiag, vecs = lanczos.lanczos_alg(lambda v: hess @ v, len(hess), order=10, rng_key=jaxrnd.PRNGKey(0))
        density, grids = density_lib.tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
        spectra.append((density, grids))
    return eigvals, spectra


def implicit_eigenspectra(X, Ys, Xgt, Ygt):
    print('computing implicit hessian spectra')
    hess_func = hessian(lambda y: loss(analytic_fixed_point(X, y), Xgt, Ygt))

    def dzdtheta(theta, zz, X):
        mat = jnp.eye(len(zz)) - jacfwd(impl_f, argnums=1)(theta, zz, X)
        return jnp.linalg.inv(mat) @ jacfwd(impl_f, argnums=0)(theta, zz, X)

    def dtheta_op(f, X):
        """assumes f(theta, z(theta)) form of f
        """
        return lambda theta,zz: jacfwd(f, 0)(theta,zz) + jacfwd(f, 1)(theta,zz) @ dzdtheta(theta, zz, X)

    eigvals = []
    spectra = []
    for Y in Ys:
        hess = hess_func(Y)
        zstar = analytic_fixed_point(X, Y)

        hess = dtheta_op(dtheta_op(lambda theta,zz: loss(impl_f(theta,zz,X), Xgt, Ygt), X), X)(Y, zstar)
        # gtgrad = hess_func(Y)

        # true eigenvalues
        vals,vecs = jnp.linalg.eigh(hess)
        eigvals.append(vals)

        # lanczos
        tridiag, vecs = lanczos.lanczos_alg(lambda v: hess @ v, len(hess), order=10, rng_key=jaxrnd.PRNGKey(0))
        density, grids = density_lib.tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
        spectra.append((density, grids))
    return eigvals, spectra


def linear_regression(rnd_seed=60, eta=0.05):
    """Optimize the data (X,Y) such that the LSQ solution
    to (X,Y+noise) fits some other data (X',Y') well.

    The more noise there is, the smaller we expect the eigenvalues
    of the hessian to be.

    eta is the learning rate for both the analytic and implicit optimizers
    """
    onp.random.seed(rnd_seed)
    X,Y,Xgt,Ygt = generate_problem(ntrain=4, ntest=6)

    Yanas = analytic_trajectory(X, Y, Xgt, Ygt, T=50, eta=eta)
    Yimpls = implicit_trajectory(X, Y, Xgt, Ygt, T=50, eta=eta)
    
    # compute eigenspectra
    eigvals,eigspectra = analytic_eigenspectra(X, Yanas, Xgt, Ygt)
    eigvals_impl,eigspectra_impl = implicit_eigenspectra(X, Yimpls, Xgt, Ygt)
    eiglim = (min((min(eig) for eig in eigvals)) - 0.1, max((max(eig) for eig in eigvals)) + 0.1)

    for t,(Yana,eigval) in enumerate(zip(Yanas, eigvals)):
        # analytic
        zstar = analytic_fixed_point(X, Yana)
        xeval = jnp.array([[-1.5, 1.0],[1.5, 1.0]])
        yeval = xeval @ zstar

        fig = plt.figure(figsize=(10, 10))
        gs = mpl.gridspec.GridSpec(2, 2)

        ax = plt.subplot(gs[0, 0])
        plt.plot(X[:, 0], Yana, '.', label=r'Learned Data ($\theta$)')
        plt.plot(xeval[:, 0], yeval, label=r'Best-Fit Line ($z^*(\theta)$)')
        plt.plot(Xgt[:, 0], Ygt, '.', label=r'Test Data')
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.5))
        ax.set_aspect('equal')
        plt.legend(loc='upper left')
        plt.title('Explicit Gradient')

        ax = plt.subplot(gs[0, 1])
        plt.plot(eigspectra[t][1], eigspectra[t][0], label='Lanczos', alpha=0.4)
        plt.hist(eigval, bins=jnp.linspace(eiglim[0], eiglim[1], 150), label='Exact')
        plt.xlim(eiglim)
        plt.ylim((0.0, 3.2))
        plt.title('Explicit Eigenspectrum')
        plt.legend(loc='upper right')

        # implicit
        zstar = analytic_fixed_point(X, Yimpls[t])
        yeval = xeval @ zstar

        ax = plt.subplot(gs[1, 0])
        plt.plot(X[:, 0], Yimpls[t], '.', label=r'Learned Data ($\theta$)')
        plt.plot(xeval[:, 0], yeval, label=r'Best-Fit Line ($z^*(\theta)$)')
        plt.plot(Xgt[:, 0], Ygt, '.', label=r'Test Data')
        plt.xlim((-1.5, 1.5))
        plt.ylim((-1.5, 1.5))
        ax.set_aspect('equal')
        plt.legend(loc='upper left')
        plt.title('Implicit Gradient')

        ax = plt.subplot(gs[1, 1])
        plt.plot(eigspectra_impl[t][1], eigspectra_impl[t][0], label='Lanczos', alpha=0.4)
        plt.hist(eigvals_impl[t], bins=jnp.linspace(eiglim[0], eiglim[1], 150), label='Exact')
        plt.xlim(eiglim)
        plt.ylim((0.0, 3.2))
        plt.title('Implicit Eigenspectrum')
        plt.legend(loc='upper right')

        imname = f'silly{t:03}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close(fig)

