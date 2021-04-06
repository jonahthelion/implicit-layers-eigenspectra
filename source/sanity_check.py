import numpy as onp
import sys
import jax.numpy as jnp
import jax
from jax import jvp, vjp, jacfwd
from jax.flatten_util import ravel_pytree

# hacky but fine for now
sys.path.append('./source/deq-jax')
from src.modules.deq import deq
from src.modules.broyden import broyden


def toy_model(dim=1):
    A = onp.random.randn(dim, dim) * 0.1
    B = onp.random.randn(dim, dim)
    b = onp.random.randn(dim)
    c = onp.random.randn(dim)

    params = [A, B, b, c]

    data = onp.random.randn(dim)

    return [jnp.array(row) for row in params], jnp.array(data)


def loss_fn(params, z, data):
    A, B, b, c = params
    return jnp.dot(c, z)


def analytic_zstar(params, data):
    A, B, b, c = params
    n = len(b)
    return jnp.linalg.solve(jnp.eye(n) - A, B @ data + b)


def impl_f(params, z, data):
    A, B, b, c = params
    return A @ z + B @ data + b


def deq_zstar(params, data):
    A, B, b, c = params
    z0 = jnp.zeros((1, len(b), 1))
    max_iter = 30
    eps = 1e-6 * jnp.sqrt(z0.size)

    # layer needs to take as input B x H x L array
    def g(z):
        out = impl_f(params, z[0, :, 0], data) - z[0, :, 0]
        return out.reshape((1, out.size, 1))

    result_info = jax.lax.stop_gradient(
        broyden(g, z0, max_iter, eps)
    )
    z = result_info['result'][0, :, 0]
    return z


def dz_dtheta_jvp(params, z, data, queryv):
    """Returns
        dz/dtheta * v
        = (I - df/dz)^-1 * df/dtheta * v
        Currently assumes z is a vector (e.g. is 1-dimensional)
    """
    n = len(z)
    out = jnp.linalg.solve(
        jnp.eye(n) - jacfwd(impl_f, 1)(params, z, data),
        jvp(lambda x: impl_f(x, z, data), [params], [queryv])[1])
    return out


def dz_dtheta_vjp(params, z, data, queryv):
    n = len(z)
    _, vjp_func = vjp(lambda x: impl_f(x, z, data), params)
    vec = jnp.linalg.solve(
        jnp.eye(n) - jacfwd(impl_f, 1)(params, z, data).T,
        queryv
    )
    return vjp_func(vec)[0]


def full_dl_dtheta(params, z, data):
    term0 = jax.grad(loss_fn, 0)(params, z, data)
    term1 = dz_dtheta_vjp(params, z, data, jax.grad(loss_fn, 1)(params, z, data))

    return [v0+v1 for v0,v1 in zip(term0,term1)]


def full_dtheta_hvp(params, z, data, queryv):
    term0 = jvp(lambda x: full_dl_dtheta(x, z, data), [params], [queryv])[1]
    term1 = jvp(lambda x: full_dl_dtheta(params, x, data),
                    [z], [dz_dtheta_jvp(params, z, data, queryv)]
                )[1]
    return [v0+v1 for v0,v1 in zip(term0,term1)]


def check_deq():
    onp.random.seed(123)
    params, data = toy_model(dim=2)
    queryv = [jnp.array(onp.random.rand(*p.shape)) for p in params]

    # 0th order
    ana_zstar = analytic_zstar(params, data)
    impl_zstar = deq_zstar(params, data)
    assert(jnp.allclose(ana_zstar, impl_zstar))
    assert(jnp.allclose(ana_zstar, impl_f(params, ana_zstar, data)))

    # check dz/dtheta vp
    impl_dzdt = dz_dtheta_jvp(params, ana_zstar, data, queryv)
    ana_dzdt = jvp(lambda x: analytic_zstar(x, data), [params], [queryv])[1]
    assert(jnp.allclose(impl_dzdt, ana_dzdt)), f'{impl_dzdt} {ana_dzdt}'

    # check vp dz/dtheta
    impl_revdzdt = dz_dtheta_vjp(params, ana_zstar, data, ana_zstar)
    ana_revdzdt = vjp(lambda x: analytic_zstar(x, data), params)[1](ana_zstar)[0]
    # TODO something weird comparing device arrays?
    for a,b in zip(impl_revdzdt,ana_revdzdt):
        assert(jnp.allclose(a, b)), f'{a} {b}'

    # check full_dl_dtheta
    impl_dldtheta = full_dl_dtheta(params, ana_zstar, data)
    ana_dldtheta = jax.grad(lambda x: loss_fn(x, analytic_zstar(x, data), data))(params)
    for a,b in zip(impl_dldtheta,ana_dldtheta):
        assert(jnp.allclose(a, b)), f'{a} {b}'

    # check hvp
    impl_hvp = full_dtheta_hvp(params, ana_zstar, data, queryv)
    ana_hvp = jvp(lambda y: jax.grad(lambda x: loss_fn(x, analytic_zstar(x, data), data))(y), [params], [queryv])[1]
    for a,b in zip(impl_dldtheta,ana_dldtheta):
        assert(jnp.allclose(a, b)), f'{a} {b}'

    print('All checks passed!')
