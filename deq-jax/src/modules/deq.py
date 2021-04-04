from typing import Callable
import jax.numpy as jnp

from src.modules.rootfind import rootfind, rootfind_grad, my_rootfind, my_rootfind_grad


def deq(params: dict, rng, z: jnp.ndarray, fun: Callable, max_iter: int, custom_vjp: bool, *args) -> jnp.ndarray:
    """
    Apply Deep Equilibrium Network to haiku function.
    :param params: params for haiku function
    :param rng: rng for init and apply of haiku function
    :param fun: func to apply in the deep equilibrium limit, f(params, rng, x, *args)
     and only a function of JAX primatives (e.g can not be passed bool)
    :param max_iter: maximum number of integers for the broyden method
    :param z: initial guess for broyden method
    :param args: all other JAX primatives which must be passed to the function
    :return: z_star: equilibrium hidden state s.t lim_{i->inf}fun(z_i) = z_star
    """

    # define equilibrium eq (f(z)-z)
    def g(_params, _rng, _x, *args): return fun(_params, _rng, _x, *args) - _x

    # find equilibrium point
    if custom_vjp:
        z_star = rootfind(g, max_iter, params, rng, z, *args)
    else:
        z_star = my_rootfind(g, max_iter, params, rng, z, *args)

    # set up correct graph for chain rule (bk pass)
    # in original implementation this is run only if in_training
    z_star = fun(params, rng, z_star, *args)
    if custom_vjp:
        z_star = rootfind_grad(g, max_iter, params, rng, z_star, *args)
    else:
        z_star = my_rootfind_grad(g, max_iter, params, rng, z_star, *args)
    return z_star


def wtie(params: dict, rng, z: jnp.ndarray, fun: Callable, feedfwd_layers: int, *args) -> jnp.ndarray:
    """
    Apply Weight Tied Network to haiku function.
    :param params: params for haiku function
    :param rng: rng for init and apply of haiku function
    :param fun: func to apply in the deep equilibrium limit, f(params, rng, x, *args)
     and only a function of JAX primatives (e.g can not be passed bool)
    :param feedfwd_layers: number of feed forward iterations
    :param z: initial guess for broyden method
    :param args: all other JAX primatives which must be passed to the function
    :return: z_star: final hidden state
    """

    for _ in range(feedfwd_layers):
        z = fun(params, rng, z, *args)
    z_star = z
    return z_star