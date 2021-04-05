# Modified from https://github.com/deepmind/dm-haiku/blob/master/examples/transformer/model.py
# ==============================================================================
"""Train a transformer for language modeling on LM1B.
This example serves to demonstrate:
  - A clean Haiku transformer implementation.
  - An example minimal training loop around it.
We have not tuned the hyperparameters for LM1B at all.
Note: Run with --alsologtostderr to see outputs.
"""

import functools
import os
import pickle
import time
from typing import Any, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import optax
import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from absl import logging

import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/deq-jax/")

from src.modules.deq import deq, wtie


def toy_model(dim=1):
    A = np.random.randn(dim, dim) * 0.1
    B = np.random.randn(dim, dim)
    b = np.random.randn(dim)
    # c = np.random.randn(dim)

    # params = (A, B, b, c)
    params = (A, B, b)

    data = np.random.randn(dim)

    return params, data


def analytic_fn(params, data):
    # (A, B, b, c) = params
    (A, B, b) = params
    c = jnp.ones_like(b)
    dim, = b.shape
    z_star = jnp.linalg.solve(jnp.eye(dim) - A, B @ data + b)
    loss = c @ z_star

    return loss


def deq_fn(params, data):
    # (A, B, b, c) = params
    (A, B, b) = params
    c = jnp.ones_like(b)
    dim, = b.shape

    def layer(inner_params, rng, z):
        _A, _B, _b = inner_params
        z = z.squeeze((0, 1))
        z_update = _A @ z + _B @ data + _b
        return jnp.expand_dims(z_update, (0, 1))

    _inner_params = (A, B, b)
    z0 = jnp.zeros((1, 1, dim))
    z_star = deq(_inner_params, jax.random.PRNGKey(0), z0, layer, max_iter=50, custom_vjp=False)
    # z_star = wtie(_inner_params, jax.random.PRNGKey(0), z0, layer, feedfwd_layers=15)
    z_star = z_star.squeeze((0, 1))

    loss = c @ z_star


    return loss


if __name__ == '__main__':
    np.random.seed(123)
    params, data = toy_model()

    analytic_loss = analytic_fn(params, data)
    deq_loss = deq_fn(params, data)
    print(analytic_loss, deq_loss)

    analytic_jac = jax.jacobian(lambda p: analytic_fn(p, data))(params)
    deq_jac = jax.jacobian(lambda p: deq_fn(p, data))(params)
    analytic_jac, _ = ravel_pytree(analytic_jac)
    deq_jac, _ = ravel_pytree(deq_jac)
    # np.testing.assert_almost_equal(analytic_jac, deq_jac)
    print(analytic_jac)
    print(deq_jac)


    analytic_hes = jax.hessian(lambda p: analytic_fn(p, data))(params)
    deq_hes = jax.hessian(lambda p: deq_fn(p, data))(params)
    analytic_hes, _ = ravel_pytree(analytic_hes)
    deq_hes, _ = ravel_pytree(deq_hes)

    dim, = analytic_hes.shape
    sqrt_dim = int(np.sqrt(dim))
    print(analytic_hes.reshape(sqrt_dim,sqrt_dim))
    print(deq_hes.reshape(sqrt_dim, sqrt_dim) + deq_hes.reshape(sqrt_dim, sqrt_dim).transpose())

    np.testing.assert_almost_equal(np.array(analytic_hes.reshape(sqrt_dim,sqrt_dim)), np.array(deq_hes.reshape(sqrt_dim, sqrt_dim) + deq_hes.reshape(sqrt_dim, sqrt_dim).transpose()), decimal=5)
