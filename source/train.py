import sys
import jax
import jax.numpy as jnp
import jax.random as jaxrnd
import haiku as hk
from jax import value_and_grad

# hacky but fine for now
sys.path.append('./source/deq-jax')
from src.modules.deq import deq


def train_deq():
    print('hello')
