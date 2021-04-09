import jax
from jax import jvp,vjp,grad
import haiku as hk
import numpy as onp
import sys
import jax.numpy as jnp

# hacky but fine for now
sys.path.append('./source/deq-jax')
from src.modules.deq import deq


def prepro_fn(batch, hidden_size):
    x = batch['image'].astype(jnp.float32) / 255.0
    x = hk.Flatten()(x)
    x = hk.Linear(hidden_size)(x)
    return x.reshape((x.shape[0], 1, x.shape[1]))


class FeedForwardBlock(hk.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, h, input_embs):
        B, _, H = h.shape
        h = hk.Linear(H, with_bias=False)(h) + input_embs
        return h


def deq_fn(x, hidden_size, max_steps, analytic):
    h = jnp.zeros_like(x)

    fcn_block = FeedForwardBlock()
    transformed_net = hk.transform(fcn_block)
    inner_params = hk.experimental.lift(
        transformed_net.init)(hk.next_rng_key(), h, x)
    def f(_params, _rng, _z, *args): return transformed_net.apply(_params, _rng, _z, *args)

    if analytic:
        matinv = jnp.linalg.inv(jnp.eye(hidden_size) - inner_params['feed_forward_block/linear']['w'])
        x0 = jnp.array([x[b,0] @ matinv for b in range(len(x))]).reshape(x.shape[0], 1, hidden_size)
    else:
        x0 = deq(inner_params, hk.next_rng_key(), h, f, max_steps, x)
    return x0


def postpro_fn(x):
    x = x.squeeze(1)
    x = hk.Linear(10)(x)
    return x


def full_deq_fn(x, kind, hidden_size, max_steps):
    if kind == 'forward':
        x = prepro_fn(x, hidden_size)
        x = deq_fn(x, hidden_size, max_steps, analytic=False)
        x = postpro_fn(x)
    if kind == 'analytic':
        x = prepro_fn(x, hidden_size)
        x = deq_fn(x, hidden_size, max_steps, analytic=True)
        x = postpro_fn(x)
    return x


def check_full_deq(rnd_seed=42, bsz=1, hidden_size=2, max_steps=20):
    onp.random.seed(rnd_seed)
    batch = {
        'image': onp.random.rand(bsz, 1, 28, 28)*255,
        'label': onp.random.randint(10, size=(bsz,)),
    }
    net = hk.transform(lambda x,kind: full_deq_fn(batch, kind, hidden_size, max_steps))
    params = net.init(jax.random.PRNGKey(rnd_seed), batch, 'forward')

    deq_f = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'forward')
    ana_f = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'analytic')
    loss_f = lambda x: x.sum()  # TODO make this xent

    def check_all_close(x,y):
        assert(jnp.allclose(x,y))

    # 0th order
    deq_out = deq_f(params, batch)
    ana_out = ana_f(params, batch)
    assert(jnp.allclose(deq_out, ana_out))

    # 1st order
    deq_grad = grad(lambda params: loss_f(deq_f(params, batch)))(params)
    ana_grad = grad(lambda params: loss_f(ana_f(params, batch)))(params)
    jax.tree_multimap(check_all_close, deq_grad, ana_grad)
