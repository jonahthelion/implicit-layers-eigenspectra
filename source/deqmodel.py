import jax
from jax import jvp,vjp,grad,jacfwd,vmap
import haiku as hk
import numpy as onp
import sys
import jax.numpy as jnp
from time import time

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


def deq_fn(x, hidden_size, max_steps, kind, z=None):
    h = jnp.zeros_like(x)

    fcn_block = FeedForwardBlock()
    transformed_net = hk.transform(fcn_block)
    inner_params = hk.experimental.lift(
        transformed_net.init)(hk.next_rng_key(), h, x)
    def f(_params, _rng, _z, *args): return transformed_net.apply(_params, _rng, _z, *args)

    if kind == 'analytic':
        matinv = jnp.eye(hidden_size) - inner_params['feed_forward_block/linear']['w']
        B, _, H = x.shape
        x0 = jnp.linalg.solve(matinv.T.reshape(1, H, H).repeat(B, 0), x.squeeze(1)).reshape((B, 1, H))
    elif kind == 'forward':
        x0 = deq(inner_params, hk.next_rng_key(), h, f, max_steps, x)
    elif kind == 'zstar':
        x0 = jax.lax.stop_gradient(deq(inner_params, hk.next_rng_key(), h, f, max_steps, x))
    elif kind == 'f':
        x0 = f(inner_params, hk.next_rng_key(), z, x)
    return x0


def postpro_fn(x):
    x = x.squeeze(1)
    x = hk.Linear(10)(x)
    return x


def full_deq_fn(x, kind, hidden_size, max_steps, z=None):
    if kind == 'forward':
        x = prepro_fn(x, hidden_size)
        x = deq_fn(x, hidden_size, max_steps, kind)
        x = postpro_fn(x)
    elif kind == 'analytic':
        x = prepro_fn(x, hidden_size)
        x = deq_fn(x, hidden_size, max_steps, kind)
        x = postpro_fn(x)
    elif kind == 'prepro':
        x = prepro_fn(x, hidden_size)
    elif kind == 'postpro':
        prepro_fn(gen_fake_batch(1), hidden_size) # dummy run needed to get the param names correct...
        x = postpro_fn(x)
    elif kind == 'zstar':
        x = deq_fn(x, hidden_size, max_steps, kind)
    elif kind == 'f':
        x = deq_fn(x, hidden_size, max_steps, kind, z)
    elif kind == 'analytic_zstar':
        x = prepro_fn(x, hidden_size)
        x = deq_fn(x, hidden_size, max_steps, 'analytic')
    return x


def gen_fake_batch(bsz):
    return {
        'image': onp.random.rand(bsz, 1, 28, 28)*255,
        'label': onp.random.randint(10, size=(bsz,)),
    }


def our_gradient(final_loss_f, f_f, zstar, x0, prepro_f, batch, params):
    B,_,H = zstar.shape
    vec0 = grad(final_loss_f)(params, zstar)

    x0zstar = jnp.concatenate((x0, zstar), 1)
    mats = vmap(lambda x: jnp.eye(H) - jacfwd(f_f, 2)(params, x[0].reshape(1,1,H),x[0].reshape(1,1,H)).reshape(H,H).T)(x0zstar)

    mats = jnp.linalg.solve(mats, grad(final_loss_f, 1)(params, zstar).squeeze(1)).reshape(B, 1, H)
    _, vjp_func = vjp(lambda x: f_f(x, x0, zstar), params)
    vec1 = vjp_func(mats)[0]

    _, vjp_func = vjp(lambda x: f_f(params, x, zstar), x0)
    temp = vjp_func(mats)[0]
    _,vjp_func = vjp(lambda x: prepro_f(x, batch), params)
    vec2 = vjp_func(temp)[0]

    return jax.tree_multimap(lambda x,y,z: x+y+z, vec0, vec1, vec2)


def our_hvp(grad_f, f_f, prepro_f, zstar, params, x0, batch, queryv):
    B,_,H = zstar.shape

    vec0 = jvp(lambda x: grad_f(zstar, x), [params], [queryv])[1]

    x0zstar = jnp.concatenate((x0, zstar), 1)
    mats = vmap(lambda x: jnp.eye(H) - jacfwd(f_f, 2)(params, x[0].reshape(1,1,H),x[0].reshape(1,1,H)).reshape(H,H))(x0zstar)

    mats_theta = jnp.linalg.solve(mats, jvp(lambda x: f_f(x, x0, zstar), [params], [queryv])[1].squeeze(1)).reshape(B, 1, H)
    vec1 = jvp(lambda x: grad_f(x, params), [zstar], [mats_theta])[1]

    vecs = jvp(lambda x: prepro_f(x, batch), [params], [queryv])[1]
    mats_x = jnp.linalg.solve(mats, jvp(lambda x: f_f(params, x, zstar), [x0], [vecs])[1].squeeze(1)).reshape(B, 1, H)
    vec2 = jvp(lambda x: grad_f(x, params), [zstar], [mats_x])[1]

    return jax.tree_multimap(lambda x,y,z: x+y+z, vec0, vec1, vec2)


def check_full_deq(rnd_seed=42, bsz=2, hidden_size=3, max_steps=20):
    onp.random.seed(rnd_seed)
    batch = gen_fake_batch(bsz)
    net = hk.transform(lambda x,kind,z=None: full_deq_fn(x, kind, hidden_size, max_steps, z))

    deq_f = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'forward')
    ana_f = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'analytic')
    prepro_f = lambda params,x: net.apply(params, None, x, 'prepro')
    postpro_f = lambda params,x: net.apply(params, None, x, 'postpro')
    zstar_f = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'zstar')
    f_f = lambda params,x,z: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'f', z)
    get_ana_zstar = lambda params,x: net.apply(params, jax.random.PRNGKey(rnd_seed), x, 'analytic_zstar')
    loss_f = lambda x: x.sum()  # TODO make this xent
    final_loss_f = lambda params,z: loss_f(postpro_f(params, z))

    def check_all_close(x,y):
        assert(jnp.allclose(x,y))
    
    # evaluation parameters
    params = net.init(jax.random.PRNGKey(rnd_seed), batch, 'forward')

    # 0th order
    deq_out = deq_f(params, batch)
    ana_out = ana_f(params, batch)
    assert(jnp.allclose(deq_out, ana_out, atol=1e-6))

    # 1st order
    deq_grad = grad(lambda params: loss_f(deq_f(params, batch)))(params)
    ana_grad = grad(lambda params: loss_f(ana_f(params, batch)))(params)
    jax.tree_multimap(check_all_close, deq_grad, ana_grad)

    # our implementation
    x0 = prepro_f(params, batch)
    zstar = zstar_f(params, x0)
    fzstar = f_f(params, x0, zstar)
    assert(jnp.allclose(zstar, fzstar, atol=1e-6))

    # check postpro
    assert(jnp.allclose(postpro_f(params, zstar), deq_out))

    # check our gradient
    implgrad_out = our_gradient(final_loss_f, f_f, zstar, x0, prepro_f, batch, params)
    jax.tree_multimap(check_all_close, deq_grad, implgrad_out)

    # check our hvp
    hvpout = our_hvp(lambda z,p: our_gradient(final_loss_f, f_f, z, prepro_f(p, batch), prepro_f, batch, p),
                     f_f, prepro_f, zstar, params, x0, batch, queryv=params)
    hvpgt1 = jvp(lambda p: grad(lambda params: loss_f(ana_f(params, batch)))(p), [params], [params])[1]
    hvpgt2 = jvp(lambda p: our_gradient(final_loss_f, f_f, get_ana_zstar(p, batch), prepro_f(p, batch), prepro_f, batch, p), [params], [params])[1]
    jax.tree_multimap(check_all_close, hvpout, hvpgt1)
    jax.tree_multimap(check_all_close, hvpgt1, hvpgt2)
    
    print('all checked!')
