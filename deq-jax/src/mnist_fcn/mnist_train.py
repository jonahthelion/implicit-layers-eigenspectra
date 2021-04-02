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
import optax
import tensorflow.compat.v2 as tf
from absl import app
from absl import flags
from absl import logging

import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from src.mnist_fcn import mnist_dataset
from src.mnist_fcn import fcn_model
from src.modules.deq import deq, wtie

flags.DEFINE_bool('use_deq', True, 'whether use DEQ or corresponding weight-tied networks')
flags.DEFINE_integer('max_iter', 15, 'max iteration for fixed point solving (both forward and backward)')
flags.DEFINE_integer('feedfwd_layers', 12, 'feedforward iterations for weight tied networks')
flags.DEFINE_integer('d_model', 128, 'model width')

flags.DEFINE_integer('batch_size', 100, 'Train batch size per core')
flags.DEFINE_float('learning_rate', 1e-3, 'Max learning-rate')
flags.DEFINE_float('grad_clip_value', 0.25, 'Gradient norm clip value')

flags.DEFINE_string('checkpoint_dir', '/tmp/haiku-mnist',
                    'Directory to store checkpoints.')

FLAGS = flags.FLAGS
LOG_EVERY = 50
EVAL_EVERY = 500
MAX_STEPS = 10 ** 5


def build_forward_fn(d_model: int, use_deq: bool, max_iter: int, feedfwd_layers: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        """Forward pass."""
        x = data["image"].astype(jnp.float32) / 255.
        x = hk.Flatten()(x)
        x = jnp.expand_dims(x, axis=1)  # add dummy sequence length dimension
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        # injected input
        x = hk.Linear(d_model, with_bias=True, w_init=initializer)(x)
        h = jnp.zeros_like(x)

        # create fcn block
        fcn_block = fcn_model.FeedForwardBlock()

        transformed_net = hk.transform(fcn_block)

        # lift params
        inner_params = hk.experimental.lift(
            transformed_net.init)(hk.next_rng_key(), h, x)

        def f(_params, _rng, _z, *args): return transformed_net.apply(_params, _rng, _z, *args)

        if use_deq:
            z_star = deq(inner_params, hk.next_rng_key(), h, f, max_iter, x)
        else:
            z_star = wtie(inner_params, hk.next_rng_key(), h, f, feedfwd_layers, x)

        return hk.Linear(10)(z_star.squeeze(1))

    return forward_fn


# Training loss (cross-entropy).
def ce_loss_fn(forward_fn,
               params,
               rng,
               data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data)
    targets = jax.nn.one_hot(data['label'], 10)
    assert logits.shape == targets.shape

    loss = -jnp.sum(targets * jax.nn.log_softmax(logits))
    loss /= targets.shape[0]

    return loss


# Evaluation metric (classification accuracy).
@functools.partial(jax.jit, static_argnums=0)
def accuracy(forward_fn, params, rng, data: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    predictions = forward_fn(params, rng, data)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == data["label"])


class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.
  This extracts some common boilerplate from the training loop.
  """

    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


class CheckpointingUpdater:
    """A didactic checkpointing wrapper around an Updater.
  A more mature checkpointing implementation might:
    - Use np.savez() to store the core data instead of pickle.
    - Not block JAX async dispatch.
    - Automatically garbage collect old checkpoints.
  """

    def __init__(self,
                 inner: Updater,
                 checkpoint_dir: str,
                 checkpoint_every_n: int = 10000):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(self._checkpoint_dir,
                                      self._checkpoint_paths()[-1])
            logging.info('Loading checkpoint from %s', checkpoint)
            with open(checkpoint, 'rb') as f:
                state = pickle.load(f)
            return state

    def update(self, state, data):
        """Update experiment state."""
        # NOTE: This blocks until `state` is computed. If you want to use JAX async
        # dispatch, maintain state['step'] as a NumPy scalar instead of a JAX array.
        # Context: https://jax.readthedocs.io/en/latest/async_dispatch.html
        step = np.array(state['step'])
        if step % self._checkpoint_every_n == 0:
            path = os.path.join(self._checkpoint_dir,
                                'checkpoint_{:07d}.pkl'.format(step))
            checkpoint_state = jax.device_get(state)
            logging.info('Serializing experiment state to %s', path)
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_state, f)

        state, out = self._inner.update(state, data)
        return state, out


def main(_):
    # Create the dataset.
    train_dataset = mnist_dataset.load("train", is_training=True, batch_size=FLAGS.batch_size)
    train_dataset_eval = mnist_dataset.load("train", is_training=False, batch_size=10000)
    test_dataset = mnist_dataset.load("test", is_training=False, batch_size=10000)

    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn(FLAGS.d_model, FLAGS.use_deq, FLAGS.max_iter, FLAGS.feedfwd_layers)
    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(ce_loss_fn, forward_fn.apply)

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip_value),
        optax.adam(FLAGS.learning_rate, b1=0.9, b2=0.99))

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)

    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    logging.info('# of params: {}'.format(sum([p.size for p in jax.tree_leaves(state['params'])])))

    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)
        # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
        # Using values from state/metrics too often will block the runahead and can
        # cause these overheads to become more prominent.
        if step % LOG_EVERY == 0:
            steps_per_sec = LOG_EVERY / (time.time() - prev_time)
            metrics.update({'steps_per_sec': steps_per_sec})
            logging.info({k: float(v) for k, v in metrics.items()})

            if step % EVAL_EVERY == 0:
                train_accuracy = accuracy(forward_fn.apply, state['params'], state['rng'], next(train_dataset_eval))
                test_accuracy = accuracy(forward_fn.apply, state['params'], state['rng'], next(test_dataset))
                train_accuracy, test_accuracy = jax.device_get(
                    (train_accuracy, test_accuracy))
                logging.info(f"[Step {step}] Train / Test accuracy: "
                             f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

            prev_time = time.time()


if __name__ == '__main__':
    tf.enable_v2_behavior()
    app.run(main)
