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
import itertools
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
import copy
import wandb
import math
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from src.lmdb_transformer import lmdb_dataset
from src.lmdb_transformer import transformer_model
from src.modules.deq import deq, wtie


flags.DEFINE_integer('batch_size', 64, 'Train batch size per core')
flags.DEFINE_integer('sequence_length', 128, 'Sequence length to learn on')

flags.DEFINE_bool('use_deq', True, 'whether use DEQ or corresponding weight-tied networks')
flags.DEFINE_integer('max_iter', 20, 'max iteration for fixed point solving (both forward and backward)')
flags.DEFINE_integer('feedfwd_layers', 8, 'feedforward iterations for weight tied networks')
flags.DEFINE_bool('use_stable_deq', False, 'whether use stable DEQ networks')
flags.DEFINE_integer('num_partitions', 4, 'number of partitions for stable DEQ')

flags.DEFINE_integer('d_model', 128, 'model width')
flags.DEFINE_integer('num_heads', 4, 'Number of attention heads')
flags.DEFINE_integer('num_layers', 1, 'Number of transformer layers')
flags.DEFINE_float('dropout_rate', 0.2, 'Dropout rate')

flags.DEFINE_float('learning_rate', 5e-4, 'Max learning-rate')
flags.DEFINE_float('grad_clip_value', 0.25, 'Gradient norm clip value')

flags.DEFINE_string('checkpoint_dir', './ckpt/haiku-lmdb',
                    'Directory to store checkpoints.')
flags.DEFINE_string('exp_name', 'deq',
                    'Experiment Name.')

FLAGS = flags.FLAGS


LOG_EVERY = 50
EVAL_EVERY = 500
SAVE_EVERY = EVAL_EVERY
MAX_STEPS = 10 * EVAL_EVERY


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float,
                     use_deq: bool, max_iter: int, feedfwd_layers: int,
                     use_stable_deq: bool, num_partitions: int):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        tokens = data['text']
        input_mask = jnp.greater(tokens, 0)
        batch_size, seq_length = tokens.shape

        # Embed the input tokens and positions.
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
        input_embeddings = token_embedding_map(tokens)
        positional_embeddings = hk.get_parameter(
            'pos_embs', [seq_length, d_model], init=embed_init)

        x = input_embeddings + positional_embeddings

        # Create transformer block
        if use_stable_deq:
            d_model_sub = math.ceil(d_model / np.sqrt(num_partitions) / num_heads) * num_heads
            d_model_all = d_model_sub * num_partitions
            x = hk.Linear(d_model_all)(x)
            transformer_block = transformer_model.EqTranformerBlock(
                num_partitions=num_partitions,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate)

        else:
            transformer_block = transformer_model.TranformerBlock(
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate)

        transformed_net = hk.transform(transformer_block)

        h = jnp.zeros_like(x)

        # lift params
        inner_params = hk.experimental.lift(
            transformed_net.init)(hk.next_rng_key(), h, x, input_mask, is_training)

        def f(_params, _rng, _z, *args):
            return transformed_net.apply(_params, _rng, _z, *args, is_training=is_training)

        if use_deq:
            z_star = deq(inner_params, hk.next_rng_key(), h, f, max_iter, True, x, input_mask)
        else:
            z_star = wtie(inner_params, hk.next_rng_key(), h, f, feedfwd_layers, x, input_mask)

        # Reverse the embeddings (untied).
        # return hk.Linear(1)(z_star.mean(axis=1)).squeeze(1)
        return hk.Linear(1)(z_star[:, -1]).squeeze(1)

    return forward_fn


def bcl_loss_accuracy_fn(forward_fn,
                         params,
                         rng,
                         data: Mapping[str, jnp.ndarray],
                         is_training: bool = True):
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = data['label']
    assert logits.shape == targets.shape, (logits.shape, targets.shape)

    predictions = jax.nn.sigmoid(logits) > 0.5
    loss = -(targets * jax.nn.log_sigmoid(logits) + (1 - targets) * jax.nn.log_sigmoid(-logits)).mean()
    accuracy = (predictions == targets).mean()

    return loss, accuracy


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
    # init logger
    wandb.init(entity='ryoungj', project="csc-2541-deq",
               name=FLAGS.exp_name, config=FLAGS, resume=True, id="____"+FLAGS.exp_name,
               dir=FLAGS.checkpoint_dir)

    # Create the dataset.
    train_dataset, test_dataset, vocab_size = lmdb_dataset.load(FLAGS.batch_size,
                                                                FLAGS.sequence_length)

    test_dataset = list(test_dataset)
    # Set up the model, loss, and updater.
    forward_fn = build_forward_fn(vocab_size, FLAGS.d_model, FLAGS.num_heads,
                                  FLAGS.num_layers, FLAGS.dropout_rate, FLAGS.use_deq, FLAGS.max_iter,
                                  FLAGS.feedfwd_layers, FLAGS.use_stable_deq, FLAGS.num_partitions)

    forward_fn = hk.transform(forward_fn)
    loss_fn = lambda params, rng, data: bcl_loss_accuracy_fn(forward_fn.apply,
                                                             params, rng, data, is_training=True)[0]

    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.grad_clip_value),
        optax.adam(FLAGS.learning_rate, b1=0.9, b2=0.99))

    updater = Updater(forward_fn.init, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name), checkpoint_every_n=SAVE_EVERY)

    # Initialize parameters.
    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)
    data = next(train_dataset)
    state = updater.init(rng, data)

    logging.info('# of params: {}'.format(sum([p.size for p in jax.tree_leaves(state['params'])])))

    best_test_loss = np.inf
    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(MAX_STEPS):
        data = next(train_dataset)
        state, metrics = updater.update(state, data)
        # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
        # Using values from state/metrics too often will block the runahead and can
        # cause these overheads to become more prominent.
        wandb.log({"train/loss": jax.device_get(metrics['loss'])}, step=step)
        if step % LOG_EVERY == 0:
            steps_per_sec = LOG_EVERY / (time.time() - prev_time)
            metrics.update({'steps_per_sec': steps_per_sec})
            logging.info({k: float(v) for k, v in metrics.items()})

            if step % EVAL_EVERY == 0:
                test_loss = 0.0
                test_accuracy = 0.0

                batch_num = 0
                for test_data in test_dataset:
                    loss, accuracy = bcl_loss_accuracy_fn(forward_fn.apply,
                                                          state['params'], rng, test_data, is_training=False)
                    test_loss += loss
                    test_accuracy += accuracy
                    batch_num += 1

                test_loss, test_accuracy = jax.device_get(
                    (test_loss / batch_num, test_accuracy / batch_num))
                logging.info(f"[Step {step}] Test loss/accuracy: "
                             f"{test_loss:.3f} / {test_accuracy:.3f}.")
                wandb.log({"test/loss": test_loss, "test/acc": test_accuracy}, step=step)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    path = os.path.join(os.path.join(FLAGS.checkpoint_dir, FLAGS.exp_name),
                                        'checkpoint_best.pkl'.format(step))
                    checkpoint_state = jax.device_get(state)
                    logging.info('Serializing experiment state to %s', path)
                    with open(path, 'wb') as f:
                        pickle.dump(checkpoint_state, f)


            prev_time = time.time()

    print("Training finished! Best loss: {:.3f}".format(best_test_loss))


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)
    tf.enable_v2_behavior()
    app.run(main)
