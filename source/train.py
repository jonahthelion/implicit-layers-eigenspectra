import sys
import jax
import jax.numpy as jnp
import jax.random as jaxrnd
import numpy as onp
import haiku as hk
from jax import value_and_grad
from typing import Generator, Mapping, Tuple
import optax
import tensorflow_datasets as tfds
import pickle
from glob import glob
import os
from jax.flatten_util import ravel_pytree
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmx

# hacky but fine for now
sys.path.append('./source/deq-jax')
from src.modules.deq import deq

# hacky but fine for now
sys.path.append('./source/spectral-density/jax')
import density as density_lib
import lanczos
import hessian_computation


def net_fn(batch) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    x = batch["image"].astype(jnp.float32) / 255.
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(30), jax.nn.relu,
        hk.Linear(10), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)


def load_dataset(
    split: str,
    is_training: bool,
    batch_size: int,
    ):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


# Training loss (cross-entropy).
def loss_function(params, batch, net) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    labels = jax.nn.one_hot(batch["label"], 10)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent #+ 1e-4 * l2_loss


def train_mlp():
    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(1e-3)

    loss = lambda params,batch: loss_function(params, batch, net)

    # Evaluation metric (classification accuracy).
    @jax.jit
    def accuracy(params, batch):
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    @jax.jit
    def update(
        params,
        opt_state,
        batch,
        ):
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=1000)
    train_eval = load_dataset("train", is_training=False, batch_size=60000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    # Initialize network and optimiser; note we draw an input to get shapes.
    params = avg_params = net.init(jax.random.PRNGKey(42), next(train))
    opt_state = opt.init(params)

    totalsteps = 10001
    save_ts = set(list((onp.power(onp.linspace(0.0, 1.0, 10), 3.0) * (totalsteps-1)).astype(onp.int)))
    print(save_ts)

    # Train/eval loop.
    for step in range(totalsteps):
        if step in save_ts:
            # Periodically evaluate classification accuracy on train & test sets.
            train_accuracy = accuracy(avg_params, next(train_eval))
            test_accuracy = accuracy(avg_params, next(test_eval))
            print(f"[Step {step}] Train / Test accuracy: {train_accuracy:.3f} / {test_accuracy:.3f}")

            # save model weights
            mname = f'storage/mlps/mlp{step:07}.pkl'
            print('saving', mname)
            pickle.dump(avg_params, open(mname, "wb" ))

        # Do SGD on a batch of training examples.
        params, opt_state = update(params, opt_state, next(train))
        avg_params = ema_update(params, avg_params)


def eval_model(mpath):
    """Check that we can re-load a model and get the same
    training and test accuracy
    """
    net = hk.without_apply_rng(hk.transform(net_fn))

    # Evaluation metric (classification accuracy).
    @jax.jit
    def accuracy(params, batch):
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    # Make datasets.
    train_eval = load_dataset("train", is_training=False, batch_size=60000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    # Initialize network and optimiser; note we draw an input to get shapes.
    avg_params = pickle.load(open(mpath, "rb" ))

    # Periodically evaluate classification accuracy on train & test sets.
    train_accuracy = accuracy(avg_params, next(train_eval))
    test_accuracy = accuracy(avg_params, next(test_eval))
    train_accuracy, test_accuracy = jax.device_get(
        (train_accuracy, test_accuracy))
    print(f"Train / Test accuracy: "
        f"{train_accuracy:.3f} / {test_accuracy:.3f}.")


def eval_mlp_spectrum(mfolder):
    net = hk.without_apply_rng(hk.transform(net_fn))

    train_eval = load_dataset("train", is_training=False, batch_size=60000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    loss = lambda params,batch: loss_function(params, batch, net)

    batch = next(train_eval)

    def batches_fn():
        yield batch

    fs = glob(os.path.join(mfolder, 'mlp*.pkl'))
    for f in fs:
        base = f.split('/')[-1].replace('mlp', '').replace('.pkl', '')
        outname = os.path.join(mfolder, f'spec{base}.pkl')
        avg_params = pickle.load(open(f, "rb" ))

        hvp, unravel, num_params = hessian_computation.get_hvp_fn(loss, avg_params, batches_fn)
        hvp_cl = lambda v: hvp(avg_params, v)
        print('hvp')
        tridiag, vecs = lanczos.lanczos_alg(hvp_cl, num_params, order=90, rng_key=jaxrnd.PRNGKey(0))
        print('discretizing')
        density, grids = density_lib.tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

        print('saving', outname)
        pickle.dump({'density': density, 'grids': grids}, open(outname, "wb" ))


def plot_mlp_spectrum(mfolder, imname='./mlpmnist.png'):
    fs = sorted(glob(os.path.join(mfolder, 'spec*.pkl')))

    f2data = {f: pickle.load(open(f, "rb" )) for f in fs}
    xlim = (min((min(row['grids']) for row in f2data.values())),
            max((max(row['grids']) for row in f2data.values()))
            )
    xlim = (xlim[0] - (xlim[1] - xlim[0])*0.02, xlim[1] + (xlim[1] - xlim[0])*0.02)
    ylim = (1e-6, 1e2)
    maxsteps = max([int(f.split('/')[-1].replace('spec', '').replace('.pkl', '')) for f in fs])

    # plot colors
    cNorm = mcolors.Normalize(vmin=-0.1, vmax=1.1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('winter'))

    fig = plt.figure(figsize=(20, 7))
    gs = mpl.gridspec.GridSpec(len(fs), 1, left=0.05, right=0.95, top=0.95, bottom=0.05)
    for fi,f in enumerate(reversed(fs)):
        numsteps = int(f.split('/')[-1].replace('spec', '').replace('.pkl', ''))
        data = f2data[f]

        ax = plt.subplot(gs[fi, 0])
        plt.fill_between(data['grids'], data['density'], color=scalarMap.to_rgba((numsteps/maxsteps)**(1/6)))
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.yscale('log')
        rect = ax.patch
        rect.set_alpha(0)
        ax.set_yticklabels([])
        ax.get_yaxis().set_ticks([])
        if fi != len(fs) - 1:
            ax.set_xticklabels([])
            ax.get_xaxis().set_ticks([])
        else:
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='x', which='minor', labelsize=14)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        plt.text(xlim[0] - (xlim[1] - xlim[0])*0.035, ylim[0] * onp.power(ylim[1]/ylim[0], 0.1), f'{numsteps}\nsteps',
                 rotation='horizontal', fontsize=14)
        if fi == 0:
            plt.title('MLP Hessian Eigenspectra (MNIST)', fontsize=24)
    gs.update(hspace=-0.5)

    # plt.tight_layout()
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)
