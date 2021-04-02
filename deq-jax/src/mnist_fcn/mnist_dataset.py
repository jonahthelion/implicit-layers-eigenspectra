"""MNIST datasets."""

import tensorflow_datasets as tfds

def load(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))
