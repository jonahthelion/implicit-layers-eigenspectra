"""LMDB datasets."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def load(batch_size: int, sequence_length: int):
    """Load LM1B dataset, returning it and vocab_size."""
    imdb, ds_info = tfds.load(
        'imdb_reviews/subwords8k',
        shuffle_files=True,
        as_supervised=True,
        with_info=True)

    train_data, test_data = imdb['train'], imdb['test']
    # load_data = lambda split: tfds.load(
    #     'imdb_reviews/subwords8k',
    #     as_supervised=True,
    #     with_info=True)
    # train_data, ds_info = load_data('train[:80%]')
    # valid_data, ds_info = load_data('train[-80%:]')
    # test_data, ds_info = load_data('test')

    vocab_size = ds_info.features['text'].encoder.vocab_size
    spec_token = vocab_size + 1

    def preprocess(ds, bs, train=True):
        if train:
            ds = ds.repeat()
        # Convert the dataset to constant-size int32 tensors.
        ds = ds.map(lambda text, label: (tf.cast(text, tf.int32), label))
        crop_size = sequence_length - 1
        ds = ds.map(lambda text, label: (_crop_or_pad(text, crop_size, pad_token=0), label))
        ds = ds.map(lambda text, label: (_crop_or_pad(text, sequence_length, pad_token=spec_token), label))
        ds = ds.map(lambda text, label: dict(text=text, label=label))
        ds = ds.shuffle(bs * 10)
        ds = ds.batch(bs, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = iter(tfds.as_numpy(ds))

        return ds

    train_data = preprocess(train_data, batch_size)
    test_data = preprocess(test_data, batch_size, train=False)

    return train_data, test_data, vocab_size + 1


def _crop_or_pad(value, size, pad_token):
    """Either crop or pad value to be of size size."""
    val_size = tf.size(value)
    pad = lambda: tf.pad(  # pylint: disable=g-long-lambda
        value, [[0, size - val_size]],
        'CONSTANT',
        constant_values=pad_token)
    return tf.cond(val_size < size, pad, lambda: value[:size])
