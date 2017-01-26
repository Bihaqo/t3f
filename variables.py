import tensorflow as tf


def get_tt_variable(name,
                    shape=None,
                    dtype=None,
                    initializer=None,
                    regularizer=None,
                    trainable=True,
                    collections=None,
                    caching_device=None,
                    validate_shape=True):
  # Returns TensorTrain object with tf.Variables as the TT-cores.
  raise NotImplementedError
