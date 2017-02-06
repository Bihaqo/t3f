import tensorflow as tf


# TODO: substitute with native implementation when it's ready.
# https://github.com/tensorflow/tensorflow/issues/2075
def unravel_index(indices, shape):
  with tf.name_scope('unravel_index'):
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    strides_shifted = tf.cumprod(shape, exclusive=True, reverse=True)
    res = (indices // strides_shifted) % shape
    return tf.transpose(res, (1, 0))
