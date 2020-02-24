import numpy as np
import tensorflow as tf


# TODO: substitute with native implementation when it's ready.
# https://github.com/tensorflow/tensorflow/issues/2075
def unravel_index(indices, shape):
  with tf.name_scope('unravel_index'):
    indices = tf.expand_dims(indices, 0)
    shape = tf.expand_dims(shape, 1)
    strides_shifted = tf.math.cumprod(shape, exclusive=True, reverse=True)
    res = (indices // strides_shifted) % shape
    return tf.transpose(res, (1, 0))


# TODO: get rid of this when TF fixes the NaN bugs in tf.svd:
# https://github.com/tensorflow/tensorflow/issues/8905
def replace_tf_svd_with_np_svd():
  """Replaces tf.svd with np.svd. Slow, but a workaround for tf.svd bugs."""
  if hasattr(tf, 'original_svd'):
    # This function has been already called and tf.svd is already replaced.
    return
  tf.original_svd = tf.linalg.svd

  def my_svd(tensor, full_matrices=False, compute_uv=True):
    dtype = tensor.dtype
    u, s, v = np.linalg.svd(tensor, full_matrices, compute_uv)
    # Converting numpy order of v dims to TF order.
    order = list(range(tensor.get_shape().ndims))
    order[-2], order[-1] = order[-1], order[-2]
    v = tf.transpose(v, order)
    return tf.constant(s), tf.constant(u), v

  tf.linalg.svd = my_svd


def in_eager_mode():
  """Checks whether tensorflow eager mode is avaialable and active."""
  try:
      from tensorflow.python.eager import context
      return context.in_eager_mode()
  except ImportError:
      return False
