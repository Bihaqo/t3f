import numpy as np
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


# TODO: get rid of this when TF fixes the NaN bugs in tf.svd:
# https://github.com/tensorflow/tensorflow/issues/8905
def replace_tf_svd_with_np_svd():
  """Replaces tf.svd with np.svd. Slow, but a workaround for tf.svd bugs."""
  if hasattr(tf, 'original_svd'):
    # This function has been already called and tf.svd is already replaced.
    return
  tf.original_svd = tf.svd

  def my_svd(tensor, full_matrices=False, compute_uv=True):
    dtype = tensor.dtype
    u, s, v = tf.py_func(np.linalg.svd, [tensor, full_matrices, compute_uv],
                         [dtype, dtype, dtype])
    s_, u_, v_ = tf.original_svd(tensor, full_matrices, compute_uv)
    s = tf.reshape(s, s_.get_shape())
    u = tf.reshape(u, u_.get_shape())
    v = tf.reshape(v, v_.get_shape())
    # Converting numpy order of v dims to TF order.
    order = range(tensor.get_shape().ndims)
    order[-2], order[-1] = order[-1], order[-2]
    v = tf.transpose(v, order)
    return s, u, v

  tf.svd = my_svd
