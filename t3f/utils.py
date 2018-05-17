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
    v_shape = v_.get_shape().as_list()
    v_shape[-2], v_shape[-1] = v_shape[-1], v_shape[-2]
    v = tf.reshape(v, v_shape)
    # Converting numpy order of v dims to TF order.
    order = list(range(tensor.get_shape().ndims))
    order[-2], order[-1] = order[-1], order[-2]
    v = tf.transpose(v, order)
    return s, u, v

  tf.svd = my_svd


def robust_cumprod(arr):
  """Cumulative product with large values replaced by the MAX_DTYPE.
  
  robust_cumprod([10] * 100) = [10, 100, 1000, ..., MAX_INT, ..., MAX_INT] 
  """

  res = np.ones(arr.size, dtype=arr.dtype)
  change_large_to = np.iinfo(arr.dtype).max
  res[0] = arr[0]
  for i in range(1, arr.size):
    next_value = np.array(res[i - 1]) * np.array(arr[i])
    if next_value / np.array(arr[i]) != np.array(res[i - 1]):
      next_value = change_large_to
    res[i] = next_value
  return res


def max_tt_ranks(raw_shape):
  """Maximal TT-ranks for a TT-object of given shape.
  
  For example, a tensor of shape (2, 3, 5, 7) has maximal TT-ranks
    (1, 2, 6, 7, 1)
  making the TT-ranks larger will not increase flexibility.
  
  If maximum TT-ranks result in integer overflows, it substitutes
  the too-large-values with MAX_INT.
  
  Args:
    shape: an integer vector

  Returns:
    tt_ranks: an integer vector, maximal tt-rank for each dimension
  """
  raw_shape = np.array(raw_shape).astype(np.int64)
  d = raw_shape.size
  tt_ranks = np.zeros(d + 1, dtype='int64')
  tt_ranks[0] = 1
  tt_ranks[d] = 1
  left_to_right = robust_cumprod(raw_shape)
  right_to_left = robust_cumprod(raw_shape[::-1])[::-1]
  tt_ranks[1:-1] = np.minimum(left_to_right[:-1], right_to_left[1:])
  return tt_ranks


def in_eager_mode():
  """Checks whether tensorflow eager mode is avaialable and active."""
  try:
      from tensorflow.python.eager import context
      return context.in_eager_mode()
  except ImportError:
      return False
