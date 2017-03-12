import numbers

import tensorflow as tf

import ops


def l2_regularizer(scale, scope=None):
  """Returns a function that applies L2 regularization to TensorTrain weights.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `l2(tt)` that applies L2 regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      tf.logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def l2(tt):
    """Applies l2 regularization to TensorTrain object."""
    with tf.name_scope(scope, 'l2_regularizer', [tt]) as name:
      my_scale = tf.convert_to_tensor(scale,
                                       dtype=tt.dtype.base_dtype,
                                       name='scale')
      return tf.mul(my_scale, ops.frobenius_norm_squared(tt), name=name)

  return l2


def cores_regularizer(core_regularizer, scale, scope=None):
  """Returns a function that applies given regularization to each TT-core.

  Args:
    core_regularizer: a function with signature `core_regularizer(core)` that
      returns the penalty for the given TT-core.
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.

  Returns:
    A function with signature `regularizer(weights)` that applies
    the regularization.

  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      tf.logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def regularizer(tt):
    """Applies the regularization to TensorTrain object."""
    with tf.name_scope(scope, 'l2_regularizer', [tt]) as name:
      my_scale = tf.convert_to_tensor(scale,
                                       dtype=tt.dtype.base_dtype,
                                       name='scale')
      penalty = 0.0
      for i in range(tt.ndims()):
        penalty += core_regularizer(tt.tt_cores[i])
      return tf.mul(my_scale, penalty, name=name)

  return regularizer
