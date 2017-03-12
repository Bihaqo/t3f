import numpy as np
import tensorflow as tf


# TODO: test all these functions.
def tt_ranks(tt):
  """Returns the TT-ranks of a TensorTrain.

  This operation returns a 1-D integer tensor representing the TT-ranks of
  the input.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A `Tensor`
  """
  num_dims = tt.ndims()
  ranks = []
  for i in range(num_dims):
    ranks.append(tf.shape(tt.tt_cores[i])[0])
  ranks.append(tf.shape(tt.tt_cores[-1])[-1])
  return tf.stack(ranks, axis=0)


def shape(tt):
  """Returns the shape of a TensorTrain.

  This operation returns a 1-D integer tensor representing the shape of
  the input. For TT-matrices the shape would have two values, see raw_shape for
  the tensor shape.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A `Tensor`
  """
  tt_raw_shape = raw_shape(tt)
  if tt.is_tt_matrix():
    return tf.reduce_prod(raw_shape, axis=1)
  else:
    return tt_raw_shape[0]


def raw_shape(tt):
  """Returns the shape of a TensorTrain.

  This operation returns a 2-D integer tensor representing the shape of
  the input.
  If the input is a TT-tensor, the shape will have 1 x ndims() elements.
  If the input is a TT-matrix, the shape will have 2 x ndims() elements
  representing the underlying tensor shape of the matrix.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A 2-D `Tensor` of size 1 x ndims() or 2 x ndims()
  """
  num_dims = tt.ndims()
  num_tensor_axis = len(tt.get_raw_shape())
  final_raw_shape = []
  for ax in range(num_tensor_axis):
    curr_raw_shape = []
    for core_idx in range(num_dims):
      curr_raw_shape.append(tf.shape(tt.tt_cores[core_idx])[ax + 1])
    final_raw_shape.append(tf.stack(curr_raw_shape, axis=0))
  return tf.stack(final_raw_shape, axis=0)


def lazy_tt_ranks(tt):
  """Returns static TT-ranks of a TensorTrain if defined, and dynamic otherwise.

  This operation returns a 1-D integer numpy array of TT-ranks if they are
  available on the graph compilation stage and 1-D integer tensor of dynamic
  TT-ranks otherwise.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A 1-D numpy array or `tf.Tensor`
  """
  static_tt_ranks = tt.get_tt_ranks()
  if static_tt_ranks.is_fully_defined():
    return np.array(static_tt_ranks.as_list())
  else:
    return tt_ranks(tt)


def lazy_shape(tt):
  """Returns static shape of a TensorTrain if defined, and dynamic otherwise.

  This operation returns a 1-D integer numpy array representing the shape of the
  input if it is available on the graph compilation stage and 1-D integer tensor
  of dynamic shape otherwise.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A 1-D numpy array or `tf.Tensor`
  """
  static_shape = tt.get_shape()
  if static_shape.is_fully_defined():
    return np.array(static_shape.as_list())
  else:
    return shape(tt)


def lazy_raw_shape(tt):
  """Returns static raw shape of a TensorTrain if defined, and dynamic otherwise.

  This operation returns a 2-D integer numpy array representing the raw shape of
  the input if it is available on the graph compilation stage and 2-D integer
  tensor of dynamic shape otherwise.
  If the input is a TT-tensor, the raw shape will have 1 x ndims() elements.
  If the input is a TT-matrix, the raw shape will have 2 x ndims() elements
  representing the underlying tensor shape of the matrix.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A 2-D numpy array or `tf.Tensor` of size 1 x ndims() or 2 x ndims()
  """
  # If get_shape is fully defined, it guaranties that all elements of raw shape
  # are defined.
  if tt.get_shape().is_fully_defined():
    return np.array([s.as_list() for s in tt.get_raw_shape()])
  else:
    return raw_shape(tt)
