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

  This operation returns a 1-D integer tensor representing the shape of
  the input. For TT-matrices the shape would have two values, see raw_shape for
  the tensor shape.

  Args:
    tt: `TensorTrain` object.

  Returns:
    A 2D `Tensor` of size ndims() x 1 or x 2
  """
  num_dims = tt.ndims()
  num_tensor_axis = len(tt.get_raw_shape())
  final_raw_shape = []
  for ax in range(num_tensor_axis):
    curr_raw_shape = []
    for core_idx in range(num_dims):
      curr_raw_shape.append(tf.shape(tt.tt_cores[core_idx])[ax + 1])
    # print('s', tf.stack(curr_raw_shape, axis=0).get_shape())
    final_raw_shape.append(tf.stack(curr_raw_shape, axis=0))
  # print('f', tf.stack(final_raw_shape, axis=0).get_shape())
  return tf.stack(final_raw_shape, axis=0)
