import numpy as np
import tensorflow as tf


# TODO: test all these functions.
def tt_ranks(tt):
  """Returns the TT-ranks of a TensorTrain.

  This operation returns a 1-D integer tensor representing the TT-ranks of
  the input.

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object.

  Returns:
    A `Tensor`
  """
  num_dims = tt.ndims()
  ranks = []
  # TODO: ugly.
  from tensor_train_batch import TensorTrainBatch
  left_rank_dim = 1 if isinstance(tt, TensorTrainBatch) else 0
  for i in range(num_dims):
    ranks.append(tf.shape(tt.tt_cores[i])[left_rank_dim])
  ranks.append(tf.shape(tt.tt_cores[-1])[-1])
  return tf.stack(ranks, axis=0)


def shape(tt):
  """Returns the shape of a TensorTrain.

  This operation returns a 1-D integer tensor representing the shape of
  the input. For TT-matrices the shape would have two values, see raw_shape for
  the tensor shape.

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object.

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
    tt: `TensorTrain` or `TensorTrainBatch` object.

  Returns:
    A 2-D `Tensor` of size 1 x ndims() or 2 x ndims()
  """
  num_dims = tt.ndims()
  num_tensor_axis = len(tt.get_raw_shape())
  final_raw_shape = []
  # TODO: ugly.
  from tensor_train import TensorTrain
  axes_shift = 1 if isinstance(tt, TensorTrain) else 2
  for ax in range(num_tensor_axis):
    curr_raw_shape = []
    for core_idx in range(num_dims):
      curr_raw_shape.append(tf.shape(tt.tt_cores[core_idx])[ax + axes_shift])
    final_raw_shape.append(tf.stack(curr_raw_shape, axis=0))
  return tf.stack(final_raw_shape, axis=0)


def batch_size(tt):
  """Return the number of elements in a TensorTrainBatch.

  Return 0-D integer tensor.

  Raises:
    ValueError if got `TensorTrain` which doesn't have batch_size as input."""
  if not hasattr(tt, 'batch_size'):
    raise ValueError('batch size is not available for a TensorTrain object.')
  first_core = tt.tt_cores[0]
  # The first dimension of any TT-core in TensorTrainBatch is the batch size.
  return tf.shape(first_core)[0]


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


def lazy_batch_size(tt):
  """Return static batch_size if available and dynamic otherwise.

  Args:
    tt: `TensorTrainBatch` object.

  Returns:
    A number or a 0-D `tf.Tensor`

  Raises:
    ValueError if got `TensorTrain` which doesn't have batch_size as input."""
  if not hasattr(tt, 'batch_size'):
    raise ValueError('batch size is not available for a TensorTrain object.')
  if tt.batch_size is not None:
    return tt.batch_size
  else:
    return batch_size(tt)


def clean_raw_shape(shape):
  """Returns a tuple of TensorShapes for any valid shape representation.

  Args:
    shape: An np.array, a tf.TensorShape (for tensors), a tuple of
      tf.TensorShapes (for TT-matrices or tensors), or None

  Returns:
    A tuple of tf.TensorShape, or None if the input is None
  """
  if shape is None:
    return None
  if isinstance(shape, tf.TensorShape) or isinstance(shape[0], tf.TensorShape):
    # Assume tf.TensorShape.
    if isinstance(shape, tf.TensorShape):
      shape = tuple((shape,))
  else:
    np_shape = np.array(shape)
    # Make sure that the shape is 2-d array both for tensors and TT-matrices.
    np_shape = np.squeeze(np_shape)
    if len(np_shape.shape) == 1:
      # A tensor.
      np_shape = [np_shape]
    shape = []
    for i in range(len(np_shape)):
      shape.append(tf.TensorShape(np_shape[i]))
    shape = tuple(shape)
  return shape


def is_batch_broadcasting_possible(tt_a, tt_b):
  """Check that the batch broadcasting possible for the given batch sizes.

  Returns true if the batch sizes are the same or if one of them is 1.

  Args:
    tt_a: TensorTrain or TensorTrainBatch
    tt_b: TensorTrain or TensorTrainBatch

  Returns:
    Bool
  """
  try:
    if tt_a.batch_size is None or tt_b.batch_size is None:
      # If one of the batch sizes is not available on the compilation stage,
      # we cannot say if broadcasting is possible.
      return True
    if tt_a.batch_size == tt_b.batch_size:
      return True
    if tt_a.batch_size == 1 or tt_b.batch_size == 1:
      return True
    return False
  except AttributeError:
    # One or both of the arguments are not batch tensor, but single TT tensors.
    # In this case broadcasting is always possible.
    return True


def squeeze_batch_dim(tt):
  """Converts batch size 1 TensorTrainBatch into TensorTrain.

  Args:
    tt: TensorTrain or TensorTrainBatch.

  Returns:
    TensorTrain if the input is a TensorTrainBatch with batch_size == 1 (known
      at compilation stage) or a TensorTrain.
    TensorTrainBatch otherwise.
    """
  try:
    if tt.batch_size == 1:
      return tt[0]
    else:
      return tt
  except AttributeError:
    # tt object does not have attribute batch_size, probably already
    # a TensorTrain.
    return tt


def expand_batch_dim(tt):
  """Creates a 1-element TensorTrainBatch from a TensorTrain.

  Args:
    tt: TensorTrain or TensorTrainBatch.

  Returns:
    TensorTrainBatch
  """
  if hasattr(tt, 'batch_size'):
    return tt
  else:
    from tensor_train_batch import TensorTrainBatch
    tt_cores = []
    for core_idx in range(tt.ndims()):
      tt_cores.append(tf.expand_dims(tt.tt_cores[core_idx], 0))
    return TensorTrainBatch(tt_cores, tt.get_raw_shape(), tt.get_tt_ranks(),
                            batch_size=1)