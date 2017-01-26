import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain


def tt_rand_tensor(shape, rank=2):
  """Generate a random TT-tensor of given shape.

  Args:
    shape: array representing the shape of the future tensor
    rank: a number or a (d-1)-element array with ranks.

  Returns:
    TensorTrain containing a TT-tensor
  """
  # TODO: good distribution to init training.
  shape = np.array(shape)
  rank = np.array(rank)
  if len(shape.shape) != 1:
    raise ValueError('shape should be 1d array')
  if np.any(shape < 1):
    raise ValueError('all elements in `shape` should be positive')
  if np.any(rank < 1):
    raise ValueError('`rank` should be positive')
  if rank.size != 1 and rank.size != (shape.size - 1):
    raise ValueError('`rank` array has inappropriate size')

  num_dims = shape.size
  if rank.size == 1:
    rank = rank * np.ones(num_dims - 1)
  # Add 1 to the beggining and end to simplify working with ranks.
  rank_ext = np.insert(rank, 0, 1)
  rank_ext = np.append(rank_ext, 1)
  # TODO: check that ints?
  shape = shape.astype(int)
  rank_ext = rank_ext.astype(int)

  tt_cores = [None] * num_dims
  for i in range(num_dims):
    curr_core_shape = (rank_ext[i], shape[i], rank_ext[i + 1])
    tt_cores[i] = tf.random_normal(curr_core_shape)

  return TensorTrain(tt_cores)


def tt_rand_matrix(shape, rank=2):
  """Generate a random TT-matrix of given shape.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
    rank: a number or a (d-1)-element array with ranks.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  shape = np.array(shape)
  rank = np.array(rank)
  if len(shape.shape) != 2:
    raise ValueError('shape should be 2d array')
  if shape[0].size != shape[1].size:
    raise ValueError('shape[0] should have the same length as shape[1]')
  if np.any(shape.flatten() < 1):
    raise ValueError('all elements in `shape` should be positive')
  if np.any(rank < 1):
    raise ValueError('`rank` should be positive')
  if rank.size != 1 and rank.size != (shape[0].size - 1):
    raise ValueError('`rank` array has inappropriate size')

  num_dims = shape[0].size
  if rank.size == 1:
    rank = rank * np.ones(num_dims - 1)
  # Add 1 to the beggining and end to simplify working with ranks.
  rank_ext = np.insert(rank, 0, 1)
  rank_ext = np.append(rank_ext, 1)
  # TODO: check that ints?
  shape = shape.astype(int)
  rank_ext = rank_ext.astype(int)

  tt_cores = [None] * num_dims
  for i in range(num_dims):
    curr_core_shape = (rank_ext[i], shape[0][i], shape[1][i], rank_ext[i + 1])
    tt_cores[i] = tf.random_normal(curr_core_shape)

  return TensorTrain(tt_cores)
