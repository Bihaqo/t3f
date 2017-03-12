import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain


def random_tensor(shape, tt_rank=2):
  """Generate a random TT-tensor of given shape.

  Args:
    shape: array representing the shape of the future tensor
    tt_rank: a number or a (d+1)-element array with ranks.

  Returns:
    TensorTrain containing a TT-tensor
  """
  # TODO: good distribution to init training.
  # TODO: support shape and tt_ranks as TensorShape?.
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  if len(shape.shape) != 1:
    raise ValueError('shape should be 1d array')
  if np.any(shape < 1):
    raise ValueError('all elements in `shape` should be positive')
  if np.any(tt_rank < 1):
    raise ValueError('`rank` should be positive')
  if tt_rank.size != 1 and tt_rank.size != (shape.size + 1):
    raise ValueError('`rank` array has inappropriate size')

  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)
  # TODO: check that ints?
  shape = shape.astype(int)
  tt_rank_ext = tt_rank.astype(int)
  # TODO: variable (name?) scope.
  tt_cores = [None] * num_dims
  for i in range(num_dims):
    curr_core_shape = (tt_rank_ext[i], shape[i], tt_rank_ext[i + 1])
    tt_cores[i] = tf.random_normal(curr_core_shape)

  return TensorTrain(tt_cores, shape, tt_rank_ext)


def random_matrix(shape, tt_rank=2):
  """Generate a random TT-matrix of given shape.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports ommiting one of the dimensions for vectors, e.g.
        tt_rand_matrix([[2, 2, 2], None])
      and
        tt_rand_matrix([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]))
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]))
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  if len(shape.shape) != 2:
    raise ValueError('shape should be 2d array')
  if shape[0].size != shape[1].size:
    raise ValueError('shape[0] should have the same length as shape[1]')
  if np.any(shape.flatten() < 1):
    raise ValueError('all elements in `shape` should be positive')
  if np.any(tt_rank < 1):
    raise ValueError('`rank` should be positive')
  if tt_rank.size != 1 and tt_rank.size != (shape[0].size + 1):
    raise ValueError('`rank` array has inappropriate size')

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1)
    tt_rank = np.concatenate([[1], tt_rank, [1]])
  # TODO: check that ints?
  shape = shape.astype(int)
  tt_rank = tt_rank.astype(int)
  # TODO: variable (name?) scope.
  tt_cores = [None] * num_dims
  for i in range(num_dims):
    curr_core_shape = (tt_rank[i], shape[0][i], shape[1][i],
                       tt_rank[i + 1])
    tt_cores[i] = tf.random_normal(curr_core_shape)

  return TensorTrain(tt_cores, shape, tt_rank)
