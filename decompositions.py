import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
import shapes


def to_tt_matrix(mat, shape, max_tt_rank=10, epsilon=None):
  """Converts a given matrix or vector to a TT-matrix.

  The matrix dimensions should factorize into d numbers.
  If e.g. the dimensions are prime numbers, it's usually better to
  pad the matrix with zeros until the dimensions factorize into
  (ideally) 3-8 numbers.

  Args:
    mat: two dimensional tf.Tensor (a matrix).
    shape: two dimensional array (np.array or list of lists)
      Represents the tensor shape of the matrix.
      E.g. for a (a1 * a2 * a3) x (b1 * b2 * b3) matrix `shape` should be
      ((a1, a2, a3), (b1, b2, b3))
      `shape[0]`` and `shape[1]`` should have the same length.
      For vectors you may use ((a1, a2, a3), (1, 1, 1)) or, equivalently,
      ((a1, a2, a3), None)
    max_tt_rank: a number or a list of numbers
      If a number, than defines the maximal TT-rank of the result.
      If a list of numbers, than `max_tt_rank` length should be d+1
      (where d is the length of `shape[0]`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th TT-rank of the result.
      The following two versions are equivalent
        `max_tt_rank = r`
      and
        `max_tt_rank = r * np.ones(d-1)`
    eps: a floating point number
      If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `eps` close to `mat`
      in terms of relative Frobenius error:
        ||res - mat||_F / ||mat||_F <= eps
      If the TT-ranks are restricted, providing a loose `eps` may reduce
      the TT-ranks of the result.
      E.g.
        to_tt_matrix(mat, shape, max_tt_rank=100, eps=1)
      will probably return you a TT-matrix with TT-ranks close to 1, not 100.

  Returns:
    `TensorTrain` object containing a TT-matrix.
  """
  mat = tf.convert_to_tensor(mat)
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1])).astype(int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0])).astype(int)

  shape = np.array(shape)
  tens = tf.reshape(mat, shape.flatten())
  d = len(shape[0])
  # transpose_idx = 0, d, 1, d+1 ...
  transpose_idx = np.arange(2 * d).reshape(2, d).T.flatten()
  transpose_idx = transpose_idx.astype(int)
  tens = tf.transpose(tens, transpose_idx)
  new_shape = np.prod(shape, axis=0)
  tens = tf.reshape(tens, new_shape)
  tt_tens = to_tt_tensor(tens, max_tt_rank, epsilon)
  tt_cores = []
  static_tt_ranks = tt_tens.get_tt_ranks()
  dynamic_tt_ranks = shapes.tt_ranks(tt_tens)
  for core_idx in range(d):
    curr_core = tt_tens.tt_cores[core_idx]
    curr_rank = static_tt_ranks[core_idx].value
    if curr_rank is None:
      curr_rank = dynamic_tt_ranks[core_idx]
    next_rank = static_tt_ranks[core_idx + 1].value
    if curr_rank is None:
      next_rank = dynamic_tt_ranks[core_idx + 1]
    curr_core_new_shape = (curr_rank, shape[0, core_idx],
                           shape[1, core_idx], next_rank)
    curr_core = tf.reshape(curr_core, curr_core_new_shape)
    tt_cores.append(curr_core)
  return TensorTrain(tt_cores, shape, tt_tens.get_tt_ranks())


# TODO: implement epsilon.
def to_tt_tensor(tens, max_tt_rank=10, epsilon=None):
  """Converts a given tf.Tensor to a TT-tensor of the same shape.

  Args:
    tens: tf.Tensor
    max_tt_rank: a number or a list of numbers
      If a number, than defines the maximal TT-rank of the result.
      If a list of numbers, than `max_tt_rank` length should be d+1
      (where d is the rank of `tens`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th TT-rank of the result.
      The following two versions are equivalent
        `max_tt_rank = r`
      and
        `max_tt_rank = np.vstack(1, r * np.ones(d-1), 1)`
    eps: a floating point number
      If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `eps` close to `tens`
      in terms of relative Frobenius error:
        ||res - tens||_F / ||tens||_F <= eps
      If the TT-ranks are restricted, providing a loose `eps` may
      reduce the TT-ranks of the result.
      E.g.
        to_tt_tensor(tens, max_tt_rank=100, eps=1)
      will probably return you a TT-tensor with TT-ranks close to 1,
      not 100.

  Returns:
    `TensorTrain` object containing a TT-tensor.

  Raises:
    ValueError if the rank of the input tensor is not defined, if max_tt_rank is
      less than 0, if max_tt_rank is not a number or a vector of length d + 1
      where d is the number of dimensions (rank) of the input tensor, if epsilon
      is less than 0.
  """
  tens = tf.convert_to_tensor(tens)
  static_shape = tens.get_shape()
  dynamic_shape = tf.shape(tens)
  # Raises ValueError if ndims is not defined.
  d = static_shape.__len__()
  max_tt_rank = np.array(max_tt_rank).astype(np.int32)
  if max_tt_rank < 1:
    raise ValueError('Maximum TT-rank should be greater or equal to 1.')
  if epsilon is not None and epsilon < 0:
    raise ValueError('Epsilon should be non-negative.')
  if max_tt_rank.size == 1:
    max_tt_rank = (max_tt_rank * np.ones(d+1)).astype(np.int32)
  elif max_tt_rank.size != d + 1:
    raise ValueError('max_tt_rank should be a number or a vector of size (d+1) '
                     'where d is the number of dimensions (rank) of the tensor.')
  ranks = [1] * (d + 1)
  tt_cores = []
  are_tt_ranks_defined = True
  for core_idx in range(d - 1):
    curr_mode = static_shape[core_idx].value
    if curr_mode is None:
      curr_mode = dynamic_shape[core_idx]
    rows = ranks[core_idx] * curr_mode
    tens = tf.reshape(tens, [rows, -1])
    columns = tens.get_shape()[1].value
    if columns is None:
      columns = tf.shape(tens)[1]
    s, u, v = tf.svd(tens, full_matrices=False)
    if max_tt_rank[core_idx + 1] == 1:
      ranks[core_idx + 1] = 1
    else:
      try:
        ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)
      except TypeError:
        # Some of the values are undefined on the compilation stage and thus
        # they are tf.tensors instead of values.
        min_dim = tf.minimum(rows, columns)
        ranks[core_idx + 1] = tf.minimum(max_tt_rank[core_idx + 1], min_dim)
        are_tt_ranks_defined = False
    u = u[:, 0:ranks[core_idx + 1]]
    s = s[0:ranks[core_idx + 1]]
    v = v[:, 0:ranks[core_idx + 1]]
    core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
    tt_cores.append(tf.reshape(u, core_shape))
    tens = tf.matmul(tf.diag(s), tf.transpose(v))
  last_mode = static_shape[-1].value
  if last_mode is None:
    last_mode = dynamic_shape[-1]
  core_shape = (ranks[d - 1], last_mode, ranks[d])
  tt_cores.append(tf.reshape(tens, core_shape))
  if not are_tt_ranks_defined:
    ranks = None
  return TensorTrain(tt_cores, static_shape, ranks)