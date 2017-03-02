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
    epsilon: a floating point number or None
      If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `epsilon` close to `mat`
      in terms of relative Frobenius error:
        ||res - mat||_F / ||mat||_F <= epsilon
      If the TT-ranks are restricted, providing a loose `epsilon` may reduce
      the TT-ranks of the result.
      E.g.
        to_tt_matrix(mat, shape, max_tt_rank=100, epsilon=0.9)
      will probably return you a TT-matrix with TT-ranks close to 1, not 100.
      Note that providing a nontrivial (= not equal to None) `epsilon` will make
      the TT-ranks of the result undefined on the compilation stage
      (e.g. res.get_tt_ranks() will return None, but t3f.tt_ranks(res).eval()
      will work).

  Returns:
    `TensorTrain` object containing a TT-matrix.

  Raises:
    ValueError if max_tt_rank is less than 0, if max_tt_rank is not a number and
      not a vector of length d + 1 where d is the number of dimensions (rank) of
      the input tensor, if epsilon is less than 0.
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
    if next_rank is None:
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
        `max_tt_rank = r * np.ones(d-1)`
    epsilon: a floating point number or None
      If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `epsilon` close to `tens`
      in terms of relative Frobenius error:
        ||res - tens||_F / ||tens||_F <= epsilon
      If the TT-ranks are restricted, providing a loose `epsilon` may
      reduce the TT-ranks of the result.
      E.g.
        to_tt_tensor(tens, max_tt_rank=100, epsilon=0.9)
      will probably return you a TT-tensor with TT-ranks close to 1, not 100.
      Note that providing a nontrivial (= not equal to None) `epsilon` will make
      the TT-ranks of the result undefined on the compilation stage
      (e.g. res.get_tt_ranks() will return None, but t3f.tt_ranks(res).eval()
      will work).

  Returns:
    `TensorTrain` object containing a TT-tensor.

  Raises:
    ValueError if the rank (number of dimensions) of the input tensor is
      not defined, if max_tt_rank is less than 0, if max_tt_rank is not a number
      and not a vector of length d + 1 where d is the number of dimensions (rank)
      of the input tensor, if epsilon is less than 0.
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


# TODO: rename round so not to shadow python.round?
def round(tt, max_tt_rank=None, epsilon=None):
  """TT-rounding procedure, returns a TT object with smaller TT-ranks.

  Args:
    tt: `TensorTrain` object, TT-tensor or TT-matrix
    max_tt_rank: a number or a list of numbers
      If a number, than defines the maximal TT-rank of the result.
      If a list of numbers, than `max_tt_rank` length should be d+1
      (where d is the rank of `tens`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th TT-rank of the result.
      The following two versions are equivalent
        `max_tt_rank = r`
      and
        `max_tt_rank = r * np.ones(d-1)`
    epsilon: a floating point number or None
      If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
      the result would be guarantied to be `epsilon` close to `tt`
      in terms of relative Frobenius error:
        ||res - tt||_F / ||tt||_F <= epsilon
      If the TT-ranks are restricted, providing a loose `epsilon` may
      reduce the TT-ranks of the result.
      E.g.
        round(tt, max_tt_rank=100, epsilon=0.9)
      will probably return you a TT-tensor with TT-ranks close to 1, not 100.
      Note that providing a nontrivial (= not equal to None) `epsilon` will make
      the TT-ranks of the result undefined on the compilation stage
      (e.g. res.get_tt_ranks() will return None, but t3f.tt_ranks(res).eval()
      will work).

  Returns:
    `TensorTrain` object containing a TT-tensor.

  Raises:
    ValueError if max_tt_rank is less than 0, if max_tt_rank is not a number and
      not a vector of length d + 1 where d is the number of dimensions (rank) of
      the input tensor, if epsilon is less than 0.
  """
  ndims = tt.ndims()
  max_tt_rank = np.array(max_tt_rank).astype(np.int32)
  if max_tt_rank < 1:
    raise ValueError('Maximum TT-rank should be greater or equal to 1.')
  if epsilon is not None and epsilon < 0:
    raise ValueError('Epsilon should be non-negative.')
  if max_tt_rank.size == 1:
    max_tt_rank = (max_tt_rank * np.ones(ndims + 1)).astype(np.int32)
  elif max_tt_rank.size != ndims + 1:
    raise ValueError('max_tt_rank should be a number or a vector of size (d+1) '
                     'where d is the number of dimensions (rank) of the tensor.')
  # Its better to use get_shape() than get_raw_shape()[0] since for matrices
  # get_shape().is_fully_defined() will check that all dimensions are defined.
  if tt.get_shape().is_fully_defined():
    raw_shape = [s.as_list() for s in tt.get_raw_shape()]
  else:
    raw_shape = shapes.raw_shape(tt)
  static_tt_ranks = tt.get_tt_ranks()
  if static_tt_ranks.is_fully_defined():
    tt_ranks = static_tt_ranks.as_list()
  else:
    tt_ranks = shapes.tt_ranks(tt)
  # Copy cores references so we can change the cores.
  tt_cores = list(tt.tt_cores)

  next_rank = tt_ranks[0]

  # Left to right orthogonalization.
  for core_idx in range(ndims - 1):
    curr_core = tt_cores[core_idx]
    # TT-ranks could have changed on the previous iteration, so `tt_ranks` can
    # be outdated for the current TT-rank, but should be valid for the next
    # TT-rank.
    curr_rank = next_rank
    next_rank = tt_ranks[core_idx + 1]
    if tt.is_tt_matrix():
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[0][core_idx]

    qr_shape = (curr_rank * curr_mode, next_rank)
    curr_core = tf.reshape(curr_core, qr_shape)
    curr_core, triang = tf.qr(curr_core)
    if triang.get_shape().is_fully_defined():
      triang_shape = triang.get_shape().as_list()
    else:
      triang_shape = tf.shape(triang)
    # The TT-rank could have changed: if qr_shape is e.g. 4 x 10, than q would
    # be of size 4 x 4 and r would be 4 x 10, which means that the next rank
    # should be changed to 4.
    next_rank = triang_shape[0]
    if tt.is_tt_matrix():
      new_core_shape = (curr_rank, curr_mode_left, curr_mode_right, next_rank)
    else:
      new_core_shape = (curr_rank, curr_mode, next_rank)
    tt_cores[core_idx] = tf.reshape(curr_core, new_core_shape)

    next_core = tf.reshape(tt_cores[core_idx + 1], (triang_shape[1], -1))
    tt_cores[core_idx + 1] = tf.matmul(triang, next_core)

  if tt.is_tt_matrix():
    last_core_shape = (next_rank, raw_shape[0][-1], raw_shape[1][-1], 1)
  else:
    last_core_shape = (next_rank, raw_shape[0][-1], 1)
  tt_cores[-1] = tf.reshape(tt_cores[-1], last_core_shape)

  ranks = [1] * (ndims + 1)
  are_tt_ranks_defined = True
  # Right to left SVD compression.
  for core_idx in range(ndims - 1, 0, -1):
    curr_core = tt_cores[core_idx]
    if tt.is_tt_matrix():
      curr_mode_left = raw_shape[0][core_idx]
      curr_mode_right = raw_shape[1][core_idx]
      curr_mode = curr_mode_left * curr_mode_right
    else:
      curr_mode = raw_shape[0][core_idx]

    columns = curr_mode * ranks[core_idx + 1]
    curr_core = tf.reshape(curr_core, [-1, columns])
    rows = curr_core.get_shape()[0].value
    if rows is None:
      rows = tf.shape(curr_core)[0]
    if max_tt_rank[core_idx] == 1:
      ranks[core_idx] = 1
    else:
      try:
        ranks[core_idx] = min(max_tt_rank[core_idx], rows, columns)
      except TypeError:
        # Some of the values are undefined on the compilation stage and thus
        # they are tf.tensors instead of values.
        min_dim = tf.minimum(rows, columns)
        ranks[core_idx] = tf.minimum(max_tt_rank[core_idx], min_dim)
        are_tt_ranks_defined = False
    s, u, v = tf.svd(curr_core, full_matrices=False)
    u = u[:, 0:ranks[core_idx]]
    s = s[0:ranks[core_idx]]
    v = v[:, 0:ranks[core_idx]]
    if tt.is_tt_matrix():
      core_shape = (ranks[core_idx], curr_mode_left, curr_mode_right,
                    ranks[core_idx + 1])
    else:
      core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
    tt_cores[core_idx] = tf.reshape(tf.transpose(v), core_shape)
    prev_core_shape = (-1, rows)
    tt_cores[core_idx - 1] = tf.reshape(tt_cores[core_idx - 1], prev_core_shape)
    tt_cores[core_idx - 1] = tf.matmul(tt_cores[core_idx - 1], u)
    tt_cores[core_idx - 1] = tf.matmul(tt_cores[core_idx - 1], tf.diag(s))

  if tt.is_tt_matrix():
    core_shape = (ranks[0], raw_shape[0][0], raw_shape[1][0], ranks[1])
  else:
    core_shape = (ranks[0], raw_shape[0][0], ranks[1])
  tt_cores[0] = tf.reshape(tt_cores[0], core_shape)
  if not are_tt_ranks_defined:
    ranks = None
  return TensorTrain(tt_cores, tt.get_raw_shape(), ranks)