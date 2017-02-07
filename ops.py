import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
import utils


# TODO: add complexities to the comments.

def to_tt_matrix(mat, shape, max_tt_rank=10, eps=1e-6):
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
      If a list of numbers, than `max_tt_rank` length should be d-1
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

  raise NotImplementedError


def to_tt_tensor(tens, max_tt_rank=10, eps=1e-6):
  """Converts a given tf.Tensor to a TT-tensor of the same shape.

  Args:
    tens: tf.Tensor
    max_tt_rank: a number or a list of numbers
      If a number, than defines the maximal TT-rank of the result.
      If a list of numbers, than `max_tt_rank` length should be d-1
      (where d is the rank of `tens`) and `max_tt_rank[i]` defines
      the maximal (i+1)-th TT-rank of the result.
      The following two versions are equivalent
        `max_tt_rank = r`
      and
        `max_tt_rank = r * np.ones(d-1)`
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
  """
  raise NotImplementedError


def full(tt):
  """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrain` object.

  Returns:
    tf.Tensor.
  """
  num_dims = tt.ndims()
  ranks = tt_ranks(tt)
  res = tt.tt_cores[0]
  for i in range(1, num_dims):
    res = tf.reshape(res, (-1, ranks[i]))
    curr_core = tf.reshape(tt.tt_cores[i], (ranks[i], -1))
    res = tf.matmul(res, curr_core)
  if tt.is_tt_matrix():
    raw_shape = tt.get_raw_shape()
    intermediate_shape = []
    for i in range(num_dims):
      intermediate_shape.append(raw_shape[0][i])
      intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, tf.TensorShape(intermediate_shape))
    transpose = []
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, tt.get_shape())
  else:
    return tf.reshape(res, tt.get_shape())


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
  return tf.stack(ranks, axis=0)


def tt_tt_matmul(tt_matrix_a, tt_matrix_b):
  """Multiplies two TT-matrices and returns the TT-matrix of the result.

  Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

  Returns
    `TensorTrain` object containing a TT-matrix of size M x P

  Raises:
    ValueError is the arguments are not TT matrices or if their sizes are not
    appropriate for a matrix-by-matrix multiplication.
  """
  if not isinstance(tt_matrix_a, TensorTrain) or not isinstance(tt_matrix_b, TensorTrain):
    raise ValueError('Arguments should be TT-matrices')

  ndims = tt_matrix_a.ndims()
  if tt_matrix_b.ndims() != ndims:
    raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_matrix_b.ndims()))
  result_cores = []
  # TODO: name the operation and the resulting tensor.
  for core_idx in range(ndims):
    a_core = tt_matrix_a.tt_cores[core_idx]
    b_core = tt_matrix_b.tt_cores[core_idx]
    curr_res_core = tf.einsum('aijb,cjkd->acikbd', a_core, b_core)

    res_left_rank = tf.shape(a_core)[0] * tf.shape(b_core)[0]
    res_right_rank = tf.shape(a_core)[-1] * tf.shape(b_core)[-1]
    left_mode = tf.shape(a_core)[1]
    right_mode = tf.shape(b_core)[2]
    core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
    curr_res_core = tf.reshape(curr_res_core, core_shape)
    result_cores.append(curr_res_core)
  res_shape = (tt_matrix_a.get_raw_shape()[0], tt_matrix_b.get_raw_shape()[1])
  a_ranks = tt_matrix_a.get_tt_ranks()
  b_ranks = tt_matrix_b.get_tt_ranks()
  res_ranks = []
  for core_idx in range(ndims + 1):
    res_ranks.append(a_ranks[core_idx] * b_ranks[core_idx])
  res_ranks = tf.TensorShape(res_ranks)
  return TensorTrain(result_cores, res_shape, res_ranks)


def tt_dense_matmul(tt_matrix_a, matrix_b):
  """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.

  Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: tf.Tensor of size N x P

  Returns
    tf.Tensor of size M x P
  """
  raise NotImplementedError


def dense_tt_matmul(matrix_a, tt_matrix_b):
  """Multiplies a regular matrix by a TT-matrix, returns a regular matrix.

  Args:
    matrix_a: tf.Tensor of size M x N
    tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

  Returns
    tf.Tensor of size M x P
  """
  raise NotImplementedError


def sparse_tt_matmul(sparse_matrix_a, tt_matrix_b):
  """Multiplies a sparse matrix by a TT-matrix, returns a regular matrix.

  Args:
    sparse_matrix_a: tf.SparseTensor of size M x N
    tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

  Returns
    tf.Tensor of size M x P
  """
  raise NotImplementedError


# TODO: add flag `return_type = (TT | dense)`?
def tt_sparse_matmul(tt_matrix_a, sparse_matrix_b):
  """Multiplies a TT-matrix by a sparse matrix, returns a regular matrix.

  Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    sparse_matrix_b: tf.SparseTensor of size N x P

  Returns
    tf.Tensor of size M x P
  """
  raise NotImplementedError


def matmul(matrix_a, matrix_b):
  """Multiplies two matrices that can be TT-, dense, or sparse.

  Note that multiplication of two TT-matrices returns a TT-matrix with much
  larger ranks.

  Args:
    matrix_a: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size M x N
    matrix_b: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size N x P

  Returns
    If both arguments are `TensorTrain` objects, returns a `TensorTrain`
      object containing a TT-matrix of size M x P
    If not, returns tf.Tensor of size M x P
  """
  raise NotImplementedError


def tt_tt_flat_inner(tt_a, tt_b):
  """Inner product between two TT-tensors or TT-matrices along all axis.

  The shapes of tt_a and tt_b should coincide.

  Args:
    tt_a: `TensorTrain` object
    tt_b: `TensorTrain` object

  Returns
    a number
    sum of products of all the elements of tt_a and tt_b

  Raises:
    ValueError if the arguments are not `TensorTrain` objects, have different
      number of TT-cores, different underlying shape, or if you are trying to
      compute inner product between a TT-matrix and a TT-tensor.
  """
  if not isinstance(tt_a, TensorTrain) or not isinstance(tt_b, TensorTrain):
    raise ValueError('Arguments should be TensorTrains')

  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('One of the arguments is a TT-tensor, the other is '
                     'a TT-matrix, disallowed')
  are_both_matrices = tt_a.is_tt_matrix() and tt_b.is_tt_matrix()

  # TODO: compare shapes and raise if not consistent.

  ndims = tt_a.ndims()
  if tt_b.ndims() != ndims:
    raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_b.ndims()))

  a_core = tt_a.tt_cores[0]
  b_core = tt_b.tt_cores[0]
  if are_both_matrices:
    res = tf.einsum('aijb,cijd->bd', a_core, b_core)
  else:
    res = tf.einsum('aib,cid->bd', a_core, b_core)
  # TODO: name the operation and the resulting tensor.
  for core_idx in range(1, ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if are_both_matrices:
      res = tf.einsum('ac,aijb,cijd->bd', res, a_core, b_core)
    else:
      res = tf.einsum('ac,aib,cid->bd', res, a_core, b_core)
  return res


def tt_dense_flat_inner(tt_a, dense_b):
  """Inner product between a TT-tensor (or TT-matrix) and tf.Tensor along all axis.

  The shapes of tt_a and dense_b should coincide.

  Args:
    tt_a: `TensorTrain` object
    dense_b: tf.Tensor

  Returns
    a number
    sum of products of all the elements of tt_a and dense_b
  """
  raise NotImplementedError


def tt_sparse_flat_inner(tt_a, sparse_b):
  """Inner product between a TT-tensor (or TT-matrix) and tf.SparseTensor along all axis.

  The shapes of tt_a and sparse_b should coincide.

  Args:
    tt_a: `TensorTrain` object
    sparse_b: tf.SparseTensor

  Returns
    a number
    sum of products of all the elements of tt_a and sparse_b
  """
  num_elements = tf.shape(sparse_b.indices)[0]
  tt_a_elements = tf.ones((num_elements, 1, 1))
  if tt_a.is_tt_matrix():
    # TODO: use t3f.shape is safer??
    tensor_shape = tt_a.get_raw_shape()
    row_idx_linear = tf.cast(sparse_b.indices[:, 0], tf.int64)
    row_idx = utils.unravel_index(row_idx_linear, tf.cast(tensor_shape[0], tf.int64))
    col_idx_linear = tf.cast(sparse_b.indices[:, 1], tf.int64)
    col_idx = utils.unravel_index(col_idx_linear, tf.cast(tensor_shape[1], tf.int64))
    for core_idx in range(tt_a.ndims()):
      # TODO: probably a very slow way to do it, wait for a reasonable gather
      # implementation
      # https://github.com/tensorflow/tensorflow/issues/206
      curr_core = tt_a.tt_cores[core_idx]
      left_rank = tf.shape(curr_core)[0]
      right_rank = tf.shape(curr_core)[-1]
      curr_core = tf.transpose(curr_core, (1, 2, 0, 3))
      curr_core = tf.reshape(curr_core, (-1, left_rank, right_rank))
      # Ravel multiindex (row_idx[:, core_idx], col_idx[:, core_idx]) into
      # a linear index to use tf.gather that supports only first dimensional
      # gather.
      curr_elements_idx = row_idx[:, core_idx] * tensor_shape[1][core_idx]
      curr_elements_idx += col_idx[:, core_idx]
      core_slices = tf.gather(curr_core, curr_elements_idx)
      tt_a_elements = tf.matmul(tt_a_elements, core_slices)
  else:
    for core_idx in range(tt_a.ndims()):
      curr_elements_idx = sparse_b.indices[:, core_idx]
      # TODO: probably a very slow way to do it, wait for a reasonable gather
      # implementation
      # https://github.com/tensorflow/tensorflow/issues/206
      curr_core = tt_a.tt_cores[core_idx]
      curr_core = tf.transpose(curr_core, (1, 0, 2))
      core_slices = tf.gather(curr_core, curr_elements_idx)
      tt_a_elements = tf.matmul(tt_a_elements, core_slices)
  tt_a_elements = tf.reshape(tt_a_elements, (1, -1))
  sparse_b_elements = tf.reshape(sparse_b.values, (-1, 1))
  result = tf.matmul(tt_a_elements, sparse_b_elements)
  # Convert a 1x1 matrix into a number.
  result = result[0, 0]
  return result


def dense_tt_flat_inner(dense_a, tt_b):
  """Inner product between a tf.Tensor and TT-tensor (or TT-matrix) along all axis.

  The shapes of dense_a and tt_b should coincide.

  Args:
    dense_a: `TensorTrain` object
    tt_b: tf.SparseTensor

  Returns
    a number
    sum of products of all the elements of dense_a and tt_b
  """
  raise NotImplementedError


def sparse_tt_flat_inner(sparse_a, tt_b):
  """Inner product between a tf.SparseTensor and TT-tensor (or TT-matrix) along all axis.

  The shapes of sparse_a and tt_b should coincide.

  Args:
    sparse_a: `TensorTrain` object
    tt_b: tf.SparseTensor

  Returns
    a number
    sum of products of all the elements of sparse_a and tt_b
  """
  raise NotImplementedError


def flat_inner(a, b):
  """Inner product along all axis.

  The shapes of a and b should coincide.

  Args:
    a: `TensorTrain`, tf.Tensor, or tf.SparseTensor
    b: `TensorTrain`, tf.Tensor, or tf.SparseTensor

  Returns
    a number
    sum of products of all the elements of a and b
  """
  raise NotImplementedError


def frobenius_norm_squared(tt):
  """Frobenius norm squared of a TensorTrain (sum of squares of all elements).

  Args:
    tt: `TensorTrain` object

  Returns
    a number
    sum of squares of all elements in `tt`
  """
  if tt.is_tt_matrix():
    running_prod = tf.einsum('aijb,cijd->bd', tt.tt_cores[0], tt.tt_cores[0])
  else:
    running_prod = tf.einsum('aib,cid->bd', tt.tt_cores[0], tt.tt_cores[0])

  for core_idx in range(1, tt.ndims()):
    curr_core = tt.tt_cores[core_idx]
    if tt.is_tt_matrix():
      running_prod = tf.einsum('ac,aijb,cijd->bd', running_prod, curr_core,
                               curr_core)
    else:
      running_prod = tf.einsum('ac,aib,cid->bd', running_prod, curr_core,
                               curr_core)
  return running_prod[0, 0]
