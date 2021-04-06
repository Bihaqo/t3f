import tensorflow as tf
import numpy as np
from t3f.tensor_train_base import TensorTrainBase
from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import shapes
from t3f import utils
from t3f import decompositions
from t3f import initializers

# TODO: add complexities to the comments.


def full(tt, name='t3f_full'):
  """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object.
    name: string, name of the Op.

  Returns:
    tf.Tensor.
  """
  with tf.name_scope(name):
    if isinstance(tt, TensorTrainBatch):
      # Batch of Tensor Trains.
      return _full_tt_batch(tt)
    else:
      # TensorTrain object (not batch).
      return _full_tt(tt)


def _full_tt(tt):
  """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrain` object.

  Returns:
    tf.Tensor.
  """
  num_dims = tt.ndims()
  ranks = shapes.lazy_tt_ranks(tt)
  shape = shapes.lazy_shape(tt)
  raw_shape = shapes.lazy_raw_shape(tt)

  res = tt.tt_cores[0]
  for i in range(1, num_dims):
    res = tf.reshape(res, (-1, ranks[i]))
    curr_core = tf.reshape(tt.tt_cores[i], (ranks[i], -1))
    res = tf.matmul(res, curr_core)
  if tt.is_tt_matrix():
    intermediate_shape = []
    for i in range(num_dims):
      intermediate_shape.append(raw_shape[0][i])
      intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, intermediate_shape)
    transpose = []
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, shape)
  else:
    return tf.reshape(res, shape)


def _full_tt_batch(tt):
  """Converts a TensorTrainBatch into a regular tensor or matrix (tf.Tensor).

  Args:
    tt: `TensorTrainBatch` object.

  Returns:
    tf.Tensor.
  """
  num_dims = tt.ndims()
  ranks = shapes.lazy_tt_ranks(tt)
  shape = shapes.lazy_shape(tt)
  raw_shape = shapes.lazy_raw_shape(tt)

  res = tt.tt_cores[0]
  batch_size = shapes.lazy_batch_size(tt)
  for i in range(1, num_dims):
    res = tf.reshape(res, (batch_size, -1, ranks[i]))
    curr_core = tf.reshape(tt.tt_cores[i], (batch_size, ranks[i], -1))
    res = tf.einsum('oqb,obw->oqw', res, curr_core)
  if tt.is_tt_matrix():
    intermediate_shape = [batch_size]
    for i in range(num_dims):
      intermediate_shape.append(raw_shape[0][i])
      intermediate_shape.append(raw_shape[1][i])
    res = tf.reshape(res, intermediate_shape)
    transpose = [0]
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i + 1)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i + 1)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, shape)
  else:
    return tf.reshape(res, shape)


def tt_tt_matmul(tt_matrix_a, tt_matrix_b):
  """Multiplies two TT-matrices and returns the TT-matrix of the result.

  Args:
    tt_matrix_a: `TensorTrain` or `TensorTrainBatch` object containing
      a TT-matrix (a batch of TT-matrices) of size M x N
    tt_matrix_b: `TensorTrain` or `TensorTrainBatch` object containing
      a TT-matrix (a batch of TT-matrices) of size N x P

  Returns
    `TensorTrain` object containing a TT-matrix of size M x P if both arguments
      are `TensorTrain`s
    `TensorTrainBatch` if any of the arguments is a `TensorTrainBatch`

  Raises:
    ValueError is the arguments are not TT matrices or if their sizes are not
    appropriate for a matrix-by-matrix multiplication.
  """
  # Both TensorTrain and TensorTrainBatch are inherited from TensorTrainBase.
  if not isinstance(tt_matrix_a, TensorTrainBase) or \
      not isinstance(tt_matrix_b, TensorTrainBase) or \
      not tt_matrix_a.is_tt_matrix() or \
      not tt_matrix_b.is_tt_matrix():
    raise ValueError('Arguments should be TT-matrices')

  if not shapes.is_batch_broadcasting_possible(tt_matrix_a, tt_matrix_b):
    raise ValueError('The batch sizes are different and not 1, broadcasting is '
                     'not available.')

  ndims = tt_matrix_a.ndims()
  if tt_matrix_b.ndims() != ndims:
    raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_matrix_b.ndims()))

  # Convert BatchSize 1 batch into TT object to simplify broadcasting.
  tt_matrix_a = shapes.squeeze_batch_dim(tt_matrix_a)
  tt_matrix_b = shapes.squeeze_batch_dim(tt_matrix_b)
  is_a_batch = isinstance(tt_matrix_a, TensorTrainBatch)
  is_b_batch = isinstance(tt_matrix_b, TensorTrainBatch)
  is_res_batch = is_a_batch or is_b_batch
  a_batch_str = 'o' if is_a_batch else ''
  b_batch_str = 'o' if is_b_batch else ''
  res_batch_str = 'o' if is_res_batch else ''
  einsum_str = '{}aijb,{}cjkd->{}acikbd'.format(a_batch_str, b_batch_str,
                                                res_batch_str)
  result_cores = []
  a_shape = shapes.lazy_raw_shape(tt_matrix_a)
  a_ranks = shapes.lazy_tt_ranks(tt_matrix_a)
  b_shape = shapes.lazy_raw_shape(tt_matrix_b)
  b_ranks = shapes.lazy_tt_ranks(tt_matrix_b)
  if is_res_batch:
    if is_a_batch:
      batch_size = shapes.lazy_batch_size(tt_matrix_a)
    if is_b_batch:
      batch_size = shapes.lazy_batch_size(tt_matrix_b)
  for core_idx in range(ndims):
    a_core = tt_matrix_a.tt_cores[core_idx]
    b_core = tt_matrix_b.tt_cores[core_idx]
    curr_res_core = tf.einsum(einsum_str, a_core, b_core)

    res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
    res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
    left_mode = a_shape[0][core_idx]
    right_mode = b_shape[1][core_idx]
    if is_res_batch:
      core_shape = (batch_size, res_left_rank, left_mode, right_mode, res_right_rank)
    else:
      core_shape = (res_left_rank, left_mode, right_mode,
                    res_right_rank)
    curr_res_core = tf.reshape(curr_res_core, core_shape)
    result_cores.append(curr_res_core)

  res_shape = (tt_matrix_a.get_raw_shape()[0], tt_matrix_b.get_raw_shape()[1])
  static_a_ranks = tt_matrix_a.get_tt_ranks()
  static_b_ranks = tt_matrix_b.get_tt_ranks()
  out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]
  if is_res_batch:
    return TensorTrainBatch(result_cores, res_shape, out_ranks, batch_size)
  else:
    return TensorTrain(result_cores, res_shape, out_ranks)


def tt_dense_matmul(tt_matrix_a, matrix_b):
  """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.

  Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: tf.Tensor of size N x P

  Returns
    tf.Tensor of size M x P
  """
  if not isinstance(tt_matrix_a, TensorTrain) or not tt_matrix_a.is_tt_matrix():
    raise ValueError('The first argument should be a TT-matrix')

  ndims = tt_matrix_a.ndims()
  a_columns = tt_matrix_a.get_shape().as_list()[1]
  b_rows = matrix_b.get_shape().as_list()[0]
  if a_columns is not None and b_rows is not None:
    if a_columns != b_rows:
      raise ValueError('Arguments shapes should align got %d and %d instead.' %
                       (tt_matrix_a.get_shape(), matrix_b.get_shape()))

  a_shape = shapes.lazy_shape(tt_matrix_a)
  a_raw_shape = shapes.lazy_raw_shape(tt_matrix_a)
  if matrix_b.get_shape().is_fully_defined():
    b_shape = matrix_b.get_shape().as_list()
  else:
    b_shape = tf.shape(matrix_b)
  a_ranks = shapes.lazy_tt_ranks(tt_matrix_a)
  # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
  # data is (K, j0, ..., jd-2) x jd-1 x 1
  data = tf.transpose(matrix_b)
  data = tf.reshape(data, (-1, a_raw_shape[1][-1], 1))
  for core_idx in reversed(range(ndims)):
    curr_core = tt_matrix_a.tt_cores[core_idx]
    # On the k = core_idx iteration, after applying einsum the shape of data
    # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
    data = tf.einsum('aijb,rjb->ira', curr_core, data)
    if core_idx > 0:
      # After reshape the shape of data becomes
      # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
      new_data_shape = (-1, a_raw_shape[1][core_idx - 1], a_ranks[core_idx])
      data = tf.reshape(data, new_data_shape)
  # At the end the shape of the data is (i0, ..., id-1) x K
  return tf.reshape(data, (a_shape[0], b_shape[1]))


def dense_tt_matmul(matrix_a, tt_matrix_b):
  """Multiplies a regular matrix by a TT-matrix, returns a regular matrix.

  Args:
    matrix_a: tf.Tensor of size M x N
    tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

  Returns
    tf.Tensor of size M x P
  """
#   TODO: make a more efficient implementation.
  a_t = tf.transpose(matrix_a)
  b_t = transpose(tt_matrix_b)
  return tf.transpose(tt_dense_matmul(b_t, a_t))


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


def matmul(a, b, name='t3f_matmul'):
  """Multiplies two matrices that can be TT-, dense, or sparse.

  Note that multiplication of two TT-matrices returns a TT-matrix with much
  larger ranks.
  Also works for multiplying two batches of TT-matrices or a product between a
  TT-matrix and a batch of TT-matrices (with broadcasting).

  Args:
    a: `TensorTrain`, `TensorTrainBatch`, tf.Tensor, or tf.SparseTensor of
      size M x N
    b: `TensorTrain`, `TensorTrainBatch`, tf.Tensor, or tf.SparseTensor of
      size N x P
    name: string, name of the Op.

  Returns
    If both arguments are `TensorTrain` objects, returns a `TensorTrain`
      object containing a TT-matrix of size M x P.
    If at least one of the arguments is a `TensorTrainBatch` object, returns
      a `TensorTrainBatch` object containing a batch of TT-matrices of size
      M x P.
    Otherwise, returns tf.Tensor of size M x P.
  """
#   TODO: is it safe to check types? What if a class is derived from TT?
  if isinstance(a, TensorTrainBase) and isinstance(b, TensorTrainBase):
    with tf.name_scope(name):
      return tt_tt_matmul(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.Tensor):
    with tf.name_scope(name):
      return tt_dense_matmul(a, b)
  elif isinstance(a, tf.Tensor) and isinstance(b, TensorTrain):
    with tf.name_scope(name):
      return dense_tt_matmul(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.SparseTensor):
    with tf.name_scope(name):
      return tt_sparse_matmul(a, b)
  elif isinstance(a, tf.SparseTensor) and isinstance(b, TensorTrain):
    with tf.name_scope(name):
      return sparse_tt_matmul(a, b)
  else:
    raise ValueError('Argument types are not supported in matmul: %s x %s' %
                     (a, b))


def tt_tt_flat_inner(tt_a, tt_b):
  """Inner product between two TT-tensors or TT-matrices along all axis.

  The shapes of tt_a and tt_b should coincide.

  Args:
    tt_a: `TensorTrain` or `TensorTrainBatch` object
    tt_b: `TensorTrain` or `TensorTrainBatch` object

  Returns
    a number or a Tensor with numbers for each element in the batch.
    sum of products of all the elements of tt_a and tt_b

  Raises:
    ValueError if the arguments are not `TensorTrain` objects, have different
      number of TT-cores, different underlying shape, or if you are trying to
      compute inner product between a TT-matrix and a TT-tensor.

  Complexity:
    Multiplying two single TT-objects is O(d r^3 n) where d is the number of
      TT-cores (tt_a.ndims()), r is the largest TT-rank
        max(tt_a.get_tt_rank(), tt_b.get_tt_rank())
      and n is the size of the axis dimension, e.g.
        for a tensor of size 4 x 4 x 4, n is 4;
        for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
      A more precise complexity is O(d r1 r2 n max(r1, r2)) where
        r1 is the largest TT-rank of tt_a and r2 is the largest TT-rank of tt_b.
    The complexity of this operation for batch input is O(batch_size d r^3 n).
  """
  if not isinstance(tt_a, TensorTrainBase) or not isinstance(tt_b,
                                                             TensorTrainBase):
    raise ValueError('Arguments should be TensorTrains')

  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('One of the arguments is a TT-tensor, the other is '
                     'a TT-matrix, disallowed')
  are_both_matrices = tt_a.is_tt_matrix() and tt_b.is_tt_matrix()

  if not shapes.is_batch_broadcasting_possible(tt_a, tt_b):
    raise ValueError('The batch sizes are different and not 1, broadcasting is '
                     'not available.')

  # TODO: compare shapes and raise if not consistent.

  ndims = tt_a.ndims()
  if tt_b.ndims() != ndims:
    raise ValueError('Arguments should have the same number of dimensions, '
                     'got %d and %d instead.' % (ndims, tt_b.ndims()))

  axes_str = 'ij' if are_both_matrices else 'i'
  # Convert BatchSize 1 batch into TT object to simplify broadcasting.
  tt_a = shapes.squeeze_batch_dim(tt_a)
  tt_b = shapes.squeeze_batch_dim(tt_b)
  is_a_batch = isinstance(tt_a, TensorTrainBatch)
  is_b_batch = isinstance(tt_b, TensorTrainBatch)
  is_res_batch = is_a_batch or is_b_batch
  a_batch_str = 'o' if is_a_batch else ''
  b_batch_str = 'o' if is_b_batch else ''
  res_batch_str = 'o' if is_res_batch else ''
  init_einsum_str = '{1}a{0}b,{2}c{0}d->{3}bd'.format(axes_str, a_batch_str,
                                                      b_batch_str,
                                                      res_batch_str)
  a_core = tt_a.tt_cores[0]
  b_core = tt_b.tt_cores[0]
  # Simplest example of this operation:
  # if both arguments are TT-tensors, then it is
  # res = tf.einsum('aib,cid->bd', a_core, b_core)
  res = tf.einsum(init_einsum_str, a_core, b_core)

  einsum_str = '{3}ac,{1}a{0}b,{2}c{0}d->{3}bd'.format(axes_str, a_batch_str,
                                                       b_batch_str,
                                                       res_batch_str)
  for core_idx in range(1, ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    # Simplest example of this operation:
    # if both arguments are TT-tensors, then it is
    # res = tf.einsum('ac,aib,cid->bd', res, a_core, b_core)
    res = tf.einsum(einsum_str, res, a_core, b_core)
  return tf.squeeze(res)


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
  if sparse_b.indices.get_shape().is_fully_defined():
    num_elements = sparse_b.indices.get_shape()[0]
  else:
    num_elements = tf.shape(sparse_b.indices)[0]
  a_shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
  if tt_a.is_tt_matrix():
    tt_a_elements = tf.ones((num_elements, 1, 1), dtype=tt_a.dtype)
    # TODO: use t3f.shape is safer??
    tensor_shape = tt_a.get_raw_shape()
    row_idx_linear = tf.cast(sparse_b.indices[:, 0], tf.int64)
    row_idx = utils.unravel_index(row_idx_linear, tf.cast(tensor_shape[0], tf.int64))
    col_idx_linear = tf.cast(sparse_b.indices[:, 1], tf.int64)
    col_idx = utils.unravel_index(col_idx_linear, tf.cast(tensor_shape[1], tf.int64))
    for core_idx in range(tt_a.ndims()):
      curr_core = tt_a.tt_cores[core_idx]
      left_rank = a_ranks[core_idx]
      right_rank = a_ranks[core_idx + 1]
      curr_core = tf.transpose(curr_core, (1, 2, 0, 3))
      curr_core_shape = (a_shape[0][core_idx]*a_shape[1][core_idx], left_rank,
                         right_rank)
      curr_core = tf.reshape(curr_core, curr_core_shape)
      # Ravel multiindex (row_idx[:, core_idx], col_idx[:, core_idx]) into
      # a linear index to use tf.gather that supports only first dimensional
      # gather.
      # TODO: use gather_nd instead.
      curr_elements_idx = row_idx[:, core_idx] * tensor_shape[1][core_idx]
      curr_elements_idx += col_idx[:, core_idx]
      core_slices = tf.gather(curr_core, curr_elements_idx)
      tt_a_elements = tf.matmul(tt_a_elements, core_slices)
  else:
    tt_a_elements = gather_nd(tt_a, sparse_b.indices)
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
    dense_a: tf.Tensor
    tt_b: `TensorTrain` object

  Returns
    a number
    sum of products of all the elements of dense_a and tt_b
  """
  raise NotImplementedError


def sparse_tt_flat_inner(sparse_a, tt_b):
  """Inner product between a tf.SparseTensor and TT-tensor (or TT-matrix) along all axis.

  The shapes of sparse_a and tt_b should coincide.

  Args:
    sparse_a: tf.SparseTensor
    tt_b: `TensorTrain` object

  Returns
    a number
    sum of products of all the elements of sparse_a and tt_b
  """
  raise NotImplementedError


def flat_inner(a, b, name='t3f_flat_inner'):
  """Inner product along all axis.

  The shapes of a and b should coincide.

  Args:
    a: `TensorTrain`, `TensorTrainBatch`, tf.Tensor, or tf.SparseTensor
    b: `TensorTrain`, `TensorTrainBatch`, tf.Tensor, or tf.SparseTensor
    name: string, name of the Op.

  Returns
    a number
      sum of products of all the elements of a and b
    OR or a tf.Tensor of size batch_size
      sum of products of all the elements of a and b for each element in the
      batch.
  """
#   TODO: is it safe to check types? What if a class is derived from TT?
  if isinstance(a, TensorTrainBase) and isinstance(b, TensorTrainBase):
    with tf.name_scope(name):
      return tt_tt_flat_inner(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.Tensor):
    with tf.name_scope(name):
      return tt_dense_flat_inner(a, b)
  elif isinstance(a, tf.Tensor) and isinstance(b, TensorTrain):
    with tf.name_scope(name):
      return dense_tt_flat_inner(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.SparseTensor):
    with tf.name_scope(name):
      return tt_sparse_flat_inner(a, b)
  elif isinstance(a, tf.SparseTensor) and isinstance(b, TensorTrain):
    with tf.name_scope(name):
      return sparse_tt_flat_inner(a, b)
  else:
    raise ValueError('Argument types are not supported in flat_inner: %s x %s' %
                     (a, b))


def _add_tensor_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-tensors.

  Does the actual assembling of the TT-cores to add two TT-tensors.
  """
  ndims = tt_a.ndims()
  dtype = tt_a.dtype
  shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = tf.concat((a_core, b_core), axis=2)
    elif core_idx == ndims - 1:
      curr_core = tf.concat((a_core, b_core), axis=0)
    else:
      upper_zeros = tf.zeros((a_ranks[core_idx], shape[0][core_idx],
                              b_ranks[core_idx + 1]), dtype)
      lower_zeros = tf.zeros((b_ranks[core_idx], shape[0][core_idx],
                              a_ranks[core_idx + 1]), dtype)
      upper = tf.concat((a_core, upper_zeros), axis=2)
      lower = tf.concat((lower_zeros, b_core), axis=2)
      curr_core = tf.concat((upper, lower), axis=0)
    tt_cores.append(curr_core)
  return tt_cores


def _add_batch_tensor_cores(tt_a, tt_b):
  """Internal function to be called from add for two batches of TT-tensors.

  Does the actual assembling of the TT-cores to add two batches of TT-tensors.
  """
  ndims = tt_a.ndims()
  dtype = tt_a.dtype
  shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)
  if isinstance(tt_a, TensorTrainBatch) and tt_a.batch_size == 1:
    # We add 1 element batch tt_a to a batch_size element batch tt_b to get
    # the answer TensorTrainBatch of batch_size == tt_b.batch_size.
    batch_size = shapes.lazy_batch_size(tt_b)
  else:
    batch_size = shapes.lazy_batch_size(tt_a)
  tt_a = shapes.expand_batch_dim(tt_a)
  tt_b = shapes.expand_batch_dim(tt_b)
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    if tt_a.batch_size == 1:
      a_core = tf.tile(a_core, (batch_size, 1, 1, 1))
    b_core = tt_b.tt_cores[core_idx]
    if tt_b.batch_size == 1:
      b_core = tf.tile(b_core, (batch_size, 1, 1, 1))
    if core_idx == 0:
      curr_core = tf.concat((a_core, b_core), axis=3)
    elif core_idx == ndims - 1:
      curr_core = tf.concat((a_core, b_core), axis=1)
    else:
      upper_zeros = tf.zeros((batch_size, a_ranks[core_idx], shape[0][core_idx],
                              b_ranks[core_idx + 1]), dtype)
      lower_zeros = tf.zeros((batch_size, b_ranks[core_idx], shape[0][core_idx],
                              a_ranks[core_idx + 1]), dtype)
      upper = tf.concat((a_core, upper_zeros), axis=3)
      lower = tf.concat((lower_zeros, b_core), axis=3)
      curr_core = tf.concat((upper, lower), axis=1)
    tt_cores.append(curr_core)
  return tt_cores, batch_size


def _add_matrix_cores(tt_a, tt_b):
  """Internal function to be called from add for two TT-matrices.

  Does the actual assembling of the TT-cores to add two TT-matrices.
  """
  ndims = tt_a.ndims()
  dtype = tt_a.dtype
  shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = tf.concat((a_core, b_core), axis=3)
    elif core_idx == ndims - 1:
      curr_core = tf.concat((a_core, b_core), axis=0)
    else:
      upper_zeros = tf.zeros((a_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]), dtype)
      lower_zeros = tf.zeros((b_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]), dtype)
      upper = tf.concat((a_core, upper_zeros), axis=3)
      lower = tf.concat((lower_zeros, b_core), axis=3)
      curr_core = tf.concat((upper, lower), axis=0)
    tt_cores.append(curr_core)
  return tt_cores


def _add_batch_matrix_cores(tt_a, tt_b):
  """Internal function to be called from add for two batches of TT-matrices.

  Does the actual assembling of the TT-cores to add two batches of TT-matrices.
  """
  ndims = tt_a.ndims()
  dtype = tt_a.dtype
  shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)
  if isinstance(tt_a, TensorTrainBatch) and tt_a.batch_size == 1:
    # We add 1 element batch tt_a to a batch_size element batch tt_b to get
    # the answer TensorTrainBatch of batch_size == tt_b.batch_size.
    batch_size = shapes.lazy_batch_size(tt_b)
  else:
    batch_size = shapes.lazy_batch_size(tt_a)
  tt_a = shapes.expand_batch_dim(tt_a)
  tt_b = shapes.expand_batch_dim(tt_b)
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    if tt_a.batch_size == 1:
      a_core = tf.tile(a_core, (batch_size, 1, 1, 1, 1))
    b_core = tt_b.tt_cores[core_idx]
    if tt_b.batch_size == 1:
      b_core = tf.tile(b_core, (batch_size, 1, 1, 1, 1))
    if core_idx == 0:
      curr_core = tf.concat((a_core, b_core), axis=4)
    elif core_idx == ndims - 1:
      curr_core = tf.concat((a_core, b_core), axis=1)
    else:
      upper_zeros = tf.zeros((batch_size, a_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], b_ranks[core_idx + 1]), dtype)
      lower_zeros = tf.zeros((batch_size, b_ranks[core_idx], shape[0][core_idx],
                              shape[1][core_idx], a_ranks[core_idx + 1]), dtype)
      upper = tf.concat((a_core, upper_zeros), axis=4)
      lower = tf.concat((lower_zeros, b_core), axis=4)
      curr_core = tf.concat((upper, lower), axis=1)
    tt_cores.append(curr_core)
  return tt_cores, batch_size


def add(tt_a, tt_b, name='t3f_add'):
  """Returns a TensorTrain corresponding to elementwise sum tt_a + tt_b.

  The shapes of tt_a and tt_b should coincide.
  Supports broadcasting:
    add(TensorTrainBatch, TensorTrain)
  adds TensorTrain to each element in the batch of TTs in TensorTrainBatch.

  Args:
    tt_a: `TensorTrain`, `TensorTrainBatch`, TT-tensor, or TT-matrix
    tt_b: `TensorTrain`, `TensorTrainBatch`, TT-tensor, or TT-matrix
    name: string, name of the Op.

  Returns
    a `TensorTrain` object corresponding to the element-wise sum of arguments if
      both arguments are `TensorTrain`s.
    OR a `TensorTrainBatch` if at least one of the arguments is
      `TensorTrainBatch`

  Raises
    ValueError if the arguments shapes do not coincide
  """
  ndims = tt_a.ndims()
  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('The arguments should be both TT-tensors or both '
                     'TT-matrices')

  if tt_a.get_raw_shape() != tt_b.get_raw_shape():
    raise ValueError('The arguments should have the same shape.')

  if not shapes.is_batch_broadcasting_possible(tt_a, tt_b):
    raise ValueError('The batch sizes are different and not 1, broadcasting is '
                     'not available.')

  with tf.name_scope(name):
    is_a_batch = isinstance(tt_a, TensorTrainBatch)
    is_b_batch = isinstance(tt_b, TensorTrainBatch)
    is_batch_case = is_a_batch or is_b_batch
    batch_size = None
    if is_batch_case:
      if tt_a.is_tt_matrix():
        tt_cores, batch_size = _add_batch_matrix_cores(tt_a, tt_b)
      else:
        tt_cores, batch_size = _add_batch_tensor_cores(tt_a, tt_b)
    else:
      if tt_a.is_tt_matrix():
        tt_cores = _add_matrix_cores(tt_a, tt_b)
      else:
        tt_cores = _add_tensor_cores(tt_a, tt_b)

    out_ranks = [1]
    static_a_ranks = tt_a.get_tt_ranks()
    static_b_ranks = tt_b.get_tt_ranks()
    for core_idx in range(1, ndims):
      out_ranks.append(static_a_ranks[core_idx] + static_b_ranks[core_idx])
    out_ranks.append(1)
    if is_batch_case:
      return TensorTrainBatch(tt_cores, tt_a.get_raw_shape(), out_ranks,
                              batch_size)
    else:
      return TensorTrain(tt_cores, tt_a.get_raw_shape(), out_ranks)


def multiply(tt_left, right, name='t3f_multiply'):
  """Returns a TensorTrain corresponding to element-wise product tt_left * right.

  Supports broadcasting:
    multiply(TensorTrainBatch, TensorTrain) returns TensorTrainBatch consisting
    of element-wise products of TT in TensorTrainBatch and TensorTrain

    multiply(TensorTrainBatch_a, TensorTrainBatch_b) returns TensorTrainBatch
    consisting of element-wise products of TT in TensorTrainBatch_a and
    TT in TensorTrainBatch_b

    Batch sizes should support broadcasting
  Args:
    tt_left: `TensorTrain` OR `TensorTrainBatch`
    right: `TensorTrain` OR `TensorTrainBatch` OR a number.
    name: string, name of the Op.

  Returns
    a `TensorTrain` or `TensorTrainBatch` object corresponding to the
    element-wise product of the arguments.

  Raises
    ValueError if the arguments shapes do not coincide or broadcasting is not
    possible.
  """
  is_left_batch = isinstance(tt_left, TensorTrainBatch)
  is_right_batch = isinstance(right, TensorTrainBatch)

  is_batch_case = is_left_batch or is_right_batch
  ndims = tt_left.ndims()
  if not isinstance(right, TensorTrainBase):
    with tf.name_scope(name):
      # Assume right is a number, not TensorTrain.
      # To squash right uniformly across TT-cores we pull its absolute value
      # and raise to the power 1/ndims. First TT-core is multiplied by the sign
      # of right.
      tt_cores = list(tt_left.tt_cores)
      fact = tf.pow(tf.cast(tf.abs(right), tt_left.dtype), 1.0 / ndims)
      sign = tf.cast(tf.sign(right), tt_left.dtype)
      for i in range(len(tt_cores)):
        tt_cores[i] = fact * tt_cores[i]

      tt_cores[0] = tt_cores[0] * sign
      out_ranks = tt_left.get_tt_ranks()
      if is_left_batch:
          out_batch_size = tt_left.batch_size
  else:
    with tf.name_scope(name):

      if tt_left.is_tt_matrix() != right.is_tt_matrix():
        raise ValueError('The arguments should be both TT-tensors or both '
                         'TT-matrices')

      if tt_left.get_raw_shape() != right.get_raw_shape():
        raise ValueError('The arguments should have the same shape.')

      out_batch_size = 1
      dependencies = []
      can_determine_if_broadcast = True
      if is_left_batch and is_right_batch:
        if tt_left.batch_size is None and right.batch_size is None:
          can_determine_if_broadcast = False
        elif tt_left.batch_size is None and right.batch_size is not None:
          if right.batch_size > 1:
              can_determine_if_broadcast = False
        elif tt_left.batch_size is not None and right.batch_size is None:
          if tt_left.batch_size > 1:
              can_determine_if_broadcast = False

      if not can_determine_if_broadcast:
        # Cannot determine if broadcasting is needed. Avoid broadcasting and
        # assume elementwise multiplication AND add execution time assert to
        # print a better error message if the batch sizes turn out to be
        # different.

        message = ('The batch sizes were unknown on compilation stage, so '
                   'assumed elementwise multiplication (i.e. no broadcasting). '
                   'Now it seems that they are different after all :')

        data = [message, shapes.lazy_batch_size(tt_left), ' x ',
                shapes.lazy_batch_size(right)]
        bs_eq = tf.assert_equal(shapes.lazy_batch_size(tt_left),
                                shapes.lazy_batch_size(right))

        dependencies.append(bs_eq)

      do_broadcast = shapes.is_batch_broadcasting_possible(tt_left, right)
      if not can_determine_if_broadcast:
        # Assume elementwise multiplication if broadcasting cannot be determined
        # on compilation stage.
        do_broadcast = False
      if not do_broadcast and can_determine_if_broadcast:
        raise ValueError('The batch sizes are different and not 1, broadcasting '
                         'is not available.')

      a_ranks = shapes.lazy_tt_ranks(tt_left)
      b_ranks = shapes.lazy_tt_ranks(right)
      shape = shapes.lazy_raw_shape(tt_left)

      output_str = ''
      bs_str_left = ''
      bs_str_right = ''

      if is_batch_case:
        if is_left_batch and is_right_batch:
          # Both arguments are batches of equal size.
          if tt_left.batch_size == right.batch_size or not can_determine_if_broadcast:
            bs_str_left = 'n'
            bs_str_right = 'n'
            output_str = 'n'
            if not can_determine_if_broadcast:
              out_batch_size = None
            else:
              out_batch_size = tt_left.batch_size
          else:
            # Broadcasting (e.g batch_sizes are 1 and n>1).
            bs_str_left = 'n'
            bs_str_right = 'm'
            output_str = 'nm'
            if tt_left.batch_size is None or tt_left.batch_size > 1:
              out_batch_size = tt_left.batch_size
            else:
              out_batch_size = right.batch_size
        else:
          # One of the arguments is TensorTrain.
          if is_left_batch:
            bs_str_left = 'n'
            bs_str_right = ''
            out_batch_size = tt_left.batch_size
          else:
            bs_str_left = ''
            bs_str_right = 'n'
            out_batch_size = right.batch_size
          output_str = 'n'

      is_matrix = tt_left.is_tt_matrix()
      tt_cores = []

      for core_idx in range(ndims):
        a_core = tt_left.tt_cores[core_idx]
        b_core = right.tt_cores[core_idx]
        left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        if is_matrix:
          with tf.control_dependencies(dependencies):
            curr_core = tf.einsum('{0}aijb,{1}cijd->{2}acijbd'.format(bs_str_left,
                                  bs_str_right, output_str), a_core, b_core)
            curr_core = tf.reshape(curr_core, (-1, left_rank,
                                               shape[0][core_idx],
                                               shape[1][core_idx],
                                               right_rank))
            if not is_batch_case:
                curr_core = tf.squeeze(curr_core, axis=0)
        else:
          with tf.control_dependencies(dependencies):
            curr_core = tf.einsum('{0}aib,{1}cid->{2}acibd'.format(bs_str_left,
                                  bs_str_right, output_str), a_core, b_core)
            curr_core = tf.reshape(curr_core, (-1, left_rank,
                                   shape[0][core_idx], right_rank))
            if not is_batch_case:
              curr_core = tf.squeeze(curr_core, axis=0)

        tt_cores.append(curr_core)

      combined_ranks = zip(tt_left.get_tt_ranks(), right.get_tt_ranks())
      out_ranks = [a * b for a, b in combined_ranks]

  if not is_batch_case:
    return TensorTrain(tt_cores, tt_left.get_raw_shape(), out_ranks)
  else:
    return TensorTrainBatch(tt_cores, tt_left.get_raw_shape(), out_ranks,
                            batch_size=out_batch_size)

def frobenius_norm_squared(tt, differentiable=False,
                           name='t3f_frobenius_norm_squared'):
  """Frobenius norm squared of `TensorTrain` or of each TT in `TensorTrainBatch`.

  Frobenius norm squared is the sum of squares of all elements in a tensor.

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object
    differentiable: bool, whether to use a differentiable implementation
      or a fast and stable implementation based on QR decomposition.
    name: string, name of the Op.

  Returns
    a number which is the Frobenius norm squared of `tt`, if it is `TensorTrain`
    OR
    a Tensor of size tt.batch_size, consisting of the Frobenius norms squared of
    each TensorTrain in `tt`, if it is `TensorTrainBatch`
  """
  with tf.name_scope(name):
    if differentiable:
      if hasattr(tt, 'batch_size'):
          bs_str = 'n'
      else:
          bs_str = ''
      if tt.is_tt_matrix():
        running_prod = tf.einsum('{0}aijb,{0}cijd->{0}bd'.format(bs_str),
                                 tt.tt_cores[0], tt.tt_cores[0])
      else:
        running_prod = tf.einsum('{0}aib,{0}cid->{0}bd'.format(bs_str),
                                 tt.tt_cores[0], tt.tt_cores[0])

      for core_idx in range(1, tt.ndims()):
        curr_core = tt.tt_cores[core_idx]
        if tt.is_tt_matrix():
          running_prod = tf.einsum('{0}ac,{0}aijb,{0}cijd->{0}bd'.format(bs_str),
                                   running_prod, curr_core, curr_core)
        else:
          running_prod = tf.einsum('{0}ac,{0}aib,{0}cid->{0}bd'.format(bs_str),
                                   running_prod, curr_core, curr_core)

      return tf.squeeze(running_prod, [-1, -2])

    else:
      orth_tt = decompositions.orthogonalize_tt_cores(tt, left_to_right=True)
      # All the cores of orth_tt except the last one are orthogonal, hence
      # the Frobenius norm of orth_tt equals to the norm of the last core.
      if hasattr(tt, 'batch_size'):
        batch_size = shapes.lazy_batch_size(tt)
        last_core = tf.reshape(orth_tt.tt_cores[-1], (batch_size, -1))
        return tf.norm(last_core, axis=1) ** 2
      else:
        return tf.norm(orth_tt.tt_cores[-1]) ** 2


def frobenius_norm(tt, epsilon=1e-5, differentiable=False,
                   name='t3f_frobenius_norm'):
  """Frobenius norm of `TensorTrain` or of each TT in `TensorTrainBatch`

  Frobenius norm is the sqrt of the sum of squares of all elements in a tensor.

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object
    epsilon: the function actually computes sqrt(norm_squared + epsilon) for
      numerical stability (e.g. gradient of sqrt at zero is inf).
    differentiable: bool, whether to use a differentiable implementation or
      a fast and stable implementation based on QR decomposition.
    name: string, name of the Op.

  Returns
    a number which is the Frobenius norm of `tt`, if it is `TensorTrain`
    OR
    a Tensor of size tt.batch_size, consisting of the Frobenius norms of
    each TensorTrain in `tt`, if it is `TensorTrainBatch`
  """
  with tf.name_scope(name):
    return tf.sqrt(frobenius_norm_squared(tt, differentiable) + epsilon)


def transpose(tt_matrix, name='t3f_transpose'):
  """Transpose a TT-matrix or a batch of TT-matrices.

  Args:
    tt_matrix: `TensorTrain` or `TensorTrainBatch` object containing a TT-matrix
      (or a batch of TT-matrices).
    name: string, name of the Op.

  Returns:
    `TensorTrain` or `TensorTrainBatch` object containing a transposed TT-matrix
      (or a batch of TT-matrices).

  Raises:
    ValueError if the argument is not a TT-matrix.
  """
  if not isinstance(tt_matrix, TensorTrainBase) or not tt_matrix.is_tt_matrix():
    raise ValueError('The argument should be a TT-matrix.')

  with tf.name_scope(name):
    transposed_tt_cores = []
    for core_idx in range(tt_matrix.ndims()):
      curr_core = tt_matrix.tt_cores[core_idx]
      if isinstance(tt_matrix, TensorTrain):
        transposed_tt_cores.append(tf.transpose(curr_core, (0, 2, 1, 3)))
      else:
        # TensorTrainBatch.
        transposed_tt_cores.append(tf.transpose(curr_core, (0, 1, 3, 2, 4)))

    tt_matrix_shape = tt_matrix.get_raw_shape()
    transposed_shape = tt_matrix_shape[1], tt_matrix_shape[0]
    tt_ranks = tt_matrix.get_tt_ranks()
    if isinstance(tt_matrix, TensorTrain):
      return TensorTrain(transposed_tt_cores, transposed_shape, tt_ranks)
    else:
      batch_size = tt_matrix.batch_size
      return TensorTrainBatch(transposed_tt_cores, transposed_shape, tt_ranks,
                              batch_size)


def quadratic_form(A, b, c, name='t3f_bilinear_form'):
  """Outdated, see `bilinear_form`."""
  print('Warning: function quadratic_form is being depricated and '
        'replaced with bilinear_form.')
  return bilinear_form(A, b, c)


def bilinear_form(A, b, c, name='t3f_bilinear_form'):
  """Bilinear form b^t A c; A is a TT-matrix, b and c can be batches.

  Args:
    A: `TensorTrain` object containing a TT-matrix of size N x M.
    b: `TensorTrain` object containing a TT-matrix of size N x 1
      or `TensorTrainBatch` with a batch of TT-matrices of size N x 1.
    c: `TensorTrain` object containing a TT-matrix of size M x 1
      or `TensorTrainBatch` with a batch of TT-matrices of size M x 1.
    name: string, name of the Op.

  Returns:
    A number, the value of the bilinear form if all the arguments are
      `TensorTrain`s.
    OR tf.Tensor of size batch_size if at least one of the arguments is
      `TensorTrainBatch`

  Raises:
    ValueError if the arguments are not TT-matrices or if the shapes are
      not consistent.

  Complexity:
       O(batch_size r_A r_c r_b n d (r_b + r_A n + r_c))
    d is the number of TT-cores (A.ndims());
    r_A is the largest TT-rank of A max(A.get_tt_rank())
    n is the size of the axis dimensions e.g.
      if b and c are tensors of shape (3, 3, 3),
      A is a 27 x 27 matrix of tensor shape (3, 3, 3) x (3, 3, 3)
      then n is 3
  """
  if not isinstance(A, TensorTrainBase) or not A.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')

  # TODO: support tf.Tensor as b and c.
  if not isinstance(b, TensorTrainBase) or not b.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')
  if not isinstance(c, TensorTrainBase) or not c.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')

  b_is_batch = isinstance(b, TensorTrainBatch)
  c_is_batch = isinstance(b, TensorTrainBatch)
  b_bs_str = 'p' if b_is_batch else ''
  c_bs_str = 'p' if c_is_batch else ''
  out_bs_str = 'p' if b_is_batch or c_is_batch else ''

  with tf.name_scope(name):
    ndims = A.ndims()
    curr_core_1 = b.tt_cores[0]
    curr_core_2 = c.tt_cores[0]
    curr_matrix_core = A.tt_cores[0]
    # We enumerate the dummy dimension (that takes 1 value) with `k`.
    # You may think that using two different k would be faster, but in my
    # experience it's even a little bit slower (but neglectable in general).
    einsum_str = '{0}aikb,cijd,{1}ejkf->{2}bdf'.format(b_bs_str, c_bs_str,
                                                       out_bs_str)
    res = tf.einsum(einsum_str, curr_core_1, curr_matrix_core, curr_core_2)
    for core_idx in range(1, ndims):
      curr_core_1 = b.tt_cores[core_idx]
      curr_core_2 = c.tt_cores[core_idx]
      curr_matrix_core = A.tt_cores[core_idx]
      einsum_str = '{2}ace,{0}aikb,cijd,{1}ejkf->{2}bdf'.format(b_bs_str,
                                                                c_bs_str,
                                                                out_bs_str)
      res = tf.einsum(einsum_str, res, curr_core_1,
                      curr_matrix_core, curr_core_2)

    # Squeeze to make the result a number instead of 1 x 1 for NON batch case
    # and to make the result a tensor of size
    #   batch_size
    # instead of
    #   batch_size x 1 x 1
    # in the batch case.
    return tf.squeeze(res)


def bilinear_form_two_mat(x, A, B, y, name='t3f_bilinear_xaby'):
  """Bilinear form x^t A B y; A are B are TT-matrices, x and y can be batches.

  Args:
    x: `TensorTrain` object containing a TT-matrix of size N x 1
      or `TensorTrainBatch` with a batch of TT-matrices of size N x 1.
    A: `TensorTrain` object containing a TT-matrix of size N x M.
    B: `TensorTrain` object containing a TT-matrix of size M x K.
    y: `TensorTrain` object containing a TT-matrix of size K x 1
      or `TensorTrainBatch` with a batch of TT-matrices of size K x 1.
    name: string, name of the Op.
  Returns:
    A number, the value of the bilinear form if all the arguments are
      `TensorTrain`s.
    OR tf.Tensor of size batch_size if at least one of the arguments is
      `TensorTrainBatch`
  Raises:
    ValueError if the arguments are not TT-matrices or if the shapes are
      not consistent.
  """
  for matrix in [A, B]:
    if not isinstance(matrix, TensorTrainBase) or not matrix.is_tt_matrix():
      raise ValueError('The arguments should be a TT-matrix.')

  # TODO: support tf.Tensor as x and y.
  for vec in [x, y]:
    if not isinstance(vec, TensorTrainBase) or not vec.is_tt_matrix():
      raise ValueError('The arguments should be a TT-matrix.')

  x_is_batch = isinstance(x, TensorTrainBatch)
  y_is_batch = isinstance(x, TensorTrainBatch)
  x_bs_str = 'p' if x_is_batch else ''
  y_bs_str = 'p' if y_is_batch else ''
  out_bs_str = 'p' if x_is_batch or y_is_batch else ''
  all_cores = x.tt_cores + A.tt_cores + B.tt_cores + y.tt_cores
  with tf.name_scope(name):
    ndims = A.ndims()
    curr_core_1 = x.tt_cores[0]
    curr_core_2 = y.tt_cores[0]
    curr_matrix_core_1 = A.tt_cores[0]
    curr_matrix_core_2 = B.tt_cores[0]
    # We enumerate the dummy dimension (that takes 1 value) with `k`.
    # You may think that using two different k would be faster, but in my
    # experience it's even a little bit slower (but neglectable in general).
    einsum_str = '{0}elnf,glph,ipoj,{1}aomb->{2}fhjb'.format(x_bs_str, y_bs_str,
                                                             out_bs_str)
    res = tf.einsum(einsum_str, curr_core_1, curr_matrix_core_1, curr_matrix_core_2,
                    curr_core_2)
    for core_idx in range(1, ndims):
      curr_core_1 = x.tt_cores[core_idx]
      curr_core_2 = y.tt_cores[core_idx]
      curr_matrix_core_1 = A.tt_cores[core_idx]
      curr_matrix_core_2 = B.tt_cores[core_idx]
      einsum_str = '{2}egia,{0}elnf,glph,ipoj,{1}aomb->{2}fhjb'.format(x_bs_str,
                                                                y_bs_str,
                                                                out_bs_str)
      res = tf.einsum(einsum_str, res, curr_core_1,
                      curr_matrix_core_1, curr_matrix_core_2,
                      curr_core_2)

    # Squeeze to make the result a number instead of 1 x 1 for NON batch case
    # and to make the result a tensor of size
    #   batch_size
    # instead of
    #   batch_size x 1 x 1
    # in the batch case.
    return tf.squeeze(res)


def cast(tt, dtype, name='t3f_cast'):
  """Casts a tt-tensor to a new type.

  Args:
    tt: `TensorTrain` object.
    dtype: The destination type.
    name: string, name of the Op.

  Raises:
    TypeError: If `tt` cannot be cast to the `dtype`.
    ValueError: If `tt` is not a `TensorTrain` or `TensorTrainBatch`.
  """
  with tf.name_scope(name):
    res_cores = []
    cores = tt.tt_cores
    for core_idx in range(tt.ndims()):
      res_cores.append(tf.cast(cores[core_idx], dtype))
    res_shape = tt.get_raw_shape()
    res_ranks = tt.get_tt_ranks()
    if isinstance(tt, TensorTrain):
      return TensorTrain(res_cores, res_shape, res_ranks)
    elif isinstance(tt, TensorTrainBatch):
      return TensorTrainBatch(res_cores, res_shape, res_ranks, tt.batch_size)
    else:
      raise ValueError('Unsupported type of input "%s", should be TensorTrain '
                       'or TensorTrainBatch.' % tt)


def gather_nd(tt, indices, name='t3f_gather_nd'):
  """out[i] = tt[indices[i, 0], indices[i, 1], ...]

  Equivalent to
      tf.gather_nd(t3f.full(tt), indices)
    but much faster, since it does not materialize the full tensor.

  For batches of TT works indices should include the batch dimension as well.

  Args:
    tt: `TensorTrain` or `TensorTrainBatch` object representing a tensor
      (TT-matrices are not implemented yet)
    indices: numpy array, tf.Tensor, placeholder with 2 or more dimensions.
      The last dimension indices.shape[-1] should be equal to the numbers of
      dimensions in TT:
        indices.shape[-1] = tt.ndims for `TensorTrain`
        indices.shape[-1] = tt.ndims + 1 for `TensorTrainBatch`
    name: string, name of the Op.

  Returns:
    tf.Tensor with elements specified by indices.

  Raises:
    ValueError if `indices` have wrong shape.
    NotImplementedError if `tt` is a TT-matrix.
  """
  with tf.name_scope(name):
    if tt.is_tt_matrix():
      raise NotImplementedError('gather_nd doesnt support TT-matrices yet '
                                '(got %s)' % tt)
    indices = tf.convert_to_tensor(indices)
    if isinstance(tt, TensorTrainBatch):
      if indices.get_shape()[-1] != tt.ndims() + 1:
        raise ValueError('The last dimension of indices (%d) should have '
                         'the same size as the number of dimensions in the tt '
                         'object (%d) + 1 (for the batch dimension).' %
                         (indices.get_shape()[-1], tt.ndims()))
    else:
      if indices.get_shape()[-1] != tt.ndims():
        raise ValueError('The last dimension of indices (%d) should have '
                         'the same size as the number of dimensions in the tt '
                         'object (%d).' % (indices.get_shape()[-1], tt.ndims()))
    tt_elements = tf.ones(tf.shape(indices)[:-1], dtype=tt.dtype)
    tt_elements = tf.reshape(tt_elements, (-1, 1, 1))
    for core_idx in range(tt.ndims()):
      curr_core = tt.tt_cores[core_idx]
      if isinstance(tt, TensorTrainBatch):
        curr_core = tf.transpose(curr_core, (0, 2, 1, 3))
        curr_idx = tf.stack((indices[:, 0], indices[:, core_idx + 1]), axis=1)
        core_slices = tf.gather_nd(curr_core, curr_idx)
      else:
        curr_core = tf.transpose(curr_core, (1, 0, 2))
        core_slices = tf.gather(curr_core, indices[:, core_idx])
      tt_elements = tf.matmul(tt_elements, core_slices)
    tt_elements = tf.reshape(tt_elements, tf.shape(indices)[:-1])
    return tt_elements


def renormalize_tt_cores(tt, epsilon=1e-8, name='t3f_renormalize_tt_cores'):
    """Renormalizes TT-cores to make them of the same Frobenius norm.

    Doesn't change the tensor represented by `tt` object, but renormalizes the
    TT-cores to make further computations more stable.

    Args:
      tt: `TensorTrain` or `TensorTrainBatch` object
      epsilon: parameter for numerical stability of sqrt
      name: string, name of the Op.

    Returns:
      `TensorTrain` or `TensorTrainBatch` which represents the same
      tensor as tt, but with all cores having equal norm. In the batch
      case applies to each TT in `TensorTrainBatch`.
    """
    # TODO: bad way to check if batch or not.
    with tf.name_scope(name):
      epsilon = tf.convert_to_tensor(epsilon, dtype=tt.dtype)
      if isinstance(tt, TensorTrain):
        new_cores = []
        running_log_norm = 0
        core_norms = []
        for core in tt.tt_cores:
          cur_core_norm = tf.sqrt(tf.maximum(tf.reduce_sum(core ** 2), epsilon))
          core_norms.append(cur_core_norm)
          running_log_norm += tf.math.log(cur_core_norm)

        running_log_norm = running_log_norm / tt.ndims()
        fact = tf.exp(running_log_norm)
        for i, core in enumerate(tt.tt_cores):
          new_cores.append(core * fact / core_norms[i])

        return TensorTrain(new_cores)
      else:
        sz = (tt.batch_size,) + (len(tt.tt_cores[0].shape) - 1) * (1,)
        running_core_log_norms = tf.zeros(sz, dtype=tt.dtype)
        ax = np.arange(len(tt.tt_cores[0].shape))[1:]
        fact_list = []
        for core in tt.tt_cores:
          cur_core_norm_sq = tf.reduce_sum(core**2, axis=ax, keepdims=True)
          cur_core_norm = tf.sqrt(tf.maximum(epsilon, cur_core_norm_sq))
          fact_list.append(cur_core_norm)
          running_core_log_norms += tf.math.log(cur_core_norm)

        new_cores = []
        exp_fact = tf.exp(running_core_log_norms / tt.ndims())
        for i, core in enumerate(tt.tt_cores):
          new_cores.append(tf.multiply(core, exp_fact / fact_list[i]))

        return TensorTrainBatch(new_cores)
