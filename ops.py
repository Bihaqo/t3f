import tensorflow as tf

from tensor_train import TensorTrain
import shapes
import utils


# TODO: add complexities to the comments.

def full(tt):
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
    res = tf.reshape(res, tf.TensorShape(intermediate_shape))
    transpose = []
    for i in range(0, 2 * num_dims, 2):
      transpose.append(i)
    for i in range(1, 2 * num_dims, 2):
      transpose.append(i)
    res = tf.transpose(res, transpose)
    return tf.reshape(res, shape)
  else:
    return tf.reshape(res, shape)


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
  a_shape = shapes.lazy_raw_shape(tt_matrix_a)
  a_ranks = shapes.lazy_tt_ranks(tt_matrix_a)
  b_shape = shapes.lazy_raw_shape(tt_matrix_b)
  b_ranks = shapes.lazy_tt_ranks(tt_matrix_b)
  for core_idx in range(ndims):
    a_core = tt_matrix_a.tt_cores[core_idx]
    b_core = tt_matrix_b.tt_cores[core_idx]
    curr_res_core = tf.einsum('aijb,cjkd->acikbd', a_core, b_core)

    res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
    res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
    left_mode = a_shape[0][core_idx]
    right_mode = b_shape[1][core_idx]
    core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
    curr_res_core = tf.reshape(curr_res_core, core_shape)
    result_cores.append(curr_res_core)

  res_shape = (tt_matrix_a.get_raw_shape()[0], tt_matrix_b.get_raw_shape()[1])
  static_a_ranks = tt_matrix_a.get_tt_ranks()
  static_b_ranks = tt_matrix_b.get_tt_ranks()
  out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]
  return TensorTrain(result_cores, res_shape, out_ranks)


def tt_dense_matmul(tt_matrix_a, matrix_b):
  """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.

  Args:
    tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
    matrix_b: tf.Tensor of size N x P

  Returns
    tf.Tensor of size M x P
  """
  if not isinstance(tt_matrix_a, TensorTrain):
    raise ValueError('The first argument should be a TT-matrix')

  ndims = tt_matrix_a.ndims()
  a_columns = tt_matrix_a.get_shape()[1].value
  b_rows = matrix_b.get_shape()[0].value
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


def matmul(a, b):
  """Multiplies two matrices that can be TT-, dense, or sparse.

  Note that multiplication of two TT-matrices returns a TT-matrix with much
  larger ranks.

  Args:
    a: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size M x N
    b: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size N x P

  Returns
    If both arguments are `TensorTrain` objects, returns a `TensorTrain`
      object containing a TT-matrix of size M x P
    If not, returns tf.Tensor of size M x P
  """
#   TODO: is it safe to check types? What if a class is derived from TT?
  if isinstance(a, TensorTrain) and isinstance(b, TensorTrain):
    return tt_tt_matmul(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.Tensor):
    return tt_dense_matmul(a, b)
  elif isinstance(a, tf.Tensor) and isinstance(b, TensorTrain):
    return dense_tt_matmul(a, b)
  elif isinstance(a, TensorTrain) and isinstance(b, tf.SparseTensor):
    return tt_sparse_matmul(a, b)
  elif isinstance(a, tf.SparseTensor) and isinstance(b, TensorTrain):
    return sparse_tt_matmul(a, b)


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
  if sparse_b.indices.get_shape().is_fully_defined():
    num_elements = sparse_b.indices.get_shape()[0]
  else:
    num_elements = tf.shape(sparse_b.indices)[0]
  tt_a_elements = tf.ones((num_elements, 1, 1))
  a_shape = shapes.lazy_raw_shape(tt_a)
  a_ranks = shapes.lazy_tt_ranks(tt_a)
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
      left_rank = a_ranks[core_idx]
      right_rank = a_ranks[core_idx + 1]
      curr_core = tf.transpose(curr_core, (1, 2, 0, 3))
      curr_core_shape = (a_shape[0][core_idx]*a_shape[1][core_idx], left_rank,
                         right_rank)
      curr_core = tf.reshape(curr_core, curr_core_shape)
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


def add(tt_a, tt_b):
  """Returns a TensorTrain corresponding to elementwise sum tt_a + tt_b.

  The shapes of tt_a and tt_b should coincide.

  Args:
    tt_a: `TensorTrain`, TT-tensor or TT-matrix
    tt_b: `TensorTrain`, TT-tensor or TT-matrix

  Returns
    a `TensorTrain` object corresponding to the element-wise sum of arguments.

  Raises
    ValueError if the arguments shapes do not coincide
  """
  ndims = tt_a.ndims()
  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('The arguments should be both TT-tensors or both '
                     'TT-matrices')

  if tt_a.get_shape() != tt_b.get_shape():
    raise ValueError('The arguments should have the same shape.')

  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)

  # If tt_a.get_shape() is fully defined, it means that even in the TT-matrix
  # case all dimensions are defined.
  if tt_a.get_shape().is_fully_defined:
    shape = [s.as_list() for s in tt_a.get_raw_shape()]
  else:
    shape = shapes.raw_shape(tt_a)

  is_matrix = tt_a.is_tt_matrix()
  last_axis = 3 if is_matrix else 2
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    if core_idx == 0:
      curr_core = tf.concat((a_core, b_core), axis=last_axis)
    elif core_idx == ndims - 1:
      curr_core = tf.concat((a_core, b_core), axis=0)
    else:
      if is_matrix:
        upper_zeros = tf.zeros((a_ranks[core_idx], shape[0][core_idx],
                                shape[1][core_idx], b_ranks[core_idx + 1]))
        lower_zeros = tf.zeros((b_ranks[core_idx], shape[0][core_idx],
                                shape[1][core_idx], a_ranks[core_idx + 1]))
      else:
        upper_zeros = tf.zeros((a_ranks[core_idx], shape[0][core_idx],
                                b_ranks[core_idx + 1]))
        lower_zeros = tf.zeros((b_ranks[core_idx], shape[0][core_idx],
                                a_ranks[core_idx + 1]))
      upper = tf.concat((a_core, upper_zeros), axis=last_axis)
      lower = tf.concat((lower_zeros, b_core), axis=last_axis)
      curr_core = tf.concat((upper, lower), axis=0)
    tt_cores.append(curr_core)

  out_ranks = [1]
  static_a_ranks = tt_a.get_tt_ranks()
  static_b_ranks = tt_b.get_tt_ranks()
  for core_idx in range(1, ndims):
    out_ranks.append(static_a_ranks[core_idx] + static_b_ranks[core_idx])
  out_ranks.append(1)
  return TensorTrain(tt_cores, tt_a.get_raw_shape(), out_ranks)


def multiply(tt_a, tt_b):
  """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b.

  The shapes of tt_a and tt_b should coincide.

  Args:
    tt_a: `TensorTrain`, TT-tensor or TT-matrix
    tt_b: `TensorTrain`, TT-tensor or TT-matrix

  Returns
    a `TensorTrain` object corresponding to the element-wise product of the
    arguments.

  Raises
    ValueError if the arguments shapes do not coincide.
  """
  ndims = tt_a.ndims()
  if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
    raise ValueError('The arguments should be both TT-tensors or both '
                     'TT-matrices')

  if tt_a.get_shape() != tt_b.get_shape():
    raise ValueError('The arguments should have the same shape.')

  a_ranks = shapes.lazy_tt_ranks(tt_a)
  b_ranks = shapes.lazy_tt_ranks(tt_b)
  shape = shapes.lazy_raw_shape(tt_a)

  is_matrix = tt_a.is_tt_matrix()
  tt_cores = []
  for core_idx in range(ndims):
    a_core = tt_a.tt_cores[core_idx]
    b_core = tt_b.tt_cores[core_idx]
    left_rank = a_ranks[core_idx] * b_ranks[core_idx]
    right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
    if is_matrix:
      curr_core = tf.einsum('aijb,cijd->acijbd', a_core, b_core)
      curr_core = tf.reshape(curr_core, (left_rank, shape[0][core_idx],
                                         shape[1][core_idx], right_rank))
    else:
      curr_core = tf.einsum('aib,cid->acibd', a_core, b_core)
      curr_core = tf.reshape(curr_core, (left_rank, shape[0][core_idx],
                                         right_rank))
    tt_cores.append(curr_core)

  out_ranks = [a * b for a, b in zip(tt_a.get_tt_ranks(), tt_b.get_tt_ranks())]
  return TensorTrain(tt_cores, tt_a.get_raw_shape(), out_ranks)


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


def frobenius_norm(tt, epsilon=1e-5):
  """Frobenius norm of a TensorTrain (sqrt of the sum of squares of all elements).

  Args:
    tt: `TensorTrain` object
    epsilon: the function actually computes sqrt(norm_squared + epsilon) for
      numerical stability (e.g. gradient of sqrt at zero is inf).

  Returns
    a number
    sqrt of the sum of squares of all elements in `tt`
  """
  return tf.sqrt(frobenius_norm_squared(tt) + epsilon)


def transpose(tt_matrix):
  """Transpose a TT-matrix.

  Args:
    tt_matrix: `TensorTrain` object containing a TT-matrix.

  Returns:
    `TensorTrain` object containing a transposed TT-matrix.

  Raises:
    ValueError if the argument is not a TT-matrix.
  """
  if not isinstance(tt_matrix, TensorTrain) or not tt_matrix.is_tt_matrix():
    raise ValueError('The argument should be a TT-matrix.')

  transposed_tt_cores = []
  for core_idx in range(tt_matrix.ndims()):
    curr_core = tt_matrix.tt_cores[core_idx]
    transposed_tt_cores.append(tf.transpose(curr_core, (0, 2, 1, 3)))

  tt_matrix_shape = tt_matrix.get_raw_shape()
  transposed_shape = tt_matrix_shape[1], tt_matrix_shape[0]
  tt_ranks = tt_matrix.get_tt_ranks()
  return TensorTrain(transposed_tt_cores, transposed_shape, tt_ranks)


def quadratic_form(A, b, c):
  """Computes the quadratic form b^t A c where A is a TT-matrix.

  Args:
    A: `TensorTrain` object containing a TT-matrix.
    b: `TensorTrain` object containing a TT-vector.
    c: `TensorTrain` object containing a TT-vector.

  Returns:
    A number, the value of the quadratic form.

  Raises:
    ValueError if the argument is not a TT-matrix or if the shapes are
      not consistent.
  """
  if not isinstance(A, TensorTrain) or not A.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')

  # TODO: support tf.Tensor as b and c.
  if not isinstance(b, TensorTrain) or not b.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')
  if not isinstance(c, TensorTrain) or not c.is_tt_matrix():
    raise ValueError('The arguments should be a TT-matrix.')

  # TODO: make a more efficient implementation taylored for this case.
  return tt_tt_flat_inner(A, tt_tt_matmul(b, transpose(c)))


def cast(tt_a, dtype):
  """Casts a tt-tensor to a new type.

  Args:
    tt_a: `TensorTrain` object.
    dtype: The destination type. 

  Raises:
    TypeError: If `tt_a` cannot be cast to the `dtype`.
    ValueError: If `tt_a` is not a `TensorTrain`
  """

  if not isinstance(tt_a, TensorTrain):
    raise ValueError('Argument should be a TensorTrain')

  res_cores = []
  cores = tt_a.tt_cores
  for core_idx in range(tt_a.ndims()):
    res_cores.append(tf.cast(cores[core_idx], dtype))
  res_shape = tt_a.get_raw_shape()
  res_ranks = tt_a.get_tt_ranks()
  return TensorTrain(res_cores, res_shape, res_ranks)
