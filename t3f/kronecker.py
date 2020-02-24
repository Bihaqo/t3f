import tensorflow as tf

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import ops


def determinant(kron_a, name='t3f_kronecker_determinant'):
  """Computes the determinant of a given Kronecker-factorized matrix. 

  Note, that this method can suffer from overflow.

  Args:
    kron_a: `TensorTrain` or `TensorTrainBatch` object containing a matrix or a
      batch of matrices of size N x N, factorized into a Kronecker product of 
      square matrices (all tt-ranks are 1 and all tt-cores are square). 
    name: string, name of the Op.
  
  Returns:
    A number or a Tensor with numbers for each element in the batch.
    The determinant of the given matrix.

  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1.
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product (tt-ranks '
                     'should be 1)')

  shapes_defined = kron_a.get_shape().is_fully_defined()
  if shapes_defined:
    i_shapes = kron_a.get_raw_shape()[0].as_list()
    j_shapes = kron_a.get_raw_shape()[1].as_list()
  else:
    i_shapes = ops.raw_shape(kron_a)[0].as_list()
    j_shapes = ops.raw_shape(kron_a)[1].as_list()

  if shapes_defined:
    if i_shapes != j_shapes:
      raise ValueError('The argument should be a Kronecker product of square '
                       'matrices (tt-cores must be square)')
      
  is_batch = isinstance(kron_a, TensorTrainBatch)
  with tf.name_scope(name):
    pows = tf.cast(tf.reduce_prod(i_shapes), kron_a.dtype)
    cores = kron_a.tt_cores
    det = 1
    for core_idx in range(kron_a.ndims()):
      core = cores[core_idx]
      if is_batch:
        core_det = tf.linalg.det(core[:, 0, :, :, 0])
      else:
        core_det = tf.linalg.det(core[0, :, :, 0])
      core_pow = pows / i_shapes[core_idx]

      det *= tf.pow(core_det, core_pow)
    return det


def slog_determinant(kron_a, name='t3f_kronecker_slog_determinant'):
  """Computes the sign and log-det of a given Kronecker-factorized matrix.

  Args:
    kron_a: `TensorTrain` or `TensorTrainBatch` object containing a matrix or a
      batch of matrices of size N x N, factorized into a Kronecker product of 
      square matrices (all tt-ranks are 1 and all tt-cores are square). 
    name: string, name of the Op.
  
  Returns:
    Two number or two Tensor with numbers for each element in the batch.
    Sign of the determinant and the log-determinant of the given 
    matrix. If the determinant is zero, then sign will be 0 and logdet will be
    -Inf. In all cases, the determinant is equal to sign * np.exp(logdet).

  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1.
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product ' 
                     '(tt-ranks should be 1)')
 
  shapes_defined = kron_a.get_shape().is_fully_defined()
  if shapes_defined:
    i_shapes = kron_a.get_raw_shape()[0].as_list()
    j_shapes = kron_a.get_raw_shape()[1].as_list()
  else:
    i_shapes = ops.raw_shape(kron_a)[0].as_list()
    j_shapes = ops.raw_shape(kron_a)[1].as_list()

  if shapes_defined:
    if i_shapes != j_shapes:
      raise ValueError('The argument should be a Kronecker product of square '
                       'matrices (tt-cores must be square)')

  is_batch = isinstance(kron_a, TensorTrainBatch)
  with tf.name_scope(name):
    pows = tf.cast(tf.reduce_prod(i_shapes), kron_a.dtype)
    logdet = 0.
    det_sign = 1.

    for core_idx in range(kron_a.ndims()):
      core = kron_a.tt_cores[core_idx]
      if is_batch:
        core_det = tf.linalg.det(core[:, 0, :, :, 0])
      else:
        core_det = tf.linalg.det(core[0, :, :, 0])
      core_abs_det = tf.abs(core_det)
      core_det_sign = tf.sign(core_det)
      core_pow = pows / i_shapes[core_idx]
      logdet += tf.math.log(core_abs_det) * core_pow
      det_sign *= core_det_sign**(core_pow)
    return det_sign, logdet


def inv(kron_a, name='t3f_kronecker_inv'):
  """Computes the inverse of a given Kronecker-factorized matrix.

  Args:
    kron_a: `TensorTrain` or `TensorTrainBatch` object containing a matrix or a
      batch of matrices of size N x N, factorized into a Kronecker product of 
      square matrices (all tt-ranks are 1 and all tt-cores are square). 
    name: string, name of the Op.

  Returns:
    `TensorTrain` object containing a TT-matrix of size N x N if the argument is
      `TensorTrain`
    `TensorTrainBatch` object, containing TT-matrices of size N x N if the 
      argument is `TensorTrainBatch`  
  
  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1.
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product ' 
                     '(tt-ranks should be 1)')
    
  shapes_defined = kron_a.get_shape().is_fully_defined()
  if shapes_defined:
    i_shapes = kron_a.get_raw_shape()[0]
    j_shapes = kron_a.get_raw_shape()[1]
  else:
    i_shapes = ops.raw_shape(kron_a)[0]
    j_shapes = ops.raw_shape(kron_a)[1]

  if shapes_defined:
    if i_shapes != j_shapes:
      raise ValueError('The argument should be a Kronecker product of square '
                       'matrices (tt-cores must be square)')

  is_batch = isinstance(kron_a, TensorTrainBatch)
  with tf.name_scope(name):
    inv_cores = []
    for core_idx in range(kron_a.ndims()):
      core = kron_a.tt_cores[core_idx]
      if is_batch:
        core_inv = tf.linalg.inv(core[:, 0, :, :, 0])
        core_inv = tf.expand_dims(tf.expand_dims(core_inv, 1), -1)
      else:
        core_inv = tf.linalg.inv(core[0, :, :, 0])
        core_inv = tf.expand_dims(tf.expand_dims(core_inv, 0), -1)
      inv_cores.append(core_inv)

    res_ranks = kron_a.get_tt_ranks() 
    res_shape = kron_a.get_raw_shape()
    if is_batch:
      return TensorTrainBatch(inv_cores, res_shape, res_ranks) 
    else:
      return TensorTrain(inv_cores, res_shape, res_ranks) 


def cholesky(kron_a, name='t3f_kronecker_cholesky'):
  """Computes the Cholesky decomposition of a given Kronecker-factorized matrix.

  Args:
    kron_a: `TensorTrain` or `TensorTrainBatch` object containing a matrix or a
      batch of matrices of size N x N, factorized into a Kronecker product of 
      square matrices (all tt-ranks are 1 and all tt-cores are square). All the 
      cores must be symmetric positive-definite.
    name: string, name of the Op.

  Returns:
    `TensorTrain` object containing a TT-matrix of size N x N if the argument is
      `TensorTrain`
    `TensorTrainBatch` object, containing TT-matrices of size N x N if the 
      argument is `TensorTrainBatch`  
    
  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1.
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product ' 
                     '(tt-ranks should be 1)')
    
  shapes_defined = kron_a.get_shape().is_fully_defined()
  if shapes_defined:
    i_shapes = kron_a.get_raw_shape()[0]
    j_shapes = kron_a.get_raw_shape()[1]
  else:
    i_shapes = ops.raw_shape(kron_a)[0]
    j_shapes = ops.raw_shape(kron_a)[1]

  if shapes_defined:
    if i_shapes != j_shapes:
      raise ValueError('The argument should be a Kronecker product of square '
                       'matrices (tt-cores must be square)')

  is_batch = isinstance(kron_a, TensorTrainBatch)
  with tf.name_scope(name):
    cho_cores = []
    for core_idx in range(kron_a.ndims()):
      core = kron_a.tt_cores[core_idx]
      if is_batch:
        core_cho = tf.linalg.cholesky(core[:, 0, :, :, 0])
        core_cho = tf.expand_dims(tf.expand_dims(core_cho, 1), -1)
      else:
        core_cho = tf.linalg.cholesky(core[0, :, :, 0])
        core_cho = tf.expand_dims(tf.expand_dims(core_cho, 0), -1)
      cho_cores.append(core_cho)

    res_ranks = kron_a.get_tt_ranks()
    res_shape = kron_a.get_raw_shape()
    if is_batch:
      return TensorTrainBatch(cho_cores, res_shape, res_ranks)
    else:
      return TensorTrain(cho_cores, res_shape, res_ranks)


def _is_kron(tt_a):
  """Returns True if the argument is a Kronecker product matrix.

  Args:
    t_a: `TensorTrain` or `TensorTrainBatch` object.
    
  Returns:
    bool
  """
  if tt_a.is_tt_matrix():
    return max(tt_a.get_tt_ranks()) == 1
  return False    

