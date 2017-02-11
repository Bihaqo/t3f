import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain

def determinant(kron_a):
  """Computes the determinant of a given Kronecker-factorized matrix  

  Note, that this method can suffer from overflow.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). 
  
  Returns:
    Number, the determinant of the given matrix

  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product (tt-ranks'
                     'should be 1)')

  cores = kron_a.tt_cores
  det, pows = 1, 1
  for core_idx in range(kron_a.ndims()):
    core = kron_a.tt_cores[core_idx]
#    if core.get_shape()[1] != core.get_shape()[2]:
#      raise ValueError('The argument should be a Kronecker product of square ' 
#                      'matrices (tt-cores must be square)')
    pows *= core.get_shape()[1].value
  for core_idx in range(kron_a.ndims()):
    core = cores[core_idx]
    core_det = tf.matrix_determinant(core[0, :, :, 0])
    core_pow = pows / core.get_shape()[1].value

    det *= tf.pow(core_det, core_pow)
  return det


def slog_determinant(kron_a):
  """Computes the sign and log-det of a given Kronecker-factorized matrix

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square)

  Returns:
    Two numbers, sign of the determinant and the log-determinant of the given 
    matrix

  Raises:
    ValueError if the tt-cores of the provided matrix are not square,
    or the tt-ranks are not 1
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product ' 
                     '(tt-ranks should be 1)')
 
  shapes_defined = kron_a.get_shape().is_fully_defined()
  if shapes_defined:
    i_shapes = kron_a.get_raw_shape()[0]
    j_shapes = kron_a.get_raw_shape()[1]
  else:
    i_shapes = raw_shape(kron_a)[0]

  if shapes_defined:
    if i_shapes != j_shapes:
      raise ValueError('The argument should be a Kronecker product of square '
                       'matrices (tt-cores must be square)')
  pows = tf.cast(tf.reduce_prod(i_shapes), kron_a.dtype)
                                                          
  logdet = 0.
  det_sign = 1.
  for core_idx in range(kron_a.ndims()):
    core = kron_a.tt_cores[core_idx]
    core_det = tf.matrix_determinant(core[0, :, :, 0])
    core_abs_det = tf.abs(core_det)
    core_det_sign = tf.sign(core_det)
    core_pow = pows / i_shapes[core_idx].value
    logdet += tf.log(core_abs_det) * core_pow
    det_sign *= core_det_sign**(core_pow)
  
  return det_sign, logdet

def inv(kron_a):
  """Computes the inverse of a given Kronecker-factorized matrix

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must be invertable

  Returns:
    `TensorTrain` object, containing a TT-matrix of size N x N    
  
  Raises:
    ValueError if the cores are not square or the ranks are not 1 
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product' 
                     '(tt-ranks should be 1)')

  inv_cores = []
  for core_idx in range(kron_a.ndims()):
    core = kron_a.tt_cores[core_idx]
    if core.get_shape()[1] != core.get_shape()[2]:
      raise ValueError('The argument should be a Kronecker product of square'
                      'matrices (tt-cores must be square)')
 
    core_inv = tf.matrix_inverse(core[0, :, :, 0])
    inv_cores.append(tf.expand_dims(tf.expand_dims(core_inv, 0), -1))

  res_ranks = tf.TensorShape([1] * (kron_a.ndims() + 1))
  res_shape = kron_a.get_raw_shape()
  return TensorTrain(inv_cores, res_shape, res_ranks) 

def cholesky(kron_a):
  """Computes the Cholesky decomposition of a given Kronecker-factorized matrix

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must be symmetric positive-definite

  Returns:
    `TensorTrain` object, containing a TT-matrix of size N x N    
  
  Raises:
    ValueError if the cores are not square or the ranks are not 1 
  """
  if not _is_kron(kron_a):
    raise ValueError('The argument should be a Kronecker product' 
                     '(tt-ranks should be 1)')

  cho_cores = []
  for core_idx in range(kron_a.ndims()):
    core = kron_a.tt_cores[core_idx]
    if core.get_shape()[1] != core.get_shape()[2]:
      raise ValueError('The argument should be a Kronecker product of square'
                      'matrices (tt-cores must be square)')
    core_cho = tf.cholesky(core[0, :, :, 0])
    cho_cores.append(tf.expand_dims(tf.expand_dims(core_cho, 0), -1))

  res_ranks = tf.TensorShape([1] * (kron_a.ndims() + 1))
  res_shape = kron_a.get_raw_shape()
  return TensorTrain(cho_cores, res_shape, res_ranks) 

def _is_kron(tt_a):
  """Returns True if the argument is a Kronecker product matrix

  Args:
    tt_a: `TensorTrain` object

  Returns:
    bool
  """
  if tt_a.is_tt_matrix():
    return max(tt_a.get_tt_ranks()) == 1
  return False    

