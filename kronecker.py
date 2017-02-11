import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain

def determinant(kron_a):
  """Computes the determinant of a given matrix, factorized into
  a Kronecker product of square matrices.
  
  Note, that this method can suffer from overflow.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). 
  Returns:
    Number, the determinant of the given matrix
  """
  raise NotImplementedError


def log_determinant(kron_a):
  """Computes the log-determinant of a given matrix, factorized into
  a Kronecker product of square matrices.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must have positive determinants
  
  Returns:
    Number, the log-determinant of the given matrix

  Raises:
    ValueError if the cores are not square, or there determinants
    are not positive
  """
  raise NotImplementedError


def inv(kron_a):
  """Computes the inverse of a given matrix, factorized into
  a Kronecker-product of square matrices.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must be invertable

  Returns:
    `TensorTrain` object, containing a TT-matrix of size N x N    
  """
  raise NotImplementedError  


def cholesky(kron_a):
  """Computes the Cholesky decomposition of a given matrix, factorized 
  into a Kronecker-product of symmetric positive-definite square matrices.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must be symmetric positive-definite

  Returns:
    `TensorTrain` object, containing a TT-matrix of size N x N    
  """
  raise NotImplementedError 


def _is_kron(tt_a):
  """Returns True if the argument is a Kronecker product matrix.

  Args:
    tt_a: `TensorTrain` object

  Returns:
    bool
  """
  if tt_a.is_tt_matrix():
    return max(tt_a.get_tt_ranks()) == 1
  return False    

