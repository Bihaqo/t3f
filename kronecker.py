import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain

def log_determinant(kron_a):
  """Computes the log-determinant of a given matrix, factorized into
  a Kronecker product of square matrices.

  Args:
    kron_a: `TensorTrain` object containing a matrix of size N x N, 
    factorized into a Kronecker product of square matrices (all 
    tt-ranks are 1 and all tt-cores are square). All the cores
    must have positive determinants.
  
  Returns:
    If the determinants of all cores are non-negative, returns the log-determinant 
  """
  raise NotImplementedError


