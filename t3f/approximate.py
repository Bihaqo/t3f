import tensorflow as tf
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import decompositions
from t3f import batch_ops


def add_n(tt_objects, max_tt_rank):
  """Adds a bunch of TT-object and round after each summation.

  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.

  Args:
    tt_objects: a list of `TensorTrainBase` objects.
    max_tt_rank: a number, TT-rank for each individual rounding.

  Returns:
    Object of the same type as each input.
  """
  prev_level = tt_objects
  while len(prev_level) > 1:
    next_level = []
    for i in range(0, len(prev_level), 2):
      curr = prev_level[i]
      if i + 1 < len(prev_level):
        curr = decompositions.round(curr + prev_level[i + 1], max_tt_rank)
      next_level.append(curr)
    prev_level = next_level
  return prev_level[0]


def reduce_sum_batch(tt_batch, max_tt_rank, coef=None):
  """Sum of all TT-objects in the batch with rounding after each summation.
  
  
  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.

  Args:
    tt_batch: `TensorTrainBatch` object.
    max_tt_rank: a number, TT-rank for each individual rounding.
    coef: tf.Tensor, its shape is either batch_size, or batch_size x N.
      If coef is a vecotor of size batch_size, the result will
      be (approximate) weighted sum.
      If coef is a matrix of shape batch_size x N, the result will be
      a `TensorTrainBatch` res such that
        res[j] ~= sum_i coef[i, j] tt_batch[i]

  Returns:
    If coefficients are absent or is a vector of numbers, returns
      a `TensorTrain` object representing (approximate) element-wise sum of all
      the objects in the batch, weighted if coef is provided.
    If coefficients is a matrix, returns `TensorTrainBatch`.
  """
  ndims = tt_batch.ndims()
  left_tt_rank_dim = tt_batch.left_tt_rank_dim
  right_tt_rank_dim = tt_batch.right_tt_rank_dim
  shape = tt_batch.get_raw_shape()
  dtype = tt_batch.dtype

  if coef is not None:
    coef = tf.convert_to_tensor(coef)
    assert len(coef.get_shape()) == 1
    tt_batch = batch_ops.multiply_along_batch_dim(tt_batch, coef)

  prev_level = tt_batch
  while prev_level.batch_size > 1:
    current_level_cores = []
    for core_idx in range(ndims):
      curr_orig_core = prev_level.tt_cores[core_idx]
      a_core = curr_orig_core[::2]
      b_core = curr_orig_core[1::2]
      if a_core.get_shape()[0] > b_core.get_shape()[0]:
        # Odd number of elements in the batch, will have to add dummy
        # TT-object with the tt-cores filled with zeros.
        zeros_shape = b_core.get_shape().as_list()
        zeros_shape[0] = 1
        zeros = tf.zeros(zeros_shape, dtype)
        b_core = tf.concat((b_core, zeros), axis=0)

      if core_idx == 0:
        curr_sum_core = tf.concat((a_core, b_core), axis=right_tt_rank_dim)
      elif core_idx == ndims - 1:
        curr_sum_core = tf.concat((a_core, b_core), axis=left_tt_rank_dim)
      else:
        zeros = tf.zeros(b_core.get_shape(), dtype)
        upper = tf.concat((a_core, zeros), axis=right_tt_rank_dim)
        lower = tf.concat((zeros, b_core), axis=right_tt_rank_dim)
        curr_sum_core = tf.concat((upper, lower), axis=left_tt_rank_dim)
      current_level_cores.append(curr_sum_core)
    current_level = TensorTrainBatch(current_level_cores, shape)
    prev_level = decompositions.round(current_level, max_tt_rank)
  return prev_level[0]
