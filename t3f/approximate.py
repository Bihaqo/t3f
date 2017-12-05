import tensorflow as tf
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import decompositions


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


def reduce_sum_batch(tt_batch, max_tt_rank):
  """Sum of all TT-objects in the batch with rounding after each summation.
  
  
  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.

  Args:
    tt_batch: `TensorTrainBatch` object.
    max_tt_rank: a number, TT-rank for each individual rounding.

  Returns:
    A `TensorTrain` object representing element-wise sum of all the objects in
    the batch.
  """
  ndims = tt_batch.ndims()
  left_tt_rank_dim = tt_batch.left_tt_rank_dim
  right_tt_rank_dim = tt_batch.right_tt_rank_dim
  tt_ranks = tt_batch.get_tt_ranks()
  shape = tt_batch.get_shape().as_list()
  dtype = tt_batch.dtype

  prev_level = tt_batch
  while prev_level.batch_size > 1:
    current_level_cores = []
    for core_idx in range(ndims):
      curr_orig_core = prev_level.tt_cores[core_idx]
      a_core = curr_orig_core[::2]
      b_core = curr_orig_core[1::2]
      if a_core.get_shape()[0] < b_core.get_shape()[0]:
        # Not even number of elements in the batch, will have to add dummy
        # TT-object with the tt-cores filled with zeros.
        zeros = tf.zeros((1, b_core.get_shape().as_list()[1:]))
        b_core = tf.concat((b_core, zeros), axis=0)

      if core_idx == 0:
        curr_sum_core = tf.concat((a_core, b_core), axis=right_tt_rank_dim)
      elif core_idx == ndims - 1:
        curr_sum_core = tf.concat((a_core, b_core), axis=left_tt_rank_dim)
      else:
        upper_zeros = tf.zeros((tt_ranks[core_idx], shape[0][core_idx],
                                shape[1][core_idx], tt_ranks[core_idx + 1]),
                               dtype)
        lower_zeros = tf.zeros((tt_ranks[core_idx], shape[0][core_idx],
                                shape[1][core_idx], tt_ranks[core_idx + 1]),
                               dtype)
        upper = tf.concat((a_core, upper_zeros), axis=right_tt_rank_dim)
        lower = tf.concat((lower_zeros, b_core), axis=right_tt_rank_dim)
        curr_sum_core = tf.concat((upper, lower), axis=left_tt_rank_dim)
      current_level_cores.append(curr_sum_core)
    current_level = TensorTrainBatch(current_level, shape, tt_ranks)
    prev_level = decompositions.round(current_level, max_tt_rank)
  return prev_level[0]
