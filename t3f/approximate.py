import itertools
import numpy as np
import tensorflow as tf
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import decompositions
from t3f import batch_ops


def add_n(tt_objects, max_tt_rank, name='t3f_approximate_add_n'):
  """Adds a bunch of TT-object and round after each summation.

  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.
  I.e. it uses the following idea:
    round(a + b + c + d) ~= round(round(a + b) + round(c + d))
  and so is able to compute the answer in log(N) parallel adds/rounds.

  Args:
    tt_objects: a list of `TensorTrainBase` objects.
    max_tt_rank: a number, TT-rank for each individual rounding.
    name: string, name of the Op.

  Returns:
    Object of the same type as each input.
  
  See Also:
    t3f.approximate.reduce_sum_batch
  """
  list_of_cores_lists = [tt.tt_cores for tt in tt_objects]
  all_cores = tuple(itertools.chain.from_iterable(list_of_cores_lists))
  with tf.name_scope(name):
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


def reduce_sum_batch(tt_batch, max_tt_rank, coef=None,
                     name='t3f_approximate_reduce_sum_batch'):
  """Sum of all TT-objects in the batch with rounding after each summation.
  
  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.
  I.e. it uses the following idea:
    round(a + b + c + d) ~= round(round(a + b) + round(c + d))
  and so is able to compute the answer in log(batch_size) parallel adds/rounds.

  Args:
    tt_batch: `TensorTrainBatch` object.
    max_tt_rank: a number, TT-rank for each individual rounding.
    coef: tf.Tensor, its shape is either batch_size, or batch_size x N.
      If coef is a vecotor of size batch_size, the result will
        be (approximate) weighted sum.
      If coef is a matrix of shape batch_size x N, the result will be
        a `TensorTrainBatch` res containing N TT-object such that
          res[j] ~= sum_i tt_batch[i] coef[i, j]
    name: string, name of the Op.

  Returns:
    If coefficients are absent or is a vector of numbers, returns
      a `TensorTrain` object representing (approximate) element-wise sum of all
      the objects in the batch, weighted if coef is provided.
    If coefficients is a matrix, returns `TensorTrainBatch`.

  See Also:
    t3f.approximate.add_n
  """
  ndims = tt_batch.ndims()
  left_tt_rank_dim = tt_batch.left_tt_rank_dim
  right_tt_rank_dim = tt_batch.right_tt_rank_dim
  shape = tt_batch.get_raw_shape()
  dtype = tt_batch.dtype

  all_tensors = tt_batch.tt_cores
  if coef is not None:
    all_tensors += (coef, )
  with tf.name_scope(name):
    is_batch_output = False
    if coef is not None:
      coef = tf.convert_to_tensor(coef, dtype=tt_batch.dtype)
      if len(coef.get_shape()) == 1:
        tt_batch = batch_ops.multiply_along_batch_dim(tt_batch, coef)
      elif len(coef.get_shape()) == 2:
        is_batch_output = True
        output_size = coef.get_shape().as_list()[1]
        # Coef is of size batch_size x N, need to duplicate the batch
        # dimension xN.
        if coef.shape[0] != tt_batch.batch_size:
          raise ValueError('If coef is a matrix, it should be of shape '
                           'batch_size x N, got %d x %d instead '
                           '(batch size is %d).' % (coef.shape[0], coef.shape[1],
                                                    tt_batch.batch_size))
        tt_batch_cores = []
        for core_idx in range(ndims):
          curr_core = tt_batch.tt_cores[core_idx]
          curr_shape = curr_core.get_shape().as_list()
          new_shape = np.insert(curr_shape, 1, 1)
          tiling = np.ones(len(new_shape), dtype=int)
          tiling[1] = output_size
          curr_core = tf.tile(tf.reshape(curr_core, new_shape), tiling)
          if core_idx == 0:
            # Multiply the first TT-core by the provided coefficients.
            # TODO: add t3f.utils.expands_dims_like(coef, curr_core)
            shaped_coef = coef
            for _ in range(len(curr_core.get_shape()) - len(coef.shape)):
              shaped_coef = tf.expand_dims(shaped_coef, -1)
            curr_core = curr_core * shaped_coef
          # Merge the first two dimensions back into one.
          raveled_shape = np.array(curr_shape).copy()
          raveled_shape[0] *= output_size
          curr_core = tf.reshape(curr_core, raveled_shape)
          tt_batch_cores.append(curr_core)
        tt_batch = TensorTrainBatch(tt_batch_cores, shape,
                                    tt_batch.get_tt_ranks())

      else:
        raise ValueError('Coef cannot be more than 2-d.')

    if not is_batch_output:
      output_size = 1

    prev_level = tt_batch
    while prev_level.batch_size > output_size:
      current_level_cores = []
      for core_idx in range(ndims):
        curr_orig_core = prev_level.tt_cores[core_idx]
        if is_batch_output:
          # Split the first dimension into batch_size x N
          unraveled_shape = curr_orig_core.get_shape().as_list()
          unraveled_shape = np.array(unraveled_shape).copy()
          unraveled_shape[0] /= output_size
          unraveled_shape = np.insert(unraveled_shape, 1, output_size)
          curr_orig_core = tf.reshape(curr_orig_core, unraveled_shape)

        a_core = curr_orig_core[::2]
        b_core = curr_orig_core[1::2]

        if a_core.get_shape()[0] > b_core.get_shape()[0]:
          # Odd number of elements in the batch, will have to add dummy
          # TT-object with the tt-cores filled with zeros.
          zeros_shape = b_core.get_shape().as_list()
          zeros_shape[0] = 1
          zeros = tf.zeros(zeros_shape, dtype)
          b_core = tf.concat((b_core, zeros), axis=0)

        if is_batch_output:
          # Merge the first two dimensions back into one.
          a_core_shape = a_core.get_shape().as_list()
          a_core_shape[0] = a_core_shape[0] * a_core_shape[1]
          a_core_shape = np.delete(a_core_shape, 1)
          a_core = tf.reshape(a_core, a_core_shape)
          b_core_shape = b_core.get_shape().as_list()
          b_core_shape[0] = b_core_shape[0] * b_core_shape[1]
          b_core_shape = np.delete(b_core_shape, 1)
          b_core = tf.reshape(b_core, b_core_shape)

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
    if is_batch_output:
      return prev_level
    else:
      return prev_level[0]
