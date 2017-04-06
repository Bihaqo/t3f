import tensorflow as tf

import shapes
import decompositions
from tensor_train import TensorTrain
from tensor_train_batch import TensorTrainBatch


def project_sum(what, where, weights=None):
  """Project sum of `what` TTs on the tangent space of `where` TT.

  project_sum(what, x) = P_x(what)
  project_sum(batch_what, x) = P_x(\sum_i batch_what[i])
  project_sum(batch_what, x, weights) = P_x(\sum_j weights[j] * batch_what[j])

  This function implements the algorithm from the paper [1], theorem 3.1.

  [1] C. Lubich, I. Oseledets and B. Vandereycken, Time integration of
    Tensor Trains.

  Args:
    what: TensorTrain or TensorTrainBatch. In the case of batch returns
      projection of the sum of elements in the batch.
    where: TensorTrain, TT-tensor or TT-matrix on which tangent space to project
    weights: python list or tf.Tensor of numbers or None, weights of the sum

  Returns:
     a TensorTrain with the TT-ranks equal 2 * tangent_space_tens.get_tt_ranks()
  """
  # Always work with batch of TT objects for simplicity.
  what = shapes.expand_batch_dim(what)

  if weights is not None:
    weights = tf.convert_to_tensor(weights)

  if not isinstance(where, TensorTrain):
    raise ValueError('The first argument should be a TensorTrain object, got '
                     '"%s".' % where)

  if where.get_raw_shape() != what.get_raw_shape():
    raise ValueError('The shapes of the tensor we want to project and of the '
                     'tensor on which tangent space we want to project should '
                     'match, got %s and %s.' %
                     (where.get_raw_shape(),
                      what.get_raw_shape()))

  if not where.dtype.is_compatible_with(what.dtype):
    raise ValueError('Dtypes of the arguments should coincide, got %s and %s.' %
                     (where.dtype,
                      what.dtype))

  left_tangent_space_tens = decompositions.orthogonalize_tt_cores(
    where)
  right_tangent_space_tens = decompositions.orthogonalize_tt_cores(
    left_tangent_space_tens, left_to_right=False)

  ndims = where.ndims()
  dtype = where.dtype
  raw_shape = shapes.lazy_raw_shape(where)
  batch_size = shapes.lazy_batch_size(what)
  right_tangent_tt_ranks = shapes.lazy_tt_ranks(right_tangent_space_tens)
  left_tangent_tt_ranks = shapes.lazy_tt_ranks(left_tangent_space_tens)

  # For einsum notation.
  mode_str = 'ij' if where.is_tt_matrix() else 'i'
  right_rank_dim = 3 if where.is_tt_matrix() else 2
  left_rank_dim = 0
  if weights is not None:
    weights_shape = weights.get_shape()
    output_is_batch = len(weights_shape) > 1 and weights_shape[1] > 1
  else:
    output_is_batch = False
  output_batch_str = 'o' if output_is_batch else ''
  if output_is_batch:
    right_rank_dim += 1
    left_rank_dim = 1
    output_batch_size = weights.get_shape()[1].value

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = what.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]
    einsum_str = 'sa{0}b,c{0}d,sbd->sac'.format(mode_str)
    rhs[core_idx] = tf.einsum(einsum_str, tens_core, right_tang_core,
                              rhs[core_idx + 1])

  # Prepare lhs vectors.
  # lhs[core_idx] is of size
  #   batch_size x tangent_tt_ranks[core_idx] x tensor_tt_ranks[core_idx]
  lhs = [None] * (ndims + 1)
  lhs[0] = tf.ones((batch_size, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1):
    tens_core = what.tt_cores[core_idx]
    left_tang_core = left_tangent_space_tens.tt_cores[core_idx]
    einsum_str = 'sab,a{0}c,sb{0}d->scd'.format(mode_str)
    lhs[core_idx + 1] = tf.einsum(einsum_str, lhs[core_idx], left_tang_core,
                                  tens_core)

  # Left to right sweep.
  res_cores_list = []
  for core_idx in range(ndims):
    tens_core = what.tt_cores[core_idx]
    left_tang_core = left_tangent_space_tens.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]

    if core_idx < ndims - 1:
      einsum_str = 'sab,sb{0}c->sa{0}c'.format(mode_str)
      proj_core = tf.einsum(einsum_str, lhs[core_idx], tens_core)
      einsum_str = 'a{0}b,sbc->sa{0}c'.format(mode_str)
      proj_core -= tf.einsum(einsum_str, left_tang_core, lhs[core_idx + 1])
      if weights is None:
        einsum_str = 'sa{0}b,sbc->a{0}c'.format(mode_str)
        proj_core = tf.einsum(einsum_str, proj_core, rhs[core_idx + 1])
      else:
        einsum_str = 'sa{0}b,sbc,s{1}->{1}a{0}c'.format(mode_str, output_batch_str)
        proj_core = tf.einsum(einsum_str, proj_core, rhs[core_idx + 1], weights)

    if core_idx == ndims - 1:
      if weights is None:
        einsum_str = 'sab,sb{0}c->a{0}c'.format(mode_str)
        proj_core = tf.einsum(einsum_str, lhs[core_idx], tens_core)
      else:
        einsum_str = 'sab,sb{0}c,s{1}->{1}a{0}c'.format(mode_str, output_batch_str)
        proj_core = tf.einsum(einsum_str, lhs[core_idx], tens_core, weights)

    if output_is_batch:
      # Add batch dimension of size output_batch_size to left_tang_core and
      # right_tang_core
      extended_left_tang_core = tf.expand_dims(left_tang_core, 0)
      extended_right_tang_core = tf.expand_dims(right_tang_core, 0)
      if where.is_tt_matrix():
        extended_left_tang_core = tf.tile(extended_left_tang_core,
                                          [output_batch_size, 1, 1, 1, 1])
        extended_right_tang_core = tf.tile(extended_right_tang_core,
                                           [output_batch_size, 1, 1, 1, 1])
      else:
        extended_left_tang_core = tf.tile(extended_left_tang_core,
                                          [output_batch_size, 1, 1, 1])
        extended_right_tang_core = tf.tile(extended_right_tang_core,
                                           [output_batch_size, 1, 1, 1])
    else:
      extended_left_tang_core = left_tang_core
      extended_right_tang_core = right_tang_core

    if core_idx == 0:
      res_core = tf.concat((proj_core, extended_left_tang_core),
                           axis=right_rank_dim)
    elif core_idx == ndims - 1:
      res_core = tf.concat((extended_right_tang_core, proj_core), axis=left_rank_dim)
    else:
      rank_1 = right_tangent_tt_ranks[core_idx]
      rank_2 = left_tangent_tt_ranks[core_idx + 1]
      if where.is_tt_matrix():
        mode_size_n = raw_shape[0][core_idx]
        mode_size_m = raw_shape[1][core_idx]
        shape = [rank_1, mode_size_n, mode_size_m, rank_2]
      else:
        mode_size = raw_shape[0][core_idx]
        shape = [rank_1, mode_size, rank_2]
      if output_is_batch:
        shape = [output_batch_size] + shape
      zeros = tf.zeros(shape, dtype)
      upper = tf.concat((extended_right_tang_core, zeros), axis=right_rank_dim)
      lower = tf.concat((proj_core, extended_left_tang_core),
                        axis=right_rank_dim)
      res_core = tf.concat((upper, lower), axis=left_rank_dim)
    res_cores_list.append(res_core)
  # TODO: TT-ranks.
  if output_is_batch:
    return TensorTrainBatch(res_cores_list, where.get_raw_shape(),
                            batch_size=output_batch_size)
  else:
    return TensorTrain(res_cores_list, where.get_raw_shape())
