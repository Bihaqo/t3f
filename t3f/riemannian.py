import tensorflow as tf

import shapes
import decompositions
from tensor_train import TensorTrain


def project(tangent_space_tens, tensor, coef=None):
  """Project TT tensor on the tangent space of the TT tensor tangent_space_tens.

  project(x, tens) = P_x(tens)
  project(x, batch) = P_x(\sum_i batch[i])
  project(x, tens, coef) = P_x(\sum_i coef[i] * batch[i])

  This function implements an algorithm from the paper [1], theorem 3.1.

  [1] C. Lubich, I. Oseledets and B. Vandereycken, Time integration of
    Tensor Trains

  Args:
    tangent_space_tens: TensorTrain.
    tensor: TensorTrain or TensorTrainBatch. In the case of batch returns
    projection of the sum of elements in the batch.

  Returns:
     a TensorTrain with the TT-ranks equal 2 * rank(tangent_space_tens).
  """
  # Always work with batch of TT for simplicity.
  tensor = shapes.expand_batch_dim(tensor)

  if not isinstance(tangent_space_tens, TensorTrain) or \
      tangent_space_tens.is_tt_matrix():
    raise ValueError('The first argument should be a TT-tensor (not matrix), '
                     'got "%s".' % tangent_space_tens)

  if tangent_space_tens.get_raw_shape() != tensor.get_raw_shape():
    raise ValueError('The shapes of the tensor we want to project and of the '
                     'tensor on which tangent space we want to project should '
                     'match, got %s and %s.' %
                     (tangent_space_tens.get_raw_shape(),
                      tensor.get_raw_shape()))

  left_tangent_space_tens = decompositions.orthogonalize_tt_cores(
    tangent_space_tens)
  right_tangent_space_tens = decompositions.orthogonalize_tt_cores(
    left_tangent_space_tens, left_to_right=False)


  ndims = tangent_space_tens.ndims()
  # TODO: use raw shape to support TT-matrices.
  shape = shapes.lazy_shape(tangent_space_tens)
  batch_size = shapes.lazy_batch_size(tensor)
  right_tangent_tt_ranks = shapes.lazy_tt_ranks(right_tangent_space_tens)
  left_tangent_tt_ranks = shapes.lazy_tt_ranks(left_tangent_space_tens)

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1))
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = tensor.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]
    rhs[core_idx] = tf.einsum('oaib,cid,obd->oac', tens_core, right_tang_core,
                              rhs[core_idx + 1])

  # lhs[core_idx] is of size
  #   batch_size x tangent_tt_ranks[core_idx] x tensor_tt_ranks[core_idx]
  lhs = tf.ones((batch_size, 1, 1))
  # Left to right sweep.
  res_cores_list = []
  for core_idx in range(ndims):
    tens_core = tensor.tt_cores[core_idx]
    left_tang_core = left_tangent_space_tens.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]

    if core_idx < ndims - 1:
      new_lhs = tf.einsum('oab,aic,obid->ocd', lhs, left_tang_core, tens_core)

      proj_core = tf.einsum('oab,obic->oaic', lhs, tens_core)
      proj_core -= tf.einsum('aib,obc->oaic', right_tang_core, new_lhs)
      # TODO: add coef here.
      proj_core = tf.einsum('oaib,obc->aic', proj_core, rhs[core_idx + 1])
      lhs = new_lhs

    if core_idx == ndims - 1:
      proj_core = tf.einsum('oab,obic->aic', lhs, tens_core)

    if core_idx == 0:
      res_core = tf.concat((proj_core, left_tang_core), axis=2)
    elif core_idx == ndims - 1:
      res_core = tf.concat((right_tang_core, proj_core), axis=0)
    else:
      mode_size = shape[core_idx]
      rank_1 = right_tangent_tt_ranks[core_idx]
      rank_2 = left_tangent_tt_ranks[core_idx + 1]
      zeros = tf.zeros((rank_1, mode_size, rank_2))
      upper = tf.concat((right_tang_core, zeros), axis=2)
      lower = tf.concat((proj_core, left_tang_core), axis=2)
      res_core = tf.concat((upper, lower), axis=0)
    res_cores_list.append(res_core)
  # TODO: TT-ranks.
  return TensorTrain(res_cores_list, tangent_space_tens.get_raw_shape())