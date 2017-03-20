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

  Args:
    tangent_space: TensorTrain.
    tensor: TensorTrain or TensorTrainBatch. In the case of batch returns
    projection of the sum of elements in the batch.

  Returns:
     a TensorTraain with the TT-ranks equal 2 * rank(tangent_space_tens).
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
                     'match, got %d %d.' % (tangent_space_tens.get_raw_shape(),
                                            tensor.get_raw_shape()))

  # Right to left orthogonalization of tangent_space_tens.
  tangent_space_tens = decompositions.orthogonalize_tt_cores(tangent_space_tens,
                                                             left_to_right=False)

  ndims = tangent_space_tens.ndims()
  shape = shapes.lazy_raw_shape(tangent_space_tens)
  batch_size = shapes.lazy_batch_size(tensor)
  tangent_tt_ranks = shapes.lazy_tt_ranks(tangent_space_tens)
  tensor_tt_ranks = shapes.lazy_tt_ranks(tensor)

  # Initialize the cores of the projection_X(sum z[i]).
  res_ranks = []
  for core_idx in range(ndims + 1):
    if core_idx == 0 or core_idx == ndims - 1:
      res_ranks.append(1)
    else:
      res_ranks.append(2 * tangent_tt_ranks[core_idx])
  res_cores = []
  for core_idx in range(ndims):
    res_cores.append(tf.zeros((res_ranks[core_idx], shape[core_idx],
                               res_ranks[core_idx + 1])))
    for value in range(modeSize[dim]):
      res_cores[core_idx, 0:r1, value, 0:r2] = tangent_space_tens.tt_cores[core_idx, :, value, :]

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1))
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = tensor.tt_cores[core_idx]
    tang_core = tangent_space_tens.tt_cores[core_idx]
    rhs[core_idx] = tf.einsum('oaib,cid,obd->oac', tens_core, tang_core,
                              rhs[core_idx + 1])

  # lhs[core_idx] is of size
  #   batch_size x tangent_tt_ranks[core_idx] x tensor_tt_ranks[core_idx]
  lhs = tf.ones((batch_size, 1, 1))
  # Left to right sweep.
  for core_idx in range(ndims):
    tang_core = tangent_space_tens.tt_cores[core_idx]
    # r1, n, r2 = cc.shape
    if core_idx < ndims - 1:
      # Left to right orthogonalization.
      cc = tf.reshape(cc, (-1, r2))
      cc, rr = np.linalg.qr(cc)
      r2 = cc.shape[1]
      # Warning: since ranks can change here, do not use X.r!
      # Use coresX[dim].shape instead.
      coresX[dim] = reshape(cc, (r1, n, r2)).copy()
      coresX[dim + 1] = np.tensordot(rr, coresX[dim + 1], 1)

      new_lhs = tf.einsum('oaib,cid,oac->obd', tens_core, tang_core, lhs)

      currPCore = tf.einsum('oab,obic->oaic', lhs, tens_core)
      # currPCore = tf.einsum('ijk,iklm->ijlm', lhs, zCoresDim[dim])
      currPCore = tf.reshape(currPCore, (len(zArr), r1 * n, -1))
      currPCore -= tf.einsum('ij,kjl->kil', cc, new_lhs)
      currPCore = tf.einsum('ijk,ikl', currPCore, rhs[dim + 1])
      currPCore = tf.reshape(currPCore, (r1, modeSize[dim], r2))
      if core_idx == 0:
        res_cores[core_idx][0:r1, :, 0:r2] += currPCore
      else:
        res_cores[core_idx][r1:, :, 0:r2] += currPCore
      lhs = new_lhs

      if core_idx == 0:
        res_cores[core_idx][0:r1, :, r2:] = tang_core
      else:
        res_cores[core_idx][r1:, :, r2:] = tang_core

    if core_idx == ndims - 1:
      res_cores[core_idx][r1:, :, 0:r2] += tf.einsum('oab,obic->oaic', lhs, tens_core)

  return TensorTrain(res_cores)