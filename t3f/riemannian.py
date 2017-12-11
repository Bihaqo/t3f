import tensorflow as tf

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import shapes
from t3f import decompositions


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
  right_rank_dim = where.right_tt_rank_dim
  left_rank_dim = where.left_tt_rank_dim
  if weights is not None:
    weights_shape = weights.get_shape()
    output_is_batch = len(weights_shape) > 1 and weights_shape[1] > 1
  else:
    output_is_batch = False
  output_batch_str = 'o' if output_is_batch else ''
  if output_is_batch:
    right_rank_dim += 1
    left_rank_dim += 1
    output_batch_size = weights.get_shape()[1].value

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = what.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]
    einsum_str = 'sa{0}b,sbd,c{0}d->sac'.format(mode_str)
    rhs[core_idx] = tf.einsum(einsum_str, tens_core, rhs[core_idx + 1],
                              right_tang_core)

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
        einsum_str = 'sa{0}b,sbc->sa{0}c'.format(mode_str, output_batch_str)
        proj_core_s = tf.einsum(einsum_str, proj_core, rhs[core_idx + 1])
        einsum_str = 's{1},sa{0}c->{1}a{0}c'.format(mode_str, output_batch_str)
        proj_core = tf.einsum(einsum_str, weights, proj_core_s)

    if core_idx == ndims - 1:
      if weights is None:
        einsum_str = 'sab,sb{0}c->a{0}c'.format(mode_str)
        proj_core = tf.einsum(einsum_str, lhs[core_idx], tens_core)
      else:
        einsum_str = 'sab,sb{0}c->sa{0}c'.format(mode_str, output_batch_str)
        proj_core_s = tf.einsum(einsum_str, lhs[core_idx], tens_core)
        einsum_str = 's{1},sa{0}c->{1}a{0}c'.format(mode_str, output_batch_str)
        proj_core = tf.einsum(einsum_str, weights, proj_core_s)

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
    res = TensorTrainBatch(res_cores_list, where.get_raw_shape(),
                            batch_size=output_batch_size)
  else:
    res = TensorTrain(res_cores_list, where.get_raw_shape())

  res.projection_on = where
  return res


def project(what, where):
  """Project `what` TTs on the tangent space of `where` TT.

  project(what, x) = P_x(what)
  project(batch_what, x) = batch(P_x(batch_what[0]), ..., P_x(batch_what[N]))

  This function implements the algorithm from the paper [1], theorem 3.1.

  [1] C. Lubich, I. Oseledets and B. Vandereycken, Time integration of
    Tensor Trains.

  Args:
    what: TensorTrain or TensorTrainBatch. In the case of batch returns
      batch with projection of each individual tensor.
    where: TensorTrain, TT-tensor or TT-matrix on which tangent space to project

  Returns:
     a TensorTrain with the TT-ranks equal 2 * tangent_space_tens.get_tt_ranks()
  """

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
  right_tangent_tt_ranks = shapes.lazy_tt_ranks(right_tangent_space_tens)
  left_tangent_tt_ranks = shapes.lazy_tt_ranks(left_tangent_space_tens)

  # For einsum notation.
  mode_str = 'ij' if where.is_tt_matrix() else 'i'
  right_rank_dim = what.right_tt_rank_dim
  left_rank_dim = what.left_tt_rank_dim
  output_is_batch = isinstance(what, TensorTrainBatch)
  if output_is_batch:
    output_batch_size = what.batch_size

  # Always work with batch of TT objects for simplicity.
  what = shapes.expand_batch_dim(what)
  batch_size = shapes.lazy_batch_size(what)

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = what.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]
    einsum_str = 'sa{0}b,sbd,c{0}d->sac'.format(mode_str)
    rhs[core_idx] = tf.einsum(einsum_str, tens_core, rhs[core_idx + 1],
                              right_tang_core)

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
      if output_is_batch:
        einsum_str = 'sa{0}b,sbc->sa{0}c'.format(mode_str)
      else:
        einsum_str = 'sa{0}b,sbc->a{0}c'.format(mode_str)
      proj_core = tf.einsum(einsum_str, proj_core, rhs[core_idx + 1])

    if core_idx == ndims - 1:
      if output_is_batch:
        einsum_str = 'sab,sb{0}c->sa{0}c'.format(mode_str)
      else:
        einsum_str = 'sab,sb{0}c->a{0}c'.format(mode_str)
      proj_core = tf.einsum(einsum_str, lhs[core_idx], tens_core)

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
    res = TensorTrainBatch(res_cores_list, where.get_raw_shape(),
                            batch_size=output_batch_size)
  else:
    res = TensorTrain(res_cores_list, where.get_raw_shape())

  res.projection_on = where
  return res


def project_matmul(what, where, matrix):
  """Project `matrix` * `what` TTs on the tangent space of `where` TT.

  project(what, x) = P_x(what)
  project(batch_what, x) = batch(P_x(batch_what[0]), ..., P_x(batch_what[N]))

  This function implements the algorithm from the paper [1], theorem 3.1.

  [1] C. Lubich, I. Oseledets and B. Vandereycken, Time integration of
    Tensor Trains.

  Args:
    what: TensorTrain or TensorTrainBatch. In the case of batch returns
      batch with projection of each individual tensor.
    where: TensorTrain, TT-tensor or TT-matrix on which tangent space to project
    matrix: TensorTrain, TT-matrix to multiply by what

  Returns:
     a TensorTrain with the TT-ranks equal 2 * tangent_space_tens.get_tt_ranks()
  """

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
  right_rank_dim = what.right_tt_rank_dim
  left_rank_dim = what.left_tt_rank_dim
  output_is_batch = isinstance(what, TensorTrainBatch)
  if output_is_batch:
    output_batch_size = what.batch_size

  # Always work with batch of TT objects for simplicity.
  what = shapes.expand_batch_dim(what)

  # Prepare rhs vectors.
  # rhs[core_idx] is of size
  #   batch_size x tensor_tt_ranks[core_idx] x matrix_tt_ranks[core_idx] x tangent_tt_ranks[core_idx]
  rhs = [None] * (ndims + 1)
  rhs[ndims] = tf.ones((batch_size, 1, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1, 0, -1):
    tens_core = what.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]
    matrix_core = matrix.tt_cores[core_idx]
    rhs[core_idx] = tf.einsum('bije,cikf,sdef,sajkd->sabc', matrix_core,
                              right_tang_core, rhs[core_idx + 1], tens_core)
  # Prepare lhs vectors.
  # lhs[core_idx] is of size
  #   batch_size x tangent_tt_ranks[core_idx] x matrix_tt_ranks[core_idx] x tensor_tt_ranks[core_idx]
  lhs = [None] * (ndims + 1)
  lhs[0] = tf.ones((batch_size, 1, 1, 1), dtype=dtype)
  for core_idx in range(ndims - 1):
    tens_core = what.tt_cores[core_idx]
    left_tang_core = left_tangent_space_tens.tt_cores[core_idx]
    matrix_core = matrix.tt_cores[core_idx]
    # TODO: brutforce order of indices in lhs??
    lhs[core_idx + 1] = tf.einsum('bije,aikd,sabc,scjkf->sdef', matrix_core,
                                  left_tang_core, lhs[core_idx], tens_core)

  # Left to right sweep.
  res_cores_list = []
  for core_idx in range(ndims):
    tens_core = what.tt_cores[core_idx]
    matrix_core = matrix.tt_cores[core_idx]
    left_tang_core = left_tangent_space_tens.tt_cores[core_idx]
    right_tang_core = right_tangent_space_tens.tt_cores[core_idx]

    if core_idx < ndims - 1:
      proj_core = tf.einsum('scjke,sabc,bijd->saikde', tens_core,
                            lhs[core_idx], matrix_core)
      proj_core -= tf.einsum('aikb,sbcd->saikcd', left_tang_core,
                             lhs[core_idx + 1])
      proj_core = tf.einsum('saikcb,sbcd->saikd', proj_core, rhs[core_idx + 1])

    if core_idx == ndims - 1:
      # d and e dimensions take 1 value, since its the last rank.
      # To make the result shape (?, ?, ?, 1), we are summing d and leaving e,
      # but we could have done the opposite -- sum e and leave d.
      proj_core = tf.einsum('sabc,bijd,scjke->saike', lhs[core_idx], matrix_core,
                            tens_core)

    if output_is_batch:
      # Add batch dimension of size output_batch_size to left_tang_core and
      # right_tang_core
      extended_left_tang_core = tf.expand_dims(left_tang_core, 0)
      extended_right_tang_core = tf.expand_dims(right_tang_core, 0)
      extended_left_tang_core = tf.tile(extended_left_tang_core,
                                        [output_batch_size, 1, 1, 1, 1])
      extended_right_tang_core = tf.tile(extended_right_tang_core,
                                         [output_batch_size, 1, 1, 1, 1])
    else:
      extended_left_tang_core = left_tang_core
      extended_right_tang_core = right_tang_core

    if core_idx == 0:
      res_core = tf.concat((proj_core, extended_left_tang_core),
                           axis=right_rank_dim)
    elif core_idx == ndims - 1:
      res_core = tf.concat((extended_right_tang_core, proj_core),
                           axis=left_rank_dim)
    else:
      rank_1 = right_tangent_tt_ranks[core_idx]
      rank_2 = left_tangent_tt_ranks[core_idx + 1]
      mode_size_n = raw_shape[0][core_idx]
      mode_size_m = raw_shape[1][core_idx]
      shape = [rank_1, mode_size_n, mode_size_m, rank_2]
      if output_is_batch:
        shape = [output_batch_size] + shape
      zeros = tf.zeros(shape, dtype)
      upper = tf.concat((extended_right_tang_core, zeros),
                        axis=right_rank_dim)
      lower = tf.concat((proj_core, extended_left_tang_core),
                        axis=right_rank_dim)
      res_core = tf.concat((upper, lower), axis=left_rank_dim)
    res_cores_list.append(res_core)

  # TODO: TT-ranks.
  if output_is_batch:
    res = TensorTrainBatch(res_cores_list, where.get_raw_shape(),
                            batch_size=output_batch_size)
  else:
    res = TensorTrain(res_cores_list, where.get_raw_shape())

  res.projection_on = where
  return res


def pairwise_flat_inner_projected(projected_tt_vectors_1,
                                  projected_tt_vectors_2):
  """Scalar products between two batches of TTs from the same tangent space.
  
    res[i, j] = t3f.flat_inner(projected_tt_vectors_1[i], projected_tt_vectors_1[j]).
  
  pairwise_flat_inner_projected(projected_tt_vectors_1, projected_tt_vectors_2)
  is equivalent to
    pairwise_flat_inner(projected_tt_vectors_1, projected_tt_vectors_2)
  , but works only on objects from the same tangent space and is much faster
  than general pairwise_flat_inner. 
  
  Args:
    projected_tt_vectors_1: TensorTrainBatch of tensors projected on the same
      tangent space as projected_tt_vectors_2.
    projected_tt_vectors_2: TensorTrainBatch.
    
  Returns:
    tf.tensor with the scalar product matrix.
      
  Complexity:
      O(batch_size^2 d r^2 n), where
    d is the number of TT-cores (projected_tt_vectors_1.ndims());
    r is the largest TT-rank max(projected_tt_vectors_1.get_tt_rank())
      (i.e. 2 * {the TT-rank of the object we projected vectors onto}.
    and n is the size of the axis dimension, e.g.
      for a tensor of size 4 x 4 x 4, n is 4;
      for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12.
  """
  if not hasattr(projected_tt_vectors_1, 'projection_on') or \
      not hasattr(projected_tt_vectors_2, 'projection_on'):
    raise ValueError('Both arguments should be projections on the tangent '
                     'space of some other TT-object. All projection* functions '
                     'leave .projection_on field in the resulting TT-object '
                     'which is not present in the arguments you\'ve provided')

  if projected_tt_vectors_1.projection_on != projected_tt_vectors_2.projection_on:
    raise ValueError('Both arguments should be projections on the tangent '
                     'space of the same TT-object. The provided arguments are '
                     'projections on different TT-objects (%s and %s). Or at '
                     'least the pointers are different.' %
                     (projected_tt_vectors_1.projection_on,
                      projected_tt_vectors_2.projection_on))

  # Always work with batches of objects for simplicity.
  projected_tt_vectors_1 = shapes.expand_batch_dim(projected_tt_vectors_1)
  projected_tt_vectors_2 = shapes.expand_batch_dim(projected_tt_vectors_2)

  ndims = projected_tt_vectors_1.ndims()
  tt_ranks = shapes.lazy_tt_ranks(projected_tt_vectors_1)

  if projected_tt_vectors_1.is_tt_matrix():
    right_size = tt_ranks[1] // 2
    curr_core_1 = projected_tt_vectors_1.tt_cores[0]
    curr_core_2 = projected_tt_vectors_2.tt_cores[0]
    curr_du_1 = curr_core_1[:, :, :, :, :right_size]
    curr_du_2 = curr_core_2[:, :, :, :, :right_size]
    res = tf.einsum('paijb,qaijb->pq', curr_du_1, curr_du_2)
    for core_idx in range(1, ndims):
      left_size = tt_ranks[core_idx] // 2
      right_size = tt_ranks[core_idx + 1] // 2
      curr_core_1 = projected_tt_vectors_1.tt_cores[core_idx]
      curr_core_2 = projected_tt_vectors_2.tt_cores[core_idx]
      curr_du_1 = curr_core_1[:, left_size:, :, :, :right_size]
      curr_du_2 = curr_core_2[:, left_size:, :, :, :right_size]
      res += tf.einsum('paijb,qaijb->pq', curr_du_1, curr_du_2)

    left_size = tt_ranks[-2] // 2
    curr_core_1 = projected_tt_vectors_1.tt_cores[-1]
    curr_core_2 = projected_tt_vectors_2.tt_cores[-1]
    curr_du_1 = curr_core_1[:, left_size:, :, :, :]
    curr_du_2 = curr_core_2[:, left_size:, :, :, :]
    res += tf.einsum('paijb,qaijb->pq', curr_du_1, curr_du_2)
  else:
    # Working with TT-tensor, not TT-matrix.
    right_size = tt_ranks[1] // 2
    curr_core_1 = projected_tt_vectors_1.tt_cores[0]
    curr_core_2 = projected_tt_vectors_2.tt_cores[0]
    curr_du_1 = curr_core_1[:, :, :, :right_size]
    curr_du_2 = curr_core_2[:, :, :, :right_size]
    res = tf.einsum('paib,qaib->pq', curr_du_1, curr_du_2)
    for core_idx in range(1, ndims):
      left_size = tt_ranks[core_idx] // 2
      right_size = tt_ranks[core_idx + 1] // 2
      curr_core_1 = projected_tt_vectors_1.tt_cores[core_idx]
      curr_core_2 = projected_tt_vectors_2.tt_cores[core_idx]
      curr_du_1 = curr_core_1[:, left_size:, :, :right_size]
      curr_du_2 = curr_core_2[:, left_size:, :, :right_size]
      res += tf.einsum('paib,qaib->pq', curr_du_1, curr_du_2)

    left_size = tt_ranks[-2] // 2
    curr_core_1 = projected_tt_vectors_1.tt_cores[-1]
    curr_core_2 = projected_tt_vectors_2.tt_cores[-1]
    curr_du_1 = curr_core_1[:, left_size:, :, :]
    curr_du_2 = curr_core_2[:, left_size:, :, :]
    res += tf.einsum('paib,qaib->pq', curr_du_1, curr_du_2)
  return res


def add_n_projected(tt_objects, coef=None):
  """Adds all input TT-objects that are projections on the same tangent space.

    add_projected((a, b)) is equivalent add(a, b) for a and b that are from the
    same tangent space, but doesn't increase the TT-ranks.

  Args:
    tt_objects: a list of TT-objects that are projections on the same tangent
      space.
    coef: a list of numbers or anything else convertable to tf.Tensor.
      If provided, computes weighted sum. The size of this array should be
        len(tt_objects) x tt_objects[0].batch_size

  Returns:
    TT-objects representing the sum of the tt_objects (weighted sum if coef is
    provided). The TT-rank of the result equals to the TT-ranks of the arguments.
  """
  for tt in tt_objects:
    if not hasattr(tt, 'projection_on'):
      raise ValueError('Both arguments should be projections on the tangent '
                       'space of some other TT-object. All projection* functions '
                       'leave .projection_on field in the resulting TT-object '
                       'which is not present in the argument you\'ve provided.')

  projection_on = tt_objects[0].projection_on
  for tt in tt_objects[1:]:
    if tt.projection_on != projection_on:
      raise ValueError('All tt_objects should be projections on the tangent '
                       'space of the same TT-object. The provided arguments are '
                       'projections on different TT-objects (%s and %s). Or at '
                       'least the pointers are different.' % (tt.projection_on,
                                                              projection_on))
  if coef is not None:
    coef = tf.convert_to_tensor(coef)
    if coef.get_shape().ndims > 1:
      # In batch case we will need to multiply each core by this coefficients
      # along the first axis. To do it need to reshape the coefs to match
      # the TT-cores number of dimensions.
      some_core = tt_objects[0].tt_cores[0]
      dim_array = [1] * (some_core.get_shape().ndims + 1)
      dim_array[0] = coef.get_shape()[0].value
      dim_array[1] = coef.get_shape()[1].value
      coef = tf.reshape(coef, dim_array)

  ndims = tt_objects[0].ndims()
  tt_ranks = shapes.lazy_tt_ranks(tt_objects[0])
  left_rank_dim = tt_objects[0].left_tt_rank_dim
  right_rank_dim = tt_objects[0].right_tt_rank_dim
  res_cores = []

  def slice_tt_core(tt_core, left_idx, right_idx):
    num_tt_core_dims = len(tt_core.get_shape())
    idx = [slice(None)] * num_tt_core_dims
    idx[left_rank_dim] = left_idx
    idx[right_rank_dim] = right_idx
    return tt_core[idx]

  right_half_rank = tt_ranks[1] // 2
  left_chunks = []
  for obj_idx, tt in enumerate(tt_objects):
    curr_core = slice_tt_core(tt.tt_cores[0], slice(None),
                              slice(0, right_half_rank))
    if coef is not None:
      curr_core *= coef[obj_idx]
    left_chunks.append(curr_core)
  left_part = tf.add_n(left_chunks)
  first_obj_core = tt_objects[0].tt_cores[0]
  right_part = slice_tt_core(first_obj_core, slice(None),
                             slice(right_half_rank, None))
  first_core = tf.concat((left_part, right_part), axis=right_rank_dim)
  res_cores.append(first_core)

  for core_idx in range(1, ndims - 1):
    first_obj_core = tt_objects[0].tt_cores[core_idx]
    left_half_rank = tt_ranks[core_idx] // 2
    right_half_rank = tt_ranks[core_idx + 1] // 2

    upper_part = slice_tt_core(tt.tt_cores[core_idx], slice(0, left_half_rank),
                               slice(None))
    lower_right_part = slice_tt_core(first_obj_core,
                                     slice(left_half_rank, None),
                                     slice(right_half_rank, None))

    lower_left_chunks = []
    for obj_idx, tt in enumerate(tt_objects):
      curr_core = slice_tt_core(tt.tt_cores[core_idx],
                                slice(left_half_rank, None),
                                slice(0, right_half_rank))
      if coef is not None:
        curr_core *= coef[obj_idx]
      lower_left_chunks.append(curr_core)
    lower_left_part = tf.add_n(lower_left_chunks)
    lower_part = tf.concat((lower_left_part, lower_right_part),
                           axis=right_rank_dim)
    curr_core = tf.concat((upper_part, lower_part), axis=left_rank_dim)
    res_cores.append(curr_core)

  left_half_rank = tt_ranks[ndims - 1] // 2
  upper_part = slice_tt_core(tt.tt_cores[-1], slice(0, left_half_rank),
                             slice(None))
  lower_chunks = []
  for obj_idx, tt in enumerate(tt_objects):
    curr_core = slice_tt_core(tt.tt_cores[-1], slice(left_half_rank, None),
                              slice(None))
    if coef is not None:
      curr_core *= coef[obj_idx]
    lower_chunks.append(curr_core)
  lower_part = tf.add_n(lower_chunks)
  last_core = tf.concat((upper_part, lower_part), axis=left_rank_dim)
  res_cores.append(last_core)

  raw_shape = tt_objects[0].get_raw_shape()
  static_tt_ranks = tt_objects[0].get_tt_ranks()
  if isinstance(tt_objects[0], TensorTrain):
    res = TensorTrain(res_cores, raw_shape, static_tt_ranks)
  elif isinstance(tt_objects[0], TensorTrainBatch):
    res = TensorTrainBatch(res_cores, raw_shape, static_tt_ranks,
                           tt_objects[0].batch_size)
  # Maintain the projection_on property.
  res.projection_on = tt_objects[0].projection_on
  return res
