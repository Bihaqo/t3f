import itertools
import tensorflow as tf

from t3f.tensor_train_base import TensorTrainBase
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import ops


def concat_along_batch_dim(tt_list, name='t3f_concat_along_batch_dim'):
  """Concat all TensorTrainBatch objects along batch dimension.

  Args:
    tt_list: a list of TensorTrainBatch objects.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch
  """
  ndims = tt_list[0].ndims()

  if isinstance(tt_list, TensorTrainBase):
    # Not a list but just one element, nothing to concat.
    return tt_list

  for batch_idx in range(len(tt_list)):
    if not isinstance(tt_list[batch_idx], TensorTrainBatch):
      raise ValueError('All objects in the list should be TTBatch objects, got '
                       '%s' % tt_list[batch_idx])
  for batch_idx in range(1, len(tt_list)):
    if tt_list[batch_idx].get_raw_shape() != tt_list[0].get_raw_shape():
      raise ValueError('Shapes of all TT-batch objects should coincide, got %s '
                       'and %s' % (tt_list[0].get_raw_shape(),
                                   tt_list[batch_idx].get_raw_shape()))
    if tt_list[batch_idx].get_tt_ranks() != tt_list[0].get_tt_ranks():
      raise ValueError('TT-ranks of all TT-batch objects should coincide, got '
                       '%s and %s' % (tt_list[0].get_tt_ranks(),
                                      tt_list[batch_idx].get_tt_ranks()))

  list_of_cores_lists = [tt.tt_cores for tt in tt_list]
  all_cores = tuple(itertools.chain.from_iterable(list_of_cores_lists))
  with tf.name_scope(name):
    res_cores = []
    for core_idx in range(ndims):
      curr_core = tf.concat([tt.tt_cores[core_idx] for tt in tt_list], axis=0)
      res_cores.append(curr_core)

    try:
      batch_size = sum([tt.batch_size for tt in tt_list])
    except TypeError:
      # The batch sizes are not defined and you can't sum Nones.
      batch_size = None

    return TensorTrainBatch(res_cores, tt_list[0].get_raw_shape(),
                            tt_list[0].get_tt_ranks(), batch_size)


def multiply_along_batch_dim(batch_tt, weights,
                             name='t3f_multiply_along_batch_dim'):
  """Multiply each TensorTrain in a batch by a number.

  Args:
    batch_tt: TensorTrainBatch object, TT-matrices or TT-tensors.
    weights: 1-D tf.Tensor (or something convertible to it like np.array) of size
     tt.batch_size with weights.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch
  """
  with tf.name_scope(name):
    weights = tf.convert_to_tensor(weights, dtype=batch_tt.dtype)
    tt_cores = list(batch_tt.tt_cores)
    if batch_tt.is_tt_matrix():
      weights = weights[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    else:
      weights = weights[:, tf.newaxis, tf.newaxis, tf.newaxis]
    tt_cores[0] = weights * tt_cores[0]
    out_shape = batch_tt.get_raw_shape()
    out_ranks = batch_tt.get_tt_ranks()
    out_batch_size = batch_tt.batch_size
    return TensorTrainBatch(tt_cores, out_shape, out_ranks, out_batch_size)


def gram_matrix(tt_vectors, matrix=None, name='t3f_gram_matrix'):
  """Computes Gramian matrix of a batch of TT-vectors.

  If matrix is None, computes
    res[i, j] = t3f.flat_inner(tt_vectors[i], tt_vectors[j]).
  If matrix is present, computes
      res[i, j] = t3f.flat_inner(tt_vectors[i], t3f.matmul(matrix, tt_vectors[j]))
    or more shortly
      res[i, j] = tt_vectors[i]^T * matrix * tt_vectors[j]
    but is more efficient.

  Args:
    tt_vectors: TensorTrainBatch.
    matrix: None, or TensorTrain matrix.
    name: string, name of the Op.

  Returns:
    tf.tensor with the Gram matrix.
      
  Complexity:
    If the matrix is not present, the complexity is O(batch_size^2 d r^3 n)
      where d is the number of
      TT-cores (tt_vectors.ndims()), r is the largest TT-rank
        max(tt_vectors.get_tt_rank())
      and n is the size of the axis dimension, e.g.
        for a tensor of size 4 x 4 x 4, n is 4;
        for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
    If the matrix of TT-rank R is present, the complexity is
        O(batch_size^2 d R r^2 n (r + nR))
      where the matrix is of raw-shape (n, n, ..., n) x (n, n, ..., n);
      r is the TT-rank of vectors tt_vectors;
      R is the TT-rank of the matrix.
  """
  return pairwise_flat_inner(tt_vectors, tt_vectors, matrix, name)


def pairwise_flat_inner(tt_1, tt_2, matrix=None,
                        name='t3f_pairwise_flat_inner'):
  """Computes all scalar products between two batches of TT-objects.

  If matrix is None, computes
    res[i, j] = t3f.flat_inner(tt_1[i], tt_2[j]).

  If matrix is present, computes
      res[i, j] = t3f.flat_inner(tt_1[i], t3f.matmul(matrix, tt_2[j]))
    or more shortly
      res[i, j] = tt_1[i]^T * matrix * tt_2[j]
    but is more efficient.

  Args:
    tt_1: TensorTrainBatch.
    tt_2: TensorTrainBatch.
    matrix: None, or TensorTrain matrix.
    name: string, name of the Op.

  Returns:
    tf.tensor with the matrix of pairwise scalar products (flat inners).
      
  Complexity:
    If the matrix is not present, the complexity is O(batch_size^2 d r^3 n)
      where d is the number of
      TT-cores (tt_vectors.ndims()), r is the largest TT-rank
        max(tt_vectors.get_tt_rank())
      and n is the size of the axis dimension, e.g.
        for a tensor of size 4 x 4 x 4, n is 4;
        for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
      A more precise complexity is
        O(batch_size^2 d r1 r2 n max(r1, r2))
      where r1 is the largest TT-rank of tt_a
      and r2 is the largest TT-rank of tt_b.
    If the matrix is present, the complexity is
        O(batch_size^2 d R r1 r2 (n r1 + n m R + m r2))
      where
      the matrix is of raw-shape (n, n, ..., n) x (m, m, ..., m) and TT-rank R;
      tt_1 is of shape (n, n, ..., n) and is of the TT-rank r1;
      tt_2 is of shape (m, m, ..., m) and is of the TT-rank r2;
  """
  all_cores = tt_1.tt_cores + tt_2.tt_cores
  if matrix is not None:
    all_cores += matrix.tt_cores
  with tf.name_scope(name):
    ndims = tt_1.ndims()
    if matrix is None:
      curr_core_1 = tt_1.tt_cores[0]
      curr_core_2 = tt_2.tt_cores[0]
      mode_string = 'ij' if tt_1.is_tt_matrix() else 'i'
      einsum_str = 'pa{0}b,qc{0}d->pqbd'.format(mode_string)
      res = tf.einsum(einsum_str, curr_core_1, curr_core_2)
      for core_idx in range(1, ndims):
        curr_core_1 = tt_1.tt_cores[core_idx]
        curr_core_2 = tt_2.tt_cores[core_idx]
        einsum_str = 'pqac,pa{0}b,qc{0}d->pqbd'.format(mode_string)
        res = tf.einsum(einsum_str, res, curr_core_1, curr_core_2)
    else:
      # res[i, j] = tt_1[i] ^ T * matrix * tt_2[j]
      are_all_matrices = tt_1.is_tt_matrix() and tt_2.is_tt_matrix()
      are_all_matrices = are_all_matrices and matrix.is_tt_matrix()
      if not are_all_matrices:
        raise ValueError('When passing three arguments to pairwise_flat_inner, '
                         'the first 2 of them should be TT-vecors and the last '
                         'should be a TT-matrix. Got %s, %s, and %s instead.' %
                         (tt_1, tt_2, matrix))
      matrix_shape = matrix.get_raw_shape()
      if not tt_1.get_raw_shape()[0].is_compatible_with(matrix_shape[0]):
        raise ValueError('The shape of the first argument should be compatible '
                         'with the shape of the TT-matrix, that is it should '
                         'be possible to do the following matmul: '
                         'transpose(tt_1) * matrix. Got the first argument '
                         '"%s" and matrix "%s"' % (tt_1, matrix))
      if not tt_2.get_raw_shape()[0].is_compatible_with(matrix_shape[1]):
        raise ValueError('The shape of the second argument should be '
                         'compatible with the shape of the TT-matrix, that is '
                         'it should be possible to do the following matmul: '
                         'matrix * tt_2. Got the second argument '
                         '"%s" and matrix "%s"' % (tt_2, matrix))

      vectors_1_shape = tt_1.get_shape()
      if vectors_1_shape[2] == 1 and vectors_1_shape[1] != 1:
        # TODO: not very efficient, better to use different order in einsum.
        tt_1 = ops.transpose(tt_1)
      vectors_1_shape = tt_1.get_shape()
      vectors_2_shape = tt_2.get_shape()
      if vectors_2_shape[2] == 1 and vectors_2_shape[1] != 1:
        # TODO: not very efficient, better to use different order in einsum.
        tt_2 = ops.transpose(tt_2)
      vectors_2_shape = tt_2.get_shape()
      if vectors_1_shape[1] != 1:
        # TODO: do something so that in case the shape is undefined on compilation
        # it still works.
        raise ValueError('The tt_vectors_1 argument should be vectors (not '
                         'matrices) with shape defined on compilation.')
      if vectors_2_shape[1] != 1:
        # TODO: do something so that in case the shape is undefined on compilation
        # it still works.
        raise ValueError('The tt_vectors_2 argument should be vectors (not '
                         'matrices) with shape defined on compilation.')
      curr_core_1 = tt_1.tt_cores[0]
      curr_core_2 = tt_2.tt_cores[0]
      curr_matrix_core = matrix.tt_cores[0]
      # We enumerate the dummy dimension (that takes 1 value) with `k`.
      res = tf.einsum('pakib,cijd,qekjf->pqbdf', curr_core_1, curr_matrix_core,
                      curr_core_2)
      for core_idx in range(1, ndims):
        curr_core_1 = tt_1.tt_cores[core_idx]
        curr_core_2 = tt_2.tt_cores[core_idx]
        curr_matrix_core = matrix.tt_cores[core_idx]
        res = tf.einsum('pqace,pakib,cijd,qekjf->pqbdf', res, curr_core_1,
                        curr_matrix_core, curr_core_2)

    # Squeeze to make the result of size batch_size x batch_size instead of
    # batch_size x batch_size x 1 x 1.
    return tf.squeeze(res)
