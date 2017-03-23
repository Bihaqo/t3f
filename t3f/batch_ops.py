import tensorflow as tf

from tensor_train_base import TensorTrainBase
from tensor_train_batch import TensorTrainBatch
import ops


def concat_along_batch_dim(tt_list):
  """Concat all TensorTrainBatch objects along batch dimension.

  Args:
    tt_list: a list of TensorTrainBatch objects.

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

  res_cores = []
  for core_idx in range(ndims):
    curr_core = tf.concat([tt.tt_cores[core_idx] for tt in tt_list], axis=0)
    res_cores.append(curr_core)

  batch_size = sum([tt.batch_size for tt in tt_list])

  return TensorTrainBatch(res_cores, tt_list[0].get_raw_shape(),
                          tt_list[0].get_tt_ranks(), batch_size)


def gram_matrix(tt_vectors, matrix=None):
  """Computes Gramian matrix of a batch of TT-vecors.

  If matrix is None, computes
    res[i, j] = t3f.flat_inner(tt_vectors[i], tt_vectors[j]).
  If matrix is present, computes
      res[i, j] = t3f.flat_inner(tt_vectors[i], t3f.matmul(matrix, tt_vectors[j]))
    or more shorly
      res[i, j] = tt_vectors[i]^T * matrix * tt_vectors[j]

  Args:
    tt_vectors: TensorTrainBatch.
    matrix: None, or TensorTrain matrix.

  Returns:
    tf.tensor with the Gram matrix.
  """
  ndims = tt_vectors.ndims()
  if matrix is None:
    curr_core = tt_vectors.tt_cores[0]
    res = tf.einsum('paijb,qcijd->pqbd', curr_core, curr_core)
    for core_idx in range(1, ndims):
      curr_core = tt_vectors.tt_cores[core_idx]
      res = tf.einsum('pqac,paijb,qcijd->pqbd', res, curr_core, curr_core)
  else:
    # res[i, j] = tt_vectors[i] ^ T * matrix * tt_vectors[j]
    vectors_shape = tt_vectors.get_shape()
    if vectors_shape[2] == 1 and vectors_shape[1] != 1:
      # TODO: not very efficient, better to use different order in einsum.
      tt_vectors = ops.transpose(tt_vectors)
    vectors_shape = tt_vectors.get_shape()
    if vectors_shape[1] != 1:
      # TODO: do something so that in case the shape is undefined on compilation
      # it still works.
      raise ValueError('The tt_vectors argument should be vectors (not '
                       'matrices) with shape defined on compilation.')
    curr_core = tt_vectors.tt_cores[0]
    curr_matrix_core = matrix.tt_cores[0]
    # We enumerate the dummy dimension (that takes 1 value) with `k`.
    res = tf.einsum('pakib,cijd,qekjf->pqbdf', curr_core, curr_matrix_core,
                    curr_core)
    for core_idx in range(1, ndims):
      curr_core = tt_vectors.tt_cores[core_idx]
      curr_matrix_core = matrix.tt_cores[core_idx]
      res = tf.einsum('pqace,pakib,cijd,qekjf->pqbdf', res, curr_core,
                      curr_matrix_core, curr_core)

  # Squeeze to make the result of size batch_size x batch_size instead of
  # batch_size x batch_size x 1 x 1.
  return tf.squeeze(res)