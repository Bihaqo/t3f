import tensorflow as tf

from tensor_train_batch import TensorTrainBatch


def concat_along_batch_dim(tt_list):
  """Concat all TensorTrainBatch objects along batch dimension.

  Args:
    tt_list: a list of TensorTrainBatch objects.

  Returns:
    TensorTrainBatch
  """
  ndims = tt_list[0].ndims()
  for core_idx in range(ndims):
    if not isinstance(tt_list[core_idx], TensorTrainBatch):
      raise ValueError('All objects in the list should be TTBatch objects, got '
                       '%s' % (tt_list[core_idx]))
  for core_idx in range(1, ndims):
    if tt_list[core_idx].get_raw_shape() != tt_list[0].get_raw_shape():
      raise ValueError('Shapes of all TT-batch objects should coincide, got %s '
                       'and %s' % (tt_list[0].get_raw_shape(),
                                   tt_list[core_idx].get_raw_shape()))
    if tt_list[core_idx].get_tt_ranks() != tt_list[0].get_tt_ranks():
      raise ValueError('TT-ranks of all TT-batch objects should coincide, got '
                       '%s and %s' % (tt_list[0].get_tt_ranks(),
                                      tt_list[core_idx].get_tt_ranks()))

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
    tf.tensor with the gram matrix.
  """
