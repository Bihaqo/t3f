import numpy as np
import tensorflow as tf

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f.tensor_train_base import TensorTrainBase
from t3f import shapes


def _validate_input_parameters(is_tensor, shape, **params):
  """Internal function for validating input parameters

  Args:
    is_tensor: bool, determines whether we attempt to construct a TT-tensor or
      a TT-matrix (needed for the correct shape checks).
    shape: array, the desired shape of the generated TT object
    params: optional, possible values:
      batch_size: int, for constructing batches
      tt_rank: array or int, desired TT-ranks
  """

  if is_tensor:
    if len(shape.shape) != 1:
      raise ValueError('shape should be 1d array, got %a' % shape)
    if np.any(shape < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)
  else:
    if len(shape.shape) != 2:
      raise ValueError('shape should be 2d array, got %a' % shape)
    if shape[0].size != shape[1].size:
      raise ValueError('shape[0] should have the same length as shape[1], but'
                       'got %d and %d' % (shape[0].size, shape[1].size))
    if np.any(shape.flatten() < 1):
      raise ValueError('all elements in `shape` should be positive, got %a' %
                       shape)
    if not all(isinstance(sh, np.integer) for sh in shape.flatten()):
      raise ValueError('all elements in `shape` should be integers, got %a' %
                       shape)

  if 'batch_size' in params:
    batch_size = params['batch_size']
    if not isinstance(batch_size, (int, np.integer)):
      raise ValueError('`batch_size` should be integer, got %f' % batch_size)
    if batch_size < 1:
      raise ValueError('Batch size should be positive, got %d' % batch_size)
  if 'tt_rank' in params:
    tt_rank = params['tt_rank']
    if tt_rank.size == 1:
      if not isinstance(tt_rank[()], np.integer):
        raise ValueError('`tt_rank` should be integer, got %f' % tt_rank[()])
    if tt_rank.size > 1:
      if not all(isinstance(tt_r, np.integer) for tt_r in tt_rank):
        raise ValueError('all elements in `tt_rank` should be integers, got'
                         ' %a' % tt_rank)
    if np.any(tt_rank < 1):
      raise ValueError('`tt_rank` should be positive, got %a' % tt_rank)

    if is_tensor:
      if tt_rank.size != 1 and tt_rank.size != (shape.size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape.size + 1, tt_rank.size))
    else:
      if tt_rank.size != 1 and tt_rank.size != (shape[0].size + 1):
        raise ValueError('`tt_rank` array has inappropriate size, expected'
                         '1 or %d, got %d' % (shape[0].size + 1, tt_rank.size))


def tensor_ones(shape, dtype=tf.float32, name='t3f_tensor_ones'):
  """Generate TT-tensor of the given shape with all entries equal to 1.

  Args:
    shape: array representing the shape of the future tensor
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrain object containing a TT-tensor
  """

  shape = np.array(shape)
  _validate_input_parameters(is_tensor=True, shape=shape)
  num_dims = shape.size
  tt_rank = np.ones(num_dims + 1, dtype=np.int)

  with tf.name_scope(name):
    tt_cores = num_dims * [None]
    for i in range(num_dims):
      curr_core_shape = (1, shape[i], 1)
      tt_cores[i] = tf.ones(curr_core_shape, dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def tensor_zeros(shape, dtype=tf.float32, name='t3f_tensor_zeros'):
  """Generate TT-tensor of the given shape with all entries equal to 0.

  Args:
    shape: array representing the shape of the future tensor
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrain object containing a TT-tensor
  """

  shape = np.array(shape)
  _validate_input_parameters(is_tensor=True, shape=shape)
  num_dims = shape.size
  tt_rank = np.ones(num_dims + 1, dtype=np.int)
  tt_cores = num_dims * [None]
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (1, shape[i], 1)
      tt_cores[i] = tf.zeros(curr_core_shape, dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def eye(shape, dtype=tf.float32, name='t3f_eye'):
  """Creates an identity TT-matrix.

  Args:
    shape: array which defines the shape of the matrix row and column
      indices.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing an identity TT-matrix of size
    np.prod(shape) x np.prod(shape)
  """
  shape = np.array(shape)
  # In this special case shape is in the same format as in the TT-tensor case
  _validate_input_parameters(is_tensor=True, shape=shape)

  num_dims = shape.size
  tt_ranks = np.ones(num_dims + 1, dtype=np.int)

  with tf.name_scope(name):
    tt_cores = num_dims * [None]
    for i in range(num_dims):
      curr_core_shape = (1, shape[i], shape[i], 1)
      tt_cores[i] = tf.reshape(tf.eye(shape[i], dtype=dtype), curr_core_shape)

    true_shape = np.vstack([shape, shape])
    return TensorTrain(tt_cores, true_shape, tt_ranks)


def matrix_ones(shape, dtype=tf.float32, name='t3f_matrix_ones'):
  """Generate a TT-matrix of the given shape with each entry equal to 1.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        matrix_ones([[2, 2, 2], None])
      and
        matrix_ones([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1]) with each entry equal to 1
  """

  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)

  _validate_input_parameters(is_tensor=False, shape=shape)

  num_dims = shape[0].size
  tt_rank = np.ones(shape[0].size + 1, dtype=np.int)

  with tf.name_scope(name):
    tt_cores = [None] * num_dims
    for i in range(num_dims):
      curr_core_shape = (1, shape[0][i], shape[1][i], 1)
      tt_cores[i] = tf.ones(curr_core_shape, dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def matrix_zeros(shape, dtype=tf.float32, name='t3f_matrix_zeros'):
  """Generate a TT-matrix of the given shape with each entry equal to 0.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        matrix_zeros([[2, 2, 2], None])
      and
        matrix_zeros([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1]) with each entry equal to 0
  """

  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)

  _validate_input_parameters(is_tensor=False, shape=shape)
  num_dims = shape[0].size
  tt_rank = np.ones(shape[0].size + 1, dtype=np.int)

  with tf.name_scope(name):
    tt_cores = [None] * num_dims
    for i in range(num_dims):
      curr_core_shape = (1, shape[0][i], shape[1][i], 1)
      tt_cores[i] = tf.zeros(curr_core_shape, dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def tensor_with_random_cores(shape, tt_rank=2, mean=0., stddev=1.,
                             dtype=tf.float32,
                             name='t3f_tensor_with_random_cores'):
  """Generate a TT-tensor of the given shape with N(mean, stddev^2) cores.

  Args:
    shape: array representing the shape of the future tensor.
    tt_rank: a number or a (d+1)-element array with the desired ranks.
    mean: a number, the mean of the normal distribution used for
      initializing TT-cores.
    stddev: a number, the standard deviation of the normal distribution used
      for initializing TT-cores.
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-tensor
  """
  # TODO: good distribution to init training.
  # TODO: support shape and tt_ranks as TensorShape?.
  # TODO: support None as a dimension.
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=True, shape=shape, tt_rank=tt_rank)
  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)

  tt_rank = tt_rank.astype(int)
  tt_cores = [None] * num_dims
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (tt_rank[i], shape[i], tt_rank[i + 1])
      tt_cores[i] = tf.random.normal(curr_core_shape, mean=mean, stddev=stddev,
                                     dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def tensor_batch_with_random_cores(shape, tt_rank=2, batch_size=1,
                                   mean=0., stddev=1., dtype=tf.float32,
                                   name='t3f_tensor_batch_with_random_cores'):
  """Generate a batch of TT-tensors of given shape with N(mean, stddev^2) cores.

  Args:
    shape: array representing the shape of the future tensor.
    tt_rank: a number or a (d+1)-element array with ranks.
    batch_size: an integer.
    mean: a number, the mean of the normal distribution used for
      initializing TT-cores.
    stddev: a number, the standard deviation of the normal distribution used
      for initializing TT-cores.
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch containing TT-tensors
  """

  # TODO: support shape and tt_ranks as TensorShape?.
  # TODO: support None as a dimension.
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=True, shape=shape, tt_rank=tt_rank,
                             batch_size=batch_size)
  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)
  tt_rank = tt_rank.astype(int)
  tt_cores = [None] * num_dims
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (batch_size, tt_rank[i], shape[i], tt_rank[i + 1])
      tt_cores[i] = tf.random.normal(curr_core_shape, mean=mean, stddev=stddev,
                                     dtype=dtype)

    return TensorTrainBatch(tt_cores, shape, tt_rank, batch_size)


def matrix_with_random_cores(shape, tt_rank=2, mean=0., stddev=1.,
                             dtype=tf.float32,
                             name='t3f_matrix_with_random_cores'):
  """Generate a TT-matrix of given shape with N(mean, stddev^2) cores.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        matrix_with_random_cores([[2, 2, 2], None])
      and
        matrix_with_random_cores([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    mean: a number, the mean of the normal distribution used for
      initializing TT-cores.
    stddev: a number, the standard deviation of the normal distribution used
      for initializing TT-cores.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.concatenate([[1], tt_rank, [1]])

  tt_rank = tt_rank.astype(int)
  tt_cores = [None] * num_dims
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (tt_rank[i], shape[0][i], shape[1][i],
                         tt_rank[i + 1])
      tt_cores[i] = tf.random.normal(curr_core_shape, mean=mean, stddev=stddev,
                                     dtype=dtype)

    return TensorTrain(tt_cores, shape, tt_rank)


def matrix_batch_with_random_cores(shape, tt_rank=2, batch_size=1,
                                   mean=0., stddev=1., dtype=tf.float32,
                                   name='t3f_matrix_batch_with_random_cores'):
  """Generate a batch of TT-matrices of given shape with N(mean, stddev^2) cores.

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        matrix_batch_with_random_cores([[2, 2, 2], None])
      and
        matrix_batch_with_random_cores([None, [2, 2, 2]])
    will create a batch of one 8-element column and row vector correspondingly.

    tt_rank: a number or a (d+1)-element array with ranks.
    batch_size: an integer.
    mean: a number, the mean of the normal distribution used for
      initializing TT-cores.
    stddev: a number, the standard deviation of the normal distribution used
      for initializing TT-cores.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch containing a batch of TT-matrices of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank,
                             batch_size=batch_size)
  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.concatenate([[1], tt_rank, [1]])
  shape = shape.astype(int)
  tt_rank = tt_rank.astype(int)
  tt_cores = [None] * num_dims
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_shape = (batch_size, tt_rank[i], shape[0][i], shape[1][i],
                         tt_rank[i + 1])
      tt_cores[i] = tf.random.normal(curr_core_shape, mean=mean, stddev=stddev,
                                     dtype=dtype)

    return TensorTrainBatch(tt_cores, shape, tt_rank, batch_size)


def ones_like(tt, name='t3f_ones_like'):
  """Constructs t3f.ones with the shape of `tt`.

  In the case when `tt` is TensorTrainBatch constructs t3f.ones with the shape
  of a TensorTrain in `tt`.

  Args:
    tt: TensorTrain object
    name: string, name of the Op.

  Returns:
    TensorTrain object of the same shape as `tt` but with all entries equal to
    1.

  """
  if not isinstance(tt, TensorTrainBase):
    raise ValueError("`tt` has to be a Tensor Train object")
  else:
    shape = shapes.lazy_raw_shape(tt)
    # I guess variables=tt.tt_cores is not needed here since the output of
    # the function doesn't depend on the values of the TT-cores, only on their
    # shapes etc. But I'm not 100% sure.
    with tf.name_scope(name):
      if tt.is_tt_matrix():
        return matrix_ones(shape, dtype=tt.dtype)
      else:
        return tensor_ones(shape[0, :], dtype=tt.dtype)


def zeros_like(tt, name='t3f_zeros_like'):
  """Constructs t3f.zeros with the shape of `tt`.

  In the case when `tt` is a TensorTrainBatch constructs t3f.zeros with
  the shape of a TensorTrain in `tt`.

  Args:
    tt: TensorTrain object
    name: string, name of the Op.

  Returns:
    TensorTrain object of the same shape as `tt` but with all entries equal to
    0.

  """
  if not isinstance(tt, TensorTrainBase):
    raise ValueError("`tt` has to be a Tensor Train object")
  else:
    shape = shapes.lazy_raw_shape(tt)
    # I guess variables=tt.tt_cores is not needed here since the output of
    # the function doesn't depend on the values of the TT-cores, only on their
    # shapes etc. But I'm not 100% sure.
    with tf.name_scope(name):
      if tt.is_tt_matrix():
        return matrix_zeros(shape, dtype=tt.dtype)
      else:
        return tensor_zeros(shape[0, :], dtype=tt.dtype)


def random_tensor(shape, tt_rank=2, mean=0., stddev=1., dtype=tf.float32,
                  name='t3f_random_tensor'):
  """Generate a random TT-tensor of the given shape with given mean and stddev.

  Entries of the generated tensor (in the full format) will be iid and satisfy
  E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
  in fact not Gaussian (but is close for large tensors).

  In the current implementation only mean 0 is supported. To get
  a random_tensor with specified mean but tt_rank greater by 1 you can
  call
  x = t3f.random_tensor(shape, tt_rank, stddev=stddev)
  x = mean * t3f.ones_like(x) + x

  Args:
    shape: array representing the shape of the future tensor.
    tt_rank: a number or a (d+1)-element array with the desired ranks.
    mean: a number, the desired mean for the distribution of entries.
    stddev: a number, the desired standard deviation for the distribution of
      entries.
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-tensor
  """
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=True, shape=shape, tt_rank=tt_rank)

  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)

  tt_rank = tt_rank.astype(int)

  # Empirically entries of a TT tensor with cores initialized from N(0, 1)
  # will have variances np.prod(tt_rank) and mean 0.
  # We scale each TT-core to obtain the desired stddev

  cr_exponent = -1.0 / (2 * num_dims)
  var = np.prod(tt_rank ** cr_exponent)
  core_stddev = stddev ** (1.0 / num_dims) * var
  with tf.name_scope(name):
    tt = tensor_with_random_cores(shape, tt_rank=tt_rank, stddev=core_stddev,
                                  dtype=dtype)

  if np.abs(mean) < 1e-8:
    return tt
  else:
    raise NotImplementedError('non-zero mean is not supported yet')


def random_tensor_batch(shape, tt_rank=2, batch_size=1, mean=0., stddev=1.,
                        dtype=tf.float32, name='t3f_random_tensor_batch'):
  """Generate a batch of TT-tensors with given shape, mean and stddev.

  Entries of the generated tensors (in the full format) will be iid and satisfy
  E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
  in fact not Gaussian (but is close for large tensors).

  In the current implementation only mean 0 is supported. To get
  a random_tensor_batch with specified mean but tt_rank greater by 1 you can
  call
  x = t3f.random_tensor_batch(shape, tt_rank, batch_size=bs, stddev=stddev)
  x = mean * t3f.ones_like(x) + x

  Args:
    shape: array representing the shape of the future tensor.
    tt_rank: a number or a (d+1)-element array with ranks.
    batch_size: an integer.
    mean: a number, the desired mean for the distribution of entries.
    stddev: a number, the desired standard deviation for the distribution of
      entries.
    dtype: [tf.float32] dtype of the resulting tensor.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch containing TT-tensors.
  """
  # TODO: support shape and tt_ranks as TensorShape?.
  # TODO: support None as a dimension.
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=True, shape=shape, tt_rank=tt_rank,
                             batch_size=batch_size)
  num_dims = shape.size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.insert(tt_rank, 0, 1)
    tt_rank = np.append(tt_rank, 1)
  tt_rank = tt_rank.astype(int)

  cr_exponent = -1.0 / (2 * num_dims)
  var = np.prod(tt_rank ** cr_exponent)
  cr_stddev = stddev ** (1.0 / num_dims) * var
  with tf.name_scope(name):
    tt = tensor_batch_with_random_cores(shape, tt_rank=tt_rank, stddev=cr_stddev,
                                        batch_size=batch_size, dtype=dtype)

  if np.abs(mean) < 1e-8:
    return tt
  else:
    raise NotImplementedError('non-zero mean is not supported yet')


def random_matrix(shape, tt_rank=2, mean=0., stddev=1.,
                  dtype=tf.float32, name='t3f_random_matrix'):
  """Generate a random TT-matrix of the given shape with given mean and stddev.

  Entries of the generated matrix (in the full format) will be iid and satisfy
  E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the distribution is
  in fact not Gaussian.

  In the current implementation only mean 0 is supported. To get
  a random_matrix with specified mean but tt_rank greater by 1 you can call
  x = t3f.random_matrix(shape, tt_rank, stddev=stddev)
  x = mean * t3f.ones_like(x) + x

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        random_matrix([[2, 2, 2], None])
      and
        random_matrix([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    mean: a number, the desired mean for the distribution of entries.
    stddev: a number, the desired standard deviation for the distribution of
      entries.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # TODO: good distribution to init training.
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)

  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)

  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.concatenate([[1], tt_rank, [1]])

  tt_rank = tt_rank.astype(int)
  var = np.prod(tt_rank)

  # Empirically entries of a TT tensor with cores initialized from N(0, 1)
  # will have variances np.prod(tt_rank) and mean 0.
  # We scale each TT-core to obtain the desired stddev

  cr_exponent = -1.0 / (2 * num_dims)
  var = np.prod(tt_rank ** cr_exponent)
  core_stddev = stddev ** (1.0 / num_dims) * var
  with tf.name_scope(name):
    tt = matrix_with_random_cores(shape, tt_rank=tt_rank, stddev=core_stddev,
                                  dtype=dtype)

  if np.abs(mean) < 1e-8:
    return tt
  else:
    raise NotImplementedError('non-zero mean is not supported yet')


def random_matrix_batch(shape, tt_rank=2, batch_size=1, mean=0., stddev=1.,
                        dtype=tf.float32, name='t3f_random_matrix_batch'):
  """Generate a batch of TT-matrices with given shape, mean and stddev.

  Entries of the generated matrices (in the full format) will be iid and
  satisfy E[x_{i1i2..id}] = mean, Var[x_{i1i2..id}] = stddev^2, but the
  distribution is in fact not Gaussian.

  In the current implementation only mean 0 is supported. To get a
  random_matrix_batch with specified mean but tt_rank greater by 1 you can call
  x = t3f.random_matrix_batch(shape, tt_rank, batch_size=bs, stddev=stddev)
  x = mean * t3f.ones_like(x) + x

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        random_matrix_batch([[2, 2, 2], None])
      and
        random_matrix_batch([None, [2, 2, 2]])
    will create a batch of one 8-element column and row vector correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    batch_size: an integer.
    mean: a number, the desired mean for the distribution of entries.
    stddev: a number, the desired standard deviation for the distribution of
      entries.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrainBatch containing a batch of TT-matrices of size
      np.prod(shape[0]) x np.prod(shape[1])
  """
  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank,
                             batch_size=batch_size)
  num_dims = shape[0].size
  if tt_rank.size == 1:
    tt_rank = tt_rank * np.ones(num_dims - 1, dtype=np.int)
    tt_rank = np.concatenate([[1], tt_rank, [1]])

  shape = shape.astype(int)
  tt_rank = tt_rank.astype(int)

  cr_exponent = -1.0 / (2 * num_dims)
  var = np.prod(tt_rank ** cr_exponent)
  core_stddev = stddev ** (1.0 / num_dims) * var
  with tf.name_scope(name):
    tt = matrix_batch_with_random_cores(shape, tt_rank=tt_rank,
                                        stddev=core_stddev,
                                        batch_size=batch_size,
                                        dtype=dtype)

  if np.abs(mean) < 1e-8:
    return tt
  else:
    raise NotImplementedError('non-zero mean is not supported yet')


def glorot_initializer(shape, tt_rank=2, dtype=tf.float32,
                       name='t3f_glorot_initializer'):
  """Constructs a random TT matrix with entrywise variance 2.0 / (n_in + n_out)

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        glorot_initializer([[2, 2, 2], None])
      and
        glorot_initializer([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """

  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)
  n_in = np.prod(shape[0])
  n_out = np.prod(shape[1])
  lamb = 2.0 / (n_in + n_out)

  with tf.name_scope(name):
    return random_matrix(shape, tt_rank=tt_rank, stddev=np.sqrt(lamb),
                         dtype=dtype)


def he_initializer(shape, tt_rank=2, dtype=tf.float32,
                   name='t3f_he_initializer'):
  """Constructs a random TT matrix with entrywise variance 2.0 / n_in

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        he_initializer([[2, 2, 2], None])
      and
        he_initializer([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """

  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)
  n_in = np.prod(shape[0])
  lamb = 2.0 / n_in

  with tf.name_scope(name):
    return random_matrix(shape, tt_rank=tt_rank, stddev=np.sqrt(lamb),
                         dtype=dtype)


def lecun_initializer(shape, tt_rank=2, dtype=tf.float32,
                      name='t3f_lecun_initializer'):
  """Constructs a random TT matrix with entrywise variance 1.0 / n_in

  Args:
    shape: 2d array, shape[0] is the shape of the matrix row-index,
      shape[1] is the shape of the column index.
      shape[0] and shape[1] should have the same number of elements (d)
      Also supports omitting one of the dimensions for vectors, e.g.
        lecun_initializer([[2, 2, 2], None])
      and
        lecun_initializer([None, [2, 2, 2]])
      will create an 8-element column and row vectors correspondingly.
    tt_rank: a number or a (d+1)-element array with ranks.
    dtype: [tf.float32] dtype of the resulting matrix.
    name: string, name of the Op.

  Returns:
    TensorTrain containing a TT-matrix of size
      np.prod(shape[0]) x np.prod(shape[1])
  """

  # In case the shape is immutable.
  shape = list(shape)
  # In case shape represents a vector, e.g. [None, [2, 2, 2]]
  if shape[0] is None:
    shape[0] = np.ones(len(shape[1]), dtype=int)
  # In case shape represents a vector, e.g. [[2, 2, 2], None]
  if shape[1] is None:
    shape[1] = np.ones(len(shape[0]), dtype=int)
  shape = np.array(shape)
  tt_rank = np.array(tt_rank)
  _validate_input_parameters(is_tensor=False, shape=shape, tt_rank=tt_rank)
  n_in = np.prod(shape[0])
  lamb = 1.0 / n_in
  with tf.name_scope(name):
    return random_matrix(shape, tt_rank=tt_rank, stddev=np.sqrt(lamb),
                         dtype=dtype)
