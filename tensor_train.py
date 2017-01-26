import numpy as np
import tensorflow as tf


# TODO: check the methods of _TensorLike
class TensorTrain:
  """Represents a Tensor Train object (a TT-tensor or TT-matrix).
  t3f represents a Tensor Train object as a tuple of TT-cores.
  ```
  @@__init__
  @@get_shape
  @@tt_cores
  @@dtype
  @@op
  @@graph
  @@get_raw_shape
  @@ndims
  @@extended_ranks
  @@is_tt_matrix
  """

  def __init__(self, tt_cores, convert_to_tensors=True):
    """Creates a `TensorTrain`.
    Args:
      tt_cores: A tuple of 3d or 4d tensor-like objects of shape
        `[r_k-1, n_k, r_k]`.
        Tensor-like can be numpy array, tf.Tensor, of tf.Variable
      convert_to_tensors: bool, if True than convert each element of the
        tt_cores tuple into a tf.Tensor (e.g. to initialize from np.array)
    Returns:
      A `TensorTrain`.
    """

    if not _are_tt_cores_valid(tt_cores):
      raise ValueError('the tt_cores provided to TensorTrain constructor are '
                       'not valid or have different dtypes.')

    tt_cores = list(tt_cores)
    if convert_to_tensors:
      with tf.name_scope("TensorTrain", tt_cores):
        # TODO: should we pass as_ref=True because we want to be able to update
        # values later if it is a VariableOp??
        for i in range(len(tt_cores)):
          name = "core%d" % i
          tt_cores[i] = tf.convert_to_tensor(
              tt_cores[i], name=name, as_ref=False)
    self._tt_cores = tuple(tt_cores)

  def get_raw_shape(self):
    """Get tuple of `TensorShapes` representing the shapes of the underlying TT-tensor.

    Tuple contains one `TensorShape` for TT-tensor and 2 `TensorShapes` for
    TT-matrix

    Returns:
      A tuple of `TensorShape` objects.
    """
    num_dims = self.ndims()
    num_tensor_shapes = len(self._tt_cores[0].get_shape().as_list()) - 2
    shapes = [[] for _ in range(num_tensor_shapes)]
    for dim in range(num_dims):
      curr_core_shape = self._tt_cores[dim].get_shape()
      for i in range(num_tensor_shapes):
        shapes[i].append(curr_core_shape[i + 1])
    for i in range(num_tensor_shapes):
      shapes[i] = tf.TensorShape(shapes[i])
    return shapes

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.
    Returns:
      A `TensorShape` object.
    """
    raw_shape = self.get_raw_shape()
    if self.is_tt_matrix():
      # TODO: return TensorShape.
      m = np.prod(raw_shape[0].as_list())
      n = np.prod(raw_shape[1].as_list())
      return m, n
    else:
      return self.get_raw_shape()[0]

  @property
  def tt_cores(self):
    """A tuple of TT-cores.
    Returns:
      A tuple of 3d or 4d tensors shape `[r_k-1, n_k, r_k]`.
    """
    return self._tt_cores

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    # TODO: where is this created?
    return self._tt_cores[0].dtype

  # TODO: it seems like instead of dense_shape we should use get_shape().
  # But maybe dense_shape() name is better?
  # @property
  # def dense_shape(self):
  #   """A 1-D Tensor of int64 representing the shape of the dense tensor."""
  #   return self._dense_shape

  @property
  def graph(self):
    """The `Graph` that contains the tt_cores tensors."""
    return self._tt_cores[0].graph

  def __str__(self):
    raise NotImplementedError
    # return "TensorTrain(indices=%s, values=%s, dense_shape=%s)" % (
    #     self._indices, self._values, self._dense_shape)

  def ndims(self):
    """Get the number of dimensions of the underlying TT-tensor.
    Returns:
      A number.
    """
    return len(self._tt_cores)

  def extended_ranks(self):
    """Get the ranks in an array of size `num_dims`+1.

    The first and the last ranks are guarantied to be 1.

    Returns:
      np.array of size `num_dims`+1.
    """
    # TODO: is TensorShape better than np array?
    num_dims = self.ndims()
    extended_ranks = np.ones(num_dims + 1).astype(int)
    for i in range(num_dims):
      extended_ranks[i] = self._tt_cores[i].get_shape().as_list()[0]
    return extended_ranks

  def is_tt_matrix(self):
    """Returns True if the TensorTrain object represents a TT-matrix.
    Returns:
      bool
    """
    return len(self._tt_cores[0].get_shape().as_list()) == 4

  def __getitem__(self, slice_spec):
    """Basic indexing, returns a `TensorTrain` containing the specified region.
    """
    new_tt_cores = []
    reminder = None
    for i in range(self.ndims()):
      curr_core = self._tt_cores[i]
      if self.is_tt_matrix():
        raise NotImplementedError
      else:
        sliced_core = curr_core[:, slice_spec[i], :]
        if len(curr_core.get_shape()) != len(sliced_core.get_shape()):
          # This index is specified exactly and we want to collapse this axis.
          if reminder is None:
            reminder = sliced_core
          else:
            reminder = tf.matmul(reminder, sliced_core)
        else:
          if reminder is not None:
            # Add reminder from the previous collapsed cores to the current
            # core.
            # TODO: is it bad to use as_list? E.g. if we dont now the ranks
            # on the graph construction stage.
            old_shape = sliced_core.get_shape().as_list()
            sliced_core = tf.reshape(sliced_core, (old_shape[0], -1))
            sliced_core = tf.matmul(reminder, sliced_core)
            sliced_core = tf.reshape(sliced_core, (-1, old_shape[1], old_shape[2]))
            reminder = None
          new_tt_cores.append(sliced_core)

    if reminder is not None:
      # The reminder obtained from collapsing the last cores.
      old_shape = new_tt_cores[-1].get_shape().as_list()
      new_tt_cores[-1] = tf.reshape(new_tt_cores[-1], (-1, old_shape[-1]))
      new_tt_cores[-1] = tf.matmul(new_tt_cores[-1], reminder)
      new_tt_cores[-1] = tf.reshape(new_tt_cores[-1], (old_shape[0], old_shape[1], 1))
      reminder = None
    return TensorTrain(new_tt_cores)

  def eval(self, feed_dict=None, session=None):
    """Evaluates this sparse tensor in a `Session`.
    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.
    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.
    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
        description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this sparse
        tensor. If none, the default session will be used.
    Returns:
      A `SparseTensorValue` object.
    """
    raise NotImplementedError
    # indices, values, dense_shape = _eval_using_default_session(
    #     [self.indices, self.values, self.dense_shape], feed_dict, self.graph,
    #     session)
    # return SparseTensorValue(indices, values, dense_shape)
  # TODO: do we need this?
  # @staticmethod
  # def _override_operator(operator, func):
  #   _override_helper(SparseTensor, operator, func)


def _are_tt_cores_valid(tt_cores):
  """Check if dimensions of the TT-cores are consistent and the dtypes coincide.

  Args:
    tt_cores: tuple of np.ndarray, tf.Tensor, or tf.Variable

  Returns:
    boolean, True if the dimensions and dtypes are consistent.
  """
  num_dims = len(tt_cores)

  def get_shape(core):
    try:
      # If core is np arrays.
      return core.shape
    except AttributeError:
      # If core is tf.Tensor or tf.Variable.
      return core.get_shape().as_list()

  for i in range(1, num_dims):
    if tt_cores[i].dtype != tt_cores[0].dtype:
      return False
    curr_core_shape = get_shape(tt_cores[i])
    prev_core_shape = get_shape(tt_cores[i - 1])
    if len(curr_core_shape) != len(prev_core_shape):
      # Shapes are inconsistent.
      return False
    if curr_core_shape[0] != prev_core_shape[-1]:
      # Ranks are inconsistent.
      return False
  if get_shape(tt_cores[0])[0] != 1 or get_shape(tt_cores[-1])[-1] != 1:
    # The first or the last rank is not 1.
    return False
  return True
