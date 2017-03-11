import numpy as np
import tensorflow as tf


# TODO: check the methods of _TensorLike
class TensorTrain(object):
  """Represents a Tensor Train object (a TT-tensor or TT-matrix).
  t3f represents a Tensor Train object as a tuple of TT-cores.
  ```
  @@__init__
  @@get_shape
  @@name
  @@tt_cores
  @@dtype
  @@op
  @@graph
  @@get_raw_shape
  @@ndims
  @@get_tt_ranks
  @@is_tt_matrix
  """

  def __init__(self, tt_cores, shape=None, tt_ranks=None, convert_to_tensors=True):
    """Creates a `TensorTrain`.
    Args:
      tt_cores: A tuple of 3d or 4d tensor-like objects of shape
        `[r_k-1, n_k, r_k]`.
        Tensor-like can be numpy array, tf.Tensor, of tf.Variable
      shape: Shape of the underlying tensor. If None, tries to infer from the
        cores (not always possible even if it should be, e.g. if ranks are
        unknown, than the whole shape of a core can be unknown).
      tt_ranks: a TensorShape of length d+1 (d is the dimensionality of
        the underlying tensor). The first and the last ranks are assumed to
        equal to 1. If None, tries to infer the ranks from the cores.
      convert_to_tensors: bool, if True than convert each element of the
        tt_cores tuple into a tf.Tensor (e.g. to initialize from np.array)

    Returns:
      A `TensorTrain`.

    Raises:
      ValueError if the provided TT-cores are not valid or inconsistent with
        the provided shape.
    """
    tt_cores = list(tt_cores)
    if convert_to_tensors:
      # TODO: what does this namescope do?
      with tf.name_scope("TensorTrain", tt_cores):
        for i in range(len(tt_cores)):
          name = "core%d" % i
          tt_cores[i] = tf.convert_to_tensor(tt_cores[i], name=name)

    if not _are_tt_cores_valid(tt_cores, shape, tt_ranks):
      raise ValueError('the tt_cores provided to TensorTrain constructor are '
                       'not valid, have different dtypes, or are inconsistent '
                       'with the provided shape.')

    self._tt_cores = tuple(tt_cores)
    self._shape = _clean_shape(shape)
    self._tt_ranks = None if tt_ranks is None else tf.TensorShape(tt_ranks)

  def get_raw_shape(self):
    """Get tuple of `TensorShapes` representing the shapes of the underlying TT-tensor.

    Tuple contains one `TensorShape` for TT-tensor and 2 `TensorShapes` for
    TT-matrix

    Returns:
      A tuple of `TensorShape` objects.
    """
    if self._shape is not None:
      return self._shape
    else:
      num_dims = self.ndims()
      num_tensor_shapes = len(self.tt_cores[0].get_shape().as_list()) - 2
      shapes = [[] for _ in range(num_tensor_shapes)]
      for dim in range(num_dims):
        curr_core_shape = self.tt_cores[dim].get_shape()
        for i in range(num_tensor_shapes):
          shapes[i].append(curr_core_shape[i + 1])
      for i in range(num_tensor_shapes):
        shapes[i] = tf.TensorShape(shapes[i])
      return tuple(shapes)

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.
    Returns:
      A `TensorShape` object.
    """
    raw_shape = self.get_raw_shape()
    if self.is_tt_matrix():
      m = np.prod(raw_shape[0].as_list())
      n = np.prod(raw_shape[1].as_list())
      return tf.TensorShape((m, n))
    else:
      return raw_shape[0]

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
    return self.tt_cores[0].dtype

  @property
  def name(self):
    """The name of the TensorTrain.

    Returns:
      String, the scope in which the TT-cores are defined.
    """
    core_name = self.tt_cores[0].name
    idx = core_name.rfind('/')
    return core_name[:idx]

  # TODO: it seems like instead of dense_shape we should use get_shape().
  # But maybe dense_shape() name is better?
  # @property
  # def dense_shape(self):
  #   """A 1-D Tensor of int64 representing the shape of the dense tensor."""
  #   return self._dense_shape

  @property
  def graph(self):
    """The `Graph` that contains the tt_cores tensors."""
    return self.tt_cores[0].graph

  def __str__(self):
    raise NotImplementedError
    # return "TensorTrain(indices=%s, values=%s, dense_shape=%s)" % (
    #     self._indices, self._values, self._dense_shape)

  def ndims(self):
    """Get the number of dimensions of the underlying TT-tensor.
    Returns:
      A number.
    """
    return len(self.tt_cores)

  def get_tt_ranks(self):
    """Get the TT-ranks in an array of size `num_dims`+1.

    The first and the last TT-rank are guarantied to be 1.

    Returns:
      TensorShape of size `num_dims`+1.
    """
    if self._tt_ranks is not None:
      return self._tt_ranks
    else:
      ranks = []
      for i in range(self.ndims()):
        ranks.append(self.tt_cores[i].get_shape()[0])
      ranks.append(self.tt_cores[-1].get_shape()[-1])
      return tf.TensorShape(ranks)

  def is_tt_matrix(self):
    """Returns True if the TensorTrain object represents a TT-matrix.
    Returns:
      bool
    """
    if self._shape is not None:
      return len(self._shape) == 2
    return len(self.tt_cores[0].get_shape().as_list()) == 4

  def __getitem__(self, slice_spec):
    """Basic indexing, returns a `TensorTrain` containing the specified region.
    """
    new_tt_cores = []
    remainder = None
    for i in range(self.ndims()):
      curr_core = self.tt_cores[i]
      if self.is_tt_matrix():
        raise NotImplementedError
      else:
        sliced_core = curr_core[:, slice_spec[i], :]
        if len(curr_core.get_shape()) != len(sliced_core.get_shape()):
          # This index is specified exactly and we want to collapse this axis.
          if remainder is None:
            remainder = sliced_core
          else:
            remainder = tf.matmul(remainder, sliced_core)
        else:
          if remainder is not None:
            # Add reminder from the previous collapsed cores to the current
            # core.
            # TODO: is it bad to use as_list? E.g. if we dont now the ranks
            # on the graph construction stage.
            old_shape = sliced_core.get_shape().as_list()
            sliced_core = tf.reshape(sliced_core, (old_shape[0], -1))
            sliced_core = tf.matmul(remainder, sliced_core)
            sliced_core = tf.reshape(sliced_core, (-1, old_shape[1], old_shape[2]))
            remainder = None
          new_tt_cores.append(sliced_core)

    if remainder is not None:
      # The reminder obtained from collapsing the last cores.
      old_shape = new_tt_cores[-1].get_shape().as_list()
      new_tt_cores[-1] = tf.reshape(new_tt_cores[-1], (-1, old_shape[-1]))
      new_tt_cores[-1] = tf.matmul(new_tt_cores[-1], remainder)
      new_tt_cores[-1] = tf.reshape(new_tt_cores[-1], (old_shape[0], old_shape[1], 1))
      remainder = None
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
      ????
    """
    # TODO: what to return, None?
    if session is None:
      session = tf.get_default_session()
    session.run(self.tt_cores)

  # TODO: do we need this?
  # @staticmethod
  # def _override_operator(operator, func):
  #   _override_helper(SparseTensor, operator, func)

  def __add__(self, other):
    """Returns a TensorTrain corresponding to element-wise sum tt_a + tt_b.

    Just calls t3f.add, see its documentation for details.
    """
    # TODO: ugly.
    # We can't import ops in the beginning since it creates cyclic dependencies.
    import ops
    return ops.add(self, other)

  def __mul__(self, other):
    """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b.

    Just calls t3f.multiply, see its documentation for details.
    """
    # TODO: ugly.
    # We can't import ops in the beginning since it creates cyclic dependencies.
    import ops
    return ops.multiply(self, other)


def _clean_shape(shape):
  """Returns a tuple of TensorShapes for any valid shape representation.

  Args:
    shape: An np.array, a tf.TensorShape (for tensors), a tuple of
      tf.TensorShapes (for TT-matrices or tensors), or None

  Returns:
    A tuple of tf.TensorShape, or None is the input is None
  """
  if shape is None:
    return None
  if isinstance(shape, tf.TensorShape) or isinstance(shape[0], tf.TensorShape):
    # Assume tf.TensorShape.
    if isinstance(shape, tf.TensorShape):
      shape = tuple((shape,))
  else:
    np_shape = np.array(shape)
    # Make sure that the shape is 2-d array both for tensors and TT-matrices.
    np_shape = np.squeeze(np_shape)
    if len(np_shape.shape) == 1:
      # A tensor.
      np_shape = [np_shape]
    shape = []
    for i in range(len(np_shape)):
      shape.append(tf.TensorShape(np_shape[i]))
    shape = tuple(shape)
  return shape


def _are_tt_cores_valid(tt_cores, shape, tt_ranks):
  """Check if dimensions of the TT-cores are consistent and the dtypes coincide.

  Args:
    tt_cores: a tuple of `Tensor` objects
    shape: An np.array, a tf.TensorShape (for tensors), a tuple of
      tf.TensorShapes (for TT-matrices or tensors), or None
    tt_ranks: An np.array or a tf.TensorShape of length len(tt_cores)+1.

  Returns:
    boolean, True if the dimensions and dtypes are consistent.
  """
  shape = _clean_shape(shape)
  num_dims = len(tt_cores)

  for core_idx in range(1, num_dims):
    if tt_cores[core_idx].dtype != tt_cores[0].dtype:
      return False
  try:
    for core_idx in range(num_dims):
      curr_core_shape = tt_cores[core_idx].get_shape()
      if len(curr_core_shape) != len(tt_cores[0].get_shape()):
        # Shapes are inconsistent.
        return False
      if shape is not None:
        for i in range(len(shape)):
          if curr_core_shape[i + 1] != shape[i][core_idx]:
            # The TT-cores are not aligned with the given shape.
            return False
      if core_idx >= 1:
        prev_core_shape = tt_cores[core_idx - 1].get_shape()
        if curr_core_shape[0] != prev_core_shape[-1]:
          # TT-ranks are inconsistent.
          return False
      if tt_ranks is not None:
        if curr_core_shape[0] != tt_ranks[core_idx]:
          # The TT-ranks are not aligned with the TT-cores shape.
          return False
        if curr_core_shape[-1] != tt_ranks[core_idx + 1]:
          # The TT-ranks are not aligned with the TT-cores shape.
          return False
    if tt_cores[0].get_shape()[0] != 1 or tt_cores[-1].get_shape()[-1] != 1:
      # The first or the last rank is not 1.
      return False
  except ValueError:
    # The shape of the TT-cores is undetermined, can not validate it.
    pass
  return True
