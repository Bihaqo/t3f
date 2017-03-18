import tensorflow as tf

from tensor_train_base import TensorTrainBase
import shapes


class TensorTrain(TensorTrainBase):
  """Represents a Tensor Train object (a TT-tensor or TT-matrix).

  t3f represents a Tensor Train object as a tuple of TT-cores.
  ```
  @@__init__
  @@get_raw_shape
  @@get_shape
  @@tt_cores
  @@dtype
  @@name
  @@graph
  @@ndims
  @@get_tt_ranks
  @@is_tt_matrix
  @@is_variable
  @@eval
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
      raise ValueError('The tt_cores provided to TensorTrain constructor are '
                       'not valid, have different dtypes, or are inconsistent '
                       'with the provided shape or TT-ranks.')

    self._tt_cores = tuple(tt_cores)
    self._raw_shape = shapes.clean_raw_shape(shape)
    if self._raw_shape is None:
      self._raw_shape = _infer_raw_shape(self._tt_cores)
    self._tt_ranks = None if tt_ranks is None else tf.TensorShape(tt_ranks)
    if self._tt_ranks is None:
      self._tt_ranks = _infer_tt_ranks(self._tt_cores)

  @property
  def tt_cores(self):
    """A tuple of TT-cores.

    Returns:
      A tuple of 3d or 4d tensors shape
        `[r_k-1, n_k, r_k]`
      or
        `[r_k-1, n_k, m_k, r_k]`
    """
    return self._tt_cores

  def __str__(self):
    """A string describing the TensorTrain object, its TT-rank, and shape."""
    shape = self.get_shape()
    tt_ranks = self.get_tt_ranks()
    variable_str = ' variable' if self.is_variable() else ''
    if self.is_tt_matrix():
      raw_shape = self.get_raw_shape()
      return "A TT-Matrix%s of size %d x %d, underlying tensor " \
             "shape: %s x %s, TT-ranks: %s" % (variable_str, shape[0], shape[1],
                                               raw_shape[0], raw_shape[1],
                                               tt_ranks)
    else:
      return "A Tensor Train%s of shape %s, TT-ranks: %s" % (variable_str,
                                                              shape, tt_ranks)

  def __getitem__(self, slice_spec):
    """Basic indexing, returns a `TensorTrain` containing the specified region.

    Examples:
      >>> a = t3f.random_tensor((2, 3, 4))
      >>> a[1, :, :]
      is a 2D TensorTrain 3 x 4.
      >>> a[1:2, :, :]
      is a 3D TensorTrain 1 x 3 x 4
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
            sliced_core = tf.einsum('ab,bid->aid', remainder, sliced_core)
            remainder = None
          new_tt_cores.append(sliced_core)

    if remainder is not None:
      # The reminder obtained from collapsing the last cores.
      new_tt_cores[-1] = tf.einsum('aib,bd->aid', new_tt_cores[-1], remainder)
      remainder = None
    # TODO: infer the output ranks and shape.
    return TensorTrain(new_tt_cores)


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
  shape = shapes.clean_raw_shape(shape)
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


def _infer_raw_shape(tt_cores):
  """Tries to infer the (static) raw shape from the TT-cores."""
  num_dims = len(tt_cores)
  num_tensor_shapes = len(tt_cores[0].get_shape().as_list()) - 2
  raw_shape = [[] for _ in range(num_tensor_shapes)]
  for dim in range(num_dims):
    curr_core_shape = tt_cores[dim].get_shape()
    for i in range(num_tensor_shapes):
      raw_shape[i].append(curr_core_shape[i + 1])
  for i in range(num_tensor_shapes):
    raw_shape[i] = tf.TensorShape(raw_shape[i])
  return tuple(raw_shape)


def _infer_tt_ranks(tt_cores):
  """Tries to infer the (static) raw shape from the TT-cores."""
  tt_ranks = []
  for i in range(len(tt_cores)):
    tt_ranks.append(tt_cores[i].get_shape()[0])
  tt_ranks.append(tt_cores[-1].get_shape()[-1])
  return tf.TensorShape(tt_ranks)
