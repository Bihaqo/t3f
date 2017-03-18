import numbers
import numpy as np
import tensorflow as tf

from tensor_train_base import TensorTrainBase
from tensor_train import TensorTrain
import shapes


class TensorTrainBatch(TensorTrainBase):
  """Represents a batch of Tensor Train objects (TT-tensors or TT-matrices).

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

  def __init__(self, tt_cores, shape=None, tt_ranks=None, batch_size=None,
               convert_to_tensors=True):
    """Creates a `TensorTrain`.

    Args:
      tt_cores: A tuple of 3d or 4d tensor-like objects of shape
        `[r_k-1, n_k, r_k]`.
        Tensor-like can be numpy array, tf.Tensor, of tf.Variable
      batch_size: number of elements in the batch. If None, tries to infer from
        the TT-cores (not always possible even if it should be, e.g. if ranks
        are unknown, than the whole shape of a core can be unknown).
      shape: Shape of the underlying tensor. If None, tries to infer from the
        TT-cores.
      tt_ranks: a TensorShape of length d+1 (d is the dimensionality of
        the underlying tensor). The first and the last ranks are assumed to
        equal to 1. If None, tries to infer the ranks from the cores.
      convert_to_tensors: bool, if True than convert each element of the
        tt_cores tuple into a tf.Tensor (e.g. to initialize from np.array)

    Returns:
      A `TensorTrainBatch`.

    Raises:
      ValueError if the provided TT-cores are not valid or inconsistent with
        the provided shape.
    """
    tt_cores = list(tt_cores)
    if convert_to_tensors:
      # TODO: what does this namescope do?
      with tf.name_scope("TensorTrainBatch", tt_cores):
        for i in range(len(tt_cores)):
          name = "core%d" % i
          tt_cores[i] = tf.convert_to_tensor(tt_cores[i], name=name)

    if not _are_batch_tt_cores_valid(tt_cores, shape, tt_ranks, batch_size):
      raise ValueError('The tt_cores provided to TensorTrainBatch constructor '
                       'are not valid, have different dtypes, or are '
                       'inconsistent with the provided batch_size, shape, or '
                       'TT-ranks.')

    self._tt_cores = tuple(tt_cores)
    if batch_size is None:
      self._batch_size = tt_cores[0].get_shape()[0].value
    else:
      self._batch_size = batch_size
    self._raw_shape = shapes.clean_raw_shape(shape)
    if self._raw_shape is None:
      self._raw_shape = _infer_batch_raw_shape(self._tt_cores)
    self._tt_ranks = None if tt_ranks is None else tf.TensorShape(tt_ranks)
    if self._tt_ranks is None:
      self._tt_ranks = _infer_batch_tt_ranks(self._tt_cores)

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    The first dimension is the batch_size.

    Returns:
      A `TensorShape` object.
    """
    shape = TensorTrainBase.get_shape(self)
    return tf.TensorShape(np.hstack((self.batch_size, shape)))

  @property
  def tt_cores(self):
    """A tuple of TT-cores.

    Returns:
      A tuple of 4d or 5d tensors shape
        `[batch_size, r_k-1, n_k, r_k]`
      or
        `[batch_size, r_k-1, n_k, m_k, r_k]`
    """
    return self._tt_cores

  @property
  def batch_size(self):
    """The number of elements or None if not known."""
    return self._batch_size

  def __str__(self):
    """A string describing the TensorTrainBatch, its TT-rank and shape."""
    shape = self.get_shape()
    tt_ranks = self.get_tt_ranks()

    if self.is_tt_matrix():
      raw_shape = self.get_raw_shape()
      type_str = 'TT-matrix variables' if self.is_variable() else 'TT-matrices'
      return "A %d element batch of %s of size %d x %d, underlying tensor " \
             "shape: %s x %s, TT-ranks: %s" % (self.batch_size, type_str,
                                               shape[0], shape[1],
                                               raw_shape[0], raw_shape[1],
                                               tt_ranks)
    else:
      if self.is_variable():
        type_str = 'Tensor Train variables'
      else:
        type_str = 'Tensor Trains'
      return "A %d element batch of %s of shape %s, TT-ranks: %s" % \
             (self.batch_size, type_str, shape, tt_ranks)

  def _batch_dim_getitem(self, element_spec):
    """__getitem__ when provided only one (batch) index.

    Examples:
      a[1]
      a[1:3]
    """

    # This object index is specified exactly and we want to collapse the
    # batch_size axis, i.e. return a TensorTrain instead of a TensorTrainBatch.
    do_collapse_batch_dim = isinstance(element_spec, numbers.Number)
    if not isinstance(element_spec, slice) and not do_collapse_batch_dim:
      raise ValueError('Expected just 1 index, got %s' % element_spec)

    new_tt_cores = []
    for core_idx in range(self.ndims()):
      curr_core = self.tt_cores[core_idx]
      if self.is_tt_matrix():
        new_tt_cores.append(curr_core[element_spec, :, :, :, :])
      else:
        new_tt_cores.append(curr_core[element_spec, :, :, :])
    if do_collapse_batch_dim:
      # This index is specified exactly and we want to collapse the batch_size
      # axis, i.e. return a TensorTrain instead of a TensorTrainBatch.
      return TensorTrain(new_tt_cores, self.get_raw_shape(),
                         self.get_tt_ranks())
    else:
      batch_size = new_tt_cores[0].get_shape()[0].value
      return TensorTrainBatch(new_tt_cores, self.get_raw_shape(),
                              self.get_tt_ranks(), batch_size)

  def _full_getitem(self, slice_spec):
    """__getitem__ when provided full index of length ndims + 1.

    Examples:
      a = t3f.random_tensor_batch((2, 3, 4), batch_size=5)
      a[:3, 1:2, 4, :]
    """
    if len(slice_spec) != self.ndims() + 1:
      raise ValueError('Expected %d indices, got %d' % (self.ndims() + 1,
                                                        len(slice_spec)))
    # This object index is specified exactly and we want to collapse the
    # batch_size axis, i.e. return a TensorTrain instead of a TensorTrainBatch.
    do_collapse_batch_dim = isinstance(slice_spec[0], numbers.Number)
    remainder = None
    new_tt_cores = []
    for core_idx in range(self.ndims()):
      curr_core = self.tt_cores[core_idx]
      if self.is_tt_matrix():
        raise NotImplementedError
      else:
        sliced_core = curr_core[slice_spec[0], :, slice_spec[core_idx + 1], :]
        do_collapse_curr_dim = isinstance(slice_spec[core_idx + 1],
                                          numbers.Number)
        if do_collapse_curr_dim:
          # This index is specified exactly and we want to collapse this axis.
          if remainder is None:
            remainder = sliced_core
          else:
            if do_collapse_batch_dim:
              remainder = tf.einsum('ab,bd->ad', remainder, sliced_core)
            else:
              remainder = tf.einsum('oab,obd->oad', remainder, sliced_core)
        else:
          if remainder is not None:
            # Add reminder from the previous collapsed cores to the current
            # core.
            if do_collapse_batch_dim:
              sliced_core = tf.einsum('ab,bid->aid', remainder, sliced_core)
            else:
              sliced_core = tf.einsum('oab,obid->oaid', remainder,
                                      sliced_core)
            remainder = None
          new_tt_cores.append(sliced_core)

    if remainder is not None:
      # The reminder obtained from collapsing the last cores.
      if do_collapse_batch_dim:
        new_tt_cores[-1] = tf.einsum('aib,bd->aid', new_tt_cores[-1],
                                     remainder)

      else:
        new_tt_cores[-1] = tf.einsum('oaib,obd->oaid', new_tt_cores[-1],
                                     remainder)
      remainder = None
    # TODO: infer the output ranks and shape.
    if do_collapse_batch_dim:
      return TensorTrain(new_tt_cores)
    else:
      return TensorTrainBatch(new_tt_cores)

  def __getitem__(self, slice_spec):
    """Basic indexing, returns a `TensorTrainBatch` with the specified region.

    Examples:
      >>> a = t3f.random_tensor_batch((2, 3, 4), batch_size=5)
      >>> a[1:3, :, :, :]
      is a 3D TensorTrainBatch 2 x 3 x 4 with batch_size = 2.
      >>> a[1:3]
      the same as above, a 3D TensorTrainBatch 2 x 3 x 4 with batch_size = 2.
      >>> a[1, :, :, :]
      is a 3D TensorTrain 2 x 3 x 4.
      >>> a[1]
      the same as above, a 3D TensorTrain 2 x 3 x 4.
      >>> a[1:3, :, 1, :]
      is a 2D TensorTrainBatch 2 x 4 with batch_size = 2.
      >>> a[1, :, 1, :]
      is a 2D TensorTrain 2 x 4.

    Returns:
      `TensorTrainBatch` or `TensorTrain` depending on whether the first
      (batch) dim was specified as a range or as a number.
    """
    new_tt_cores = []
    slice_only_batch_dim = isinstance(slice_spec, slice) or \
                           isinstance(slice_spec, numbers.Number)

    if slice_only_batch_dim:
      # Indexing only for the batch_size axis, e.g. a[1:3].
      return self._batch_dim_getitem(slice_spec)
    elif len(slice_spec) == self.ndims() + 1:
      return self._full_getitem(slice_spec)
    else:
      raise ValueError('TensorTrainBatch.__getitem__: wrong number of '
                       'dimensions, expected 1 or %d, got %d' %
                       (self.ndims() + 1, len(slice_spec)))


def _are_batch_tt_cores_valid(tt_cores, shape, tt_ranks, batch_size):
  """Check if dimensions of the TT-cores are consistent and the dtypes coincide.

  Args:
    tt_cores: a tuple of `Tensor` objects
    shape: An np.array, a tf.TensorShape (for tensors), a tuple of
      tf.TensorShapes (for TT-matrices or tensors), or None
    tt_ranks: An np.array or a tf.TensorShape of length len(tt_cores)+1.
    batch_size: a number or None

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
      if batch_size is not None and curr_core_shape[0].value is not None:
        if curr_core_shape[0].value != batch_size:
          # The TT-cores are not aligned with the given batch_size.
          return False
      if shape is not None:
        for i in range(len(shape)):
          if curr_core_shape[i + 2] != shape[i][core_idx]:
            # The TT-cores are not aligned with the given shape.
            return False
      if core_idx >= 1:
        prev_core_shape = tt_cores[core_idx - 1].get_shape()
        if curr_core_shape[1] != prev_core_shape[-1]:
          # TT-ranks are inconsistent.
          return False
      if tt_ranks is not None:
        if curr_core_shape[1] != tt_ranks[core_idx]:
          # The TT-ranks are not aligned with the TT-cores shape.
          return False
        if curr_core_shape[-1] != tt_ranks[core_idx + 1]:
          # The TT-ranks are not aligned with the TT-cores shape.
          return False
    if tt_cores[0].get_shape()[1] != 1 or tt_cores[-1].get_shape()[-1] != 1:
      # The first or the last rank is not 1.
      return False
  except ValueError:
    # The shape of the TT-cores is undetermined, can not validate it.
    pass
  return True


def _infer_batch_raw_shape(tt_cores):
  """Tries to infer the (static) raw shape from the TT-cores."""
  num_dims = len(tt_cores)
  num_tensor_shapes = len(tt_cores[0].get_shape().as_list()) - 3
  raw_shape = [[] for _ in range(num_tensor_shapes)]
  for dim in range(num_dims):
    curr_core_shape = tt_cores[dim].get_shape()
    for i in range(num_tensor_shapes):
      raw_shape[i].append(curr_core_shape[i + 2])
  for i in range(num_tensor_shapes):
    raw_shape[i] = tf.TensorShape(raw_shape[i])
  return tuple(raw_shape)


def _infer_batch_tt_ranks(tt_cores):
  """Tries to infer the (static) raw shape from the TT-cores."""
  tt_ranks = []
  for i in range(len(tt_cores)):
    tt_ranks.append(tt_cores[i].get_shape()[1])
  tt_ranks.append(tt_cores[-1].get_shape()[-1])
  return tf.TensorShape(tt_ranks)
