import numpy as np
import tensorflow as tf


# TODO: check the methods of _TensorLike
class TensorTrainBase(object):
  """An abstract class that represents a collection of Tensor Train cores.
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

  def __init__(self, tt_cores):
    """Creates a `TensorTrainBase`."""
    pass

  def get_raw_shape(self):
    """Get tuple of `TensorShapes` representing the shapes of the underlying TT-tensor.

    Tuple contains one `TensorShape` for TT-tensor and 2 `TensorShapes` for
    TT-matrix

    Returns:
      A tuple of `TensorShape` objects.
    """
    return self._raw_shape

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    raw_shape = self.get_raw_shape()
    if self.is_tt_matrix():
      # TODO: as list is not available if shape is partly known.
      m = np.prod(raw_shape[0].as_list())
      n = np.prod(raw_shape[1].as_list())
      return tf.TensorShape((m, n))
    else:
      return raw_shape[0]

  @property
  def tt_cores(self):
    """A tuple of TT-cores."""
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

  @property
  def graph(self):
    """The `Graph` that contains the tt_cores tensors."""
    # TODO: check in init that the other cores are from the same graph.
    return self.tt_cores[0].graph

  def __str__(self):
    """A string describing the TensorTrain object, its TT-rank and shape."""
    return NotImplementedError

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
    return self._tt_ranks

  def is_tt_matrix(self):
    """Returns True if the TensorTrain object represents a TT-matrix."""
    return len(self.get_raw_shape()) == 2

  def is_variable(self):
    """True if the TensorTrain object is a variable (e.g. is trainable)."""
    return isinstance(self.tt_cores[0], tf.Variable)

  @property
  def op(self):
    """The `Operation` that evaluates all the cores."""
    return tf.group(*[c.op for c in self.tt_cores])

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
    """
    # TODO: implement feed_dict
    if session is None:
      session = tf.get_default_session()
    session.run(self.tt_cores)

  # TODO: do we need this?
  # @staticmethod
  # def _override_operator(operator, func):
  #   _override_helper(SparseTensor, operator, func)

  def __add__(self, other):
    """Returns a TensorTrain corresponding to element-wise sum tt_a + tt_b.

    Supports broadcasting (e.g. you can add TensorTrainBatch and TensorTrain).
    Just calls t3f.add, see its documentation for details.
    """
    # TODO: ugly.
    # We can't import ops in the beginning since it creates cyclic dependencies.
    import ops
    return ops.add(self, other)

  def __mul__(self, other):
    """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b.

    Supports broadcasting (e.g. you can multiply TensorTrainBatch and
    TensorTrain).
    Just calls t3f.multiply, see its documentation for details.
    """
    # TODO: ugly.
    # We can't import ops in the beginning since it creates cyclic dependencies.
    import ops
    return ops.multiply(self, other)

  # To support 'TT * 4' as well as '4 * TT'.
  __rmul__ = __mul__
