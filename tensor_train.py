import tensorflow as tf


# TODO: check the methods of _TensorLike
class TensorTrain():
  """Represents a Tensor Train object (a TT-tensor or TT-matrix).
  t3f represents a Tensor Train object as a tuple of TT-cores.
  ```
  @@__init__
  @@get_shape
  @@tt_cores
  @@dense_shape
  @@dtype
  @@op
  @@graph
  """

  def __init__(self, tt_cores):
    """Creates a `TensorTrain`.
    Args:
      tt_cores: A tuple of 3d or 4d tensors shape `[r_k-1, n_k, r_k]`.
    Returns:
      A `TensorTrain`.
    """
    tt_cores = tuple(tt_cores)
    with tf.name_scope("TensorTrain", tt_cores):
      # TODO: should we pass as_ref=True because we want to be able to update
      # values later if it is a VariableOp??
      for i in range(len(tt_cores)):
        tt_cores[i] = tf.convert_to_tensor(
            tt_cores[i], name="indices", as_ref=False)
    self._tt_cores = tt_cores

    # TODO: check the cores are valid and all have the same dtype.

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.
    Returns:
      A `TensorShape` object.
    """
    raise NotImplementedError
    # return tensor_util.constant_value_as_shape(self._dense_shape)

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

  @property
  def dense_shape(self):
    """A 1-D Tensor of int64 representing the shape of the dense tensor."""
    return self._dense_shape

  @property
  def graph(self):
    """The `Graph` that contains the tt_cores tensors."""
    return self._tt_cores[0].graph

  def __str__(self):
    raise NotImplementedError
    # return "TensorTrain(indices=%s, values=%s, dense_shape=%s)" % (
    #     self._indices, self._values, self._dense_shape)

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
