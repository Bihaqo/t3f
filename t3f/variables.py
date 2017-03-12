import tensorflow as tf

import tensor_train


def get_variable(name,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 validate_shape=True):
  """Returns TensorTrain object with tf.Variables as the TT-cores.

  Args:
    name: The name of the new or existing TensorTrain variable.
      Used to name the TT-cores.
    dtype: Type of the new or existing TensorTrain variable TT-cores (defaults
      to DT_FLOAT).
    initializer: Initializer for the variable if one is created.
    regularizer: A (TensorTrain -> Tensor or None) function; the result of
      applying it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If True also add the variable to the graph collection
      GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
    collections:  List of graph collections keys to add the Variables
      (underlying TT-cores). Defaults to [GraphKeys.GLOBAL_VARIABLES]
      (see tf.Variable).
    caching_device: Optional device string or function describing where
      the Variable should be cached for reading. Defaults to the Variable's
      device. If not None, caches on another device. Typical use is to cache
      on the device where the Ops using the Variable reside, to deduplicate
      copying through Switch and other conditional statements.
    validate_shape: If False, allows the variable to be initialized with a value
      of unknown shape. If True, the default, the shape of initial_value must be
      known.

  Returns:
    The created or existing `TensorTrain` object with tf.Variables TT-cores.

  Raises:
    `ValueError`: when creating a new variable and shape is not declared, when
      violating reuse during variable creation, or when initializer dtype and
      dtype don't match. Reuse is set inside variable_scope.
  """
  # TODO: support validate shape: check that the tensor dimensions are correct,
  # but ignore the ranks.
  # TODO: add validate ranks flag.
  variable_cores = []
  if initializer is None:
    # Find an existing variable in the collection.
    path = tf.get_variable_scope().name
    if path != '' and path[-1] != '/':
      path += '/'
    path += name

    found_v = None
    for v in tf.get_collection('TensorTrainVariables'):
      if v.name == path:
        found_v = v
        break
    if found_v is None:
      raise ValueError('ValueError: Variable %s does not exist, or was not '
                       'created with t3f.get_tt_variable(). Did you mean to '
                       'set reuse=None in VarScope?' % name)
    with tf.variable_scope(name):
      # Try to get the first core through tf.get_variable to check that we don't
      # violate reuse: it will raise a ValueError otherwise.
      tf.get_variable('core_0')
    return found_v
  else:
    # Create new variable.
    with tf.variable_scope(name):
      num_dims = initializer.ndims()
      for i in range(num_dims):
        curr_core_var = tf.get_variable('core_%d' % i,
                                        initializer=initializer.tt_cores[i],
                                        dtype=dtype, trainable=trainable,
                                        collections=collections,
                                        caching_device=caching_device)
        variable_cores.append(curr_core_var)
    v = tensor_train.TensorTrain(variable_cores, initializer.get_raw_shape(),
                                 initializer.get_tt_ranks(),
                                 convert_to_tensors=False)

    # Add the create TensorTrain object into a collection so that we can
    # retrieve it in the future by get_tt_variable('name').
    tf.add_to_collection('TensorTrainVariables', v)

    # Run the regularizer if requested and save the resulting loss.
    if regularizer:
      with tf.name_scope(name + "/Regularizer/"):
        loss = regularizer(v)
      if loss is not None:
        tf.logging.vlog(1, "Applied regularizer to %s and added the result %s "
                        "to REGULARIZATION_LOSSES.", v.name, loss.name)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
    return v


def assign(ref, value, validate_shape=None, use_locking=None, name=None):
  new_cores = []
  if name is None:
    name = ''
  with tf.variable_scope(name):
    for i in range(ref.ndims()):
      new_cores.append(tf.assign(ref.tt_cores[i], value.tt_cores[i],
                                 use_locking=use_locking))
  return tensor_train.TensorTrain(new_cores, value.get_raw_shape(),
                                  value.get_tt_ranks(),
                                  convert_to_tensors=False)
