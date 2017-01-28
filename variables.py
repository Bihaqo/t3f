import re
import tensorflow as tf

import tensor_train
from initializers import tt_rand_tensor

def get_tt_variable(name,
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
    regularizer: Not supported yet.
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
  # TODO: support regularizer (a TensorTrain -> Tensor function).
  # TODO: Provide basic regularizers (like apply_to_cores(func)).

  variable_cores = []
  if initializer is not None:
    with tf.variable_scope(name):
      num_dims = initializer.ndims()
      for i in range(num_dims):
        curr_core_var = tf.get_variable('core_%d' % i,
                                        initializer=initializer.tt_cores[i],
                                        dtype=dtype, trainable=trainable,
                                        collections=collections,
                                        caching_device=caching_device,
                                        validate_shape=validate_shape)
        variable_cores.append(curr_core_var)
  else:
    with tf.variable_scope(name):
      i = 0
      while True:
        try:
          curr_core_var = tf.get_variable('core_%d' % i,
                                          dtype=dtype, trainable=trainable,
                                          collections=collections,
                                          caching_device=caching_device)
          variable_cores.append(curr_core_var)
          i += 1
        except ValueError as e:
          if i == 0:
            # The variable doesn't exist or it does but scope.reuse == False,
            # raise ValueError.
            raise e
          else:
            # We found all the cores, the i-th core doesn't exist.
            break
  return tensor_train.TensorTrain(variable_cores, convert_to_tensors=False)


def assign(ref, value, validate_shape=None, use_locking=None, name=None):
  output = tensor_train.TensorTrain(ref.tt_cores, convert_to_tensors=False)
  new_cores = []
  if name is None:
    name = ''
  with tf.variable_scope(name):
    for i in range(output.ndims()):
      new_cores.append(tf.assign(output.tt_cores[i], value.tt_cores[i],
                                 validate_shape, use_locking))
  output.tt_cores = tuple(new_cores)
  return output
