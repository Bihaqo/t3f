import tensorflow as tf

import tensor_train
from initializers import tt_rand_tensor

def get_tt_variable(name,
                    shape=None,
                    rank=2,
                    dtype=None,
                    initializer=None,
                    regularizer=None,
                    trainable=True,
                    collections=None,
                    caching_device=None,
                    validate_shape=True):
  # Returns TensorTrain object with tf.Variables as the TT-cores.
  if initializer is None:
    initializer = tt_rand_tensor(shape, rank)

  num_dims = initializer.ndims()
  variable_cores = []
  with tf.name_scope(name):
    for i in range(num_dims):
      curr_core_var = tf.get_variable('core_%d' % i,
                                      initializer=initializer.tt_cores[i],
                                      dtype=dtype, trainable=trainable,
                                      collections=collections,
                                      caching_device=caching_device,
                                      validate_shape=validate_shape)
      variable_cores.append(curr_core_var)

  return tensor_train.TensorTrain(variable_cores, convert_to_tensors=False)

