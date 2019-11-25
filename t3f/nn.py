"""Utils for simplifying building neural networks with TT-layers"""

from itertools import count
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
import t3f
import tensorflow.compat.v1 as tf


class KerasDense(Layer):
  _counter = count(0)

  def __init__(self, input_dims, output_dims, tt_rank=2,
               activation=None, use_bias=True, kernel_initializer='glorot',
               bias_initializer=0.1, **kwargs):
    """Creates a TT-Matrix based Dense Keras layer.

    Args:
      input_dims: an array, tensor shape of the matrix row index
      ouput_dims: an array, tensor shape of the matrix column index
      tt_rank: a number or an array, desired tt-rank of the TT-Matrix
      activation: [None] string or None, specifies the activation function.
      use_bias: bool, whether to use bias
      kernel_initializer: string specifying initializer for the TT-Matrix.
          Possible values are 'glorot', 'he', and 'lecun'.
      bias_initializer: a number, initialization value of the bias

    Returns:
      Layer object corresponding to multiplication by a TT-Matrix
          followed by addition of a bias and applying
          an elementwise activation

    Raises:
        ValueError if the provided activation or kernel_initializer is
        unknown.
    """
    self.counter = next(self._counter)
    self.tt_shape = [input_dims, output_dims]
    self.output_dim = np.prod(output_dims)
    self.tt_rank = tt_rank
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    super(KerasDense, self).__init__(**kwargs)

  def build(self, input_shape):
    if self.kernel_initializer == 'glorot':
      initializer = t3f.glorot_initializer(self.tt_shape,
                                           tt_rank=self.tt_rank)
    elif self.kernel_initializer == 'he':
      initializer = t3f.he_initializer(self.tt_shape,
                                       tt_rank=self.tt_rank)
    elif self.kernel_initializer == 'lecun':
      initializer = t3f.lecun_initializer(self.tt_shape,
                                          tt_rank=self.tt_rank)
    else:
      raise ValueError('Unknown kernel_initializer "%s", only "glorot",'
                       '"he", and "lecun"  are supported'
                       % self.kernel_initializer)
    name = 'tt_dense_{}'.format(self.counter)
    with tf.variable_scope(name):
      self.matrix = t3f.get_variable('matrix', initializer=initializer)
      cores_ = []
      for i,  v in enumerate(self.matrix.tt_cores):
        def _initializer(*args, **kwargs):
          return v.initialized_value()
          
        cores_.append(self.add_weight('%d' % i, 
                        shape=v.shape, 
                        trainable=True,
                        dtype=tf.float32,
                        initializer=_initializer
                        ))
      self.matrix = t3f.TensorTrain(cores_)
      self.b = None
      if self.use_bias:
        b_init = tf.constant_initializer(self.bias_initializer)
        self.b = self.add_weight('bias', shape=(self.output_dim,),
                                 initializer=b_init)

  def call(self, x):
    res = t3f.matmul(x, self.matrix)
    if self.use_bias:
      res += self.b
    if self.activation is not None:
      res = Activation(self.activation)(res)
    return res

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)
