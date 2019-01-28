"""Utils for simplifying building neural networks with TT-layers"""

from itertools import count
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Activation
import t3f
import tensorflow as tf
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
from itertools import cycle, islice
from scipy.stats import entropy


MODES = ['ascending', 'descending', 'mixed']
CRITERIONS = ['entropy', 'var']


def _to_list(p):
  res = []
  for k, v in p.items():
      res += [k, ] * v
  return res


def _roundup(n, k):
  return int(np.ceil(n / 10**k)) * 10**k


def _roundrobin(*iterables):
  "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
  # Recipe credited to George Sakkis
  pending = len(iterables)
  nexts = cycle(iter(it).__next__ for it in iterables)
  while pending:
      try:
          for next in nexts:
              yield next()
      except StopIteration:
          pending -= 1
          nexts = cycle(islice(nexts, pending))


def _get_all_factors(n, d=3, mode='ascending'):
  p = _factorint2(n)
  if len(p) < d:
      p = p + [1, ] * (d - len(p))

  if mode == 'ascending':
      def prepr(x):
          return tuple(sorted([np.prod(_) for _ in x]))
  elif mode == 'descending':
      def prepr(x):
          return tuple(sorted([np.prod(_) for _ in x], reverse=True))

  elif mode == 'mixed':
      def prepr(x):
          x = sorted(np.prod(_) for _ in x)
          N = len(x)
          xf, xl = x[:N//2], x[N//2:]
          return tuple(_roundrobin(xf, xl))

  else:
      raise ValueError('Wrong mode specified, only {} are available'.format(MODES))

  raw_factors = multiset_partitions(p, d)
  clean_factors = [prepr(f) for f in raw_factors]
  clean_factors = list(set(clean_factors))
  return clean_factors


def _factorint2(p):
  return _to_list(factorint(p))


def auto_shape(n, d=3, criterion='entropy', mode='ascending'):
  factors = _get_all_factors(n, d=d, mode=mode)
  if criterion == 'entropy':
      weights = [entropy(f) for f in factors]
  elif criterion == 'var':
      weights = [-np.var(f) for f in factors]
  else:
      raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))

  i = np.argmax(weights)
  return list(factors[i])


def suggest_shape(n, d=3, criterion='entropy', mode='ascending'):
  weights = []
  for i in range(len(str(n))):

      n_i = _roundup(n, i)
      if criterion == 'entropy':
          weights.append(entropy(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
      elif criterion == 'var':
          weights.append(-np.var(auto_shape(n_i, d=d, mode=mode, criterion=criterion)))
      else:
          raise ValueError('Wrong criterion specified, only {} are available'.format(CRITERIONS))

  i = np.argmax(weights)
  factors = auto_shape(int(_roundup(n, i)), d=d, mode=mode, criterion=criterion)
  return factors


class KerasDense(Layer):
  _counter = count(0)

  def __init__(self, in_dim=None, out_dim=None, d=None, mode='mixed',
               criterion='entropy',input_dims=None, output_dims=None, tt_rank=8,
               activation=None, use_bias=True, kernel_initializer='glorot',
               bias_initializer=0.1, **kwargs):
    """Creates a TT-Matrix based Dense Keras layer.
    If in_dim, out_dim and d are provided, will determine the (quasi)optimal
    factorizations automatically using 'mode' factorization style, and 'criterion'
    for optimality. Default settings are recommended.
    Otherwise, the desired factorizations has to be specified as input_dims
    and output_dims lists.

    Args:
      in_dim: an int, number of input neurons
      out_dim: an int, number of output neurons
      d: number of factors in shape factorizations
      mode: string, specifies the way of factorizing in_dim and out_dim.
          Possible values are 'ascending', 'descending', 'mixed'.
      criterion: string, specifies the shape optimality criterion.
          Possible values are 'entropy', 'var'.
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
    if in_dim and out_dim and d:
      input_dims = auto_shape(in_dim, d=d, mode=mode, criterion=criterion)
      output_dims = auto_shape(out_dim, d=d, mode=mode, criterion=criterion)

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
      self.b = None
      if self.use_bias:
        b_init = tf.constant_initializer(self.bias_initializer)
        self.b = tf.get_variable('bias', shape=self.output_dim,
                                 initializer=b_init)
    self.trainable_weights = list(self.matrix.tt_cores)
    if self.b is not None:
      self.trainable_weights.append(self.b)

  def call(self, x):
    res = t3f.matmul(x, self.matrix)
    if self.use_bias:
      res += self.b
    if self.activation is not None:
      res = Activation(self.activation)(res)
    return res

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_dim)
