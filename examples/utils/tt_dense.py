from keras.engine.topology import Layer
from keras.layers import Activation
import t3f
import tensorflow as tf
import numpy as np

activations = ['relu', 'sigmoid', 'tanh', 'softmax']
inits = ['glorot', 'he', 'lecun']


class TTDense(Layer):
    counter = 0

    def __init__(self, row_dims, column_dims, tt_rank=2, init='glorot',
                 activation='relu', bias=True, bias_init=0.1, **kwargs):
        """Creates a TT-Matrix based Dense layer.

        Args:
            row_dims: an array, shape of the matrix row index
            column_dims: an array, shape of the matrix column index
            tt_rank: a number or an array, desired tt-rank of the TT-Matrix
            init: string specifying initializer for the TT-Matrix. Possible
                values are 'glorot', 'he', 'lecun'.
            activation: string, specifies the activation function. Possible
                values are 'relu', 'sigmoid', 'tanh', 'softmax' and None
            bias: bool, whether to use bias
            bias_init: a number, initialization value of the bias

        Returns:
            Layer object corresponding to multiplication by a TT-Matrix
                followed by addition of a bias and applying
                an elementwise activation

        Raises:
            ValueError if the provided activation or init is unknown
        """
        self.tt_shape = [row_dims, column_dims]
        self.output_dim = np.prod(column_dims)
        self.tt_rank = tt_rank
        self.activation = activation
        self.bias = bias
        self.bias_init = bias_init
        self.init = init
        super(TTDense, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.init == 'glorot':
            initializer = t3f.glorot_initializer(self.tt_shape,
                                                 tt_rank=self.tt_rank)
        elif self.init == 'he':
            initializer = t3f.he_initializer(self.tt_shape,
                                             tt_rank=self.tt_rank)
        elif self.init == 'lecun':
            initializer = t3f.lecun_initializer(self.tt_shape,
                                                tt_rank=self.tt_rank)
        else:
            raise ValueError('Unknown init "%s", only %s are supported'
                             % (self.init, inits))
        name = 'tt_dense_matrix_{}'.format(TTDense.counter)
        self.W = t3f.get_variable(name, initializer=initializer)
        self.b = None
        if self.bias:
            b_name = 'tt_dense_b_{}'.format(TTDense.counter)
            b_init = tf.constant_initializer(self.bias_init)
            self.b = tf.get_variable(b_name, shape=self.output_dim,
                                     initializer=b_init)
        TTDense.counter += 1
        self.trainable_weights = list(self.W.tt_cores)
        if self.b is not None:
            self.trainable_weights.append(self.b)

    def call(self, x):
        if self.bias:
            h = t3f.matmul(x, self.W) + self.b
        else:
            h = t3f.matmul(x, self.W)
        if self.activation is not None:
            if self.activation in activations:
                h = Activation(self.activation)(h)
            else:
                raise ValueError('Unknown activation "%s", only %s and None '
                                 'are supported'
                                 % (self.activation, activations))
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
