import numpy as np
import tensorflow as tf

from t3f import neural


class _NeuralTest():

  def testKerasDense(self):
    # Try to create the layer twice to check that it won't crush saying the
    # variable already exist.
    x = tf.random_normal((20, 28*28))
    layer = neural.KerasDense(input_dims=[7, 4, 7, 4], output_dims=[5, 5, 5, 5])
    layer(x)
    layer = neural.KerasDense(input_dims=[7, 4, 7, 4], output_dims=[5, 5, 5, 5])
    layer(x)


class NeuralTestFloat32(tf.test.TestCase, _NeuralTest):
  dtype = tf.float32


class NeuralTestFloat64(tf.test.TestCase, _NeuralTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
