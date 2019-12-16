import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util
tf.enable_eager_execution()

from t3f import initializers
from t3f import shapes


class _ShapesTest():

  @test_util.run_in_graph_and_eager_modes
  def testLazyShapeOverflow(self):
    large_shape = [10] * 20
    tensor = initializers.random_matrix_batch([large_shape, large_shape],
    										  batch_size=5, dtype=self.dtype)
    self.assertAllEqual([5, 10 ** 20, 10 ** 20], shapes.lazy_shape(tensor))


class ShapesTestFloat32(tf.test.TestCase, _ShapesTest):
  dtype = tf.float32


class ShapesTestFloat64(tf.test.TestCase, _ShapesTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
