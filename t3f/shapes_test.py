import tensorflow as tf

from t3f import initializers
from t3f import shapes


class ShapesTest(tf.test.TestCase):

  def testLazyShapeOverflow(self):
    large_shape = [10] * 20
    tensor = initializers.random_matrix_batch([large_shape, large_shape], batch_size=5)
    self.assertAllEqual([5, 10 ** 20, 10 ** 20], shapes.lazy_shape(tensor))


if __name__ == "__main__":
  tf.test.main()
