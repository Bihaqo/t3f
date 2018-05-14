import numpy as np
import tensorflow as tf

from t3f import ops
from t3f import initializers
from t3f import variables
from t3f import riemannian
from t3f import autodiff


class AutodiffTest(tf.test.TestCase):

  def testGradients(self):
    w = initializers.random_matrix(([5] * 3, None))
    x = initializers.random_matrix(([5] * 3, None))

    def func(x):
      return 0.5 * ops.flat_inner(x, w) ** 2

    actual = ops.full(autodiff.gradients(func, x))
    desired = ops.full(ops.flat_inner(x, w) * riemannian.project(w, x))
    with self.test_session() as sess:
      actual_v, desired_v = sess.run([actual, desired])
      np.testing.assert_allclose(actual_v, desired_v, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()

