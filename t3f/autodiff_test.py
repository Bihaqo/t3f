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
    A = initializers.random_matrix(([5] * 3, [5] * 3))
    x = initializers.random_matrix(([5] * 3, None))
    z = initializers.random_matrix(([5] * 3, None))

    def func(x):
      return 0.5 * ops.flat_inner(x, w) ** 2

    actual = ops.full(autodiff.gradients(func, x))
    desired = ops.full(ops.flat_inner(x, w) * riemannian.project(w, x))
    with self.test_session() as sess:
      actual_v, desired_v = sess.run([actual, desired])
      np.testing.assert_allclose(actual_v, desired_v, rtol=1e-4)

    def func2(x):
      return ops.quadratic_form(A, x, x)

    actual2 = ops.full(autodiff.gradients(func2, x))
    grad = ops.matmul(ops.transpose(A) + A, x)
    desired2 = ops.full(riemannian.project(grad, x))
    with self.test_session() as sess:
      actual_v2, desired_v2 = sess.run([actual2, desired2])
      np.testing.assert_allclose(actual_v2, desired_v2, rtol=1e-4)


if __name__ == "__main__":
  tf.test.main()

