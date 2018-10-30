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

    def func1(x):
      return 0.5 * ops.flat_inner(x, w) ** 2

    actual1 = ops.full(autodiff.gradients(func1, x))
    desired1 = ops.full(ops.flat_inner(x, w) * riemannian.project(w, x))
    with self.test_session() as sess:
      actual1_v, desired1_v = sess.run([actual1, desired1])
      np.testing.assert_allclose(actual1_v, desired1_v, rtol=1e-4)

    def func2(x):
      return ops.quadratic_form(A, x, x)

    actual2 = ops.full(autodiff.gradients(func2, x))
    grad = ops.matmul(ops.transpose(A) + A, x)
    desired2 = ops.full(riemannian.project(grad, x))
    with self.test_session() as sess:
      actual_v2, desired_v2 = sess.run([actual2, desired2])
      np.testing.assert_allclose(actual_v2, desired_v2, rtol=1e-4)

  def testHessianVectorProduct(self):
    w = initializers.random_matrix(([5] * 3, None))
    A = initializers.random_matrix(([5] * 3, [5] * 3))
    x = initializers.random_matrix(([5] * 3, None))
    z = initializers.random_matrix(([5] * 3, None))
    projected_vector = ops.full(riemannian.project(z, x))

    def func1(x):
      return 0.5 * ops.flat_inner(x, w) ** 2
    # Grad: <x, w> w
    # Hessian: w w.T
    # Hessian by vector: w <w, P_x z>

    actual1 = ops.full(autodiff.hessian_vector_product(func1, x, z))
    projected_z = riemannian.project(z, x)
    desired1 = riemannian.project(ops.flat_inner(projected_z, w) * w, x)
    desired1 = ops.full(desired1)
    with self.test_session() as sess:
      actual1_v, desired1_v = sess.run([actual1, desired1])
      np.testing.assert_allclose(actual1_v, desired1_v, rtol=1e-4)

    def func2(x):
      return ops.quadratic_form(A, x, x)
    # Hessian of <x, Ax> is A + A.T
    actual2 = ops.full(autodiff.hessian_vector_product(func2, x, z))
    projected_vector = riemannian.project(z, x)
    hessian_by_vector = ops.matmul(ops.transpose(A) + A, projected_vector)
    desired2 = ops.full(riemannian.project(hessian_by_vector, x))
    with self.test_session() as sess:
      actual2_v, desired2_v = sess.run([actual2, desired2])
      np.testing.assert_allclose(actual2_v, desired2_v, rtol=1e-3)


if __name__ == "__main__":
  tf.test.main()

