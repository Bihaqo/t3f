import numpy as np
import tensorflow as tf

from t3f import ops
from t3f import initializers
from t3f import variables
from t3f import riemannian
from t3f import autodiff


class _AutodiffTest():

  def testGradients(self):
    w = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    A = initializers.random_matrix(([5] * 3, [5] * 3), dtype=self.dtype)
    x = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    z = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)

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

    def func3(x):
      # A function which is not invariant to different representations of the
      # same tensor, i.e. it does not even have a Riemannian gradient.
      return tf.add_n([tf.reduce_sum(c) for c in x.tt_cores]) ** 2
    actual3 = ops.full(autodiff.gradients(func3, x))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      with self.test_session() as sess:
        sess.run(actual3)


  def testHessianVectorProduct(self):
    w = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    A = initializers.random_matrix(([5] * 3, [5] * 3), dtype=self.dtype)
    x = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    z = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    projected_vector = riemannian.project(z, x)

    def func1(x):
      return 0.5 * ops.flat_inner(x, w) ** 2
    # Grad: <x, w> w
    # Hessian: w w.T
    # Hessian by vector: w <w, P_x z>

    actual1 = ops.full(autodiff.hessian_vector_product(func1, x, z))
    desired1 = riemannian.project(ops.flat_inner(projected_vector, w) * w, x)
    desired1 = ops.full(desired1)
    with self.test_session() as sess:
      actual1_v, desired1_v = sess.run([actual1, desired1])
      np.testing.assert_allclose(actual1_v, desired1_v, rtol=1e-4)

    def func2(x):
      return ops.quadratic_form(A, x, x)
    # Hessian of <x, Ax> is A + A.T
    actual2 = ops.full(autodiff.hessian_vector_product(func2, x, z))
    hessian_by_vector = ops.matmul(ops.transpose(A) + A, projected_vector)
    desired2 = ops.full(riemannian.project(hessian_by_vector, x))
    with self.test_session() as sess:
      actual2_v, desired2_v = sess.run([actual2, desired2])
      np.testing.assert_allclose(actual2_v, desired2_v, rtol=1e-3)

    def func3(x):
      # A function which is not invariant to different representations of the
      # same tensor, i.e. it does not even have a Riemannian gradient or
      # hessian.
      return tf.add_n([tf.reduce_sum(c) for c in x.tt_cores]) ** 2
    actual3 = ops.full(autodiff.hessian_vector_product(func3, x, z))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      with self.test_session() as sess:
        sess.run(actual3)


class AutodiffTestFloat32(tf.test.TestCase, _AutodiffTest):
  dtype = tf.float32


class AutodiffTestFloat64(tf.test.TestCase, _AutodiffTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()

