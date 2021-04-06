import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import ops
from t3f import initializers
from t3f import riemannian
from t3f import autodiff


class _AutodiffTest():

  def _TestSingleGradient(self, func, x, desired):
    actual1 = ops.full(autodiff.gradients(func, x, runtime_check=False))
    actual2 = ops.full(autodiff.gradients(func, x, runtime_check=True))

    desired_v, actual1_v, actual2_v = self.evaluate([desired, actual1, actual2])
    self.assertAllClose(desired_v, actual1_v, rtol=1e-4)
    self.assertAllClose(desired_v, actual2_v, rtol=1e-4)

  def testGradients(self):
    w = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)
    A = initializers.random_matrix(([5] * 3, [5] * 3), dtype=self.dtype)
    x = initializers.random_matrix(([5] * 3, None), dtype=self.dtype)

    def func1(x):
      return 0.5 * ops.flat_inner(x, w) ** 2
    desired1 = ops.full(riemannian.project(w, x) * ops.flat_inner(x, w))

    self._TestSingleGradient(func1, x, desired1)

    def func2(x):
      return ops.bilinear_form(A, x, x)
    grad = ops.matmul(ops.transpose(A) + A, x)
    desired2 = ops.full(riemannian.project(grad, x))
    self._TestSingleGradient(func2, x, desired2)

    def func3(x):
      # A function which is not invariant to different representations of the
      # same tensor, i.e. it does not even have a Riemannian gradient.
      return tf.add_n([tf.reduce_sum(c) for c in x.tt_cores]) ** 2
    with self.assertRaises(tf.errors.InvalidArgumentError):
      actual3 = ops.full(autodiff.gradients(func3, x))
      self.evaluate(actual3)

  def _TestSingleHessianByVector(self, func, x, z, desired):
    actual1 = ops.full(autodiff.hessian_vector_product(
        func, x, z, runtime_check=False))
    actual2 = ops.full(autodiff.hessian_vector_product(func, x, z,
        runtime_check=True))

    desired_v, actual1_v, actual2_v = self.evaluate([desired, actual1, actual2])
    self.assertAllClose(desired_v, actual1_v, rtol=1e-4)
    self.assertAllClose(desired_v, actual2_v, rtol=1e-4)

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
    desired1 = riemannian.project(w * ops.flat_inner(projected_vector, w), x)
    desired1 = ops.full(desired1)
    self._TestSingleHessianByVector(func1, x, z, desired1)

    def func2(x):
      return ops.bilinear_form(A, x, x)
    # Hessian of <x, Ax> is A + A.T
    hessian_by_vector = ops.matmul(ops.transpose(A) + A, projected_vector)
    desired2 = ops.full(riemannian.project(hessian_by_vector, x))
    self._TestSingleHessianByVector(func1, x, z, desired1)

    def func3(x):
      # A function which is not invariant to different representations of the
      # same tensor, i.e. it does not even have a Riemannian gradient or
      # hessian.
      return tf.add_n([tf.reduce_sum(c) for c in x.tt_cores]) ** 2
    with self.assertRaises(tf.errors.InvalidArgumentError):
      actual3 = ops.full(autodiff.hessian_vector_product(func3, x, z))
      self.evaluate(actual3)


class AutodiffTestFloat32(tf.test.TestCase, _AutodiffTest):
  dtype = tf.float32


class AutodiffTestFloat64(tf.test.TestCase, _AutodiffTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
