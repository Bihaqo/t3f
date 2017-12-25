import numpy as np
import tensorflow as tf

from t3f import initializers
from t3f import ops


class InitializersTest(tf.test.TestCase):

  def testTensorOnesAndZeros(self):
    tt_ones = initializers.tensor_ones([2, 3, 4])
    tt_zeros = initializers.tensor_zeros([2, 3, 4])

    ones_desired = np.ones((2, 3, 4))
    zeros_desired = np.zeros((2, 3, 4))
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertAllClose(tt_zeros_full, zeros_desired)

  def testMatrixOnesAndZeros(self):
    tt_ones = initializers.matrix_ones([[2, 3, 4], [1, 2, 5]])
    tt_zeros = initializers.matrix_zeros([[2, 3, 4], [1, 2, 5]])

    ones_desired = np.ones((24, 10))
    zeros_desired = np.zeros((24, 10))
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertAllClose(tt_zeros_full, zeros_desired)

  def testEye(self):
      tt_eye = initializers.eye([4, 5, 6])
      eye_desired = np.eye(120)
      with self.test_session() as sess:
        eye_full = sess.run(ops.full(tt_eye))
        self.assertAllClose(eye_full, eye_desired)

  def testOnesLikeAndZerosLike(self):
    a = initializers.random_tensor([2, 3, 4])
    b = ops.ones_like(a)
    c = ops.zeros_like(a)
    var_list = [ops.full(b), ops.full(c)]
    with self.test_session() as sess:
      bf, cf = sess.run(var_list)
      self.assertAllClose(bf, np.ones((2, 3, 4)))
      self.assertAllClose(cf, np.zeros((2, 3, 4)))


if __name__ == "__main__":
  tf.test.main()
