import numpy as np
import tensorflow.compat.v1 as tf

from t3f import utils


class UtilsTest(tf.test.TestCase):

  def testUnravelIndex(self):
    with self.test_session():
      # 2D.
      shape = (7, 6)
      linear_idx = [22, 41, 37]
      desired = [[3, 4], [6, 5], [6, 1]]
      actual = utils.unravel_index(linear_idx, shape)
      self.assertAllEqual(desired, actual.eval())
      # 3D.
      shape = (2, 3, 4)
      linear_idx = [19, 17, 0, 23]
      desired = [[1, 1, 3], [1, 1, 1], [0, 0, 0], [1, 2, 3]]
      actual = utils.unravel_index(linear_idx, shape)
      self.assertAllEqual(desired, actual.eval())

  def testReplaceTfSvdWithNpSvd(self):
    with self.test_session() as sess:
      mat = tf.constant([[3., 4], [5, 6]])
      desired = sess.run(tf.svd(mat))
      utils.replace_tf_svd_with_np_svd()
      actual = sess.run(tf.svd(mat))
      self.assertAllClose(actual[0], desired[0])
      self.assertAllClose(np.abs(np.dot(actual[1].T, desired[1])), np.eye(2))
      self.assertAllClose(np.abs(np.dot(actual[2].T, desired[2])), np.eye(2))


if __name__ == "__main__":
  tf.test.main()
