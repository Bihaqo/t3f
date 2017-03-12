import numpy as np
import tensorflow as tf

import utils


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


if __name__ == "__main__":
  tf.test.main()
