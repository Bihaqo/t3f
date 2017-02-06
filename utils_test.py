import numpy as np
import tensorflow as tf

import utils


class UtilsTest(tf.test.TestCase):

  def testUnravelIndex(self):
    with self.test_session():
      shape = (7, 6)
      linear_idx = [22, 41, 37]
      desired = [[3, 6, 6], [4, 5, 1]]
      actual = utils.unravel_index(linear_idx, shape)
      self.assertAllEqual(desired, actual.eval())


if __name__ == "__main__":
  tf.test.main()
