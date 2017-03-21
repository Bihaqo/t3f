import tensorflow as tf

import ops
import initializers
import riemannian


class RiemannianTest(tf.test.TestCase):

  def testProject(self):
    tens = initializers.random_tensor((2, 3, 4))
    # Projection of X into the tangent space of itself is X: P_x(x) = x.
    proj = riemannian.project(tens, tens)
    with self.test_session() as sess:
      actual_val, desired_val = sess.run((ops.full(proj), ops.full(tens)))
      self.assertAllClose(desired_val, actual_val)

if __name__ == "__main__":
  tf.test.main()
